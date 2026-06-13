import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class SAMModelManager:
    _instance = None
    _sam_models = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_sam_model(cls, sam_checkpoint_path, device='cuda'):
        if sam_checkpoint_path is None:
            return None


        if sam_checkpoint_path in cls._sam_models:
            print(f"[SAM-ASE] Reusing existing SAM model from {sam_checkpoint_path}")
            return cls._sam_models[sam_checkpoint_path]


        try:
            from segment_anything import sam_model_registry, SamPredictor


            if 'vit_h' in sam_checkpoint_path or 'huge' in sam_checkpoint_path:
                model_type = 'vit_h'
            elif 'vit_l' in sam_checkpoint_path or 'large' in sam_checkpoint_path:
                model_type = 'vit_l'
            else:
                model_type = 'vit_b'

            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
            sam.to(device)
            sam.eval()


            for param in sam.parameters():
                param.requires_grad = False

            predictor = SamPredictor(sam)


            cls._sam_models[sam_checkpoint_path] = (sam, predictor)

            print(f"[SAM-ASE] Successfully loaded SAM model (Singleton) from {sam_checkpoint_path}")
            print(f"[SAM-ASE] Model type: {model_type.upper()}, Parameters: {sum(p.numel() for p in sam.parameters()) / 1e6:.1f}M")
            print(f"[SAM-ASE] This SAM model will be shared across all SAM-ASE modules")

            return sam, predictor

        except Exception as e:
            print(f"[SAM-ASE] Warning: Failed to load SAM model: {e}")
            print("[SAM-ASE] Will use learnable adapter only")
            return None

    @classmethod
    def clear_cache(cls):
        cls._sam_models.clear()


_sam_manager = SAMModelManager()


def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


class LearnablePromptGenerator(nn.Module):
    def __init__(self, num_prompts=16, input_channels=4, feature_dim=64):
        super().__init__()
        self.num_prompts = num_prompts
        self.input_channels = input_channels


        self.attention_head = nn.Sequential(
            nn.Conv2d(input_channels, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, 1, kernel_size=1),
        )


        self.prompt_encoding = nn.Parameter(torch.randn(num_prompts, 2))


        nn.init.uniform_(self.prompt_encoding, 0.2, 0.8)

    def forward(self, x):
        B, C, H, W = x.shape


        importance_map = self.attention_head(x)
        importance_map = torch.sigmoid(importance_map)


        adaptive_coords = []

        for i in range(self.num_prompts):

            base_coord = self.prompt_encoding[i]


            y = int(base_coord[0].item() * H)
            x_coord = int(base_coord[1].item() * W)


            y = max(0, min(H - 1, y))
            x_coord = max(0, min(W - 1, x_coord))

            adaptive_coords.append([y, x_coord])

        return torch.tensor(adaptive_coords, device=x.device, dtype=torch.float32)


def semantic_neighbor(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x


class SAMFeatureExtractor(nn.Module):
    def __init__(
        self,
        sam_checkpoint_path=None,
        feature_dim=256,
        output_dim=64,
        use_frozen_sam=True,
        use_adapter=True,
        use_learnable_prompts=False,
        num_learnable_prompts=16,
        use_soft_masks=False,
        num_soft_regions=8,
        input_channels=4,
        device='cuda'
    ):
        super().__init__()
        self.use_frozen_sam = use_frozen_sam
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.device = device
        self.sam_checkpoint_path = sam_checkpoint_path
        self.use_learnable_prompts = use_learnable_prompts
        self.use_soft_masks = use_soft_masks
        self.num_soft_regions = num_soft_regions


        self._timing_epoch = -1
        self._timing_probe_limit = 2
        self._timing_probe_count = 0
        self._extract_time_sum = 0.0
        self._extract_time_calls = 0
        self._last_extract_time = 0.0


        self.sam_model = None
        self.predictor = None
        if use_frozen_sam and sam_checkpoint_path is not None:
            sam_model, predictor = _sam_manager.get_sam_model(sam_checkpoint_path, device)
            self.sam_model = sam_model
            self.predictor = predictor

            if self.sam_model is None:
                print("[SAM-ASE] Will use learnable adapter only")


        if use_learnable_prompts:
            self.prompt_generator = LearnablePromptGenerator(
                num_prompts=num_learnable_prompts,
                input_channels=input_channels,
                feature_dim=64
            )
            print(f"[SAM-ASE] Learnable prompt generator enabled (num_prompts={num_learnable_prompts})")


        if use_soft_masks:
            self.soft_mask_head = nn.Sequential(
                nn.Conv2d(feature_dim, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, num_soft_regions, kernel_size=1),
            )
            print(f"[SAM-ASE] Soft mask head enabled (num_regions={num_soft_regions})")


        if use_adapter:
            self.adapter = nn.Sequential(
                nn.Conv2d(feature_dim, output_dim, 1, 1, 0),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_dim, output_dim, 1, 1, 0),
            )


            if use_soft_masks:

                self.mask_embedding = nn.Sequential(
                    nn.Conv2d(num_soft_regions, 32, 3, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, output_dim, 1, 1, 0),
                )
            else:

                self.mask_embedding = nn.Sequential(
                    nn.Conv2d(1, 32, 3, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, output_dim, 1, 1, 0),
                )
        else:
            self.adapter = None
            self.mask_embedding = None
        self._last_prior_bank = None

    def set_timing_epoch(self, epoch):
        if epoch != self._timing_epoch:
            self._timing_epoch = epoch
            self._timing_probe_count = 0
            self._extract_time_sum = 0.0
            self._extract_time_calls = 0
            self._last_extract_time = 0.0

    def _ensure_mask_3d(self, mask_tensor):
        if mask_tensor is None:
            return None
        if mask_tensor.dim() == 4:
            if mask_tensor.shape[1] == 1:
                return mask_tensor.squeeze(1)
            return mask_tensor.mean(dim=1)
        if mask_tensor.dim() == 2:
            return mask_tensor.unsqueeze(0)
        return mask_tensor

    def _normalize_prior_map(self, map_tensor):
        if map_tensor is None:
            return None
        if map_tensor.dim() == 2:
            map_tensor = map_tensor.unsqueeze(0).unsqueeze(0)
        elif map_tensor.dim() == 3:
            map_tensor = map_tensor.unsqueeze(1)
        elif map_tensor.dim() != 4:
            return None

        flat = map_tensor.flatten(2)
        min_val = flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        max_val = flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        return (map_tensor - min_val) / (max_val - min_val + 1e-6)

    def _build_boundary_map_from_regions(self, region_map):
        if region_map is None:
            return None
        dx = torch.abs(region_map[:, :, :, 1:] - region_map[:, :, :, :-1])
        dy = torch.abs(region_map[:, :, 1:, :] - region_map[:, :, :-1, :])
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        return self._normalize_prior_map(dx + dy)

    def _build_boundary_map_from_soft_masks(self, soft_masks):
        if soft_masks is None:
            return None
        dx = torch.abs(soft_masks[:, :, :, 1:] - soft_masks[:, :, :, :-1])
        dy = torch.abs(soft_masks[:, :, 1:, :] - soft_masks[:, :, :-1, :])
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        boundary_map = dx.mean(dim=1, keepdim=True) + dy.mean(dim=1, keepdim=True)
        return self._normalize_prior_map(boundary_map)

    def _build_prompt_strength_map(self, semantic_prompt, H, W):
        if semantic_prompt is None:
            return None
        B, L, C = semantic_prompt.shape
        if L != H * W:
            return None
        prompt_2d = semantic_prompt.permute(0, 2, 1).reshape(B, C, H, W)
        prompt_strength = torch.linalg.norm(prompt_2d, dim=1, keepdim=True)
        return self._normalize_prior_map(prompt_strength)

    def _build_prior_bank(self, semantic_prompt, semantic_masks, H, W, soft_masks=None):
        if semantic_prompt is None or semantic_masks is None:
            return None

        semantic_masks = self._ensure_mask_3d(semantic_masks)
        if semantic_masks is None:
            return None

        region_map = self._normalize_prior_map(semantic_masks)
        prompt_strength_map = self._build_prompt_strength_map(semantic_prompt, H, W)

        if soft_masks is not None:
            boundary_map = self._build_boundary_map_from_soft_masks(soft_masks)
            confidence_map = self._normalize_prior_map(soft_masks.max(dim=1, keepdim=True)[0])
        else:
            boundary_map = self._build_boundary_map_from_regions(region_map)
            confidence_map = prompt_strength_map
            if confidence_map is None and region_map is not None:
                confidence_map = torch.ones_like(region_map)

        return {
            'region_map': region_map,
            'boundary_map': boundary_map,
            'confidence_map': confidence_map,
            'prompt_strength_map': prompt_strength_map,
        }

    def extract_sam_features(self, x):
        if self.sam_model is None:

            return None, None

        B, C, H, W = x.shape
        L = H * W


        sam_features, sam_masks = None, None
        try:
            B_inner, C_inner, H_inner, W_inner = x.shape
            features_list = []
            masks_list = []


            for b in range(B_inner):


                img_tensor = x[b].detach()


                if C_inner >= 3:
                    img_3ch = img_tensor[:3, :, :]
                else:

                    img_3ch = img_tensor[0:1, :, :].repeat(3, 1, 1)


                img = img_3ch.permute(1, 2, 0).cpu().numpy()


                img_min = img.min()
                img_max = img.max()
                if img_max > img_min:
                    img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img = np.zeros_like(img, dtype=np.uint8)


                self.predictor.set_image(img)


                if self.use_learnable_prompts and hasattr(self, 'prompt_generator'):

                    points_tensor = self.prompt_generator(x[b:b+1])
                    points = points_tensor.cpu().numpy().tolist()
                else:

                    points = []
                    for i in range(1, 4):
                        for j in range(1, 4):
                            points.append([i * H_inner // 4, j * W_inner // 4])


                masks, scores, _ = self.predictor.predict(
                    point_coords=np.array(points),
                    point_labels=np.ones(len(points)),
                    multimask_output=True,
                )


                features = self.predictor.features


                best_idx = scores.argmax()
                best_mask = masks[best_idx]


                if isinstance(features, np.ndarray):
                    features_tensor = torch.from_numpy(features).float().to(self.device)
                else:

                    features_tensor = features.float().to(self.device)

                if isinstance(best_mask, np.ndarray):
                    mask_tensor = torch.from_numpy(best_mask).float().to(self.device)
                else:
                    mask_tensor = best_mask.float().to(self.device)

                features_list.append(features_tensor)
                masks_list.append(mask_tensor)


            features_batch = torch.cat(features_list, dim=0)
            masks_stacked = torch.stack(masks_list, dim=0)
            masks_batch = masks_stacked.unsqueeze(1)

            sam_features = features_batch
            sam_masks = masks_batch.squeeze(1)

        except Exception as e:
            print(f"[SAM-ASE] Warning: SAM feature extraction failed: {e}")
            sam_features = None
            sam_masks = None


        return self.compose_semantic_from_raw(
            x,
            sam_features=sam_features,
            sam_masks=sam_masks,
        )


        if sam_features is not None and self.adapter is not None:

            sam_features = F.interpolate(sam_features, size=(H, W), mode='bilinear', align_corners=True)
            semantic_prompt = self.adapter(sam_features)


            if self.use_soft_masks and hasattr(self, 'soft_mask_head'):

                soft_masks = self.soft_mask_head(sam_features)
                soft_masks = F.softmax(soft_masks, dim=1)


                if self.mask_embedding is not None:
                    mask_embed = self.mask_embedding(soft_masks)
                    semantic_prompt = semantic_prompt + mask_embed


                semantic_prompt = semantic_prompt.reshape(B, self.output_dim, H * W).permute(0, 2, 1)


                semantic_masks = soft_masks.argmax(dim=1).float()

                semantic_masks = semantic_masks / (self.num_soft_regions - 1)


            elif sam_masks is not None and self.mask_embedding is not None:

                if sam_masks.dim() == 4:

                    sam_masks = sam_masks.mean(dim=1, keepdim=False)
                elif sam_masks.dim() == 3 and sam_masks.shape[0] == 1 and B > 1:

                    sam_masks = sam_masks.repeat(B, 1, 1)


                if sam_masks.dim() == 3:
                    sam_masks = sam_masks.unsqueeze(1)

                mask_embed = self.mask_embedding(sam_masks)
                semantic_prompt = semantic_prompt + mask_embed


            semantic_prompt = semantic_prompt.reshape(B, self.output_dim, L).permute(0, 2, 1)


            if sam_masks is not None:
                semantic_masks = sam_masks
            else:

                semantic_prompt_spatial = semantic_prompt.reshape(B, H, W, self.output_dim).mean(dim=-1)
                threshold = semantic_prompt_spatial.mean()
                semantic_masks = (semantic_prompt_spatial > threshold).float()

        elif self.adapter is not None:


            if C != self.feature_dim:
                if not hasattr(self, 'channel_adapter'):
                    self.channel_adapter = nn.Conv2d(C, self.feature_dim, 1, 1, 0).to(x.device)
                x_adapted = self.channel_adapter(x)
            else:
                x_adapted = x

            semantic_prompt = self.adapter(x_adapted)


            semantic_prompt = semantic_prompt.reshape(B, self.output_dim, L).permute(0, 2, 1)


            semantic_prompt_spatial = semantic_prompt.reshape(B, H, W, self.output_dim).mean(dim=-1)
            threshold = semantic_prompt_spatial.mean()
            semantic_masks = (semantic_prompt_spatial > threshold).float()
        else:

            return None, None

        return semantic_prompt, semantic_masks

    def extract_raw_sam_outputs(self, x):
        if self.sam_model is None:
            return None, None

        sam_features, sam_masks = None, None
        try:
            B_inner, C_inner, H_inner, W_inner = x.shape
            features_list = []
            masks_list = []

            for b in range(B_inner):
                img_tensor = x[b].detach()
                if C_inner >= 3:
                    img_3ch = img_tensor[:3, :, :]
                else:
                    img_3ch = img_tensor[0:1, :, :].repeat(3, 1, 1)

                img = img_3ch.permute(1, 2, 0).cpu().numpy()
                img_min = img.min()
                img_max = img.max()
                if img_max > img_min:
                    img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img = np.zeros_like(img, dtype=np.uint8)

                self.predictor.set_image(img)

                points = []
                for i in range(1, 4):
                    for j in range(1, 4):
                        points.append([i * H_inner // 4, j * W_inner // 4])

                masks, scores, _ = self.predictor.predict(
                    point_coords=np.array(points),
                    point_labels=np.ones(len(points)),
                    multimask_output=True,
                )

                features = self.predictor.features
                best_idx = scores.argmax()
                best_mask = masks[best_idx]

                if isinstance(features, np.ndarray):
                    features_tensor = torch.from_numpy(features).float().to(self.device)
                else:
                    features_tensor = features.float().to(self.device)

                if isinstance(best_mask, np.ndarray):
                    mask_tensor = torch.from_numpy(best_mask).float().to(self.device)
                else:
                    mask_tensor = best_mask.float().to(self.device)

                features_list.append(features_tensor)
                masks_list.append(mask_tensor)

            sam_features = torch.cat(features_list, dim=0)
            sam_masks = torch.stack(masks_list, dim=0)

        except Exception as e:
            print(f"[SAM-ASE] Warning: SAM raw feature extraction failed: {e}")
            sam_features = None
            sam_masks = None

        return sam_features, sam_masks

    def compose_semantic_from_raw(self, x, sam_features=None, sam_masks=None):
        B, C, H, W = x.shape
        L = H * W
        soft_masks = None
        self._last_prior_bank = None

        if sam_features is not None:
            if sam_features.dim() == 3:
                sam_features = sam_features.unsqueeze(0)
            sam_features = sam_features.to(x.device).float()

        if sam_masks is not None:
            if sam_masks.dim() == 2:
                sam_masks = sam_masks.unsqueeze(0)
            sam_masks = sam_masks.to(x.device).float()

        if sam_features is not None and self.adapter is not None:
            sam_features = F.interpolate(sam_features, size=(H, W), mode='bilinear', align_corners=True)
            semantic_prompt = self.adapter(sam_features)

            if self.use_soft_masks and hasattr(self, 'soft_mask_head'):
                soft_masks = self.soft_mask_head(sam_features)
                soft_masks = F.softmax(soft_masks, dim=1)

                if self.mask_embedding is not None:
                    mask_embed = self.mask_embedding(soft_masks)
                    semantic_prompt = semantic_prompt + mask_embed

                semantic_prompt = semantic_prompt.reshape(B, self.output_dim, H * W).permute(0, 2, 1)
                semantic_masks = soft_masks.argmax(dim=1).float()
                if self.num_soft_regions > 1:
                    semantic_masks = semantic_masks / (self.num_soft_regions - 1)
            else:
                if sam_masks is not None and self.mask_embedding is not None:
                    if sam_masks.dim() == 4:
                        sam_masks_for_embed = sam_masks
                        semantic_masks = sam_masks.mean(dim=1)
                    else:
                        sam_masks_for_embed = sam_masks.unsqueeze(1)
                        semantic_masks = sam_masks
                    mask_embed = self.mask_embedding(sam_masks_for_embed)
                    semantic_prompt = semantic_prompt + mask_embed
                else:
                    semantic_masks = sam_masks

                semantic_prompt = semantic_prompt.reshape(B, self.output_dim, L).permute(0, 2, 1)

            if semantic_masks is None:
                semantic_prompt_spatial = semantic_prompt.reshape(B, H, W, self.output_dim).mean(dim=-1)
                threshold = semantic_prompt_spatial.mean()
                semantic_masks = (semantic_prompt_spatial > threshold).float()

            semantic_masks = self._ensure_mask_3d(semantic_masks)
            self._last_prior_bank = self._build_prior_bank(
                semantic_prompt,
                semantic_masks,
                H,
                W,
                soft_masks=soft_masks,
            )
            return semantic_prompt, semantic_masks

        if self.adapter is not None:
            if C != self.feature_dim:
                if not hasattr(self, 'channel_adapter'):
                    self.channel_adapter = nn.Conv2d(C, self.feature_dim, 1, 1, 0).to(x.device)
                x_adapted = self.channel_adapter(x)
            else:
                x_adapted = x

            semantic_prompt = self.adapter(x_adapted)
            semantic_prompt = semantic_prompt.reshape(B, self.output_dim, L).permute(0, 2, 1)
            semantic_prompt_spatial = semantic_prompt.reshape(B, H, W, self.output_dim).mean(dim=-1)
            threshold = semantic_prompt_spatial.mean()
            semantic_masks = (semantic_prompt_spatial > threshold).float()
            semantic_masks = self._ensure_mask_3d(semantic_masks)
            self._last_prior_bank = self._build_prior_bank(
                semantic_prompt,
                semantic_masks,
                H,
                W,
                soft_masks=None,
            )
            return semantic_prompt, semantic_masks

        return None, None

    def forward(self, x, cached_raw_features=None, cached_raw_masks=None):
        self._last_prior_bank = None
        use_cached_raw = cached_raw_features is not None and cached_raw_masks is not None

        if use_cached_raw and self.use_learnable_prompts:
            if not hasattr(self, '_cache_prompt_warning_issued'):
                self._cache_prompt_warning_issued = False
            if not self._cache_prompt_warning_issued:
                print("[SAM-ASE] WARNING: offline raw SAM cache is incompatible with learnable prompts; fallback to online SAM")
                self._cache_prompt_warning_issued = True
            use_cached_raw = False

        if not use_cached_raw and x.is_cuda:
            torch.cuda.synchronize(x.device)
        start_t = time.perf_counter()

        if use_cached_raw:
            semantic_prompt, semantic_masks = self.compose_semantic_from_raw(
                x,
                sam_features=cached_raw_features,
                sam_masks=cached_raw_masks,
            )
        else:
            semantic_prompt, semantic_masks = self.extract_sam_features(x)

        if not use_cached_raw and x.is_cuda:
            torch.cuda.synchronize(x.device)
        elapsed = 0.0 if use_cached_raw else (time.perf_counter() - start_t)

        self._last_extract_time = elapsed
        self._extract_time_sum += elapsed
        self._extract_time_calls += 1

        return semantic_prompt, semantic_masks


class SelectiveScanWithSAM(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        expand=2.,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank


        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in [self.x_proj]], dim=0))


        self.dt_projs = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in [self.dt_projs]], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in [self.dt_projs]], dim=0))


        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device)
        A = A[None, :].repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = A_log[None, :, :].repeat(copies, 1, 1)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = D[None, :].repeat(copies, 1)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, sam_prompt):
        B, L, C = x.shape
        K = 1

        xs = x.permute(0, 2, 1).view(B, 1, C, L).contiguous()

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)


        Cs = Cs.float().view(B, K, -1, L) + sam_prompt

        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, sam_prompt, **kwargs):


        if sam_prompt is None:
            B, L, C = x.shape

            sam_prompt_zero = torch.zeros(B, L, self.d_state, dtype=x.dtype, device=x.device)

            sam_prompt_reshaped = sam_prompt_zero.permute(0, 2, 1).contiguous().view(B, 1, self.d_state, L)
            y = self.forward_core(x, sam_prompt_reshaped)
            return y


        b, l, c = sam_prompt.shape
        sam_prompt = sam_prompt.permute(0, 2, 1).contiguous().view(b, 1, c, l)
        y = self.forward_core(x, sam_prompt)
        y = y.permute(0, 2, 1).contiguous()
        return y


class PriorRefinerV1(nn.Module):

    def __init__(self, d_state, hidden_channels=None):
        super().__init__()
        hidden_channels = max(int(hidden_channels or d_state), 8)

        self.in_proj = nn.Conv2d(2, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.mix = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_channels,
        )
        self.out_proj = nn.Conv2d(hidden_channels, d_state, kernel_size=1, stride=1, padding=0)
        self.act = nn.GELU()

        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, region_map, prompt_strength_map):
        x = torch.cat([region_map, prompt_strength_map], dim=1)
        x = self.act(self.in_proj(x))
        x = self.act(self.mix(x))
        return self.out_proj(x)


class SAMASEModule(nn.Module):
    def __init__(self, dim, d_state, input_resolution, sam_prompt_dim=64, mlp_ratio=2., use_sam=True, sam_checkpoint=None,
                 use_learnable_prompts=False, num_learnable_prompts=16, use_soft_masks=False, num_soft_regions=8, input_channels=4,
                 use_structure_guided_sam_ase=False, structure_texture_weight=0.25):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.sam_prompt_dim = sam_prompt_dim
        self.input_resolution = input_resolution
        self.use_sam = use_sam
        self.use_structure_guided_sam_ase = use_structure_guided_sam_ase
        self.structure_texture_weight = structure_texture_weight


        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)


        if use_sam:
            self.sam_extractor = SAMFeatureExtractor(
                sam_checkpoint_path=sam_checkpoint,
                feature_dim=256,
                output_dim=sam_prompt_dim,
                use_frozen_sam=True,
                use_adapter=True,
                use_learnable_prompts=use_learnable_prompts,
                num_learnable_prompts=num_learnable_prompts,
                use_soft_masks=use_soft_masks,
                num_soft_regions=num_soft_regions,
                input_channels=input_channels,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:

            self.sam_extractor = SAMFeatureExtractor(
                sam_checkpoint_path=None,
                feature_dim=dim,
                output_dim=sam_prompt_dim,
                use_frozen_sam=False,
                use_adapter=True,
                use_learnable_prompts=use_learnable_prompts,
                num_learnable_prompts=num_learnable_prompts,
                use_soft_masks=use_soft_masks,
                num_soft_regions=num_soft_regions,
                input_channels=input_channels,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )


        self.selectiveScan = SelectiveScanWithSAM(d_model=hidden, d_state=self.d_state, expand=1)


        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)


        self.in_proj = nn.Sequential(
            nn.Conv2d(self.dim, hidden, 1, 1, 0),
        )


        self.CPE = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
        )


        self.prompt_proj = nn.Sequential(
            nn.Linear(sam_prompt_dim, d_state),
            nn.LayerNorm(d_state),
        )


        self.prompt_to_hidden = nn.Linear(sam_prompt_dim, hidden)
        self.prior_refiner = PriorRefinerV1(d_state=d_state, hidden_channels=max(d_state, 8))
        self.use_sam_prior_c_refiner = False
        self.sam_prior_c_refiner_scale = 0.1
        self.prior_c_refiner_role = 'unknown'
        self.current_sam_prior_bank = None

    def _normalize_spatial_map(self, spatial_map, batch_size, target_hw, device, dtype):
        target_h, target_w = target_hw
        if spatial_map is None:
            return torch.zeros(batch_size, 1, target_h, target_w, device=device, dtype=dtype)

        if spatial_map.dim() == 3:
            spatial_map = spatial_map.unsqueeze(1)
        elif spatial_map.dim() != 4:
            return torch.zeros(batch_size, 1, target_h, target_w, device=device, dtype=dtype)

        if spatial_map.shape[1] != 1:
            spatial_map = spatial_map.mean(dim=1, keepdim=True)

        spatial_map = spatial_map.to(device=device, dtype=dtype)
        if spatial_map.shape[-2:] != (target_h, target_w):
            spatial_map = F.interpolate(spatial_map, size=(target_h, target_w), mode='bilinear', align_corners=False)

        if spatial_map.shape[0] != batch_size:
            return torch.zeros(batch_size, 1, target_h, target_w, device=device, dtype=dtype)

        flat = spatial_map.flatten(1)
        min_val = flat.min(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
        max_val = flat.max(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
        return (spatial_map - min_val) / (max_val - min_val + 1e-6)

    def _build_prior_bias_d_state(self, target_hw, batch_size, device, dtype):
        if not self.use_sam_prior_c_refiner:
            return None

        prior_bank = getattr(self, 'current_sam_prior_bank', None)
        if not isinstance(prior_bank, dict):
            return None

        region_map = self._normalize_spatial_map(
            prior_bank.get('region_map'),
            batch_size,
            target_hw,
            device,
            dtype,
        )
        prompt_strength_map = self._normalize_spatial_map(
            prior_bank.get('prompt_strength_map'),
            batch_size,
            target_hw,
            device,
            dtype,
        )

        prior_bias = self.prior_refiner(region_map, prompt_strength_map)
        prior_bias = torch.tanh(prior_bias) * float(self.sam_prior_c_refiner_scale)
        return prior_bias.reshape(batch_size, self.d_state, -1).permute(0, 2, 1).contiguous()

    def _resize_wavelet_guidance(self, wavelet_guidance, target_hw, batch_size, device, dtype):
        target_h, target_w = target_hw
        if wavelet_guidance is None:
            return torch.zeros(batch_size, 1, target_h, target_w, device=device, dtype=dtype)

        if wavelet_guidance.dim() == 3:
            guidance_4d = wavelet_guidance.unsqueeze(1)
        elif wavelet_guidance.dim() == 4:
            guidance_4d = wavelet_guidance if wavelet_guidance.shape[1] == 1 else wavelet_guidance.mean(dim=1, keepdim=True)
        else:
            return torch.zeros(batch_size, 1, target_h, target_w, device=device, dtype=dtype)

        guidance_4d = guidance_4d.to(device=device, dtype=dtype)
        if guidance_4d.shape[-2:] != (target_h, target_w):
            guidance_4d = F.interpolate(guidance_4d, size=(target_h, target_w), mode='bilinear', align_corners=False)

        if guidance_4d.shape[0] != batch_size:
            return torch.zeros(batch_size, 1, target_h, target_w, device=device, dtype=dtype)

        flat = guidance_4d.flatten(1)
        min_val = flat.min(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
        max_val = flat.max(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
        return (guidance_4d - min_val) / (max_val - min_val + 1e-6)

    def _build_structure_sort_score(self, semantic_masks, wavelet_guidance, target_hw, batch_size, device, dtype):
        semantic_4d = semantic_masks.unsqueeze(1) if semantic_masks.dim() == 3 else semantic_masks
        semantic_4d = semantic_4d.to(device=device, dtype=dtype)
        if semantic_4d.shape[-2:] != target_hw:
            semantic_4d = F.interpolate(semantic_4d, size=target_hw, mode='bilinear', align_corners=False)
        if semantic_4d.shape[1] != 1:
            semantic_4d = semantic_4d.mean(dim=1, keepdim=True)

        if not self.use_structure_guided_sam_ase:
            return semantic_4d

        texture_map = self._resize_wavelet_guidance(wavelet_guidance, target_hw, batch_size, device, dtype)
        semantic_bins = torch.round(semantic_4d * 255.0) / 255.0
        return semantic_bins + self.structure_texture_weight * texture_map

    def forward(self, x, x_size, sam_prompt=None, semantic_masks=None, original_msi=None, wavelet_guidance=None):
        B, n, C = x.shape
        H, W = x_size


        if sam_prompt is None or semantic_masks is None:

            if original_msi is not None:

                sam_prompt, semantic_masks = self.sam_extractor(original_msi)
                self.current_sam_prior_bank = getattr(self.sam_extractor, '_last_prior_bank', None)
            else:

                print(f"[SAM-ASE] WARNING: original_msi not provided, trying to extract from features (C={C})")

                x_2d = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

                if C != 4:
                    print(f"[SAM-ASE] ERROR: Cannot extract SAM features from {C}-channel features (expected 4 channels)")
                    sam_prompt, semantic_masks = None, None
                else:
                    sam_prompt, semantic_masks = self.sam_extractor(x_2d)
                    self.current_sam_prior_bank = getattr(self.sam_extractor, '_last_prior_bank', None)


        if sam_prompt is None or semantic_masks is None:
            print(f"[SAM-ASE] ERROR: SAM feature extraction returned None!")

            sam_prompt = torch.zeros(B, n, self.sam_prompt_dim, device=x.device, dtype=x.dtype)
            semantic_masks = torch.zeros(B, H, W, device=x.device, dtype=x.dtype)


        _, L_sam, C_sam = sam_prompt.shape

        if L_sam != n:

            H_sam = int(np.sqrt(L_sam))
            W_sam = L_sam // H_sam

            sam_prompt_spatial = sam_prompt.permute(0, 2, 1).reshape(B, C_sam, H_sam, W_sam)
            sam_prompt_resized = F.interpolate(sam_prompt_spatial, size=(H, W), mode='bilinear', align_corners=False)
            sam_prompt = sam_prompt_resized.reshape(B, C_sam, H * W).permute(0, 2, 1)


        if semantic_masks.shape[-2:] != (H, W):

            masks_4d = semantic_masks.unsqueeze(1) if semantic_masks.dim() == 3 else semantic_masks
            semantic_masks_resized = F.interpolate(masks_4d, size=(H, W), mode='bilinear', align_corners=False)
            semantic_masks = semantic_masks_resized.squeeze(1)


        sam_prompt_d_state = self.prompt_proj(sam_prompt)
        prior_bias_d_state = self._build_prior_bias_d_state(
            (H, W),
            B,
            sam_prompt_d_state.device,
            sam_prompt_d_state.dtype,
        )
        if prior_bias_d_state is not None:
            sam_prompt_d_state = sam_prompt_d_state + prior_bias_d_state


        sam_prompt_hidden = self.prompt_to_hidden(sam_prompt)


        structure_score = self._build_structure_sort_score(
            semantic_masks,
            wavelet_guidance,
            (H, W),
            B,
            x.device,
            x.dtype,
        )
        semantic_masks_flat = structure_score.view(B, H * W)


        _, semantic_indices = torch.sort(semantic_masks_flat, dim=-1, stable=True)
        semantic_indices_reverse = index_reverse(semantic_indices)


        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.in_proj(x)
        x = x * torch.sigmoid(self.CPE(x))
        cc = x.shape[1]
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)


        semantic_x = semantic_neighbor(x, semantic_indices)


        y = self.selectiveScan(semantic_x, sam_prompt_d_state)
        y = self.out_proj(self.out_norm(y))


        x = semantic_neighbor(y, semantic_indices_reverse)

        return x


class SAMASEMambaBlock(nn.Module):
    def __init__(self, d_model, sam_prompt_dim=64, modal_type='single', d_state=8, use_sam=True, sam_checkpoint=None,
                 use_wavelet=False, use_learnable_prompts=False, num_learnable_prompts=16,
                 use_soft_masks=False, num_soft_regions=8, input_channels=4,
                 use_structure_guided_sam_ase=False, structure_texture_weight=0.25):
        super().__init__()
        self.d_model = d_model
        self.modal_type = modal_type
        self.use_wavelet = use_wavelet
        self.use_sam = use_sam


        self.sam_ase_module = SAMASEModule(
            dim=d_model,
            d_state=d_state,
            input_resolution=(64, 64),
            sam_prompt_dim=sam_prompt_dim,
            use_sam=use_sam,
            sam_checkpoint=sam_checkpoint,
            use_learnable_prompts=use_learnable_prompts,
            num_learnable_prompts=num_learnable_prompts,
            use_soft_masks=use_soft_masks,
            num_soft_regions=num_soft_regions,
            input_channels=input_channels
        )


        self.out_proj = nn.Linear(d_model, d_model)
        self.sam_ase_module.use_structure_guided_sam_ase = use_structure_guided_sam_ase
        self.sam_ase_module.structure_texture_weight = structure_texture_weight
        self.prior_c_refiner_role = 'unknown'
        self.sam_ase_module.prior_c_refiner_role = 'unknown'
        self.current_sam_prior_bank = None


    def forward(self, x, extra_emb=None, sam_prompt=None, semantic_masks=None, original_msi=None, wavelet_guidance=None):
        B, L, C = x.shape


        import math
        H = W = int(round(math.sqrt(L)))

        if H * W != L:

            found = False
            for h in range(int(math.sqrt(L)), 0, -1):
                if L % h == 0:
                    H = h
                    W = L // h
                    found = True
                    break

            if not found:
                H = int(math.sqrt(L))
                W = L // H
                if H * W < L:
                    W += 1

        x_size = (H, W)


        self.sam_ase_module.current_sam_prior_bank = getattr(self, 'current_sam_prior_bank', None)
        output = self.sam_ase_module(x, x_size, sam_prompt, semantic_masks, original_msi, wavelet_guidance)


        if extra_emb is not None:
            output = output + extra_emb


        output = self.out_proj(output)


        return output
