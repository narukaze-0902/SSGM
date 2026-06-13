import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mamba_ssm.modules.mamba_simple import Mamba


from model.sam_ase_mamba import (
    SAMFeatureExtractor,
    SelectiveScanWithSAM,
    LearnablePromptGenerator
)


from model.haar_dwt import HaarDWT, HaarIDWT, SimpleDWT, SimpleIDWT


class SparseMambaScan(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.mamba = Mamba(d_model, expand=1, d_state=d_state, bimamba_type='v2',
                         if_devide_out=True, use_norm=True)

    def forward(self, features, mask, sam_prompt=None):
        B, C, H, W = features.shape


        features_flat = features.flatten(2)
        mask_flat = mask.flatten(1)


        active_indices_list = []
        num_active_per_batch = []

        for b in range(B):
            active_mask_b = mask_flat[b] > 0
            num_active = active_mask_b.sum().item()
            num_active_per_batch.append(num_active)

            if num_active > 0:
                active_indices = active_mask_b.nonzero(as_tuple=False).squeeze(-1)
                active_indices_list.append(active_indices)
            else:

                active_indices_list.append(torch.empty(0, dtype=torch.long, device=features.device))


        max_active = max(num_active_per_batch) if num_active_per_batch else 1


        if max_active == 0:
            print(f"[SparseMambaScan] Warning: No active tokens in entire batch, using dense fallback")

            return torch.zeros_like(features)


        active_tokens_list = []
        for b in range(B):
            if num_active_per_batch[b] > 0:
                active_indices = active_indices_list[b]
                active_tokens = features_flat[b, :, active_indices].T
            else:

                active_tokens = torch.zeros(0, C, device=features.device)


            if active_tokens.shape[0] < max_active and active_tokens.shape[0] > 0:
                padding = max_active - active_tokens.shape[0]
                active_tokens = F.pad(active_tokens, (0, 0, 0, padding))
            elif active_tokens.shape[0] == 0:

                active_tokens = torch.zeros(max_active, C, device=features.device)

            active_tokens_list.append(active_tokens)


        active_tokens = torch.stack(active_tokens_list, dim=0)


        output_tokens = self.mamba(active_tokens)


        output_flat = torch.zeros_like(features_flat)

        for b in range(B):
            num_active = num_active_per_batch[b]
            if num_active > 0:
                active_indices = active_indices_list[b]
                output_flat[b, :, active_indices] = output_tokens[b, :num_active].T


        output = output_flat.reshape(B, C, H, W)

        return output


class DenseMambaScan(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.mamba = Mamba(d_model, expand=1, d_state=d_state, bimamba_type='v2',
                         if_devide_out=True, use_norm=True)

    def forward(self, features, sam_prompt=None):
        B, C, H, W = features.shape


        features_seq = features.flatten(2).permute(0, 2, 1)


        output_seq = self.mamba(features_seq)


        output = output_seq.permute(0, 2, 1).reshape(B, C, H, W)

        return output


class SparsifiedSAMASEModule(nn.Module):
    def __init__(
        self,
        dim=32,
        d_state=8,
        sam_prompt_dim=64,
        use_sam=True,
        sam_checkpoint=None,
        use_wavelet=False,
        use_learnable_prompts=False,
        num_learnable_prompts=16,
        use_soft_masks=False,
        num_soft_regions=8,
        input_channels=4,


        use_fass_sparse=True,
        fass_compression_ratio=2,
        fass_threshold=0.5,
        fass_sparsity_target=0.3,
        fass_ll_sparsity=0.25,
        fass_hf_sparsity=0.08,
        fass_d_state=16,
        train_mode='auto',
        dense_epochs=100,
        gating_loss_weight=1.0
        ,
        gating_input_mode='energy',
        gating_use_semantic_mask=True,
        gating_use_prompt_strength=True,
        gating_use_local_contrast=True,
        use_structure_guided_sam_ase=False,
        structure_texture_weight=0.25,
    ):
        super().__init__()

        self.dim = dim
        self.d_state = d_state
        self.use_wavelet = use_wavelet
        self.use_sam = use_sam
        self.use_structure_guided_sam_ase = use_structure_guided_sam_ase
        self.structure_texture_weight = structure_texture_weight


        self.use_fass_sparse = use_fass_sparse
        self.threshold = fass_threshold
        self.ll_sparsity = fass_ll_sparsity
        self.hf_sparsity = fass_hf_sparsity
        self.train_mode = train_mode
        self.dense_epochs = dense_epochs
        self.gating_loss_weight = gating_loss_weight
        self.gating_input_mode = gating_input_mode
        self.gating_use_semantic_mask = gating_use_semantic_mask
        self.gating_use_prompt_strength = gating_use_prompt_strength
        self.gating_use_local_contrast = gating_use_local_contrast
        self.use_sam_prior_bank = False
        self.sam_prior_use_boundary = True
        self.sam_prior_use_confidence = True
        self.use_semantic_frequency_adaptive_scanning = False
        self.semantic_frequency_semantic_weight = 1.0
        self.semantic_frequency_wavelet_weight = 1.0
        self.semantic_frequency_boundary_weight = 0.5
        self.semantic_frequency_confidence_weight = 0.25
        self.semantic_frequency_prompt_weight = 0.5
        if self.gating_input_mode not in ('energy', 'hybrid_v2', 'semantic_frequency_v1'):
            raise ValueError(f"Unsupported gating_input_mode: {self.gating_input_mode}")
        self.is_gating_frozen = False
        self.current_epoch = 0
        self._last_mask_ll = None
        self._last_mask_hf = None
        self._last_mask_stats = None
        self._current_epoch_gating_stats = None
        self._prev_epoch_gating_stats = None
        self._epoch_gating_sums = None
        self._no_active_warning_count = {'ll': 0, 'hf': 0}
        self._batch_count = 0


        self.expand = 2
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


        if use_fass_sparse:


            self.dwt = SimpleDWT()
            self.idwt = SimpleIDWT()


            self.ll_conv = nn.Conv2d(dim, dim, kernel_size=1)


            hf_dim_raw = dim * 3
            hf_compressed_dim = hf_dim_raw // fass_compression_ratio
            self.hf_proj = nn.Conv2d(hf_dim_raw, hf_compressed_dim, 1, 1, 0)
            self.hf_backproj = nn.Conv2d(hf_compressed_dim, hf_dim_raw, 1, 1, 0)


            self.hf_residual_proj = nn.Conv2d(hf_dim_raw, hf_compressed_dim, 1, 1, 0)


            nn.init.normal_(self.hf_residual_proj.weight, mean=0, std=0.01)
            if self.hf_residual_proj.bias is not None:
                nn.init.zeros_(self.hf_residual_proj.bias)


            self.in_proj_ll = nn.Sequential(
                nn.Conv2d(dim, hidden, 1, 1, 0),
            )

            self.in_proj_hf = nn.Sequential(
                nn.Conv2d(hf_compressed_dim, hidden, 1, 1, 0),
            )


            self.gating_net_ll = nn.Sequential(
                nn.Conv2d(1, hidden // 4, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(hidden // 4, 1, 1, 1, 0),
                nn.Sigmoid()
            )


            self.gating_net_ll_hybrid = nn.Sequential(
                nn.Conv2d(4, hidden // 4, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(hidden // 4, 1, 1, 1, 0),
                nn.Sigmoid()
            )

            self.gating_net_hf = nn.Sequential(
                nn.Conv2d(1, hidden // 4, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(hidden // 4, 1, 1, 1, 0),
                nn.Sigmoid()
            )


            self.gating_net_hf_hybrid = nn.Sequential(
                nn.Conv2d(5, hidden // 4, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(hidden // 4, 1, 1, 1, 0),
                nn.Sigmoid()
            )
            self._bootstrap_hybrid_gating(self.gating_net_ll, self.gating_net_ll_hybrid)
            self._bootstrap_hybrid_gating(self.gating_net_hf, self.gating_net_hf_hybrid)

            self.sparse_mamba_scan = SparseMambaScan(hidden, fass_d_state)
            self.dense_mamba_scan = DenseMambaScan(hidden, fass_d_state)


            self.sparse_output_proj_ll = nn.Conv2d(hidden, dim, 1, 1, 0)
            self.sparse_output_proj_hf = nn.Conv2d(hidden, hf_compressed_dim, 1, 1, 0)

            print(f"[FASS-SAM-ASE] Sparse mode enabled: train_mode={train_mode}, dense_epochs={dense_epochs}")
        else:
            print(f"[FASS-SAM-ASE] Dense mode only (no sparsification)")

    def freeze_gating(self):
        if self.use_fass_sparse:
            for param in self.gating_net_ll.parameters():
                param.requires_grad = False
            for param in self.gating_net_hf.parameters():
                param.requires_grad = False
            for param in self.gating_net_ll_hybrid.parameters():
                param.requires_grad = False
            for param in self.gating_net_hf_hybrid.parameters():
                param.requires_grad = False
            self.is_gating_frozen = True
            print(f"[FASS-SAM-ASE] Gating networks frozen (switching to sparse training)")

    def _stash_previous_epoch_stats(self):
        if self._current_epoch_gating_stats is not None:
            self._prev_epoch_gating_stats = dict(self._current_epoch_gating_stats)

    def _reset_epoch_gating_buffers(self):
        self._current_epoch_gating_stats = None
        self._epoch_gating_sums = None
        self._last_mask_stats = None

    def _ensure_epoch_gating_sums(self, mode):
        if self._epoch_gating_sums is None or self._epoch_gating_sums.get('mode') != mode:
            self._epoch_gating_sums = {
                'mode': mode,
                'num_batches': 0,
                'll_mean_sum': 0.0,
                'll_std_sum': 0.0,
                'hf_mean_sum': 0.0,
                'hf_std_sum': 0.0,
                'll_keep_sum': 0.0,
                'hf_keep_sum': 0.0,
            }
            if mode == 'sparse':
                self._epoch_gating_sums.update({
                    'll_active_sum': 0.0,
                    'hf_active_sum': 0.0,
                    'll_total': 0,
                    'hf_total': 0,
                })
        return self._epoch_gating_sums

    def _update_dense_gating_stats(self, mask_ll, mask_hf):
        mask_ll_detached = mask_ll.detach()
        mask_hf_detached = mask_hf.detach()

        ll_mean = float(mask_ll_detached.mean().item())
        ll_std = float(mask_ll_detached.std().item())
        hf_mean = float(mask_hf_detached.mean().item())
        hf_std = float(mask_hf_detached.std().item())
        ll_keep_ratio = float((mask_ll_detached > 0.5).float().mean().item())
        hf_keep_ratio = float((mask_hf_detached > 0.5).float().mean().item())

        self._last_mask_stats = {
            'mode': 'dense',
            'll_mean': ll_mean,
            'll_std': ll_std,
            'hf_mean': hf_mean,
            'hf_std': hf_std,
            'll_keep_ratio': ll_keep_ratio,
            'hf_keep_ratio': hf_keep_ratio,
            'healthy': bool(ll_std >= 0.01 and hf_std >= 0.01),
        }

        stats_sum = self._ensure_epoch_gating_sums('dense')
        stats_sum['num_batches'] += 1
        stats_sum['ll_mean_sum'] += ll_mean
        stats_sum['ll_std_sum'] += ll_std
        stats_sum['hf_mean_sum'] += hf_mean
        stats_sum['hf_std_sum'] += hf_std
        stats_sum['ll_keep_sum'] += ll_keep_ratio
        stats_sum['hf_keep_sum'] += hf_keep_ratio

        num_batches = stats_sum['num_batches']
        avg_ll_std = stats_sum['ll_std_sum'] / num_batches
        avg_hf_std = stats_sum['hf_std_sum'] / num_batches
        self._current_epoch_gating_stats = {
            'mode': 'dense',
            'num_batches': num_batches,
            'll_mean': stats_sum['ll_mean_sum'] / num_batches,
            'll_std': avg_ll_std,
            'hf_mean': stats_sum['hf_mean_sum'] / num_batches,
            'hf_std': avg_hf_std,
            'll_keep_ratio': stats_sum['ll_keep_sum'] / num_batches,
            'hf_keep_ratio': stats_sum['hf_keep_sum'] / num_batches,
            'healthy': bool(avg_ll_std >= 0.01 and avg_hf_std >= 0.01),
        }

    def _update_sparse_gating_stats(self, mask_ll, mask_hf, ll_active, ll_total, hf_active, hf_total):
        mask_ll_detached = mask_ll.detach()
        mask_hf_detached = mask_hf.detach()

        ll_mean = float(mask_ll_detached.mean().item())
        ll_std = float(mask_ll_detached.std().item())
        hf_mean = float(mask_hf_detached.mean().item())
        hf_std = float(mask_hf_detached.std().item())
        ll_keep_ratio = float(ll_active / ll_total) if ll_total > 0 else 0.0
        hf_keep_ratio = float(hf_active / hf_total) if hf_total > 0 else 0.0

        self._last_mask_stats = {
            'mode': 'sparse',
            'll_mean': ll_mean,
            'll_std': ll_std,
            'hf_mean': hf_mean,
            'hf_std': hf_std,
            'll_keep_ratio': ll_keep_ratio,
            'hf_keep_ratio': hf_keep_ratio,
            'll_active': int(ll_active),
            'll_total': int(ll_total),
            'hf_active': int(hf_active),
            'hf_total': int(hf_total),
        }

        stats_sum = self._ensure_epoch_gating_sums('sparse')
        stats_sum['num_batches'] += 1
        stats_sum['ll_mean_sum'] += ll_mean
        stats_sum['ll_std_sum'] += ll_std
        stats_sum['hf_mean_sum'] += hf_mean
        stats_sum['hf_std_sum'] += hf_std
        stats_sum['ll_keep_sum'] += ll_keep_ratio
        stats_sum['hf_keep_sum'] += hf_keep_ratio
        stats_sum['ll_active_sum'] += float(ll_active)
        stats_sum['hf_active_sum'] += float(hf_active)
        stats_sum['ll_total'] = int(ll_total)
        stats_sum['hf_total'] = int(hf_total)

        num_batches = stats_sum['num_batches']
        avg_ll_keep = stats_sum['ll_keep_sum'] / num_batches
        avg_hf_keep = stats_sum['hf_keep_sum'] / num_batches
        avg_ll_active = int(round(stats_sum['ll_active_sum'] / num_batches))
        avg_hf_active = int(round(stats_sum['hf_active_sum'] / num_batches))
        self._current_epoch_gating_stats = {
            'mode': 'sparse',
            'num_batches': num_batches,
            'll_mean': stats_sum['ll_mean_sum'] / num_batches,
            'll_std': stats_sum['ll_std_sum'] / num_batches,
            'hf_mean': stats_sum['hf_mean_sum'] / num_batches,
            'hf_std': stats_sum['hf_std_sum'] / num_batches,
            'll_keep_ratio': avg_ll_keep,
            'hf_keep_ratio': avg_hf_keep,
            'll_active': avg_ll_active,
            'll_total': stats_sum['ll_total'],
            'll_sparsity': (1.0 - avg_ll_keep) * 100.0,
            'hf_active': avg_hf_active,
            'hf_total': stats_sum['hf_total'],
            'hf_sparsity': (1.0 - avg_hf_keep) * 100.0,
        }

    def _resize_semantic_mask(self, semantic_masks, target_hw, batch_size, device, dtype):
        target_h, target_w = target_hw
        if semantic_masks is None or not self.gating_use_semantic_mask:
            return torch.zeros(batch_size, 1, target_h, target_w, device=device, dtype=dtype)

        if semantic_masks.dim() == 3:
            masks_4d = semantic_masks.unsqueeze(1)
        elif semantic_masks.dim() == 4:
            masks_4d = semantic_masks if semantic_masks.shape[1] == 1 else semantic_masks.mean(dim=1, keepdim=True)
        else:
            return torch.zeros(batch_size, 1, target_h, target_w, device=device, dtype=dtype)

        masks_4d = masks_4d.to(device=device, dtype=dtype)
        if masks_4d.shape[-2:] != (target_h, target_w):
            masks_4d = F.interpolate(masks_4d, size=(target_h, target_w), mode='bilinear', align_corners=False)
        if masks_4d.shape[0] != batch_size:
            return torch.zeros(batch_size, 1, target_h, target_w, device=device, dtype=dtype)
        return masks_4d

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

    def _normalize_spatial_map(self, spatial_map, batch_size):
        if spatial_map is None:
            return None
        flat = spatial_map.flatten(1)
        min_val = flat.min(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
        max_val = flat.max(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
        return (spatial_map - min_val) / (max_val - min_val + 1e-6)

    def _resize_prior_bank_map(self, key, target_hw, batch_size, device, dtype):
        if not self.use_sam_prior_bank:
            return None

        sam_prior_bank = getattr(self, 'current_sam_prior_bank', None)
        if not isinstance(sam_prior_bank, dict):
            return None

        prior_map = sam_prior_bank.get(key, None)
        if prior_map is None:
            return None

        target_h, target_w = target_hw
        if prior_map.dim() == 3:
            prior_map = prior_map.unsqueeze(1)
        elif prior_map.dim() == 4:
            prior_map = prior_map if prior_map.shape[1] == 1 else prior_map.mean(dim=1, keepdim=True)
        else:
            return None

        prior_map = prior_map.to(device=device, dtype=dtype)
        if prior_map.shape[-2:] != (target_h, target_w):
            prior_map = F.interpolate(prior_map, size=(target_h, target_w), mode='bilinear', align_corners=False)
        if prior_map.shape[0] != batch_size:
            return None
        return self._normalize_spatial_map(prior_map, batch_size)

    def _compose_structure_semantic_map(self, semantic_masks, wavelet_guidance, target_hw, batch_size, device, dtype):
        if not self.gating_use_semantic_mask:
            target_h, target_w = target_hw
            return torch.zeros(batch_size, 1, target_h, target_w, device=device, dtype=dtype)

        semantic_map = self._resize_semantic_mask(semantic_masks, target_hw, batch_size, device, dtype)
        if self.use_sam_prior_bank:
            region_map = self._resize_prior_bank_map('region_map', target_hw, batch_size, device, dtype)
            if region_map is not None:
                semantic_map = region_map

            if self.sam_prior_use_boundary:
                boundary_map = self._resize_prior_bank_map('boundary_map', target_hw, batch_size, device, dtype)
                if boundary_map is not None:
                    semantic_map = semantic_map + 0.5 * boundary_map

            if self.use_structure_guided_sam_ase:
                texture_map = self._resize_wavelet_guidance(wavelet_guidance, target_hw, batch_size, device, dtype)
                semantic_bins = torch.round(semantic_map * 255.0) / 255.0
                semantic_map = semantic_bins + self.structure_texture_weight * texture_map

            if self.sam_prior_use_confidence:
                confidence_map = self._resize_prior_bank_map('confidence_map', target_hw, batch_size, device, dtype)
                if confidence_map is not None:
                    semantic_map = semantic_map * (0.5 + 0.5 * confidence_map)

            return self._normalize_spatial_map(semantic_map, batch_size)

        if not self.use_structure_guided_sam_ase:
            return semantic_map

        texture_map = self._resize_wavelet_guidance(wavelet_guidance, target_hw, batch_size, device, dtype)
        semantic_bins = torch.round(semantic_map * 255.0) / 255.0
        return semantic_bins + self.structure_texture_weight * texture_map

    def _build_semantic_frequency_components(self, semantic_masks, sam_prompt, wavelet_guidance, target_hw, batch_size, device, dtype):
        semantic_map = self._compose_structure_semantic_map(
            semantic_masks,
            wavelet_guidance,
            target_hw,
            batch_size,
            device,
            dtype,
        )
        boundary_map = self._resize_prior_bank_map('boundary_map', target_hw, batch_size, device, dtype)
        confidence_map = self._resize_prior_bank_map('confidence_map', target_hw, batch_size, device, dtype)
        prompt_strength = self._build_prompt_strength_map(
            sam_prompt,
            target_hw,
            batch_size,
            device,
            dtype,
        )
        wavelet_map = self._resize_wavelet_guidance(
            wavelet_guidance,
            target_hw,
            batch_size,
            device,
            dtype,
        )
        if boundary_map is None:
            boundary_map = torch.zeros_like(semantic_map)
        if confidence_map is None:
            confidence_map = torch.zeros_like(semantic_map)
        semantic_component = (
            self.semantic_frequency_semantic_weight * semantic_map
            + self.semantic_frequency_boundary_weight * boundary_map
            + self.semantic_frequency_confidence_weight * confidence_map
        )
        prompt_frequency_component = (
            self.semantic_frequency_prompt_weight * prompt_strength
            + self.semantic_frequency_wavelet_weight * wavelet_map
        )
        semantic_component = self._normalize_spatial_map(semantic_component, batch_size)
        prompt_frequency_component = self._normalize_spatial_map(prompt_frequency_component, batch_size)
        return semantic_component, prompt_frequency_component

    def _build_prompt_strength_map(self, sam_prompt, target_hw, batch_size, device, dtype):
        target_h, target_w = target_hw
        if not self.gating_use_prompt_strength:
            return torch.zeros(batch_size, 1, target_h, target_w, device=device, dtype=dtype)

        prior_prompt_strength = self._resize_prior_bank_map(
            'prompt_strength_map',
            target_hw,
            batch_size,
            device,
            dtype,
        )
        if prior_prompt_strength is not None:
            return prior_prompt_strength

        if sam_prompt is None:
            return torch.zeros(batch_size, 1, target_h, target_w, device=device, dtype=dtype)

        if sam_prompt.dim() != 3 or sam_prompt.shape[0] != batch_size:
            return torch.zeros(batch_size, 1, target_h, target_w, device=device, dtype=dtype)

        prompt_len = sam_prompt.shape[1]
        prompt_hw = int(math.sqrt(prompt_len))
        if prompt_hw * prompt_hw != prompt_len:
            return torch.zeros(batch_size, 1, target_h, target_w, device=device, dtype=dtype)

        prompt_strength = sam_prompt.to(device=device, dtype=dtype).abs().mean(dim=-1)
        prompt_strength = prompt_strength.view(batch_size, 1, prompt_hw, prompt_hw)
        if (prompt_hw, prompt_hw) != (target_h, target_w):
            prompt_strength = F.interpolate(
                prompt_strength,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False,
            )
        return prompt_strength

    def _compute_local_contrast(self, feature_map):
        local_mean = F.avg_pool2d(feature_map, kernel_size=3, stride=1, padding=1)
        return (feature_map - local_mean).abs()

    def _bootstrap_hybrid_gating(self, source_net, target_net):
        with torch.no_grad():
            source_in = source_net[0]
            target_in = target_net[0]
            target_in.weight.zero_()
            target_in.weight[:, :1].copy_(source_in.weight)
            if source_in.bias is not None and target_in.bias is not None:
                target_in.bias.copy_(source_in.bias)

            source_out = source_net[2]
            target_out = target_net[2]
            target_out.weight.copy_(source_out.weight)
            if source_out.bias is not None and target_out.bias is not None:
                target_out.bias.copy_(source_out.bias)

    def _build_ll_gating_input(self, Yl_cpe, semantic_masks, sam_prompt, wavelet_guidance=None):
        batch_size, _, target_h, target_w = Yl_cpe.shape
        ll_energy = Yl_cpe.pow(2).mean(dim=1, keepdim=True)
        if self.gating_input_mode == 'energy':
            return ll_energy

        if self.gating_input_mode == 'semantic_frequency_v1':
            ll_feature_norm = Yl_cpe.abs().mean(dim=1, keepdim=True)
            semantic_component, prompt_frequency_component = self._build_semantic_frequency_components(
                semantic_masks,
                sam_prompt,
                wavelet_guidance,
                (target_h, target_w),
                batch_size,
                Yl_cpe.device,
                Yl_cpe.dtype,
            )
            return torch.cat([ll_energy, ll_feature_norm, semantic_component, prompt_frequency_component], dim=1)

        ll_feature_norm = Yl_cpe.abs().mean(dim=1, keepdim=True)
        semantic_map = self._compose_structure_semantic_map(
            semantic_masks, wavelet_guidance, (target_h, target_w), batch_size, Yl_cpe.device, Yl_cpe.dtype
        )
        prompt_strength = self._build_prompt_strength_map(
            sam_prompt, (target_h, target_w), batch_size, Yl_cpe.device, Yl_cpe.dtype
        )
        return torch.cat([ll_energy, ll_feature_norm, semantic_map, prompt_strength], dim=1)

    def _build_hf_gating_input(self, Yh_cat, semantic_masks, sam_prompt, wavelet_guidance=None):
        batch_size, _, target_h, target_w = Yh_cat.shape
        hf_energy = Yh_cat.pow(2).mean(dim=1, keepdim=True)
        hf_log_energy = torch.log(hf_energy + 1e-8)
        if self.gating_input_mode == 'energy':
            return hf_log_energy

        if self.gating_input_mode == 'semantic_frequency_v1':
            hf_abs_mean = Yh_cat.abs().mean(dim=1, keepdim=True)
            if self.gating_use_local_contrast:
                hf_local_contrast = self._compute_local_contrast(hf_abs_mean)
            else:
                hf_local_contrast = torch.zeros_like(hf_abs_mean)
            semantic_component, prompt_frequency_component = self._build_semantic_frequency_components(
                semantic_masks,
                sam_prompt,
                wavelet_guidance,
                (target_h, target_w),
                batch_size,
                Yh_cat.device,
                Yh_cat.dtype,
            )
            return torch.cat(
                [hf_log_energy, hf_abs_mean, hf_local_contrast, semantic_component, prompt_frequency_component],
                dim=1,
            )

        hf_abs_mean = Yh_cat.abs().mean(dim=1, keepdim=True)
        if self.gating_use_local_contrast:
            hf_local_contrast = self._compute_local_contrast(hf_abs_mean)
        else:
            hf_local_contrast = torch.zeros_like(hf_abs_mean)
        semantic_map = self._compose_structure_semantic_map(
            semantic_masks, wavelet_guidance, (target_h, target_w), batch_size, Yh_cat.device, Yh_cat.dtype
        )
        prompt_strength = self._build_prompt_strength_map(
            sam_prompt, (target_h, target_w), batch_size, Yh_cat.device, Yh_cat.dtype
        )
        return torch.cat(
            [hf_log_energy, hf_abs_mean, hf_local_contrast, semantic_map, prompt_strength],
            dim=1,
        )

    def _predict_ll_mask(self, Yl_cpe, semantic_masks, sam_prompt, wavelet_guidance=None):
        gating_input = self._build_ll_gating_input(Yl_cpe, semantic_masks, sam_prompt, wavelet_guidance)
        if self.gating_input_mode in ('hybrid_v2', 'semantic_frequency_v1'):
            return self.gating_net_ll_hybrid(gating_input)
        return self.gating_net_ll(gating_input)

    def _predict_hf_mask(self, Yh_cat, semantic_masks, sam_prompt, wavelet_guidance=None):
        gating_input = self._build_hf_gating_input(Yh_cat, semantic_masks, sam_prompt, wavelet_guidance)
        if self.gating_input_mode in ('hybrid_v2', 'semantic_frequency_v1'):
            return self.gating_net_hf_hybrid(gating_input)
        return self.gating_net_hf(gating_input)

    def forward(self, x, x_size, sam_prompt=None, semantic_masks=None, original_msi=None, wavelet_guidance=None, epoch=0):
        B, n, C = x.shape
        H, W = x_size


        if epoch != self.current_epoch:
            self._stash_previous_epoch_stats()
            self.current_epoch = epoch
            self._no_active_warning_count = {'ll': 0, 'hf': 0}
            self._batch_count = 0

            self._reset_epoch_gating_buffers()

        self._batch_count += 1


        if self.use_fass_sparse:
            if self.train_mode == 'auto':
                current_mode = 'dense' if epoch < self.dense_epochs else 'sparse'
            else:
                current_mode = self.train_mode


            if current_mode == 'sparse' and not self.is_gating_frozen:

                stats = self._current_epoch_gating_stats
                if stats is None and self._prev_epoch_gating_stats is not None:
                    stats = self._prev_epoch_gating_stats
                if stats is not None and stats.get('mode') == 'dense':
                    print(f"[FASS-SAM-ASE] Epoch {epoch}: Final Dense stats - "
                          f"LL={stats['ll_mean']:.3f}±{stats['ll_std']:.3f}, "
                          f"HF={stats['hf_mean']:.3f}±{stats['hf_std']:.3f}")

                    if stats['ll_std'] < 0.01 or stats['hf_std'] < 0.01:
                        print(f"[FASS-SAM-ASE] WARNING: Gating networks may not be well trained. "
                              f"Consider increasing dense_epochs or gating_loss_weight.")
                else:
                    print(f"[FASS-SAM-ASE] Epoch {epoch}: Switching to Sparse mode (no previous dense stats available)")

                self.freeze_gating()

                if not hasattr(self, '_has_printed_switch'):
                    print(f"[FASS-SAM-ASE] Epoch {epoch}: Switching to Sparse mode, Gating networks frozen")
                    self._has_printed_switch = True
        else:
            current_mode = 'dense'


        if current_mode == 'dense' or not self.use_fass_sparse:
            return self.forward_dense(x, H, W, sam_prompt, semantic_masks, wavelet_guidance)
        else:
            return self.forward_sparse(x, H, W, sam_prompt, semantic_masks, wavelet_guidance)

    def forward_dense(self, x, H, W, sam_prompt, semantic_masks, wavelet_guidance=None):
        B, L, C = x.shape


        hidden = int(self.dim * self.expand)


        x_2d = x.permute(0, 2, 1).reshape(B, C, H, W)


        Yl, Yh_list = self.dwt(x_2d)


        Yl_out = self.ll_conv(Yl)


        Yl_proj = self.in_proj_ll(Yl_out)


        Yl_cpe = self.CPE(Yl_proj) + Yl_proj


        _, _, H_l, W_l = Yl_cpe.shape


        energy_ll = Yl_cpe.pow(2).mean(dim=1, keepdim=True)


        mask_ll = self._predict_ll_mask(Yl_cpe, semantic_masks, sam_prompt, wavelet_guidance)
        self._last_mask_ll = mask_ll
        self._last_mask_ll = mask_ll


        self._last_mask_ll = mask_ll


        mask_ll_expanded = mask_ll.expand(B, hidden, H_l, W_l)
        Yl_masked = Yl_cpe * mask_ll_expanded


        Yl_seq = Yl_masked.flatten(2).permute(0, 2, 1)


        sam_prompt_d_state = None
        sam_prompt_hidden = None

        if sam_prompt is not None:

            B_sam, L_sam, C_sam = sam_prompt.shape
            H_sam = W_sam = int(math.sqrt(L_sam))


            if H_sam != H_l or W_sam != W_l:

                sam_prompt_2d = sam_prompt.permute(0, 2, 1).reshape(B, C_sam, H_sam, W_sam)
                sam_prompt_resized = F.interpolate(sam_prompt_2d, size=(H_l, W_l), mode='bilinear', align_corners=False)
                sam_prompt = sam_prompt_resized.reshape(B, C_sam, H_l * W_l).permute(0, 2, 1)

            sam_prompt_d_state = self.prompt_proj(sam_prompt)
            sam_prompt_hidden = self.prompt_to_hidden(sam_prompt)

            Yl_seq = Yl_seq + sam_prompt_hidden

        Yl_output_seq = self.selectiveScan(Yl_seq, sam_prompt_d_state)


        Yl_output = Yl_output_seq.reshape(B, H_l, W_l, -1).permute(0, 3, 1, 2)


        mask_ll_expanded = mask_ll.expand(B, hidden, H_l, W_l)
        Yl_out_final = Yl_output * mask_ll_expanded + Yl_cpe * (1 - mask_ll_expanded)

        Yl_out_final = self.sparse_output_proj_ll(Yl_out_final)


        Yh_cat = torch.cat(Yh_list, dim=1)


        energy_hf = Yh_cat.pow(2).mean(dim=1, keepdim=True)
        log_energy_hf = torch.log(energy_hf + 1e-8)


        mask_hf = self._predict_hf_mask(Yh_cat, semantic_masks, sam_prompt, wavelet_guidance)
        self._last_mask_hf = mask_hf
        self._last_mask_hf = mask_hf


        self._last_mask_hf = mask_hf


        Yh_compressed = self.hf_proj(Yh_cat)


        Yh_proj = self.in_proj_hf(Yh_compressed)


        Yh_cpe = self.CPE(Yh_proj) + Yh_proj


        mask_hf_expanded = mask_hf.expand(B, hidden, H_l, W_l)
        Yh_masked = Yh_cpe * mask_hf_expanded


        Yh_seq = Yh_masked.flatten(2).permute(0, 2, 1)


        if sam_prompt_hidden is not None:
            Yh_seq = Yh_seq + sam_prompt_hidden
            Yh_output_seq = self.selectiveScan(Yh_seq, sam_prompt_d_state)
        else:
            Yh_output_seq = self.selectiveScan(Yh_seq, None)


        Yh_output = Yh_output_seq.reshape(B, H_l, W_l, -1).permute(0, 3, 1, 2)


        Yh_out_final_hidden = Yh_output * mask_hf_expanded + Yh_cpe * (1 - mask_hf_expanded)

        Yh_out_final = self.sparse_output_proj_hf(Yh_out_final_hidden)


        Yh_out_final = self.hf_backproj(Yh_out_final)


        C_hf = Yh_out_final.shape[1] // 3
        Yh_lh, Yh_hl, Yh_hh = torch.split(Yh_out_final, C_hf, dim=1)
        Yh_list_out = [Yh_lh, Yh_hl, Yh_hh]


        output = self.idwt(Yl_out_final, Yh_list_out)


        output_seq = output.flatten(2).permute(0, 2, 1)


        if self._batch_count == 1:
            ll_mean = mask_ll.mean().item()
            ll_std = mask_ll.std().item()
            hf_mean = mask_hf.mean().item()
            hf_std = mask_hf.std().item()

            self._current_epoch_gating_stats = {
                'mode': 'dense',
                'll_mean': ll_mean,
                'll_std': ll_std,
                'hf_mean': hf_mean,
                'hf_std': hf_std
            }

        self._update_dense_gating_stats(mask_ll, mask_hf)
        return output_seq

    def forward_sparse(self, x, H, W, sam_prompt, semantic_masks, wavelet_guidance=None):
        B, L, C = x.shape


        x_2d = x.permute(0, 2, 1).reshape(B, C, H, W)

        Yl, Yh_list = self.dwt(x_2d)


        Yl_out = self.ll_conv(Yl)


        Yl_proj = self.in_proj_ll(Yl_out)


        Yl_cpe = self.CPE(Yl_proj) + Yl_proj


        B, _, H_l, W_l = Yl_cpe.shape
        N = H_l * W_l
        energy_ll = Yl_cpe.pow(2).mean(dim=1, keepdim=True)


        mask_ll = self._predict_ll_mask(Yl_cpe, semantic_masks, sam_prompt, wavelet_guidance)


        if self.ll_sparsity >= 1.0:


            mask_ll_binary = torch.ones_like(mask_ll)
            mask_ll_ste = mask_ll_binary


            num_active_ll = mask_ll_binary.sum().item()
            total_ll = mask_ll_binary.numel()
            sparsity_ll = 0.0


            Yl_sparse = self.dense_mamba_scan(Yl_cpe, sam_prompt=None)
        else:

            K_ll = int(N * self.ll_sparsity)
            K_ll = max(1, min(N, K_ll))


            mask_ll_flat = mask_ll.view(B, -1)
            _, topk_indices_ll = torch.topk(mask_ll_flat, K_ll, dim=1)
            mask_ll_binary_flat = torch.zeros_like(mask_ll_flat)
            mask_ll_binary_flat.scatter_(1, topk_indices_ll, 1.0)
            mask_ll_binary = mask_ll_binary_flat.view(B, 1, H_l, W_l)


            mask_ll_ste = mask_ll_binary - mask_ll.detach() + mask_ll


            num_active_ll = mask_ll_binary.sum().item()

            Yl_sparse = self.sparse_mamba_scan(Yl_cpe, mask_ll_ste, sam_prompt=None)


        num_active_ll = mask_ll_binary.sum().item()
        total_ll = mask_ll_binary.numel()
        sparsity_ll = (1.0 - num_active_ll / total_ll) * 100 if total_ll > 0 else 0.0
        self._ll_stats = (int(num_active_ll), int(total_ll), sparsity_ll)


        Yl_sparse = self.sparse_output_proj_ll(Yl_sparse)

        Yl_out_final = Yl_sparse + Yl_out


        Yh_cat = torch.cat(Yh_list, dim=1)


        energy_hf = Yh_cat.pow(2).mean(dim=1, keepdim=True)
        log_energy_hf = torch.log(energy_hf + 1e-8)


        mask_hf = self._predict_hf_mask(Yh_cat, semantic_masks, sam_prompt, wavelet_guidance)


        K_hf = int(N * self.hf_sparsity)
        K_hf = max(1, min(N, K_hf))


        mask_hf_flat = mask_hf.view(B, -1)
        _, topk_indices_hf = torch.topk(mask_hf_flat, K_hf, dim=1)
        mask_hf_binary_flat = torch.zeros_like(mask_hf_flat)
        mask_hf_binary_flat.scatter_(1, topk_indices_hf, 1.0)
        mask_hf_binary = mask_hf_binary_flat.view(B, 1, H_l, W_l)


        mask_hf_ste = mask_hf_binary - mask_hf.detach() + mask_hf


        num_active_hf = mask_hf_binary.sum().item()
        total_hf = mask_hf_binary.numel()
        sparsity_hf = (1.0 - num_active_hf / total_hf) * 100 if total_hf > 0 else 0.0

        if num_active_hf == 0:

            mask_hf_ste = torch.ones_like(mask_hf)


        if self._batch_count == 1:
            self._hf_stats = (num_active_hf, total_hf, sparsity_hf)


        Yh_compressed = self.hf_proj(Yh_cat)


        if hasattr(self, 'hf_residual_proj'):
            Yh_residual = self.hf_residual_proj(Yh_cat)
            Yh_compressed = Yh_compressed + Yh_residual


        Yh_proj = self.in_proj_hf(Yh_compressed)


        Yh_cpe = self.CPE(Yh_proj) + Yh_proj


        Yh_sparse = self.sparse_mamba_scan(Yh_cpe, mask_hf_ste, sam_prompt=None)


        Yh_sparse = self.sparse_output_proj_hf(Yh_sparse)


        Yh_out_final = self.hf_backproj(Yh_sparse)


        C = Yh_out_final.shape[1] // 3
        Yh_lh, Yh_hl, Yh_hh = torch.split(Yh_out_final, C, dim=1)
        Yh_list_out = [Yh_lh, Yh_hl, Yh_hh]


        output = self.idwt(Yl_out_final, Yh_list_out)


        output_seq = output.flatten(2).permute(0, 2, 1)


        if self._batch_count == 1:
            if hasattr(self, '_ll_stats') and hasattr(self, '_hf_stats'):
                ll_active, ll_total, ll_sparsity = self._ll_stats
                hf_active, hf_total, hf_sparsity = self._hf_stats

                self._current_epoch_gating_stats = {
                    'mode': 'sparse',
                    'll_active': ll_active,
                    'll_total': ll_total,
                    'll_sparsity': ll_sparsity,
                    'hf_active': hf_active,
                    'hf_total': hf_total,
                    'hf_sparsity': hf_sparsity
                }

        self._update_sparse_gating_stats(
            mask_ll,
            mask_hf,
            num_active_ll,
            total_ll,
            num_active_hf,
            total_hf,
        )
        return output_seq


class SparsifiedSAMASEMambaBlock(nn.Module):
    def __init__(
        self,
        d_model=32,
        sam_prompt_dim=64,
        modal_type='single',
        d_state=8,
        use_sam=True,
        sam_checkpoint=None,
        use_wavelet=False,
        use_learnable_prompts=False,
        num_learnable_prompts=16,
        use_soft_masks=False,
        num_soft_regions=8,
        input_channels=4,


        fass_compression_ratio=2,
        fass_threshold=0.5,
        fass_sparsity_target=0.3,
        fass_ll_sparsity=0.25,
        fass_hf_sparsity=0.08,
        fass_d_state=16,
        train_mode='auto',
        dense_epochs=100,
        gating_loss_weight=1.0
        ,
        gating_input_mode='energy',
        gating_use_semantic_mask=True,
        gating_use_prompt_strength=True,
        gating_use_local_contrast=True,
        use_structure_guided_sam_ase=False,
        structure_texture_weight=0.25
    ):
        super().__init__()
        self.d_model = d_model
        self.modal_type = modal_type
        self.use_wavelet = use_wavelet
        self.use_sam = use_sam


        self.sam_ase_module = SparsifiedSAMASEModule(
            dim=d_model,
            d_state=d_state,
            sam_prompt_dim=sam_prompt_dim,
            use_sam=use_sam,
            sam_checkpoint=sam_checkpoint,
            use_wavelet=use_wavelet,
            use_learnable_prompts=use_learnable_prompts,
            num_learnable_prompts=num_learnable_prompts,
            use_soft_masks=use_soft_masks,
            num_soft_regions=num_soft_regions,
            input_channels=input_channels,
            use_fass_sparse=True,
            fass_compression_ratio=fass_compression_ratio,
            fass_threshold=fass_threshold,
            fass_sparsity_target=fass_sparsity_target,
            fass_ll_sparsity=fass_ll_sparsity,
            fass_hf_sparsity=fass_hf_sparsity,
            fass_d_state=fass_d_state,
            train_mode=train_mode,
            dense_epochs=dense_epochs,
            gating_loss_weight=gating_loss_weight,
            gating_input_mode=gating_input_mode,
            gating_use_semantic_mask=gating_use_semantic_mask,
            gating_use_prompt_strength=gating_use_prompt_strength,
            gating_use_local_contrast=gating_use_local_contrast,
            use_structure_guided_sam_ase=use_structure_guided_sam_ase,
            structure_texture_weight=structure_texture_weight
        )


        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, original_msi=None, sam_prompt=None, semantic_masks=None, wavelet_guidance=None, current_epoch=0):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))

        x_size = (H, W)
        self.sam_ase_module.current_sam_prior_bank = getattr(self, 'current_sam_prior_bank', None)


        output = self.sam_ase_module(
            x, x_size,
            sam_prompt=sam_prompt,
            semantic_masks=semantic_masks,
            original_msi=original_msi,
            wavelet_guidance=wavelet_guidance,
            epoch=current_epoch
        )


        output = self.out_proj(output)


        output = output + x

        return output
