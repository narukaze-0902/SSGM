import math
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn


from model.ase_mamba import ASEMambaBlock, index_reverse, semantic_neighbor

from model.sam_ase_mamba import SAMASEMambaBlock

from model.fass_sam_ase_mamba import FASSSAMASEMambaBlock

from model.sparsified_sam_ase_mamba import SparsifiedSAMASEMambaBlock

class SingleMambaBlock(nn.Module):
    def __init__(self, dim, H=64, W=64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v6',
                           if_devide_out=True, use_norm=True, input_h=H, input_w=W)

    def forward(self, input):

        skip = input
        input = self.norm(input)


        if hasattr(self.block, 'input_h'):
            b, seq_len, c = input.shape
            h = w = int(math.sqrt(seq_len))

            if h * w == seq_len:
                self.block.input_h = h
                self.block.input_w = w

        output = self.block(input)
        return output + skip


class CrossMambaBlock(nn.Module):
    def __init__(self, dim, H=64, W=64):
        super().__init__()
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)

        self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v7',
                         if_devide_out=True, use_norm=True, input_h=H, input_w=W)

    def forward(self, input0, input1):

        b, seq_len, c = input0.shape
        h = w = int(math.sqrt(seq_len))


        if h * w != seq_len:
            raise ValueError(f"序列长度{seq_len}不能分解为{h}x{w}的网格")


        if hasattr(self.block, 'input_h'):
            self.block.input_h = h
            self.block.input_w = w


        skip = input0
        input0 = self.norm0(input0)
        input1 = self.norm1(input1)
        output = self.block(input0, extra_emb=input1)
        return output + skip


class SAMSemanticPromptBankRefiner(nn.Module):
    def __init__(self, feature_dim, num_tokens, d_state):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_tokens = num_tokens
        self.d_state = d_state

        self.feature_proj = nn.Conv2d(feature_dim, feature_dim, 1, 1, 0)
        self.prior_proj = nn.Sequential(
            nn.Conv2d(2, feature_dim, 1, 1, 0),
            nn.LeakyReLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(feature_dim, d_state, 1, 1, 0),
        )
        self.slot_pool = nn.AdaptiveAvgPool2d((num_tokens, 1))

        nn.init.zeros_(self.fuse[-1].weight)
        if self.fuse[-1].bias is not None:
            nn.init.zeros_(self.fuse[-1].bias)

    @staticmethod
    def _ensure_map_4d(map_tensor):
        if map_tensor is None:
            return None
        if map_tensor.dim() == 3:
            return map_tensor.unsqueeze(1)
        if map_tensor.dim() == 4:
            return map_tensor
        return None

    def forward(self, fusion_tokens, spatial_hw, prior_bank):
        if prior_bank is None:
            return None

        region_map = self._ensure_map_4d(prior_bank.get('region_map'))
        prompt_strength_map = self._ensure_map_4d(prior_bank.get('prompt_strength_map'))
        if region_map is None or prompt_strength_map is None:
            return None

        h, w = spatial_hw
        fusion_2d = rearrange(fusion_tokens, 'b (h w) c -> b c h w', h=h, w=w)
        prior = torch.cat([region_map, prompt_strength_map], dim=1).to(device=fusion_tokens.device, dtype=torch.float32)
        if prior.shape[-2:] != (h, w):
            prior = F.interpolate(prior, size=(h, w), mode='bilinear', align_corners=False)

        fused = self.feature_proj(fusion_2d.float()) + self.prior_proj(prior)
        bias_map = self.fuse(fused)
        bias_slots = self.slot_pool(bias_map).squeeze(-1).permute(0, 2, 1).contiguous()
        return torch.tanh(bias_slots).to(dtype=fusion_tokens.dtype)


class SAMRegionPrototypePromptConditioner(nn.Module):
    def __init__(self, feature_dim, num_tokens, d_state, sam_feature_dim=256, num_prototypes=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_tokens = num_tokens
        self.d_state = d_state
        self.num_prototypes = num_prototypes

        self.sam_feature_proj = nn.Conv2d(sam_feature_dim, feature_dim, 1, 1, 0)
        self.prototype_fuse = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, d_state),
        )
        self.out_proj = nn.Linear(d_state, d_state)

        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, fusion_tokens, sam_region_context, wavelet_scale=0.0):
        if sam_region_context is None:
            return None

        sam_features = sam_region_context.get('sam_features')
        sam_masks = sam_region_context.get('sam_masks')
        wavelet_guidance = sam_region_context.get('semantic_frequency_prior_map')
        if wavelet_guidance is None:
            wavelet_guidance = sam_region_context.get('wavelet_guidance')
        if sam_features is None or sam_masks is None:
            return None

        if sam_features.dim() == 3:
            sam_features = sam_features.unsqueeze(0)
        if sam_masks.dim() == 3:
            sam_masks = sam_masks.unsqueeze(0)
        if sam_features.dim() != 4 or sam_masks.dim() != 4:
            return None

        sam_features = sam_features.to(device=fusion_tokens.device, dtype=torch.float32)
        sam_masks = sam_masks.to(device=fusion_tokens.device, dtype=torch.float32)

        projected_features = self.sam_feature_proj(sam_features)
        if sam_masks.shape[-2:] != projected_features.shape[-2:]:
            sam_masks = F.interpolate(
                sam_masks,
                size=projected_features.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

        bsz, num_masks, feat_h, feat_w = sam_masks.shape
        if num_masks <= 0:
            return None

        proto_count = min(self.num_prototypes, num_masks)
        mask_area = sam_masks.flatten(-2).mean(dim=-1)
        topk_idx = torch.topk(mask_area, k=proto_count, dim=1).indices
        gather_idx = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, feat_h, feat_w)
        selected_masks = torch.gather(sam_masks, 1, gather_idx)

        norm = selected_masks.flatten(-2).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        masked_features = projected_features.unsqueeze(1) * selected_masks.unsqueeze(2)
        prototypes = masked_features.flatten(-2).sum(dim=-1) / norm

        if wavelet_guidance is not None and wavelet_scale > 0:
            if wavelet_guidance.dim() == 3:
                wavelet_guidance = wavelet_guidance.unsqueeze(1)
            if wavelet_guidance.dim() == 4:
                wavelet_guidance = wavelet_guidance.to(device=fusion_tokens.device, dtype=torch.float32)
                if wavelet_guidance.shape[1] != 1:
                    wavelet_guidance = wavelet_guidance.mean(dim=1, keepdim=True)
                if wavelet_guidance.shape[-2:] != (feat_h, feat_w):
                    wavelet_guidance = F.interpolate(
                        wavelet_guidance,
                        size=(feat_h, feat_w),
                        mode='bilinear',
                        align_corners=False,
                    )
                region_energy = (selected_masks * wavelet_guidance.squeeze(1).unsqueeze(1)).flatten(-2).sum(dim=-1)
                region_energy = region_energy / norm.squeeze(-1)
                prototypes = prototypes * (1.0 + wavelet_scale * region_energy.unsqueeze(-1))

        fusion_context = fusion_tokens.mean(dim=1, keepdim=True).to(dtype=prototypes.dtype)
        fusion_context = fusion_context.expand(-1, proto_count, -1)
        prototype_state = self.prototype_fuse(torch.cat([prototypes, fusion_context], dim=-1))

        prototype_state = prototype_state.permute(0, 2, 1)
        if prototype_state.shape[-1] != self.num_tokens:
            prototype_state = F.interpolate(
                prototype_state,
                size=self.num_tokens,
                mode='linear',
                align_corners=False,
            )
        prototype_state = prototype_state.permute(0, 2, 1).contiguous()
        return torch.tanh(self.out_proj(prototype_state)).to(dtype=fusion_tokens.dtype)


class SAMRegionPromptMixer(nn.Module):
    def __init__(self, feature_dim, num_tokens, sam_feature_dim=256, num_prototypes=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_tokens = num_tokens
        self.num_prototypes = num_prototypes

        self.sam_feature_proj = nn.Conv2d(sam_feature_dim, feature_dim, 1, 1, 0)
        self.prototype_fuse = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LeakyReLU(),
        )
        self.out_proj = nn.Linear(feature_dim, num_tokens)

        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, fusion_tokens, sam_region_context):
        if sam_region_context is None:
            return None

        sam_features = sam_region_context.get('sam_features')
        sam_masks = sam_region_context.get('sam_masks')
        if sam_features is None or sam_masks is None:
            return None

        if sam_features.dim() == 3:
            sam_features = sam_features.unsqueeze(0)
        if sam_masks.dim() == 3:
            sam_masks = sam_masks.unsqueeze(0)
        if sam_features.dim() != 4 or sam_masks.dim() != 4:
            return None

        sam_features = sam_features.to(device=fusion_tokens.device, dtype=torch.float32)
        sam_masks = sam_masks.to(device=fusion_tokens.device, dtype=torch.float32)

        projected_features = self.sam_feature_proj(sam_features)
        if sam_masks.shape[-2:] != projected_features.shape[-2:]:
            sam_masks = F.interpolate(
                sam_masks,
                size=projected_features.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

        bsz, num_masks, feat_h, feat_w = sam_masks.shape
        if num_masks <= 0:
            return None

        proto_count = min(self.num_prototypes, num_masks)
        mask_area = sam_masks.flatten(-2).mean(dim=-1)
        topk_idx = torch.topk(mask_area, k=proto_count, dim=1).indices
        gather_idx = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, feat_h, feat_w)
        selected_masks = torch.gather(sam_masks, 1, gather_idx)

        norm = selected_masks.flatten(-2).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        selected_masks = selected_masks.unsqueeze(2)
        masked_features = projected_features.unsqueeze(1) * selected_masks
        prototypes = masked_features.flatten(-2).sum(dim=-1) / norm

        fusion_context = fusion_tokens.mean(dim=1, keepdim=True).to(dtype=prototypes.dtype)
        fusion_context = fusion_context.expand(-1, proto_count, -1)
        fused_proto = self.prototype_fuse(torch.cat([prototypes, fusion_context], dim=-1))
        pooled_proto = fused_proto.mean(dim=1)
        return self.out_proj(pooled_proto).to(dtype=fusion_tokens.dtype)


class SAMSemanticScanner(nn.Module):
    def __init__(self, num_regions=6, mask_threshold=0.05):
        super().__init__()
        self.num_regions = num_regions
        self.mask_threshold = mask_threshold

    def _build_region_ids(self, sam_region_context, spatial_hw, device):
        if sam_region_context is None:
            return None

        sam_masks = sam_region_context.get('sam_masks')
        if sam_masks is None:
            return None

        if sam_masks.dim() == 3:
            sam_masks = sam_masks.unsqueeze(0)
        if sam_masks.dim() != 4:
            return None

        sam_masks = sam_masks.to(device=device, dtype=torch.float32)
        if sam_masks.shape[-2:] != spatial_hw:
            sam_masks = F.interpolate(
                sam_masks,
                size=spatial_hw,
                mode='bilinear',
                align_corners=False,
            )

        bsz, num_masks, _, _ = sam_masks.shape
        if num_masks <= 0:
            return None

        region_count = min(self.num_regions, num_masks)
        mask_area = sam_masks.flatten(-2).mean(dim=-1)
        topk_idx = torch.topk(mask_area, k=region_count, dim=1).indices
        gather_idx = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, spatial_hw[0], spatial_hw[1])
        selected_masks = torch.gather(sam_masks, 1, gather_idx)

        region_strength, region_index = selected_masks.max(dim=1)
        region_ids = region_index + 1
        region_ids = torch.where(
            region_strength > self.mask_threshold,
            region_ids,
            torch.zeros_like(region_ids),
        )
        return region_ids.reshape(bsz, -1).contiguous()

    def forward(self, sam_region_context, spatial_hw, device):
        region_ids = self._build_region_ids(sam_region_context, spatial_hw, device)
        if region_ids is None:
            return None, None, None

        _, scan_indices = torch.sort(region_ids, dim=-1, stable=True)
        reverse_indices = index_reverse(scan_indices)
        return region_ids, scan_indices, reverse_indices


class SAMFeatureClusteringScanner(nn.Module):
    def __init__(self, num_clusters=6, num_iters=2, spatial_weight=0.05):
        super().__init__()
        self.num_clusters = int(max(num_clusters, 1))
        self.num_iters = int(max(num_iters, 1))
        self.spatial_weight = float(max(spatial_weight, 0.0))

    def _prepare_feature_tokens(self, sam_region_context, spatial_hw, device):
        if sam_region_context is None:
            return None

        sam_features = sam_region_context.get('sam_features')
        if sam_features is None:
            return None

        if sam_features.dim() == 3:
            sam_features = sam_features.unsqueeze(0)
        if sam_features.dim() != 4:
            return None

        sam_features = sam_features.to(device=device, dtype=torch.float32)
        if sam_features.shape[-2:] != spatial_hw:
            sam_features = F.interpolate(
                sam_features,
                size=spatial_hw,
                mode='bilinear',
                align_corners=False,
            )

        bsz, channels, height, width = sam_features.shape
        tokens = sam_features.flatten(2).transpose(1, 2).contiguous()
        tokens = F.normalize(tokens, dim=-1, eps=1e-6)

        if self.spatial_weight > 0.0:
            ys = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=tokens.dtype)
            xs = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=tokens.dtype)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
            spatial_tokens = torch.stack([grid_y, grid_x], dim=-1).view(1, height * width, 2)
            spatial_tokens = spatial_tokens.expand(bsz, -1, -1)
            tokens = torch.cat([tokens, self.spatial_weight * spatial_tokens], dim=-1)
            tokens = F.normalize(tokens, dim=-1, eps=1e-6)

        return tokens

    def _build_anchor_indices(self, spatial_hw, num_clusters, device):
        height, width = spatial_hw
        grid_side = int(math.ceil(math.sqrt(num_clusters)))
        ys = torch.linspace(0, max(height - 1, 0), steps=grid_side, device=device).round().long()
        xs = torch.linspace(0, max(width - 1, 0), steps=grid_side, device=device).round().long()
        anchor_indices = []
        for y in ys.tolist():
            for x in xs.tolist():
                anchor_indices.append(int(y) * width + int(x))
                if len(anchor_indices) >= num_clusters:
                    return torch.tensor(anchor_indices, device=device, dtype=torch.long)
        return torch.tensor(anchor_indices, device=device, dtype=torch.long)

    def _cluster_assign(self, tokens, prototypes):
        similarities = torch.bmm(tokens, prototypes.transpose(1, 2))
        assignments = torch.argmax(similarities, dim=-1)
        confidence = torch.gather(similarities, 2, assignments.unsqueeze(-1)).squeeze(-1)
        return assignments, confidence

    def _update_prototypes(self, tokens, prototypes, assignments):
        batch_size, _, feature_dim = tokens.shape
        cluster_count = prototypes.shape[1]
        updated = []
        for cluster_idx in range(cluster_count):
            mask = (assignments == cluster_idx).float().unsqueeze(-1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            proto = (tokens * mask).sum(dim=1) / denom
            empty_mask = (mask.sum(dim=1).squeeze(-1) <= 0)
            if empty_mask.any():
                proto[empty_mask] = prototypes[empty_mask, cluster_idx, :]
            updated.append(proto)
        updated = torch.stack(updated, dim=1)
        return F.normalize(updated, dim=-1, eps=1e-6)

    def forward(self, sam_region_context, spatial_hw, device):
        tokens = self._prepare_feature_tokens(sam_region_context, spatial_hw, device)
        if tokens is None:
            return None, None, None

        batch_size, token_count, _ = tokens.shape
        cluster_count = min(self.num_clusters, token_count)
        if cluster_count <= 0:
            return None, None, None

        anchor_indices = self._build_anchor_indices(spatial_hw, cluster_count, device)
        prototypes = torch.gather(
            tokens,
            dim=1,
            index=anchor_indices.view(1, cluster_count, 1).expand(batch_size, cluster_count, tokens.shape[-1]),
        )
        prototypes = F.normalize(prototypes, dim=-1, eps=1e-6)

        assignments = None
        for _ in range(self.num_iters):
            assignments, _ = self._cluster_assign(tokens, prototypes)
            prototypes = self._update_prototypes(tokens, prototypes, assignments)

        region_ids = assignments + 1
        _, scan_indices = torch.sort(region_ids, dim=-1, stable=True)
        reverse_indices = index_reverse(scan_indices)
        return region_ids.contiguous(), scan_indices, reverse_indices


class WaveletAugmentedSemanticScanner(SAMSemanticScanner):
    def __init__(
        self,
        num_regions=6,
        mask_threshold=0.05,
        topk_ratio=0.25,
        strength=0.5,
        mode='stable_intra_region',
    ):
        super().__init__(num_regions=num_regions, mask_threshold=mask_threshold)
        self.topk_ratio = float(max(min(topk_ratio, 1.0), 0.0))
        self.strength = float(max(min(strength, 1.0), 0.0))
        self.mode = mode

    def _build_wavelet_scores(self, sam_region_context, spatial_hw, device, target_batch):
        if sam_region_context is None:
            return None

        prior_map = sam_region_context.get('semantic_frequency_prior_map')
        if prior_map is None:
            prior_map = sam_region_context.get('wavelet_guidance')
        if prior_map is None:
            prior_map = sam_region_context.get('wavelet_prior_map')
        if prior_map is None:
            return None

        if prior_map.dim() == 3:
            prior_map = prior_map.unsqueeze(1)
        if prior_map.dim() != 4:
            return None

        prior_map = prior_map.to(device=device, dtype=torch.float32)
        if prior_map.shape[0] != target_batch:
            if prior_map.shape[0] == 1:
                prior_map = prior_map.expand(target_batch, -1, -1, -1).contiguous()
            else:
                return None
        if prior_map.shape[1] != 1:
            prior_map = prior_map.mean(dim=1, keepdim=True)
        if prior_map.shape[-2:] != spatial_hw:
            prior_map = F.interpolate(
                prior_map,
                size=spatial_hw,
                mode='bilinear',
                align_corners=False,
            )

        wavelet_scores = prior_map.flatten(1)
        min_val = wavelet_scores.min(dim=-1, keepdim=True)[0]
        max_val = wavelet_scores.max(dim=-1, keepdim=True)[0]
        return (wavelet_scores - min_val) / (max_val - min_val + 1e-6)

    def _stable_intra_region_refine(self, region_ids, scan_indices, wavelet_scores):
        if (
            wavelet_scores is None
            or self.topk_ratio <= 0.0
            or self.strength <= 0.0
            or self.mode != 'stable_intra_region'
        ):
            return scan_indices

        refined_indices = scan_indices.clone()
        batch_size = refined_indices.shape[0]
        for batch_idx in range(batch_size):
            ordered_tokens = refined_indices[batch_idx].clone()
            ordered_region_ids = region_ids[batch_idx, ordered_tokens]
            ordered_wavelet_scores = wavelet_scores[batch_idx, ordered_tokens]

            for region_id in torch.unique_consecutive(ordered_region_ids).tolist():
                if region_id <= 0:
                    continue
                region_positions = torch.nonzero(
                    ordered_region_ids == region_id,
                    as_tuple=False,
                ).flatten()
                region_size = int(region_positions.numel())
                if region_size <= 1:
                    continue

                region_tokens = ordered_tokens[region_positions].clone()
                region_scores = ordered_wavelet_scores[region_positions]
                topk = max(1, int(math.ceil(region_size * self.topk_ratio)))
                topk = min(topk, region_size)

                local_order = torch.arange(region_size, device=region_tokens.device)
                position_prior = torch.linspace(
                    1.0,
                    0.0,
                    steps=region_size,
                    device=region_scores.device,
                    dtype=region_scores.dtype,
                )
                blended_scores = self.strength * region_scores + (1.0 - self.strength) * position_prior
                sorted_local = torch.argsort(blended_scores, descending=True, stable=True)

                promoted = sorted_local[:topk]
                promoted_mask = torch.zeros(region_size, device=region_tokens.device, dtype=torch.bool)
                promoted_mask[promoted] = True
                remaining = local_order[~promoted_mask]
                final_local_order = torch.cat([promoted, remaining], dim=0)

                ordered_tokens[region_positions] = region_tokens[final_local_order]

            refined_indices[batch_idx] = ordered_tokens
        return refined_indices

    def forward(self, sam_region_context, spatial_hw, device):
        region_ids = self._build_region_ids(sam_region_context, spatial_hw, device)
        if region_ids is None:
            return None, None, None

        _, scan_indices = torch.sort(region_ids, dim=-1, stable=True)
        wavelet_scores = self._build_wavelet_scores(
            sam_region_context,
            spatial_hw,
            device,
            target_batch=region_ids.shape[0],
        )
        scan_indices = self._stable_intra_region_refine(region_ids, scan_indices, wavelet_scores)
        reverse_indices = index_reverse(scan_indices)
        return region_ids, scan_indices, reverse_indices


class BaseSemanticStateOrganizer(nn.Module):
    def __init__(self, num_regions=6, mask_threshold=0.05):
        super().__init__()
        self.num_regions = num_regions
        self.mask_threshold = mask_threshold

    @staticmethod
    def _ensure_4d_map(map_tensor):
        if map_tensor is None:
            return None
        if map_tensor.dim() == 3:
            return map_tensor.unsqueeze(1)
        if map_tensor.dim() == 4:
            return map_tensor
        return None

    @staticmethod
    def _match_batch_dim(tensor, target_batch, name="tensor"):
        if tensor is None:
            return None
        if tensor.shape[0] == target_batch:
            return tensor
        if tensor.shape[0] == 1:
            expand_shape = [target_batch] + [-1] * (tensor.dim() - 1)
            return tensor.expand(*expand_shape).contiguous()
        raise ValueError(
            f"{name} batch mismatch: got {tuple(tensor.shape)}, expected batch {target_batch}"
        )

    def _prepare_masks(self, sam_region_context, spatial_hw, device):
        if sam_region_context is None:
            return None
        sam_masks = sam_region_context.get('sam_masks')
        if sam_masks is None:
            return None
        if sam_masks.dim() == 3:
            sam_masks = sam_masks.unsqueeze(0)
        if sam_masks.dim() != 4:
            return None
        sam_masks = sam_masks.to(device=device, dtype=torch.float32)
        if sam_masks.shape[-2:] != spatial_hw:
            sam_masks = F.interpolate(
                sam_masks,
                size=spatial_hw,
                mode='bilinear',
                align_corners=False,
            )
        return sam_masks

    def _compute_region_scores(self, sam_masks, sam_region_context):
        return sam_masks.flatten(-2).mean(dim=-1)

    def _select_region_masks(self, sam_masks, sam_region_context):
        bsz, num_masks, feat_h, feat_w = sam_masks.shape
        if num_masks <= 0:
            return None
        region_count = min(self.num_regions, num_masks)
        region_scores = self._compute_region_scores(sam_masks, sam_region_context)
        topk_idx = torch.topk(region_scores, k=region_count, dim=1).indices
        gather_idx = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, feat_h, feat_w)
        selected_masks = torch.gather(sam_masks, 1, gather_idx)
        return selected_masks

    def _build_region_ids_and_scan(self, selected_masks):
        if selected_masks is None:
            return None, None, None
        region_strength, region_index = selected_masks.max(dim=1)
        region_ids = region_index + 1
        region_ids = torch.where(
            region_strength > self.mask_threshold,
            region_ids,
            torch.zeros_like(region_ids),
        )
        region_ids = region_ids.reshape(region_ids.shape[0], -1).contiguous()
        _, scan_indices = torch.sort(region_ids, dim=-1, stable=True)
        reverse_indices = index_reverse(scan_indices)
        return region_ids, scan_indices, reverse_indices

    def _derive_boundary_map_from_regions(self, region_ids, spatial_hw):
        if region_ids is None:
            return None
        bsz = region_ids.shape[0]
        h, w = spatial_hw
        region_map = region_ids.view(bsz, h, w)
        boundary = torch.zeros_like(region_map, dtype=torch.float32)
        horizontal = (region_map[:, :, 1:] != region_map[:, :, :-1]).to(boundary.dtype)
        vertical = (region_map[:, 1:, :] != region_map[:, :-1, :]).to(boundary.dtype)
        boundary[:, :, 1:] = torch.maximum(boundary[:, :, 1:], horizontal)
        boundary[:, :, :-1] = torch.maximum(boundary[:, :, :-1], horizontal)
        boundary[:, 1:, :] = torch.maximum(boundary[:, 1:, :], vertical)
        boundary[:, :-1, :] = torch.maximum(boundary[:, :-1, :], vertical)
        return boundary.unsqueeze(1)

    def _resize_prior_map(self, prior_bank, key, spatial_hw, device):
        if prior_bank is None:
            return None
        prior_map = self._ensure_4d_map(prior_bank.get(key))
        if prior_map is None:
            return None
        prior_map = prior_map.to(device=device, dtype=torch.float32)
        if prior_map.shape[-2:] != spatial_hw:
            prior_map = F.interpolate(
                prior_map,
                size=spatial_hw,
                mode='bilinear',
                align_corners=False,
            )
        return prior_map

    def _build_token_gates(self, region_ids, spatial_hw, device, sam_prior_bank,
                           boundary_scale=0.0, reset_scale=0.0, wavelet_guidance=None,
                           wavelet_scale=0.0):
        if region_ids is None or (boundary_scale <= 0 and reset_scale <= 0):
            return None, None

        boundary_map = self._resize_prior_map(sam_prior_bank, 'boundary_map', spatial_hw, device)
        prompt_strength = self._resize_prior_map(sam_prior_bank, 'prompt_strength_map', spatial_hw, device)
        confidence_map = self._resize_prior_map(sam_prior_bank, 'confidence_map', spatial_hw, device)

        if boundary_map is None:
            boundary_map = self._derive_boundary_map_from_regions(region_ids, spatial_hw)
            if boundary_map is None:
                return None, None

        gate_source = boundary_map
        if prompt_strength is not None:
            gate_source = gate_source * (1.0 + 0.5 * prompt_strength)
        if confidence_map is not None:
            gate_source = gate_source * confidence_map.clamp(min=0.0, max=1.0)
        wavelet_map = self._ensure_4d_map(wavelet_guidance)
        if wavelet_map is not None and wavelet_scale > 0:
            wavelet_map = wavelet_map.to(device=device, dtype=torch.float32)
            if wavelet_map.shape[-2:] != spatial_hw:
                wavelet_map = F.interpolate(
                    wavelet_map,
                    size=spatial_hw,
                    mode='bilinear',
                    align_corners=False,
                )
            if wavelet_map.shape[1] != 1:
                wavelet_map = wavelet_map.mean(dim=1, keepdim=True)
            gate_source = gate_source * (1.0 + wavelet_scale * wavelet_map.clamp(min=0.0))

        gate_flat = gate_source.flatten(1)
        gate_max = gate_flat.max(dim=1, keepdim=True)[0].clamp(min=1e-6)
        gate_flat = (gate_flat / gate_max).clamp(min=0.0, max=1.0)

        boundary_gate = (boundary_scale * gate_flat).clamp(min=0.0, max=1.0) if boundary_scale > 0 else None
        reset_gate = (reset_scale * gate_flat).clamp(min=0.0, max=1.0) if reset_scale > 0 else None
        return boundary_gate, reset_gate


class SAMStateOrganizerV1(BaseSemanticStateOrganizer):
    def __init__(self, num_regions=6, mask_threshold=0.05, boundary_scale=0.1, reset_scale=0.15):
        super().__init__(num_regions=num_regions, mask_threshold=mask_threshold)
        self.boundary_scale = boundary_scale
        self.reset_scale = reset_scale

    def forward(self, sam_region_context, sam_prior_bank, spatial_hw, device):
        sam_masks = self._prepare_masks(sam_region_context, spatial_hw, device)
        if sam_masks is None:
            return None
        selected_masks = self._select_region_masks(sam_masks, sam_region_context)
        region_ids, scan_indices, reverse_indices = self._build_region_ids_and_scan(selected_masks)
        boundary_gate, reset_gate = self._build_token_gates(
            region_ids,
            spatial_hw,
            device,
            sam_prior_bank,
            boundary_scale=self.boundary_scale,
            reset_scale=self.reset_scale,
        )
        return {
            'region_ids': region_ids,
            'scan_indices': scan_indices,
            'scan_reverse_indices': reverse_indices,
            'boundary_gate': boundary_gate,
            'reset_gate': reset_gate,
        }


class WaveletGuidedSemanticStateOrganizer(BaseSemanticStateOrganizer):
    def __init__(self, num_regions=6, mask_threshold=0.05, wavelet_scale=0.05,
                 boundary_scale=0.1, reset_scale=0.15):
        super().__init__(num_regions=num_regions, mask_threshold=mask_threshold)
        self.wavelet_scale = wavelet_scale
        self.boundary_scale = boundary_scale
        self.reset_scale = reset_scale

    def _compute_region_scores(self, sam_masks, sam_region_context):
        mask_area = sam_masks.flatten(-2).mean(dim=-1)
        wavelet_guidance = None
        if sam_region_context is not None:
            wavelet_guidance = sam_region_context.get('semantic_frequency_prior_map')
            if wavelet_guidance is None:
                wavelet_guidance = sam_region_context.get('wavelet_guidance')
        wavelet_map = self._ensure_4d_map(wavelet_guidance)
        if wavelet_map is None or self.wavelet_scale <= 0:
            return mask_area
        wavelet_map = wavelet_map.to(device=sam_masks.device, dtype=torch.float32)
        if wavelet_map.shape[-2:] != sam_masks.shape[-2:]:
            wavelet_map = F.interpolate(
                wavelet_map,
                size=sam_masks.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        if wavelet_map.shape[1] != 1:
            wavelet_map = wavelet_map.mean(dim=1, keepdim=True)
        region_energy = (sam_masks * wavelet_map).flatten(-2).sum(dim=-1)
        region_energy = region_energy / sam_masks.flatten(-2).sum(dim=-1).clamp(min=1e-6)
        return mask_area * (1.0 + self.wavelet_scale * region_energy)

    def forward(self, sam_region_context, sam_prior_bank, spatial_hw, device):
        sam_masks = self._prepare_masks(sam_region_context, spatial_hw, device)
        if sam_masks is None:
            return None
        selected_masks = self._select_region_masks(sam_masks, sam_region_context)
        region_ids, scan_indices, reverse_indices = self._build_region_ids_and_scan(selected_masks)
        boundary_gate, reset_gate = self._build_token_gates(
            region_ids,
            spatial_hw,
            device,
            sam_prior_bank,
            boundary_scale=self.boundary_scale,
            reset_scale=self.reset_scale,
            wavelet_guidance=(
                sam_region_context.get('semantic_frequency_prior_map')
                if sam_region_context is not None and sam_region_context.get('semantic_frequency_prior_map') is not None
                else (sam_region_context.get('wavelet_guidance') if sam_region_context is not None else None)
            ),
            wavelet_scale=self.wavelet_scale,
        )
        return {
            'region_ids': region_ids,
            'scan_indices': scan_indices,
            'scan_reverse_indices': reverse_indices,
            'boundary_gate': boundary_gate,
            'reset_gate': reset_gate,
        }


class SAMRegionSpecificPromptSubspace(nn.Module):
    def __init__(self, feature_dim, d_state, sam_feature_dim=256, num_regions=6):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_state = d_state
        self.num_regions = num_regions

        self.sam_feature_proj = nn.Conv2d(sam_feature_dim, feature_dim, 1, 1, 0)
        self.region_fuse = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, d_state),
        )
        self.out_proj = nn.Linear(d_state, d_state)

        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, fusion_tokens, sam_region_context, spatial_hw):
        if sam_region_context is None:
            return None

        sam_features = sam_region_context.get('sam_features')
        sam_masks = sam_region_context.get('sam_masks')
        if sam_features is None or sam_masks is None:
            return None

        if sam_features.dim() == 3:
            sam_features = sam_features.unsqueeze(0)
        if sam_masks.dim() == 3:
            sam_masks = sam_masks.unsqueeze(0)
        if sam_features.dim() != 4 or sam_masks.dim() != 4:
            return None

        sam_features = sam_features.to(device=fusion_tokens.device, dtype=torch.float32)
        sam_masks = sam_masks.to(device=fusion_tokens.device, dtype=torch.float32)
        projected_features = self.sam_feature_proj(sam_features)
        if sam_masks.shape[-2:] != projected_features.shape[-2:]:
            sam_masks = F.interpolate(
                sam_masks,
                size=projected_features.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

        bsz, num_masks, feat_h, feat_w = sam_masks.shape
        if num_masks <= 0:
            return None

        region_count = min(self.num_regions, num_masks)
        mask_area = sam_masks.flatten(-2).mean(dim=-1)
        topk_idx = torch.topk(mask_area, k=region_count, dim=1).indices
        gather_idx = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, feat_h, feat_w)
        selected_masks = torch.gather(sam_masks, 1, gather_idx)

        norm = selected_masks.flatten(-2).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        masked_features = projected_features.unsqueeze(1) * selected_masks.unsqueeze(2)
        prototypes = masked_features.flatten(-2).sum(dim=-1) / norm

        fusion_context = fusion_tokens.mean(dim=1, keepdim=True).to(dtype=prototypes.dtype)
        fusion_context = fusion_context.expand(-1, region_count, -1)
        region_subspace = self.region_fuse(torch.cat([prototypes, fusion_context], dim=-1))

        if selected_masks.shape[-2:] != spatial_hw:
            selected_masks = F.interpolate(
                selected_masks,
                size=spatial_hw,
                mode='bilinear',
                align_corners=False,
            )
        region_weights = selected_masks / selected_masks.sum(dim=1, keepdim=True).clamp(min=1e-6)
        token_prompt_map = torch.einsum('bkhw,bkd->bdhw', region_weights, region_subspace)
        token_prompt_seq = rearrange(token_prompt_map, 'b d h w -> b (h w) d').contiguous()
        return torch.tanh(self.out_proj(token_prompt_seq)).to(dtype=fusion_tokens.dtype)


class SemanticFrequencyDualPrototypeConditioner(nn.Module):
    def __init__(self, feature_dim, num_tokens, d_state, sam_feature_dim=256, num_prototypes=6):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_tokens = num_tokens
        self.d_state = d_state
        self.num_prototypes = num_prototypes

        self.sam_feature_proj = nn.Conv2d(sam_feature_dim, feature_dim, 1, 1, 0)
        self.semantic_proj = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, d_state),
        )
        self.frequency_proj = nn.Sequential(
            nn.Linear(3, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, d_state),
        )
        self.fuse_proj = nn.Sequential(
            nn.Linear(d_state * 2, d_state),
            nn.LeakyReLU(),
            nn.Linear(d_state, d_state),
        )
        self.out_proj = nn.Linear(d_state, d_state)

        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _compute_region_frequency_stats(self, selected_masks, sam_region_context):
        region_frequency_stats = sam_region_context.get('region_frequency_stats') if sam_region_context is not None else None
        if region_frequency_stats is not None and region_frequency_stats.dim() == 3:
            total_regions = region_frequency_stats.shape[1]
            target_regions = selected_masks.shape[1]
            if total_regions == target_regions:
                return region_frequency_stats.to(device=selected_masks.device, dtype=selected_masks.dtype)

        spatial_prior = sam_region_context.get('wavelet_prior_map') if sam_region_context is not None else None
        spectral_prior = sam_region_context.get('spectral_prior_map') if sam_region_context is not None else None
        joint_prior = sam_region_context.get('semantic_frequency_prior_map') if sam_region_context is not None else None

        pooled_stats = []
        for prior_map in (spatial_prior, spectral_prior, joint_prior):
            if prior_map is None:
                pooled_stats.append(
                    torch.zeros(
                        selected_masks.shape[0],
                        selected_masks.shape[1],
                        device=selected_masks.device,
                        dtype=selected_masks.dtype,
                    )
                )
                continue
            if prior_map.dim() == 3:
                prior_map = prior_map.unsqueeze(1)
            if prior_map.shape[-2:] != selected_masks.shape[-2:]:
                prior_map = F.interpolate(
                    prior_map,
                    size=selected_masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            if prior_map.shape[1] != 1:
                prior_map = prior_map.mean(dim=1, keepdim=True)
            prior_map = prior_map.to(device=selected_masks.device, dtype=selected_masks.dtype)
            weighted = (selected_masks * prior_map).flatten(-2).sum(dim=-1)
            norm = selected_masks.flatten(-2).sum(dim=-1).clamp(min=1e-6)
            pooled_stats.append(weighted / norm)
        return torch.stack(pooled_stats, dim=-1).contiguous()

    def forward(self, fusion_tokens, sam_region_context, semantic_scale=0.05, frequency_scale=0.05):
        if sam_region_context is None:
            return None

        sam_features = sam_region_context.get('sam_features')
        sam_masks = sam_region_context.get('sam_masks')
        if sam_features is None or sam_masks is None:
            return None

        if sam_features.dim() == 3:
            sam_features = sam_features.unsqueeze(0)
        if sam_masks.dim() == 3:
            sam_masks = sam_masks.unsqueeze(0)
        if sam_features.dim() != 4 or sam_masks.dim() != 4:
            return None

        sam_features = sam_features.to(device=fusion_tokens.device, dtype=torch.float32)
        sam_masks = sam_masks.to(device=fusion_tokens.device, dtype=torch.float32)
        projected_features = self.sam_feature_proj(sam_features)
        if sam_masks.shape[-2:] != projected_features.shape[-2:]:
            sam_masks = F.interpolate(
                sam_masks,
                size=projected_features.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

        bsz, num_masks, feat_h, feat_w = sam_masks.shape
        if num_masks <= 0:
            return None

        proto_count = min(self.num_prototypes, num_masks)
        mask_area = sam_masks.flatten(-2).mean(dim=-1)
        topk_idx = torch.topk(mask_area, k=proto_count, dim=1).indices
        gather_idx = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, feat_h, feat_w)
        selected_masks = torch.gather(sam_masks, 1, gather_idx)

        norm = selected_masks.flatten(-2).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        masked_features = projected_features.unsqueeze(1) * selected_masks.unsqueeze(2)
        semantic_prototypes = masked_features.flatten(-2).sum(dim=-1) / norm

        fusion_context = fusion_tokens.mean(dim=1, keepdim=True).to(dtype=semantic_prototypes.dtype)
        fusion_context = fusion_context.expand(-1, proto_count, -1)
        semantic_state = self.semantic_proj(torch.cat([semantic_prototypes, fusion_context], dim=-1))

        frequency_stats = self._compute_region_frequency_stats(selected_masks, sam_region_context)
        frequency_state = self.frequency_proj(frequency_stats.to(dtype=semantic_state.dtype))

        fused_proto = self.fuse_proj(
            torch.cat(
                [
                    float(semantic_scale) * semantic_state,
                    float(frequency_scale) * frequency_state,
                ],
                dim=-1,
            )
        )
        fused_proto = fused_proto.permute(0, 2, 1)
        if fused_proto.shape[-1] != self.num_tokens:
            fused_proto = F.interpolate(
                fused_proto,
                size=self.num_tokens,
                mode='linear',
                align_corners=False,
            )
        fused_proto = fused_proto.permute(0, 2, 1).contiguous()
        return torch.tanh(self.out_proj(fused_proto)).to(dtype=fusion_tokens.dtype)


class SemanticFrequencyStateModulator(BaseSemanticStateOrganizer):
    def __init__(
        self,
        num_regions=6,
        mask_threshold=0.05,
        hidden_dim=16,
        write_scale=0.08,
        read_scale=0.08,
        delta_scale=0.05,
    ):
        super().__init__(num_regions=num_regions, mask_threshold=mask_threshold)
        self.write_scale = write_scale
        self.read_scale = read_scale
        self.delta_scale = delta_scale
        self.gate_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.gate_mlp[-1].weight)
        if self.gate_mlp[-1].bias is not None:
            nn.init.zeros_(self.gate_mlp[-1].bias)

    def _resolve_prior_map(self, sam_region_context, spatial_hw, device):
        if sam_region_context is None:
            return None
        prior_map = sam_region_context.get('semantic_frequency_prior_map')
        if prior_map is None:
            prior_map = sam_region_context.get('wavelet_guidance')
        prior_map = self._ensure_4d_map(prior_map)
        if prior_map is None:
            return None
        prior_map = prior_map.to(device=device, dtype=torch.float32)
        if prior_map.shape[-2:] != spatial_hw:
            prior_map = F.interpolate(prior_map, size=spatial_hw, mode='bilinear', align_corners=False)
        if prior_map.shape[1] != 1:
            prior_map = prior_map.mean(dim=1, keepdim=True)
        return prior_map

    def forward(self, sam_region_context, spatial_hw, device):
        sam_masks = self._prepare_masks(sam_region_context, spatial_hw, device)
        if sam_masks is None:
            return None
        selected_masks = self._select_region_masks(sam_masks, sam_region_context)
        region_ids, _, _ = self._build_region_ids_and_scan(selected_masks)
        if region_ids is None:
            return None

        prior_map = self._resolve_prior_map(sam_region_context, spatial_hw, device)
        boundary_map = self._derive_boundary_map_from_regions(region_ids, spatial_hw)
        if prior_map is None or boundary_map is None:
            return None

        target_batch = max(selected_masks.shape[0], prior_map.shape[0], boundary_map.shape[0])
        selected_masks = self._match_batch_dim(selected_masks, target_batch, "selected_masks")
        prior_map = self._match_batch_dim(prior_map, target_batch, "semantic_frequency_prior_map")
        boundary_map = self._match_batch_dim(boundary_map, target_batch, "semantic_boundary_map")

        if selected_masks.shape[-2:] != spatial_hw:
            selected_masks = F.interpolate(
                selected_masks,
                size=spatial_hw,
                mode='bilinear',
                align_corners=False,
            )

        norm = selected_masks.flatten(-2).sum(dim=-1).clamp(min=1e-6)
        region_prior = (selected_masks * prior_map).flatten(-2).sum(dim=-1) / norm
        region_boundary = (selected_masks * boundary_map).flatten(-2).sum(dim=-1) / norm
        region_stats = torch.stack([region_prior, region_boundary], dim=-1)

        region_delta = torch.tanh(self.gate_mlp(region_stats))
        region_write = (1.0 + self.write_scale * region_delta[..., 0]).clamp(min=0.5, max=1.5)
        region_read = (1.0 + self.read_scale * region_delta[..., 1]).clamp(min=0.5, max=1.5)
        region_step = (1.0 + self.delta_scale * region_delta[..., 2]).clamp(min=0.5, max=1.5)

        region_weights = selected_masks / selected_masks.sum(dim=1, keepdim=True).clamp(min=1e-6)
        token_write = torch.einsum('bkhw,bk->bhw', region_weights, region_write).reshape(region_weights.shape[0], -1)
        token_read = torch.einsum('bkhw,bk->bhw', region_weights, region_read).reshape(region_weights.shape[0], -1)
        token_step = torch.einsum('bkhw,bk->bhw', region_weights, region_step).reshape(region_weights.shape[0], -1)
        return {
            'state_write_gate': token_write,
            'state_read_gate': token_read,
            'state_delta_gate': token_step,
        }

class FusionMamba(nn.Module):
    def __init__(self, dim, H=64, W=64, depth=1, final=False, use_ase=False, num_ase_prompts=32, ase_rank=8,
                 ase_prompt_mode='hard', ase_route_temperature=1.0, ase_prompt_soft_mix=0.5,
                 ase_scope='all', use_ase_fusion_residual=False, ase_fusion_res_scale=0.3,
                 use_learnable_ase_fusion_res_scale=False,
                 use_sam_semantic_prompt_bank=False, sam_semantic_prompt_bank_scale=0.1,
                 use_sam_region_prototype_bank=False, sam_region_prototype_bank_scale=0.1,
                 sam_region_prototype_count=8,
                 use_wavelet_guided_sam_prototype_scaling=False, wavelet_guided_sam_prototype_scale=0.1,
                 use_sam_region_prompt_mixture=False, sam_region_prompt_mixture_scale=0.05,
                 sam_region_prompt_mixture_count=8,
                 use_sam_guided_semantic_scanning=False, sam_semantic_scanning_count=6,
                 use_sam_feature_cluster_scanning=False, sam_feature_cluster_count=6,
                 sam_feature_cluster_iters=2, sam_feature_cluster_spatial_weight=0.05,
                 use_wavelet_augmented_ss1=False, wavelet_augmented_ss1_count=6,
                 wavelet_augmented_ss1_topk_ratio=0.25,
                 wavelet_augmented_ss1_strength=0.5,
                 wavelet_augmented_ss1_mode='stable_intra_region',
                 use_sam_boundary_aware_state_propagation=False, sam_boundary_aware_state_scale=0.2,
                 use_sam_state_reset_stronger=False, sam_state_reset_scale=0.35,
                 use_sam_state_organizer_v1=False, sam_state_organizer_count=6,
                 sam_state_organizer_boundary_scale=0.1, sam_state_organizer_reset_scale=0.15,
                 use_sam_region_prompt_subspace=False, sam_region_prompt_subspace_scale=0.05,
                 sam_region_prompt_subspace_count=6,
                 use_wavelet_guided_semantic_state_organization=False, wavelet_guided_semantic_state_count=6,
                 wavelet_guided_semantic_state_scale=0.05,
                 wavelet_guided_semantic_boundary_scale=0.1,
                 wavelet_guided_semantic_reset_scale=0.15,
                 use_dual_prototype_bank=False, dual_prototype_semantic_scale=0.05,
                 dual_prototype_frequency_scale=0.05, dual_prototype_count=6,
                 use_semantic_frequency_state_modulation=False, semantic_frequency_state_count=6,
                 semantic_frequency_state_write_scale=0.08,
                 semantic_frequency_state_read_scale=0.08,
                 semantic_frequency_state_delta_scale=0.05,
                 use_wavelet_local_bias=False, wavelet_local_bias_scale=0.1,
                 use_wavelet_local_gate=False, wavelet_local_gate_scale=0.1,
                 use_sam_local_gate=False, sam_local_gate_scale=0.1,
                 use_sam_ase=False, sam_checkpoint=None, sam_prompt_dim=64,
                 use_learnable_prompts=False, num_learnable_prompts=16,
                 use_soft_masks=False, num_soft_regions=8,
                 input_channels=4, use_wavelet=False,
                 use_structure_guided_sam_ase=False, structure_texture_weight=0.25,

                 use_fass=False, fass_compression_ratio=2, fass_threshold=0.5,
                 fass_sparsity_target=0.3, fass_ll_sparsity=0.25, fass_hf_sparsity=0.08,
                 fass_d_state=16,

                 train_mode='auto', dense_epochs=100,
                 gating_loss_weight=None):
        super().__init__()
        self.final = final
        self.dim = dim
        self.depth = depth
        self.use_ase = use_ase
        self.use_sam_ase = use_sam_ase
        self.use_learnable_prompts = use_learnable_prompts
        self.num_learnable_prompts = num_learnable_prompts
        self.use_soft_masks = use_soft_masks
        self.num_soft_regions = num_soft_regions
        self.input_channels = input_channels
        self.use_wavelet = use_wavelet
        self.use_fass = use_fass
        self.fass_compression_ratio = fass_compression_ratio
        self.fass_threshold = fass_threshold
        self.fass_sparsity_target = fass_sparsity_target
        self.fass_ll_sparsity = fass_ll_sparsity
        self.fass_hf_sparsity = fass_hf_sparsity
        self.fass_d_state = fass_d_state
        self.train_mode = train_mode
        self.dense_epochs = dense_epochs
        self.gating_loss_weight = gating_loss_weight
        self.use_structure_guided_sam_ase = use_structure_guided_sam_ase
        self.structure_texture_weight = structure_texture_weight
        self.ase_prompt_mode = ase_prompt_mode
        self.ase_route_temperature = ase_route_temperature
        self.ase_prompt_soft_mix = ase_prompt_soft_mix
        self.ase_scope = ase_scope
        self.use_ase_fusion_residual = use_ase_fusion_residual
        self.ase_fusion_res_scale = ase_fusion_res_scale
        self.use_learnable_ase_fusion_res_scale = use_learnable_ase_fusion_res_scale
        self.use_sam_semantic_prompt_bank = use_sam_semantic_prompt_bank
        self.sam_semantic_prompt_bank_scale = sam_semantic_prompt_bank_scale
        self.use_sam_region_prototype_bank = use_sam_region_prototype_bank
        self.sam_region_prototype_bank_scale = sam_region_prototype_bank_scale
        self.sam_region_prototype_count = sam_region_prototype_count
        self.use_wavelet_guided_sam_prototype_scaling = use_wavelet_guided_sam_prototype_scaling
        self.wavelet_guided_sam_prototype_scale = wavelet_guided_sam_prototype_scale
        self.use_sam_region_prompt_mixture = use_sam_region_prompt_mixture
        self.sam_region_prompt_mixture_scale = sam_region_prompt_mixture_scale
        self.sam_region_prompt_mixture_count = sam_region_prompt_mixture_count
        self.use_sam_guided_semantic_scanning = use_sam_guided_semantic_scanning
        self.sam_semantic_scanning_count = sam_semantic_scanning_count
        self.use_sam_feature_cluster_scanning = use_sam_feature_cluster_scanning
        self.sam_feature_cluster_count = sam_feature_cluster_count
        self.sam_feature_cluster_iters = sam_feature_cluster_iters
        self.sam_feature_cluster_spatial_weight = sam_feature_cluster_spatial_weight
        self.use_wavelet_augmented_ss1 = use_wavelet_augmented_ss1
        self.wavelet_augmented_ss1_count = wavelet_augmented_ss1_count
        self.wavelet_augmented_ss1_topk_ratio = wavelet_augmented_ss1_topk_ratio
        self.wavelet_augmented_ss1_strength = wavelet_augmented_ss1_strength
        self.wavelet_augmented_ss1_mode = wavelet_augmented_ss1_mode
        self.use_sam_boundary_aware_state_propagation = use_sam_boundary_aware_state_propagation
        self.sam_boundary_aware_state_scale = sam_boundary_aware_state_scale
        self.use_sam_state_reset_stronger = use_sam_state_reset_stronger
        self.sam_state_reset_scale = sam_state_reset_scale
        self.use_sam_state_organizer_v1 = use_sam_state_organizer_v1
        self.sam_state_organizer_count = sam_state_organizer_count
        self.sam_state_organizer_boundary_scale = sam_state_organizer_boundary_scale
        self.sam_state_organizer_reset_scale = sam_state_organizer_reset_scale
        self.use_sam_region_prompt_subspace = use_sam_region_prompt_subspace
        self.sam_region_prompt_subspace_scale = sam_region_prompt_subspace_scale
        self.sam_region_prompt_subspace_count = sam_region_prompt_subspace_count
        self.use_wavelet_guided_semantic_state_organization = use_wavelet_guided_semantic_state_organization
        self.wavelet_guided_semantic_state_count = wavelet_guided_semantic_state_count
        self.wavelet_guided_semantic_state_scale = wavelet_guided_semantic_state_scale
        self.wavelet_guided_semantic_boundary_scale = wavelet_guided_semantic_boundary_scale
        self.wavelet_guided_semantic_reset_scale = wavelet_guided_semantic_reset_scale
        self.use_dual_prototype_bank = use_dual_prototype_bank
        self.dual_prototype_semantic_scale = dual_prototype_semantic_scale
        self.dual_prototype_frequency_scale = dual_prototype_frequency_scale
        self.dual_prototype_count = dual_prototype_count
        self.use_semantic_frequency_state_modulation = use_semantic_frequency_state_modulation
        self.semantic_frequency_state_count = semantic_frequency_state_count
        self.semantic_frequency_state_write_scale = semantic_frequency_state_write_scale
        self.semantic_frequency_state_read_scale = semantic_frequency_state_read_scale
        self.semantic_frequency_state_delta_scale = semantic_frequency_state_delta_scale
        self.use_wavelet_local_bias = use_wavelet_local_bias
        self.wavelet_local_bias_scale = wavelet_local_bias_scale
        self.use_wavelet_local_gate = use_wavelet_local_gate
        self.wavelet_local_gate_scale = wavelet_local_gate_scale
        self.use_sam_local_gate = use_sam_local_gate
        self.sam_local_gate_scale = sam_local_gate_scale
        self.current_ase_routing_probs = None

        if self.use_ase_fusion_residual and self.use_learnable_ase_fusion_res_scale:
            init_scale = float(max(min(ase_fusion_res_scale, 1.0 - 1e-4), 1e-4))
            init_logit = math.log(init_scale / (1.0 - init_scale))
            self.ase_fusion_res_scale_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        else:
            self.ase_fusion_res_scale_logit = None


        self.spa_mamba_layers = nn.ModuleList([])
        self.spe_mamba_layers = nn.ModuleList([])


        for _ in range(depth):
            self.spa_mamba_layers.append(SingleMambaBlock(dim, H, W))
            self.spe_mamba_layers.append(SingleMambaBlock(dim, H, W))


        self.spa_cross_mamba = CrossMambaBlock(dim, H, W)
        self.spe_cross_mamba = CrossMambaBlock(dim, H, W)


        if self.use_ase and not self.use_sam_ase and not self.use_fass:

            self.spa_ase_mamba = ASEMambaBlock(
                dim,
                num_prompts=num_ase_prompts,
                rank=ase_rank,
                modal_type='single',
                use_wavelet=use_wavelet,
                prompt_mode=ase_prompt_mode,
                route_temperature=ase_route_temperature,
                prompt_soft_mix=ase_prompt_soft_mix,
            )
            shared_embeddingA = self.spa_ase_mamba.ase_module.embeddingA
            self.spe_ase_mamba = ASEMambaBlock(
                dim,
                num_prompts=num_ase_prompts,
                rank=ase_rank,
                modal_type='single',
                use_wavelet=use_wavelet,
                shared_embeddingA=shared_embeddingA,
                prompt_mode=ase_prompt_mode,
                route_temperature=ase_route_temperature,
                prompt_soft_mix=ase_prompt_soft_mix,
            )
            self.fusion_ase_mamba = ASEMambaBlock(
                dim,
                num_prompts=num_ase_prompts,
                rank=ase_rank,
                modal_type='cross',
                use_wavelet=use_wavelet,
                shared_embeddingA=shared_embeddingA,
                prompt_mode=ase_prompt_mode,
                route_temperature=ase_route_temperature,
                prompt_soft_mix=ase_prompt_soft_mix,
            )
            self.ase_shared_embeddingA = shared_embeddingA
            if self.use_sam_semantic_prompt_bank and self.ase_scope == 'fusion_only':
                self.sam_semantic_prompt_bank_refiner = SAMSemanticPromptBankRefiner(
                    feature_dim=dim,
                    num_tokens=self.fusion_ase_mamba.num_prompts,
                    d_state=self.fusion_ase_mamba.prompt_d_state,
                )
            else:
                self.sam_semantic_prompt_bank_refiner = None
            if self.use_sam_region_prototype_bank and self.ase_scope == 'fusion_only':
                self.sam_region_prototype_prompt_conditioner = SAMRegionPrototypePromptConditioner(
                    feature_dim=dim,
                    num_tokens=self.fusion_ase_mamba.num_prompts,
                    d_state=self.fusion_ase_mamba.prompt_d_state,
                    num_prototypes=self.sam_region_prototype_count,
                )
            else:
                self.sam_region_prototype_prompt_conditioner = None
            if self.use_sam_region_prompt_mixture and self.ase_scope == 'fusion_only':
                self.sam_region_prompt_mixer = SAMRegionPromptMixer(
                    feature_dim=dim,
                    num_tokens=self.fusion_ase_mamba.num_prompts,
                    num_prototypes=self.sam_region_prompt_mixture_count,
                )
            else:
                self.sam_region_prompt_mixer = None
            if self.use_sam_region_prompt_subspace and self.ase_scope == 'fusion_only':
                self.sam_region_prompt_subspace = SAMRegionSpecificPromptSubspace(
                    feature_dim=dim,
                    d_state=self.fusion_ase_mamba.prompt_d_state,
                    num_regions=self.sam_region_prompt_subspace_count,
                )
            else:
                self.sam_region_prompt_subspace = None
            if (self.use_sam_guided_semantic_scanning or self.use_sam_boundary_aware_state_propagation or self.use_sam_state_reset_stronger) and self.ase_scope == 'fusion_only':
                if self.use_sam_feature_cluster_scanning and self.use_sam_guided_semantic_scanning:
                    self.sam_semantic_scanner = SAMFeatureClusteringScanner(
                        num_clusters=self.sam_feature_cluster_count,
                        num_iters=self.sam_feature_cluster_iters,
                        spatial_weight=self.sam_feature_cluster_spatial_weight,
                    )
                elif self.use_wavelet_augmented_ss1 and self.use_sam_guided_semantic_scanning:
                    self.sam_semantic_scanner = WaveletAugmentedSemanticScanner(
                        num_regions=self.wavelet_augmented_ss1_count,
                        mask_threshold=0.05,
                        topk_ratio=self.wavelet_augmented_ss1_topk_ratio,
                        strength=self.wavelet_augmented_ss1_strength,
                        mode=self.wavelet_augmented_ss1_mode,
                    )
                else:
                    self.sam_semantic_scanner = SAMSemanticScanner(
                        num_regions=self.sam_semantic_scanning_count,
                        mask_threshold=0.05,
                    )
            else:
                self.sam_semantic_scanner = None
            if self.use_sam_state_organizer_v1 and self.ase_scope == 'fusion_only':
                self.sam_state_organizer = SAMStateOrganizerV1(
                    num_regions=self.sam_state_organizer_count,
                    mask_threshold=0.05,
                    boundary_scale=self.sam_state_organizer_boundary_scale,
                    reset_scale=self.sam_state_organizer_reset_scale,
                )
            else:
                self.sam_state_organizer = None
            if self.use_wavelet_guided_semantic_state_organization and self.ase_scope == 'fusion_only':
                self.wavelet_guided_semantic_state_organizer = WaveletGuidedSemanticStateOrganizer(
                    num_regions=self.wavelet_guided_semantic_state_count,
                    mask_threshold=0.05,
                    wavelet_scale=self.wavelet_guided_semantic_state_scale,
                    boundary_scale=self.wavelet_guided_semantic_boundary_scale,
                    reset_scale=self.wavelet_guided_semantic_reset_scale,
                )
            else:
                self.wavelet_guided_semantic_state_organizer = None
            if self.use_dual_prototype_bank and self.ase_scope == 'fusion_only':
                self.semantic_frequency_dual_prototype_conditioner = SemanticFrequencyDualPrototypeConditioner(
                    feature_dim=dim,
                    num_tokens=self.fusion_ase_mamba.num_prompts,
                    d_state=self.fusion_ase_mamba.prompt_d_state,
                    num_prototypes=self.dual_prototype_count,
                )
            else:
                self.semantic_frequency_dual_prototype_conditioner = None
            if self.use_semantic_frequency_state_modulation and self.ase_scope == 'fusion_only':
                self.semantic_frequency_state_modulator = SemanticFrequencyStateModulator(
                    num_regions=self.semantic_frequency_state_count,
                    mask_threshold=0.05,
                    write_scale=self.semantic_frequency_state_write_scale,
                    read_scale=self.semantic_frequency_state_read_scale,
                    delta_scale=self.semantic_frequency_state_delta_scale,
                )
            else:
                self.semantic_frequency_state_modulator = None
        elif self.use_sam_ase and not self.use_fass:

            self.spa_ase_mamba = SAMASEMambaBlock(dim, sam_prompt_dim=sam_prompt_dim, modal_type='single', use_sam=True, sam_checkpoint=sam_checkpoint,
                                                   use_wavelet=use_wavelet, use_learnable_prompts=use_learnable_prompts, num_learnable_prompts=num_learnable_prompts,
                                                   use_soft_masks=use_soft_masks, num_soft_regions=num_soft_regions, input_channels=input_channels,
                                                   use_structure_guided_sam_ase=use_structure_guided_sam_ase, structure_texture_weight=structure_texture_weight)
            self.spe_ase_mamba = SAMASEMambaBlock(dim, sam_prompt_dim=sam_prompt_dim, modal_type='single', use_sam=True, sam_checkpoint=sam_checkpoint,
                                                   use_wavelet=use_wavelet, use_learnable_prompts=use_learnable_prompts, num_learnable_prompts=num_learnable_prompts,
                                                   use_soft_masks=use_soft_masks, num_soft_regions=num_soft_regions, input_channels=input_channels,
                                                   use_structure_guided_sam_ase=use_structure_guided_sam_ase, structure_texture_weight=structure_texture_weight)
            self.fusion_ase_mamba = SAMASEMambaBlock(dim, sam_prompt_dim=sam_prompt_dim, modal_type='cross', use_sam=True, sam_checkpoint=sam_checkpoint,
                                                      use_wavelet=use_wavelet, use_learnable_prompts=use_learnable_prompts, num_learnable_prompts=num_learnable_prompts,
                                                      use_soft_masks=use_soft_masks, num_soft_regions=num_soft_regions, input_channels=input_channels,
                                                      use_structure_guided_sam_ase=use_structure_guided_sam_ase, structure_texture_weight=structure_texture_weight)
        elif self.use_fass:

            self.spa_ase_mamba = SparsifiedSAMASEMambaBlock(dim, sam_prompt_dim=sam_prompt_dim, modal_type='single', use_sam=True, sam_checkpoint=sam_checkpoint,
                                                            use_wavelet=use_wavelet, use_learnable_prompts=use_learnable_prompts, num_learnable_prompts=num_learnable_prompts,
                                                            use_soft_masks=use_soft_masks, num_soft_regions=num_soft_regions, input_channels=input_channels,
                                                            use_structure_guided_sam_ase=use_structure_guided_sam_ase, structure_texture_weight=structure_texture_weight,
                                                            fass_compression_ratio=self.fass_compression_ratio,
                                                            fass_threshold=self.fass_threshold, fass_sparsity_target=self.fass_sparsity_target,
                                                            fass_ll_sparsity=self.fass_ll_sparsity, fass_hf_sparsity=self.fass_hf_sparsity,
                                                            fass_d_state=self.fass_d_state,
                                                            train_mode=self.train_mode, dense_epochs=self.dense_epochs,
                                                            gating_loss_weight=self.gating_loss_weight)
            self.spe_ase_mamba = SparsifiedSAMASEMambaBlock(dim, sam_prompt_dim=sam_prompt_dim, modal_type='single', use_sam=True, sam_checkpoint=sam_checkpoint,
                                                            use_wavelet=use_wavelet, use_learnable_prompts=use_learnable_prompts, num_learnable_prompts=num_learnable_prompts,
                                                            use_soft_masks=use_soft_masks, num_soft_regions=num_soft_regions, input_channels=input_channels,
                                                            use_structure_guided_sam_ase=use_structure_guided_sam_ase, structure_texture_weight=structure_texture_weight,
                                                            fass_compression_ratio=self.fass_compression_ratio,
                                                            fass_threshold=self.fass_threshold, fass_sparsity_target=self.fass_sparsity_target,
                                                            fass_ll_sparsity=self.fass_ll_sparsity, fass_hf_sparsity=self.fass_hf_sparsity,
                                                            fass_d_state=self.fass_d_state,
                                                            train_mode=self.train_mode, dense_epochs=self.dense_epochs,
                                                            gating_loss_weight=self.gating_loss_weight)
            self.fusion_ase_mamba = SparsifiedSAMASEMambaBlock(dim, sam_prompt_dim=sam_prompt_dim, modal_type='cross', use_sam=True, sam_checkpoint=sam_checkpoint,
                                                               use_wavelet=use_wavelet, use_learnable_prompts=use_learnable_prompts, num_learnable_prompts=num_learnable_prompts,
                                                               use_soft_masks=use_soft_masks, num_soft_regions=num_soft_regions, input_channels=input_channels,
                                                               use_structure_guided_sam_ase=use_structure_guided_sam_ase, structure_texture_weight=structure_texture_weight,
                                                               fass_compression_ratio=self.fass_compression_ratio,
                                                               fass_threshold=self.fass_threshold, fass_sparsity_target=self.fass_sparsity_target,
                                                               fass_ll_sparsity=self.fass_ll_sparsity, fass_hf_sparsity=self.fass_hf_sparsity,
                                                               fass_d_state=self.fass_d_state,
                                                               train_mode=self.train_mode, dense_epochs=self.dense_epochs,
                                                               gating_loss_weight=self.gating_loss_weight)

        if hasattr(self, 'spa_ase_mamba'):
            self._assign_prior_c_refiner_role(self.spa_ase_mamba, 'spa')
            self._assign_prior_c_refiner_role(self.spe_ase_mamba, 'spe')
            self._assign_prior_c_refiner_role(self.fusion_ase_mamba, 'fusion')

        self.out_proj = nn.Linear(dim, dim)

    def _get_ase_fusion_res_scale(self):
        if self.ase_fusion_res_scale_logit is not None:
            return torch.sigmoid(self.ase_fusion_res_scale_logit)
        return self.ase_fusion_res_scale

    def _assign_prior_c_refiner_role(self, block, role):
        if hasattr(block, 'prior_c_refiner_role'):
            block.prior_c_refiner_role = role
        sam_module = getattr(block, 'sam_ase_module', None)
        if sam_module is not None and hasattr(sam_module, 'prior_c_refiner_role'):
            sam_module.prior_c_refiner_role = role

    def _extract_feature_tensor(self, module_output):
        if isinstance(module_output, tuple):
            if len(module_output) == 0:
                raise ValueError("ASE/SAM module returned an empty tuple output")
            return module_output[0]
        return module_output

    def _extract_feature_and_routing(self, module_output):
        if isinstance(module_output, tuple):
            if len(module_output) == 0:
                raise ValueError("ASE/SAM module returned an empty tuple output")
            if len(module_output) == 1:
                return module_output[0], None
            return module_output[0], module_output[1]
        return module_output, None

    def forward(self, pan, ms, sam_prompt=None, semantic_masks=None, original_msi=None, wavelet_guidance=None, current_epoch=0):

        b, c, h, w = pan.shape


        pan_seq = rearrange(pan, 'b c h w -> b (h w) c')
        ms_seq = rearrange(ms, 'b c h w -> b (h w) c')
        if wavelet_guidance is None:
            wavelet_guidance = getattr(self, 'current_wavelet_guidance', None)
        wavelet_local_bias = getattr(self, 'current_wavelet_local_bias', None)
        wavelet_local_gate = getattr(self, 'current_wavelet_local_gate', None)
        sam_local_gate = getattr(self, 'current_sam_local_gate', None)
        sam_prior_bank = getattr(self, 'current_sam_prior_bank', None)
        sam_region_context = getattr(self, 'current_sam_region_context', None)
        if hasattr(self, 'spa_ase_mamba'):
            for ase_module in [self.spa_ase_mamba, self.spe_ase_mamba, self.fusion_ase_mamba]:
                if hasattr(ase_module, 'current_sam_prior_bank'):
                    ase_module.current_sam_prior_bank = sam_prior_bank


        for spa_layer, spe_layer in zip(self.spa_mamba_layers, self.spe_mamba_layers):
            pan_seq = spa_layer(pan_seq)
            ms_seq = spe_layer(ms_seq)

        pan_route = None
        ms_route = None
        fusion_route = None
        self.current_ase_routing_probs = None


        if self.use_ase or self.use_sam_ase:
            if self.use_fass:

                pan_seq = self.spa_ase_mamba(
                    pan_seq,
                    sam_prompt=sam_prompt,
                    semantic_masks=semantic_masks,
                    original_msi=original_msi,
                    wavelet_guidance=wavelet_guidance,
                    current_epoch=current_epoch,
                )
                ms_seq = self.spe_ase_mamba(
                    ms_seq,
                    sam_prompt=sam_prompt,
                    semantic_masks=semantic_masks,
                    original_msi=original_msi,
                    wavelet_guidance=wavelet_guidance,
                    current_epoch=current_epoch,
                )
            elif self.use_sam_ase:

                pan_seq = self.spa_ase_mamba(
                    pan_seq,
                    sam_prompt=sam_prompt,
                    semantic_masks=semantic_masks,
                    original_msi=original_msi,
                    wavelet_guidance=wavelet_guidance,
                )
                ms_seq = self.spe_ase_mamba(
                    ms_seq,
                    sam_prompt=sam_prompt,
                    semantic_masks=semantic_masks,
                    original_msi=original_msi,
                    wavelet_guidance=wavelet_guidance,
                )
            elif self.use_ase:

                if self.ase_scope != 'fusion_only':
                    pan_seq, pan_route = self._extract_feature_and_routing(self.spa_ase_mamba(pan_seq))
                    ms_seq, ms_route = self._extract_feature_and_routing(self.spe_ase_mamba(ms_seq))


        spa_fusion = self.spa_cross_mamba(pan_seq, ms_seq)
        spe_fusion = self.spe_cross_mamba(ms_seq, pan_seq)


        fusion_base = (spa_fusion + spe_fusion) / 2
        fusion = fusion_base
        if self.use_ase or self.use_sam_ase:
            if self.use_fass:
                fusion = self.fusion_ase_mamba(
                    fusion,
                    sam_prompt=sam_prompt,
                    semantic_masks=semantic_masks,
                    original_msi=original_msi,
                    wavelet_guidance=wavelet_guidance,
                    current_epoch=current_epoch,
                )
            elif self.use_sam_ase:
                fusion = self.fusion_ase_mamba(
                    fusion,
                    sam_prompt=sam_prompt,
                    semantic_masks=semantic_masks,
                    original_msi=original_msi,
                    wavelet_guidance=wavelet_guidance,
                )
            elif self.use_ase:
                semantic_prompt_bank_bias = None
                semantic_route_logit_bias = None
                semantic_token_prompt_residual = None
                semantic_scan_indices = None
                semantic_scan_reverse_indices = None
                semantic_region_ids = None
                semantic_boundary_gate = None
                semantic_state_reset_gate = None
                semantic_state_write_gate = None
                semantic_state_read_gate = None
                semantic_state_delta_gate = None

                def _match_semantic_batch(tensor, expected_batch, name):
                    if tensor is None:
                        return None
                    if tensor.shape[0] == expected_batch:
                        return tensor
                    if tensor.shape[0] == 1:
                        expand_shape = [expected_batch] + [-1] * (tensor.dim() - 1)
                        return tensor.expand(*expand_shape).contiguous()
                    raise ValueError(
                        f"{name} batch mismatch: got {tuple(tensor.shape)}, expected batch {expected_batch}"
                    )
                if (
                    self.ase_scope == 'fusion_only'
                    and self.use_sam_semantic_prompt_bank
                    and self.sam_semantic_prompt_bank_refiner is not None
                    and sam_prior_bank is not None
                ):
                    semantic_prompt_bank_bias = self.sam_semantic_prompt_bank_refiner(
                        fusion_base,
                        (h, w),
                        sam_prior_bank,
                    )
                    if semantic_prompt_bank_bias is not None:
                        semantic_prompt_bank_bias = self.sam_semantic_prompt_bank_scale * semantic_prompt_bank_bias
                if (
                    self.ase_scope == 'fusion_only'
                    and self.use_sam_region_prototype_bank
                    and self.sam_region_prototype_prompt_conditioner is not None
                    and sam_region_context is not None
                ):
                    prototype_prompt_bank_bias = self.sam_region_prototype_prompt_conditioner(
                        fusion_base,
                        sam_region_context,
                        wavelet_scale=self.wavelet_guided_sam_prototype_scale if self.use_wavelet_guided_sam_prototype_scaling else 0.0,
                    )
                    if prototype_prompt_bank_bias is not None:
                        prototype_prompt_bank_bias = self.sam_region_prototype_bank_scale * prototype_prompt_bank_bias
                        semantic_prompt_bank_bias = (
                            prototype_prompt_bank_bias
                            if semantic_prompt_bank_bias is None
                            else semantic_prompt_bank_bias + prototype_prompt_bank_bias
                        )
                if (
                    self.ase_scope == 'fusion_only'
                    and self.use_dual_prototype_bank
                    and self.semantic_frequency_dual_prototype_conditioner is not None
                    and sam_region_context is not None
                ):
                    dual_prompt_bank_bias = self.semantic_frequency_dual_prototype_conditioner(
                        fusion_base,
                        sam_region_context,
                        semantic_scale=self.dual_prototype_semantic_scale,
                        frequency_scale=self.dual_prototype_frequency_scale,
                    )
                    if dual_prompt_bank_bias is not None:
                        dual_prompt_bank_bias = _match_semantic_batch(
                            dual_prompt_bank_bias,
                            fusion_base.shape[0],
                            "dual_prompt_bank_bias",
                        )
                        semantic_prompt_bank_bias = (
                            dual_prompt_bank_bias
                            if semantic_prompt_bank_bias is None
                            else semantic_prompt_bank_bias + dual_prompt_bank_bias
                        )
                if (
                    self.ase_scope == 'fusion_only'
                    and self.use_sam_region_prompt_subspace
                    and self.sam_region_prompt_subspace is not None
                    and sam_region_context is not None
                ):
                    semantic_token_prompt_residual = self.sam_region_prompt_subspace(
                        fusion_base,
                        sam_region_context,
                        (h, w),
                    )
                    if semantic_token_prompt_residual is not None:
                        semantic_token_prompt_residual = _match_semantic_batch(
                            semantic_token_prompt_residual,
                            fusion_base.shape[0],
                            "semantic_token_prompt_residual",
                        )
                        semantic_token_prompt_residual = self.sam_region_prompt_subspace_scale * semantic_token_prompt_residual
                if (
                    self.ase_scope == 'fusion_only'
                    and self.use_sam_region_prompt_mixture
                    and self.sam_region_prompt_mixer is not None
                    and sam_region_context is not None
                ):
                    semantic_route_logit_bias = self.sam_region_prompt_mixer(
                        fusion_base,
                        sam_region_context,
                    )
                    if semantic_route_logit_bias is not None:
                        semantic_route_logit_bias = _match_semantic_batch(
                            semantic_route_logit_bias,
                            fusion_base.shape[0],
                            "semantic_route_logit_bias",
                        )
                        semantic_route_logit_bias = self.sam_region_prompt_mixture_scale * semantic_route_logit_bias
                organizer_outputs = None
                if (
                    self.ase_scope == 'fusion_only'
                    and self.wavelet_guided_semantic_state_organizer is not None
                    and sam_region_context is not None
                ):
                    organizer_outputs = self.wavelet_guided_semantic_state_organizer(
                        sam_region_context,
                        sam_prior_bank,
                        (h, w),
                        fusion_base.device,
                    )
                elif (
                    self.ase_scope == 'fusion_only'
                    and self.sam_state_organizer is not None
                    and sam_region_context is not None
                ):
                    organizer_outputs = self.sam_state_organizer(
                        sam_region_context,
                        sam_prior_bank,
                        (h, w),
                        fusion_base.device,
                    )
                if organizer_outputs is not None:
                    semantic_region_ids = _match_semantic_batch(
                        organizer_outputs.get('region_ids'),
                        fusion_base.shape[0],
                        "semantic_region_ids",
                    )
                    semantic_scan_indices = _match_semantic_batch(
                        organizer_outputs.get('scan_indices'),
                        fusion_base.shape[0],
                        "semantic_scan_indices",
                    )
                    semantic_scan_reverse_indices = _match_semantic_batch(
                        organizer_outputs.get('scan_reverse_indices'),
                        fusion_base.shape[0],
                        "semantic_scan_reverse_indices",
                    )
                    semantic_boundary_gate = _match_semantic_batch(
                        organizer_outputs.get('boundary_gate'),
                        fusion_base.shape[0],
                        "semantic_boundary_gate",
                    )
                    semantic_state_reset_gate = _match_semantic_batch(
                        organizer_outputs.get('reset_gate'),
                        fusion_base.shape[0],
                        "semantic_state_reset_gate",
                    )
                if (
                    self.ase_scope == 'fusion_only'
                    and self.semantic_frequency_state_modulator is not None
                    and sam_region_context is not None
                ):
                    modulation_outputs = self.semantic_frequency_state_modulator(
                        sam_region_context,
                        (h, w),
                        fusion_base.device,
                    )
                    if modulation_outputs is not None:
                        semantic_state_write_gate = _match_semantic_batch(
                            modulation_outputs.get('state_write_gate'),
                            fusion_base.shape[0],
                            "semantic_state_write_gate",
                        )
                        semantic_state_read_gate = _match_semantic_batch(
                            modulation_outputs.get('state_read_gate'),
                            fusion_base.shape[0],
                            "semantic_state_read_gate",
                        )
                        semantic_state_delta_gate = _match_semantic_batch(
                            modulation_outputs.get('state_delta_gate'),
                            fusion_base.shape[0],
                            "semantic_state_delta_gate",
                        )
                if (
                    self.ase_scope == 'fusion_only'
                    and self.sam_semantic_scanner is not None
                    and sam_region_context is not None
                    and organizer_outputs is None
                ):
                    semantic_region_ids, semantic_scan_indices, semantic_scan_reverse_indices = self.sam_semantic_scanner(
                        sam_region_context,
                        (h, w),
                        fusion_base.device,
                    )
                    if semantic_region_ids is not None and semantic_region_ids.shape[0] != fusion_base.shape[0]:
                        if semantic_region_ids.shape[0] == 1:
                            semantic_region_ids = semantic_region_ids.expand(fusion_base.shape[0], -1).contiguous()
                        else:
                            raise ValueError(
                                f"semantic_region_ids batch mismatch: "
                                f"got {tuple(semantic_region_ids.shape)}, expected batch {fusion_base.shape[0]}"
                            )
                    if semantic_scan_indices is not None and semantic_scan_indices.shape[0] != fusion_base.shape[0]:
                        if semantic_scan_indices.shape[0] == 1:
                            semantic_scan_indices = semantic_scan_indices.expand(fusion_base.shape[0], -1).contiguous()
                        else:
                            raise ValueError(
                                f"semantic_scan_indices batch mismatch: "
                                f"got {tuple(semantic_scan_indices.shape)}, expected batch {fusion_base.shape[0]}"
                            )
                    if semantic_scan_reverse_indices is not None and semantic_scan_reverse_indices.shape[0] != fusion_base.shape[0]:
                        if semantic_scan_reverse_indices.shape[0] == 1:
                            semantic_scan_reverse_indices = semantic_scan_reverse_indices.expand(fusion_base.shape[0], -1).contiguous()
                        else:
                            raise ValueError(
                                f"semantic_scan_reverse_indices batch mismatch: "
                                f"got {tuple(semantic_scan_reverse_indices.shape)}, expected batch {fusion_base.shape[0]}"
                            )
                fusion_out, fusion_route = self._extract_feature_and_routing(
                    self.fusion_ase_mamba(
                        fusion_base,
                        semantic_prompt_bank_bias=semantic_prompt_bank_bias,
                        semantic_route_logit_bias=semantic_route_logit_bias,
                        semantic_token_prompt_residual=semantic_token_prompt_residual,
                        semantic_scan_indices=semantic_scan_indices if (
                            self.use_sam_guided_semantic_scanning
                            or organizer_outputs is not None
                        ) else None,
                        semantic_scan_reverse_indices=semantic_scan_reverse_indices if (
                            self.use_sam_guided_semantic_scanning
                            or organizer_outputs is not None
                        ) else None,
                        semantic_region_ids=semantic_region_ids if (
                            self.use_sam_boundary_aware_state_propagation
                            or self.use_sam_state_reset_stronger
                            or organizer_outputs is not None
                        ) else None,
                        semantic_boundary_scale=self.sam_boundary_aware_state_scale if (
                            self.use_sam_boundary_aware_state_propagation and organizer_outputs is None
                        ) else 0.0,
                        semantic_state_reset_scale=self.sam_state_reset_scale if (
                            self.use_sam_state_reset_stronger and organizer_outputs is None
                        ) else 0.0,
                        semantic_boundary_gate=semantic_boundary_gate,
                        semantic_state_reset_gate=semantic_state_reset_gate,
                        semantic_state_write_gate=semantic_state_write_gate,
                        semantic_state_read_gate=semantic_state_read_gate,
                        semantic_state_delta_gate=semantic_state_delta_gate,
                    )
                )
                if self.use_ase_fusion_residual:
                    ase_delta = fusion_out - fusion_base
                    if (
                        self.ase_scope == 'fusion_only'
                        and self.use_wavelet_local_gate
                        and wavelet_local_gate is not None
                        and wavelet_local_gate.shape[1] == self.dim
                    ):
                        if wavelet_local_gate.shape[-2:] != (h, w):
                            wavelet_local_gate = F.interpolate(
                                wavelet_local_gate,
                                size=(h, w),
                                mode='bilinear',
                                align_corners=False,
                            )
                        wavelet_gate_seq = rearrange(wavelet_local_gate, 'b c h w -> b (h w) c')
                        ase_delta = wavelet_gate_seq * ase_delta
                    if (
                        self.ase_scope == 'fusion_only'
                        and self.use_sam_local_gate
                        and sam_local_gate is not None
                        and sam_local_gate.shape[1] == self.dim
                    ):
                        if sam_local_gate.shape[-2:] != (h, w):
                            sam_local_gate = F.interpolate(
                                sam_local_gate,
                                size=(h, w),
                                mode='bilinear',
                                align_corners=False,
                            )
                        sam_gate_seq = rearrange(sam_local_gate, 'b c h w -> b (h w) c')
                        ase_delta = sam_gate_seq * ase_delta
                    fusion = fusion_base + self._get_ase_fusion_res_scale() * ase_delta
                else:
                    fusion = fusion_out

        if (
            self.use_ase
            and self.ase_scope == 'fusion_only'
            and self.use_ase_fusion_residual
            and self.use_wavelet_local_bias
            and not self.use_wavelet_local_gate
            and wavelet_local_bias is not None
        ):
            if wavelet_local_bias.shape[1] == self.dim:
                if wavelet_local_bias.shape[-2:] != (h, w):
                    wavelet_local_bias = F.interpolate(
                        wavelet_local_bias,
                        size=(h, w),
                        mode='bilinear',
                        align_corners=False,
                    )
                wavelet_bias_seq = rearrange(wavelet_local_bias, 'b c h w -> b (h w) c')
                fusion = fusion + self.wavelet_local_bias_scale * wavelet_bias_seq

        if self.use_ase:
            self.current_ase_routing_probs = [route for route in [pan_route, ms_route, fusion_route] if route is not None]


        fusion = self.out_proj(fusion)


        output = rearrange(fusion, 'b (h w) c -> b c h w', h=h, w=w)


        pan_img = rearrange(pan_seq, 'b (h w) c -> b c h w', h=h, w=w)
        ms_img = rearrange(ms_seq, 'b (h w) c -> b c h w', h=h, w=w)

        if self.final:
            return output
        else:
            return (pan_img + output) / 2, (ms_img + output) / 2
