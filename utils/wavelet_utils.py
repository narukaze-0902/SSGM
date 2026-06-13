import torch
import torch.nn.functional as F

from model.haar_dwt import SimpleDWT


def build_haar_wavelet_coeffs(x):
    added_batch = False
    if x.dim() == 3:
        x = x.unsqueeze(0)
        added_batch = True
    if x.dim() != 4:
        raise ValueError(f"build_haar_wavelet_coeffs expects 3D/4D tensor, got {tuple(x.shape)}")

    dwt = SimpleDWT().to(x.device)
    approx, detail_list = dwt(x)
    details = torch.stack(detail_list, dim=1)

    if added_batch:
        approx = approx.squeeze(0)
        details = details.squeeze(0)
    return approx.contiguous(), details.contiguous()


def build_wavelet_guidance_map(lr_hsi_details, target_hw=None, device=None, dtype=None):
    if lr_hsi_details is None:
        return None

    details = lr_hsi_details.abs()
    if device is not None or dtype is not None:
        details = details.to(device=device if device is not None else details.device,
                             dtype=dtype if dtype is not None else details.dtype)

    if details.dim() == 5:
        guidance = details.mean(dim=1).mean(dim=1, keepdim=True)
    elif details.dim() == 4:
        if details.shape[0] == 3:
            guidance = details.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True)
        else:
            guidance = details.mean(dim=1, keepdim=True)
    elif details.dim() == 3:
        guidance = details.mean(dim=0, keepdim=True).unsqueeze(0)
    else:
        return None

    if target_hw is not None and guidance.shape[-2:] != target_hw:
        guidance = F.interpolate(guidance, size=target_hw, mode='bilinear', align_corners=False)

    flat = guidance.flatten(1)
    min_val = flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    max_val = flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    guidance = (guidance - min_val) / (max_val - min_val + 1e-6)
    return guidance


def _normalize_prior_map(map_tensor):
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


def build_spectral_variation_map(lr_hsi, target_hw=None, device=None, dtype=None):
    if lr_hsi is None:
        return None

    if lr_hsi.dim() == 3:
        lr_hsi = lr_hsi.unsqueeze(0)
    if lr_hsi.dim() != 4:
        return None

    spectral = lr_hsi
    if device is not None or dtype is not None:
        spectral = spectral.to(
            device=device if device is not None else spectral.device,
            dtype=dtype if dtype is not None else spectral.dtype,
        )

    if spectral.shape[1] <= 1:
        variation = spectral.abs().mean(dim=1, keepdim=True)
    else:
        band_diff = torch.abs(spectral[:, 1:, :, :] - spectral[:, :-1, :, :])
        variation = band_diff.mean(dim=1, keepdim=True)

    if target_hw is not None and variation.shape[-2:] != target_hw:
        variation = F.interpolate(variation, size=target_hw, mode='bilinear', align_corners=False)
    return _normalize_prior_map(variation)


def build_joint_spatial_spectral_prior(
    lr_hsi,
    lr_hsi_details,
    target_hw=None,
    device=None,
    dtype=None,
    spatial_weight=1.0,
    spectral_weight=1.0,
):
    spatial_prior = build_wavelet_guidance_map(
        lr_hsi_details,
        target_hw=target_hw,
        device=device,
        dtype=dtype,
    )
    spectral_prior = build_spectral_variation_map(
        lr_hsi,
        target_hw=target_hw,
        device=device,
        dtype=dtype,
    )

    if spatial_prior is None and spectral_prior is None:
        return None, None, None
    if spatial_prior is None:
        return None, spectral_prior, spectral_prior
    if spectral_prior is None:
        return spatial_prior, None, spatial_prior

    sw = float(max(spatial_weight, 0.0))
    tw = float(max(spectral_weight, 0.0))
    joint_prior = (sw * spatial_prior + tw * spectral_prior) / (sw + tw + 1e-6)
    joint_prior = _normalize_prior_map(joint_prior)
    return spatial_prior, spectral_prior, joint_prior


def build_region_frequency_statistics(
    sam_masks,
    spatial_prior_map=None,
    spectral_prior_map=None,
    joint_prior_map=None,
    target_hw=None,
):
    if sam_masks is None:
        return None

    if sam_masks.dim() == 3:
        sam_masks = sam_masks.unsqueeze(0)
    if sam_masks.dim() != 4:
        return None

    masks = sam_masks.float()
    if target_hw is not None and masks.shape[-2:] != target_hw:
        masks = F.interpolate(masks, size=target_hw, mode='bilinear', align_corners=False)

    def _prepare_prior(prior_map):
        prior_map = _normalize_prior_map(prior_map)
        if prior_map is None:
            return None
        if prior_map.shape[-2:] != masks.shape[-2:]:
            prior_map = F.interpolate(prior_map, size=masks.shape[-2:], mode='bilinear', align_corners=False)
        if prior_map.shape[1] != 1:
            prior_map = prior_map.mean(dim=1, keepdim=True)
        return prior_map.to(device=masks.device, dtype=masks.dtype)

    spatial_prior = _prepare_prior(spatial_prior_map)
    spectral_prior = _prepare_prior(spectral_prior_map)
    joint_prior = _prepare_prior(joint_prior_map)

    pooled_stats = []
    for prior_map in (spatial_prior, spectral_prior, joint_prior):
        if prior_map is None:
            pooled_stats.append(torch.zeros(masks.shape[0], masks.shape[1], device=masks.device, dtype=masks.dtype))
            continue
        weighted = (masks * prior_map).flatten(-2).sum(dim=-1)
        norm = masks.flatten(-2).sum(dim=-1).clamp(min=1e-6)
        pooled_stats.append(weighted / norm)
    return torch.stack(pooled_stats, dim=-1).contiguous()


def should_use_wavelet_priors(
    use_wavelet_legacy=False,
    use_wavelet_priors=False,
    use_joint_spatial_spectral_wavelet_prior=False,
    use_structure_guided_sam_ase=False,
    use_wavelet_local_bias=False,
    use_wavelet_local_gate=False,
    use_wavelet_guided_sam_prototype_scaling=False,
    use_wavelet_guided_semantic_state_organization=False,
    use_dual_prototype_bank=False,
    use_semantic_frequency_state_modulation=False,
    use_hf_wavelet_loss=False,
    use_semantic_region_weighted_hf_wavelet_loss=False,
    use_boundary_selective_wavelet_loss=False,
    use_semantic_frequency_adaptive_scanning=False,
):
    return any([
        use_wavelet_legacy,
        use_wavelet_priors,
        use_joint_spatial_spectral_wavelet_prior,
        use_structure_guided_sam_ase,
        use_wavelet_local_bias,
        use_wavelet_local_gate,
        use_wavelet_guided_sam_prototype_scaling,
        use_wavelet_guided_semantic_state_organization,
        use_dual_prototype_bank,
        use_semantic_frequency_state_modulation,
        use_hf_wavelet_loss,
        use_semantic_region_weighted_hf_wavelet_loss,
        use_boundary_selective_wavelet_loss,
        use_semantic_frequency_adaptive_scanning,
    ])
