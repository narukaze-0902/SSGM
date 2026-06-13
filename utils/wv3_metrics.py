from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _ensure_image_tensor(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4:
        raise ValueError(f"Expected 2D/3D/4D tensor, got shape {tuple(image.shape)}")
    return image.float()


def _block_uqi(img_a: torch.Tensor, img_b: torch.Tensor, block_size: int = 32, eps: float = 1e-8) -> torch.Tensor:
    img_a = _ensure_image_tensor(img_a)
    img_b = _ensure_image_tensor(img_b)
    if img_a.shape != img_b.shape:
        raise ValueError(f"UQI shape mismatch: {tuple(img_a.shape)} vs {tuple(img_b.shape)}")

    _, _, h, w = img_a.shape
    if h < block_size or w < block_size:
        block_size = min(h, w)
    stride = block_size
    if block_size <= 0:
        return torch.tensor(0.0, device=img_a.device)

    patches_a = F.unfold(img_a, kernel_size=block_size, stride=stride)
    patches_b = F.unfold(img_b, kernel_size=block_size, stride=stride)

    mean_a = patches_a.mean(dim=1)
    mean_b = patches_b.mean(dim=1)
    centered_a = patches_a - mean_a.unsqueeze(1)
    centered_b = patches_b - mean_b.unsqueeze(1)
    var_a = (centered_a ** 2).mean(dim=1)
    var_b = (centered_b ** 2).mean(dim=1)
    cov_ab = (centered_a * centered_b).mean(dim=1)

    numerator = 4.0 * cov_ab * mean_a * mean_b
    denominator = (var_a + var_b) * (mean_a ** 2 + mean_b ** 2) + eps
    uqi = numerator / denominator
    return uqi.mean()


def compute_d_lambda(
    fused: torch.Tensor,
    lms: torch.Tensor,
    block_size: int = 32,
    p: int = 1,
) -> torch.Tensor:
    fused = fused.float()
    lms = lms.float()
    if fused.dim() != 3 or lms.dim() != 3:
        raise ValueError(f"D_lambda expects CHW tensors, got {tuple(fused.shape)} and {tuple(lms.shape)}")
    if fused.shape != lms.shape:
        raise ValueError(f"D_lambda shape mismatch: {tuple(fused.shape)} vs {tuple(lms.shape)}")

    bands = fused.shape[0]
    diffs = []
    for i in range(bands):
        for j in range(i + 1, bands):
            q_fused = _block_uqi(fused[i], fused[j], block_size=block_size)
            q_lms = _block_uqi(lms[i], lms[j], block_size=block_size)
            diffs.append(torch.abs(q_fused - q_lms) ** p)

    if not diffs:
        return torch.tensor(0.0, device=fused.device)
    return (torch.stack(diffs).mean()) ** (1.0 / p)


def compute_d_s(
    fused: torch.Tensor,
    lms: torch.Tensor,
    pan: torch.Tensor,
    block_size: int = 32,
    p: int = 1,
) -> torch.Tensor:
    fused = fused.float()
    lms = lms.float()
    pan = pan.float()
    if fused.dim() != 3 or lms.dim() != 3:
        raise ValueError(f"D_s expects CHW tensors, got {tuple(fused.shape)} and {tuple(lms.shape)}")
    if pan.dim() == 3:
        pan = pan[0]
    if pan.dim() != 2:
        raise ValueError(f"D_s expects PAN HW or 1HW tensor, got {tuple(pan.shape)}")
    if fused.shape != lms.shape:
        raise ValueError(f"D_s shape mismatch: {tuple(fused.shape)} vs {tuple(lms.shape)}")

    diffs = []
    for i in range(fused.shape[0]):
        q_fused = _block_uqi(fused[i], pan, block_size=block_size)
        q_lms = _block_uqi(lms[i], pan, block_size=block_size)
        diffs.append(torch.abs(q_fused - q_lms) ** p)

    return (torch.stack(diffs).mean()) ** (1.0 / p)


def compute_qnr(
    fused: torch.Tensor,
    lms: torch.Tensor,
    pan: torch.Tensor,
    block_size: int = 32,
    p: int = 1,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Dict[str, float]:
    d_lambda = compute_d_lambda(fused, lms, block_size=block_size, p=p)
    d_s = compute_d_s(fused, lms, pan, block_size=block_size, p=p)
    qnr = ((1.0 - d_lambda).clamp(min=0.0) ** alpha) * ((1.0 - d_s).clamp(min=0.0) ** beta)
    return {
        "D_lambda": float(d_lambda.item()),
        "D_s": float(d_s.item()),
        "QNR": float(qnr.item()),
    }


def summarize_real_metrics(metrics_list: list[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not metrics_list:
        return {}
    keys = sorted(metrics_list[0].keys())
    summary: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = torch.tensor([item[key] for item in metrics_list], dtype=torch.float32)
        summary[key] = {
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).item()),
        }
    return summary
