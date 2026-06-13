from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from model.u2net import U2Net as Net
from utils.load_wv3_data import (
    WV3TrainValDataset,
    apply_display_range,
    chw_to_rgb,
    compute_display_range,
)
from utils.tools import ERGAS, compute_psnr, compute_rmse, compute_sam, compute_ssim
from utils.wavelet_utils import should_use_wavelet_priors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reduced-resolution evaluation on WV3.")
    parser.add_argument("--h5_path", type=str, default="./data/WV3/reduced_examples/reduced_examples/test_wv3_multiExm1.h5")
    parser.add_argument("--weight", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--use_ase", action="store_true")
    parser.add_argument("--ase_prompt_mode", type=str, default="soft")
    parser.add_argument("--ase_route_temperature", type=float, default=1.2)
    parser.add_argument("--ase_scope", type=str, default="fusion_only")
    parser.add_argument("--ase_stage_scope", type=str, default="all_stages")
    parser.add_argument("--use_ase_fusion_residual", action="store_true")
    parser.add_argument("--ase_fusion_res_scale", type=float, default=0.4)
    parser.add_argument("--use_wavelet", action="store_true")
    parser.add_argument("--use_wavelet_priors", action="store_true")
    parser.add_argument("--use_wavelet_local_bias", action="store_true")
    parser.add_argument("--wavelet_local_bias_scale", type=float, default=0.1)
    parser.add_argument("--use_wavelet_local_gate", action="store_true")
    parser.add_argument("--wavelet_local_gate_scale", type=float, default=0.1)
    parser.add_argument("--use_joint_spatial_spectral_wavelet_prior", action="store_true")
    parser.add_argument("--joint_wavelet_spatial_weight", type=float, default=1.0)
    parser.add_argument("--joint_wavelet_spectral_weight", type=float, default=0.7)
    parser.add_argument("--use_offline_sam_cache", action="store_true")
    parser.add_argument("--sam_cache_path", type=str, default=None)
    parser.add_argument("--sam_cache_strict", action="store_true")
    parser.add_argument("--use_sam_region_prototype_bank", action="store_true")
    parser.add_argument("--sam_region_prototype_bank_scale", type=float, default=0.05)
    parser.add_argument("--sam_region_prototype_count", type=int, default=6)
    parser.add_argument("--use_sam_guided_semantic_scanning", action="store_true")
    parser.add_argument("--sam_semantic_scanning_count", type=int, default=6)
    parser.add_argument("--use_wavelet_augmented_ss1", action="store_true")
    parser.add_argument("--wavelet_augmented_ss1_count", type=int, default=3)
    parser.add_argument("--wavelet_augmented_ss1_topk_ratio", type=float, default=0.08)
    parser.add_argument("--wavelet_augmented_ss1_strength", type=float, default=0.10)
    parser.add_argument("--wavelet_augmented_ss1_mode", type=str, default="stable_intra_region")
    parser.add_argument("--rgb_indices", type=int, nargs=3, default=[4, 2, 1], help="0-based RGB band indices.")
    parser.set_defaults(
        use_ase=True,
        use_ase_fusion_residual=True,
        use_wavelet=True,
        use_wavelet_priors=True,
        use_wavelet_local_bias=True,
        use_joint_spatial_spectral_wavelet_prior=True,
    )
    return parser.parse_args()


def _extract_state_dict(checkpoint: Dict) -> Dict[str, torch.Tensor]:
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _apply_checkpoint_args(args: argparse.Namespace, checkpoint: Dict) -> argparse.Namespace:
    ckpt_args = checkpoint.get("args", {})
    for key in [
        "channels",
        "ratio",
        "use_ase",
        "ase_prompt_mode",
        "ase_route_temperature",
        "ase_scope",
        "ase_stage_scope",
        "use_ase_fusion_residual",
        "ase_fusion_res_scale",
        "use_wavelet",
        "use_wavelet_priors",
        "use_wavelet_local_bias",
        "wavelet_local_bias_scale",
        "use_wavelet_local_gate",
        "wavelet_local_gate_scale",
        "use_joint_spatial_spectral_wavelet_prior",
        "joint_wavelet_spatial_weight",
        "joint_wavelet_spectral_weight",
        "use_sam_region_prototype_bank",
        "sam_region_prototype_bank_scale",
        "sam_region_prototype_count",
        "use_sam_guided_semantic_scanning",
        "sam_semantic_scanning_count",
        "use_wavelet_augmented_ss1",
        "wavelet_augmented_ss1_count",
        "wavelet_augmented_ss1_topk_ratio",
        "wavelet_augmented_ss1_strength",
        "wavelet_augmented_ss1_mode",
    ]:
        if key in ckpt_args:
            setattr(args, key, ckpt_args[key])
    return args


def _use_wavelet_side_inputs(args: argparse.Namespace) -> bool:
    return bool(
        should_use_wavelet_priors(
            use_wavelet_legacy=args.use_wavelet,
            use_wavelet_priors=args.use_wavelet_priors,
            use_joint_spatial_spectral_wavelet_prior=args.use_joint_spatial_spectral_wavelet_prior,
            use_wavelet_local_bias=args.use_wavelet_local_bias,
            use_wavelet_local_gate=args.use_wavelet_local_gate,
        )
        or args.use_wavelet_augmented_ss1
    )


def _build_model(args: argparse.Namespace, device: torch.device) -> Net:
    return Net(
        dim=args.channels,
        lr_hsi_dim=8,
        hr_msi_dim=1,
        scale=args.ratio,
        use_ase=args.use_ase,
        ase_prompt_mode=args.ase_prompt_mode,
        ase_route_temperature=args.ase_route_temperature,
        ase_scope=args.ase_scope,
        ase_stage_scope=args.ase_stage_scope,
        use_ase_fusion_residual=args.use_ase_fusion_residual,
        ase_fusion_res_scale=args.ase_fusion_res_scale,
        use_wavelet=args.use_wavelet,
        use_wavelet_priors=args.use_wavelet_priors,
        use_wavelet_local_bias=args.use_wavelet_local_bias,
        wavelet_local_bias_scale=args.wavelet_local_bias_scale,
        use_wavelet_local_gate=args.use_wavelet_local_gate,
        wavelet_local_gate_scale=args.wavelet_local_gate_scale,
        use_joint_spatial_spectral_wavelet_prior=args.use_joint_spatial_spectral_wavelet_prior,
        joint_wavelet_spatial_weight=args.joint_wavelet_spatial_weight,
        joint_wavelet_spectral_weight=args.joint_wavelet_spectral_weight,
        use_sam_region_prototype_bank=args.use_sam_region_prototype_bank,
        sam_region_prototype_bank_scale=args.sam_region_prototype_bank_scale,
        sam_region_prototype_count=args.sam_region_prototype_count,
        use_sam_guided_semantic_scanning=args.use_sam_guided_semantic_scanning,
        sam_semantic_scanning_count=args.sam_semantic_scanning_count,
        use_wavelet_augmented_ss1=args.use_wavelet_augmented_ss1,
        wavelet_augmented_ss1_count=args.wavelet_augmented_ss1_count,
        wavelet_augmented_ss1_topk_ratio=args.wavelet_augmented_ss1_topk_ratio,
        wavelet_augmented_ss1_strength=args.wavelet_augmented_ss1_strength,
        wavelet_augmented_ss1_mode=args.wavelet_augmented_ss1_mode,
    ).to(device)


def _forward_model(model: Net, batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    hr_msi = batch["hr_msi"].to(device).float()
    lr_hsi = batch["lr_hsi"].to(device).float()
    lr_hsi_approx = batch.get("lr_hsi_approx", None)
    lr_hsi_details = batch.get("lr_hsi_details", None)
    cached_sam_features = batch.get("cached_sam_features", None)
    cached_sam_masks = batch.get("cached_sam_masks", None)

    if lr_hsi_approx is not None:
        lr_hsi_approx = lr_hsi_approx.to(device).float()
    if lr_hsi_details is not None:
        lr_hsi_details = lr_hsi_details.to(device).float()
    if cached_sam_features is not None:
        cached_sam_features = cached_sam_features.to(device).float()
    if cached_sam_masks is not None:
        cached_sam_masks = cached_sam_masks.to(device).float()

    output = model(
        hr_msi,
        lr_hsi,
        lr_hsi_approx,
        lr_hsi_details,
        cached_sam_features=cached_sam_features,
        cached_sam_masks=cached_sam_masks,
    )
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output


def _select_informative_crop(pan: np.ndarray, crop_size: int = 128) -> tuple[int, int, int, int]:
    gy, gx = np.gradient(pan)
    energy = gx * gx + gy * gy
    best_score = -1.0
    best_box = (0, 0, crop_size, crop_size)
    stride = max(16, crop_size // 4)
    h, w = pan.shape
    crop_size = min(crop_size, h, w)
    for top in range(0, max(h - crop_size, 0) + 1, stride):
        for left in range(0, max(w - crop_size, 0) + 1, stride):
            score = float(energy[top:top + crop_size, left:left + crop_size].mean())
            if score > best_score:
                best_score = score
                best_box = (left, top, crop_size, crop_size)
    return best_box


def _save_rr_visual(
    pan: torch.Tensor,
    lms: torch.Tensor,
    fused: torch.Tensor,
    gt: torch.Tensor,
    save_path: Path,
    rgb_indices: Sequence[int],
) -> None:
    pan_np = pan.detach().cpu().float().numpy()
    if pan_np.ndim == 3:
        pan_np = pan_np[0]
    lms_raw = chw_to_rgb(lms, rgb_indices)
    gt_raw = chw_to_rgb(gt, rgb_indices)
    fused_raw = chw_to_rgb(fused, rgb_indices)

    low, high = compute_display_range(gt_raw)
    lms_rgb = apply_display_range(lms_raw, low, high)
    gt_rgb = apply_display_range(gt_raw, low, high)
    fused_rgb = apply_display_range(fused_raw, low, high)
    error_rgb = np.abs(fused_rgb - gt_rgb)

    left, top, width, height = _select_informative_crop(pan_np, crop_size=min(128, pan_np.shape[0], pan_np.shape[1]))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(pan_np, cmap="gray")
    axes[0, 0].set_title("PAN")
    axes[0, 1].imshow(lms_rgb)
    axes[0, 1].set_title("LMS")
    axes[0, 2].imshow(fused_rgb)
    axes[0, 2].set_title("Fused")
    axes[0, 3].imshow(gt_rgb)
    axes[0, 3].set_title("GT")
    for ax in axes[0]:
        rect = plt.Rectangle((left, top), width, height, fill=False, edgecolor="red", linewidth=1.5)
        ax.add_patch(rect)
        ax.axis("off")

    axes[1, 0].imshow(error_rgb)
    axes[1, 0].set_title("Abs Error")
    axes[1, 1].imshow(lms_rgb[top:top + height, left:left + width])
    axes[1, 1].set_title("LMS Zoom")
    axes[1, 2].imshow(fused_rgb[top:top + height, left:left + width])
    axes[1, 2].set_title("Fused Zoom")
    axes[1, 3].imshow(gt_rgb[top:top + height, left:left + width])
    axes[1, 3].set_title("GT Zoom")
    for ax in axes[1]:
        ax.axis("off")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.weight, map_location="cpu")
    args = _apply_checkpoint_args(args, checkpoint if isinstance(checkpoint, dict) else {})

    model = _build_model(args, device)
    model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
    model.eval()

    save_dir = Path(args.save_dir) if args.save_dir else Path(args.weight).resolve().parent / "test_wv3_rr"
    viz_dir = save_dir / "visualization"
    viz_dir.mkdir(parents=True, exist_ok=True)

    dataset = WV3TrainValDataset(
        args.h5_path,
        use_wavelet=_use_wavelet_side_inputs(args),
        use_offline_sam_cache=args.use_offline_sam_cache,
        sam_cache_path=args.sam_cache_path,
        sam_cache_strict=args.sam_cache_strict,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    ergas_metric = ERGAS(ratio=args.ratio).to(device)

    per_sample = []
    with torch.no_grad():
        for batch in loader:
            idx = int(batch["sample_idx"].item())
            hr_hsi = batch["hr_hsi"].to(device).float()
            sr = _forward_model(model, batch, device)
            ssim = float(compute_ssim(sr, hr_hsi).item())
            sam = float(compute_sam(sr, hr_hsi).item())
            rmse = float(compute_rmse(sr, hr_hsi).item())
            psnr = float(compute_psnr(sr, hr_hsi).item())
            ergas = float(ergas_metric(sr, hr_hsi).item())

            per_sample.append({
                "sample": idx,
                "ssim": ssim,
                "sam": sam,
                "rmse": rmse,
                "psnr": psnr,
                "ergas": ergas,
            })
            print(f"Sample {idx + 1}/{len(dataset)} | SSIM={ssim:.4f} SAM={sam:.4f} RMSE={rmse:.4f} PSNR={psnr:.4f} ERGAS={ergas:.4f}")

            _save_rr_visual(
                pan=batch["hr_msi"].squeeze(0).cpu(),
                lms=batch["lms"].squeeze(0).cpu(),
                fused=sr.squeeze(0).cpu(),
                gt=hr_hsi.squeeze(0).cpu(),
                save_path=viz_dir / f"sample_{idx:04d}.png",
                rgb_indices=args.rgb_indices,
            )

    summary = {}
    for key in ["ssim", "sam", "rmse", "psnr", "ergas"]:
        values = np.array([row[key] for row in per_sample], dtype=np.float32)
        summary[key] = {"mean": float(values.mean()), "std": float(values.std())}

    with (save_dir / "metrics_per_sample.json").open("w", encoding="utf-8") as handle:
        json.dump(per_sample, handle, indent=2, ensure_ascii=False)
    with (save_dir / "metrics_per_sample.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample", "ssim", "sam", "rmse", "psnr", "ergas"])
        writer.writeheader()
        writer.writerows(per_sample)
    with (save_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with (save_dir / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"平均 SSIM: {summary['ssim']['mean']:.4f} +/- {summary['ssim']['std']:.4f}\n")
        handle.write(f"平均 SAM: {summary['sam']['mean']:.4f} deg +/- {summary['sam']['std']:.4f} deg\n")
        handle.write(f"平均 RMSE: {summary['rmse']['mean']:.4f} +/- {summary['rmse']['std']:.4f}\n")
        handle.write(f"平均 PSNR: {summary['psnr']['mean']:.4f} dB +/- {summary['psnr']['std']:.4f} dB\n")
        handle.write(f"平均 ERGAS: {summary['ergas']['mean']:.4f} +/- {summary['ergas']['std']:.4f}\n")

    print("[DONE] WV3 reduced-resolution evaluation finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
