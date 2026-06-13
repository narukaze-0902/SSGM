from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader

from model.u2net import U2Net as Net
from utils.load_wv3_data import (
    WV3RealDataset,
    apply_display_range,
    chw_to_rgb,
    compute_display_range,
)
from utils.wavelet_utils import should_use_wavelet_priors
from utils.wv3_metrics import compute_qnr, summarize_real_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-world/original-scale evaluation on WV3.")
    parser.add_argument("--h5_path", type=str, default="./data/WV3/full_examples/full_examples/test_wv3_OrigScale_multiExm1.h5")
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
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--crop_manifest_in", type=str, default=None, help="Optional JSON manifest with fixed crop boxes.")
    parser.add_argument(
        "--crop_manifest_out",
        type=str,
        default=None,
        help="Optional JSON path to export crop boxes used by this run. Defaults to save_dir/visualization_manifest.json.",
    )
    parser.add_argument("--qnr_block_size", type=int, default=32)
    parser.add_argument("--qnr_p", type=int, default=1)
    parser.add_argument("--qnr_alpha", type=float, default=1.0)
    parser.add_argument("--qnr_beta", type=float, default=1.0)
    parser.add_argument("--save_mat", action="store_true", help="Save fused outputs to MAT files.")
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
            score = float(energy[top : top + crop_size, left : left + crop_size].mean())
            if score > best_score:
                best_score = score
                best_box = (left, top, crop_size, crop_size)
    return best_box


def _load_crop_manifest(path: str | None) -> Dict[int, Dict[str, int]]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "samples" in payload:
        payload = payload["samples"]

    loaded: Dict[int, Dict[str, int]] = {}
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict) or "sample" not in item:
                continue
            sample_idx = int(item["sample"])
            loaded[sample_idx] = {
                "left": int(item["left"]),
                "top": int(item["top"]),
                "width": int(item["width"]),
                "height": int(item["height"]),
            }
    elif isinstance(payload, dict):
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            sample_idx = int(key)
            loaded[sample_idx] = {
                "left": int(value["left"]),
                "top": int(value["top"]),
                "width": int(value["width"]),
                "height": int(value["height"]),
            }
    return loaded


def _save_real_visual(
    pan: torch.Tensor,
    lms: torch.Tensor,
    fused: torch.Tensor,
    save_path: Path,
    rgb_indices: Sequence[int],
    crop_size: int,
    fixed_crop_box: Dict[str, int] | None = None,
) -> Dict[str, int]:
    pan_np = pan.detach().cpu().float().numpy()
    if pan_np.ndim == 3:
        pan_np = pan_np[0]
    lms_raw = chw_to_rgb(lms, rgb_indices)
    fused_raw = chw_to_rgb(fused, rgb_indices)
    low, high = compute_display_range(lms_raw)
    lms_rgb = apply_display_range(lms_raw, low, high)
    fused_rgb = apply_display_range(fused_raw, low, high)

    if fixed_crop_box is None:
        left, top, width, height = _select_informative_crop(
            pan_np,
            crop_size=min(crop_size, pan_np.shape[0], pan_np.shape[1]),
        )
    else:
        left = int(fixed_crop_box["left"])
        top = int(fixed_crop_box["top"])
        width = int(fixed_crop_box["width"])
        height = int(fixed_crop_box["height"])

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(pan_np, cmap="gray")
    axes[0, 0].set_title("PAN")
    axes[0, 1].imshow(lms_rgb)
    axes[0, 1].set_title("LMS")
    axes[0, 2].imshow(fused_rgb)
    axes[0, 2].set_title("Fused")
    for ax in axes[0]:
        rect = plt.Rectangle((left, top), width, height, fill=False, edgecolor="red", linewidth=1.5)
        ax.add_patch(rect)
        ax.axis("off")

    axes[1, 0].imshow(pan_np[top : top + height, left : left + width], cmap="gray")
    axes[1, 0].set_title("PAN Zoom")
    axes[1, 1].imshow(lms_rgb[top : top + height, left : left + width])
    axes[1, 1].set_title("LMS Zoom")
    axes[1, 2].imshow(fused_rgb[top : top + height, left : left + width])
    axes[1, 2].set_title("Fused Zoom")
    for ax in axes[1]:
        ax.axis("off")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"left": left, "top": top, "width": width, "height": height}


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.weight, map_location="cpu")
    args = _apply_checkpoint_args(args, checkpoint if isinstance(checkpoint, dict) else {})

    model = _build_model(args, device)
    model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
    model.eval()

    save_dir = Path(args.save_dir) if args.save_dir else Path(args.weight).resolve().parent / "test_wv3_real"
    viz_dir = save_dir / "visualization"
    mat_dir = save_dir / "mat_outputs"
    manifest_out = Path(args.crop_manifest_out) if args.crop_manifest_out else save_dir / "visualization_manifest.json"
    viz_dir.mkdir(parents=True, exist_ok=True)
    if args.save_mat:
        mat_dir.mkdir(parents=True, exist_ok=True)
    fixed_crops = _load_crop_manifest(args.crop_manifest_in)

    dataset = WV3RealDataset(
        args.h5_path,
        use_wavelet=_use_wavelet_side_inputs(args),
        use_offline_sam_cache=args.use_offline_sam_cache,
        sam_cache_path=args.sam_cache_path,
        sam_cache_strict=args.sam_cache_strict,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    per_sample = []
    visualization_manifest = {
        "dataset": "wv3_origscale",
        "rgb_indices_0_based": [int(v) for v in args.rgb_indices],
        "rgb_indices_1_based": [int(v) + 1 for v in args.rgb_indices],
        "crop_size": int(args.crop_size),
        "crop_selection_rule": "fixed_from_manifest" if fixed_crops else "auto_pan_gradient_energy_stride32",
        "samples": [],
    }

    with torch.no_grad():
        for batch in loader:
            idx = int(batch["sample_idx"].item())
            hr_msi = batch["hr_msi"]
            lms = batch["lms"]
            sr = _forward_model(model, batch, device).clamp(0.0, 1.0)
            qnr_metrics = compute_qnr(
                fused=sr.squeeze(0).cpu(),
                lms=lms.squeeze(0),
                pan=hr_msi.squeeze(0).cpu(),
                block_size=args.qnr_block_size,
                p=args.qnr_p,
                alpha=args.qnr_alpha,
                beta=args.qnr_beta,
            )
            crop_box = _save_real_visual(
                pan=hr_msi.squeeze(0).cpu(),
                lms=lms.squeeze(0).cpu(),
                fused=sr.squeeze(0).cpu(),
                save_path=viz_dir / f"sample_{idx:04d}.png",
                rgb_indices=args.rgb_indices,
                crop_size=args.crop_size,
                fixed_crop_box=fixed_crops.get(idx),
            )
            row = {"sample": idx, **qnr_metrics, **crop_box}
            per_sample.append(row)
            visualization_manifest["samples"].append({"sample": idx, **crop_box})
            print(
                f"Sample {idx + 1}/{len(dataset)} | "
                f"D_lambda={qnr_metrics['D_lambda']:.4f} "
                f"D_s={qnr_metrics['D_s']:.4f} "
                f"QNR={qnr_metrics['QNR']:.4f}"
            )

            if args.save_mat:
                fused_hwc = sr.squeeze(0).cpu().permute(1, 2, 0).numpy()
                sio.savemat(mat_dir / f"sample_{idx:04d}.mat", {"fused": fused_hwc})

    summary = summarize_real_metrics(per_sample)
    with (save_dir / "metrics_per_sample.json").open("w", encoding="utf-8") as handle:
        json.dump(per_sample, handle, indent=2, ensure_ascii=False)
    with (save_dir / "metrics_per_sample.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample", "D_lambda", "D_s", "QNR", "left", "top", "width", "height"])
        writer.writeheader()
        writer.writerows(per_sample)
    with (save_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with manifest_out.open("w", encoding="utf-8") as handle:
        json.dump(visualization_manifest, handle, indent=2, ensure_ascii=False)
    with (save_dir / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"平均 D_lambda: {summary['D_lambda']['mean']:.4f} +/- {summary['D_lambda']['std']:.4f}\n")
        handle.write(f"平均 D_s: {summary['D_s']['mean']:.4f} +/- {summary['D_s']['std']:.4f}\n")
        handle.write(f"平均 QNR: {summary['QNR']['mean']:.4f} +/- {summary['QNR']['std']:.4f}\n")
        handle.write(f"visualization_manifest: {manifest_out.as_posix()}\n")

    print("[DONE] WV3 real-world evaluation finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
