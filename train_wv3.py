from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.haar_dwt import SimpleDWT
from model.u2net import U2Net as Net
from utils.load_wv3_data import WV3TrainValDataset
from utils.tools import ERGAS, compute_psnr, compute_rmse, compute_sam, compute_ssim
from utils.wavelet_utils import should_use_wavelet_priors


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FusionMamba/U2Net on WV3 pansharpening data.")
    parser.add_argument("--exp_code", type=str, default="WV3F3")
    parser.add_argument("--train_h5", type=str, default="./data/WV3/train_wv3.h5")
    parser.add_argument("--val_h5", type=str, default="./data/WV3/valid_wv3.h5")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val_freq", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--save_root", type=str, default="./weights")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--keep_recent_epochs", type=int, default=25)
    parser.add_argument("--save_freq", type=int, default=5)
    parser.add_argument("--disable_persistent_workers", action="store_true")
    parser.add_argument("--disable_pin_memory", action="store_true")
    parser.add_argument("--preload_train_data_to_ram", action="store_true")
    parser.add_argument("--preload_train_sam_cache_to_ram", action="store_true")
    parser.add_argument("--preload_val_data_to_ram", action="store_true")
    parser.add_argument("--preload_val_sam_cache_to_ram", action="store_true")
    parser.add_argument("--train_subset_size", type=int, default=0, help="use a fixed random subset of the training set; 0 means full training set")
    parser.add_argument("--train_subset_seed", type=int, default=3407)

    parser.add_argument("--use_ase", action="store_true")
    parser.add_argument("--ase_prompt_mode", type=str, default="soft")
    parser.add_argument("--ase_route_temperature", type=float, default=1.2)
    parser.add_argument("--ase_scope", type=str, default="fusion_only")
    parser.add_argument("--ase_stage_scope", type=str, default="all_stages")
    parser.add_argument("--use_ase_fusion_residual", action="store_true")
    parser.add_argument("--ase_fusion_res_scale", type=float, default=0.4)
    parser.add_argument("--route_reg_weight", type=float, default=0.0)

    parser.add_argument("--use_wavelet", action="store_true")
    parser.add_argument("--use_wavelet_priors", action="store_true")
    parser.add_argument("--use_wavelet_local_bias", action="store_true")
    parser.add_argument("--wavelet_local_bias_scale", type=float, default=0.1)
    parser.add_argument("--use_wavelet_local_gate", action="store_true")
    parser.add_argument("--wavelet_local_gate_scale", type=float, default=0.1)
    parser.add_argument("--use_joint_spatial_spectral_wavelet_prior", action="store_true")
    parser.add_argument("--joint_wavelet_spatial_weight", type=float, default=1.0)
    parser.add_argument("--joint_wavelet_spectral_weight", type=float, default=0.7)
    parser.add_argument("--use_hf_wavelet_loss", action="store_true")
    parser.add_argument("--hf_wavelet_loss_weight", type=float, default=0.01)
    parser.add_argument("--hf_wavelet_loss_start_epoch", type=int, default=5)

    parser.add_argument("--use_offline_sam_cache", action="store_true")
    parser.add_argument("--sam_cache_path", type=str, default=None)
    parser.add_argument("--val_sam_cache_path", type=str, default=None)
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

    parser.add_argument("--use_boundary_selective_wavelet_loss", action="store_true")
    parser.add_argument("--boundary_selective_wavelet_loss_weight", type=float, default=0.002)
    parser.add_argument("--boundary_selective_wavelet_loss_start_epoch", type=int, default=20)
    parser.add_argument("--boundary_selective_wavelet_boundary_boost", type=float, default=0.60)
    parser.add_argument("--boundary_selective_wavelet_frequency_boost", type=float, default=0.40)
    parser.set_defaults(
        use_ase=True,
        use_ase_fusion_residual=True,
        use_wavelet=True,
        use_wavelet_priors=True,
        use_wavelet_local_bias=True,
        use_joint_spatial_spectral_wavelet_prior=True,
        use_hf_wavelet_loss=True,
    )
    return parser.parse_args()


class HFWaveletLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwt = SimpleDWT()

    @staticmethod
    def _crop_to_even_size(x: torch.Tensor) -> torch.Tensor:
        h = x.shape[-2] - (x.shape[-2] % 2)
        w = x.shape[-1] - (x.shape[-1] % 2)
        return x[..., :h, :w]

    def _weighted_l1(self, pred_band: torch.Tensor, target_band: torch.Tensor, spatial_weight: Optional[torch.Tensor]) -> torch.Tensor:
        diff = torch.abs(pred_band - target_band)
        if spatial_weight is None:
            return diff.mean()

        if spatial_weight.dim() == 3:
            spatial_weight = spatial_weight.unsqueeze(1)
        if spatial_weight.shape[-2:] != diff.shape[-2:]:
            spatial_weight = F.interpolate(spatial_weight, size=diff.shape[-2:], mode="bilinear", align_corners=False)
        spatial_weight = spatial_weight.to(device=diff.device, dtype=diff.dtype)
        weighted = diff * spatial_weight
        norm = spatial_weight.sum() * diff.shape[1] + 1e-6
        return weighted.sum() / norm

    def forward(self, pred: torch.Tensor, target: torch.Tensor, spatial_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred = self._crop_to_even_size(pred)
        target = self._crop_to_even_size(target)
        if spatial_weight is not None:
            spatial_weight = self._crop_to_even_size(spatial_weight)

        _, pred_hf_list = self.dwt(pred)
        _, target_hf_list = self.dwt(target)

        hf_loss = pred.new_tensor(0.0)
        for pred_band, target_band in zip(pred_hf_list, target_hf_list):
            hf_loss = hf_loss + self._weighted_l1(pred_band, target_band, spatial_weight)
        return hf_loss / len(pred_hf_list)


def _normalize_spatial_prior(map_tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
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


def _build_boundary_map_from_regions(region_map: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if region_map is None:
        return None
    dx = torch.abs(region_map[:, :, :, 1:] - region_map[:, :, :, :-1])
    dy = torch.abs(region_map[:, :, 1:, :] - region_map[:, :, :-1, :])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return _normalize_spatial_prior(dx + dy)


def build_boundary_selective_wavelet_weight_map(
    sam_region_context: Optional[Dict],
    cached_sam_masks: Optional[torch.Tensor],
    boundary_boost: float = 0.60,
    frequency_boost: float = 0.40,
) -> Optional[torch.Tensor]:
    if sam_region_context is None and cached_sam_masks is None:
        return None

    prior_map = None
    if sam_region_context is not None:
        prior_map = sam_region_context.get("semantic_frequency_prior_map")
        if prior_map is None:
            prior_map = sam_region_context.get("wavelet_prior_map")
        if prior_map is None:
            prior_map = sam_region_context.get("wavelet_guidance")

    confidence_map = None
    boundary_map = None
    if cached_sam_masks is not None:
        if cached_sam_masks.dim() == 3:
            cached_sam_masks = cached_sam_masks.unsqueeze(1)
        if cached_sam_masks.dim() == 4:
            sam_masks = cached_sam_masks.float()
            confidence_map, region_index = sam_masks.max(dim=1, keepdim=True)
            region_ids = region_index.float() + 1.0
            region_ids = torch.where(confidence_map > 1e-5, region_ids, torch.zeros_like(region_ids))
            boundary_map = _build_boundary_map_from_regions(region_ids)

    if prior_map is not None:
        prior_map = _normalize_spatial_prior(prior_map)
    if confidence_map is not None:
        confidence_map = _normalize_spatial_prior(confidence_map)

    weight_map = None
    for component, scale in (
        (prior_map, max(float(frequency_boost), 0.0)),
        (boundary_map, max(float(boundary_boost), 0.0)),
        (confidence_map, 0.25),
    ):
        if component is None or scale <= 0:
            continue
        component = component.float()
        weight_map = scale * component if weight_map is None else weight_map + scale * component

    if weight_map is None:
        return None
    return _normalize_spatial_prior(weight_map).clamp(0.0, 1.0)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_resume_out_dir(resume_path: str | None) -> Optional[Path]:
    if not resume_path:
        return None
    resume_file = Path(resume_path).resolve()
    if not resume_file.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_file}")
    out_dir = resume_file.parent
    if out_dir.name == "checkpoints":
        out_dir = out_dir.parent
    if not out_dir.is_dir():
        raise NotADirectoryError(f"Resume directory not found: {out_dir}")
    return out_dir


def _make_output_dir(args: argparse.Namespace) -> Path:
    resume_out_dir = _resolve_resume_out_dir(getattr(args, "resume", None))
    if resume_out_dir is not None:
        return resume_out_dir
    out_dir = Path(args.save_root) / f"{args.exp_code}_wv3_x{args.ratio}_{_timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_args(args: argparse.Namespace, out_dir: Path) -> None:
    with (out_dir / "training_params.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)


def _use_wavelet_side_inputs(args: argparse.Namespace) -> bool:
    return bool(
        should_use_wavelet_priors(
            use_wavelet_legacy=args.use_wavelet,
            use_wavelet_priors=args.use_wavelet_priors,
            use_joint_spatial_spectral_wavelet_prior=args.use_joint_spatial_spectral_wavelet_prior,
            use_wavelet_local_bias=args.use_wavelet_local_bias,
            use_wavelet_local_gate=args.use_wavelet_local_gate,
            use_hf_wavelet_loss=args.use_hf_wavelet_loss,
            use_boundary_selective_wavelet_loss=args.use_boundary_selective_wavelet_loss,
        )
        or args.use_wavelet_augmented_ss1
    )


def _build_train_subset_indices(dataset_length: int, subset_size: int, subset_seed: int) -> Optional[np.ndarray]:
    if subset_size <= 0 or subset_size >= dataset_length:
        return None
    rng = np.random.default_rng(subset_seed)
    indices = np.sort(rng.choice(dataset_length, size=subset_size, replace=False).astype(np.int64))
    return indices


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


def _extract_state_dict(checkpoint: Dict) -> Dict[str, torch.Tensor]:
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _safe_torch_save(payload: Dict, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f".{target_path.name}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _iter_resume_candidates(resume_path: str | None) -> List[Path]:
    if not resume_path:
        return []
    resume_file = Path(resume_path).resolve()
    candidates: List[Path] = [resume_file]
    try:
        out_dir = _resolve_resume_out_dir(resume_path)
    except Exception:
        return candidates

    best_path = out_dir / "best_by_psnr.pth"
    if best_path not in candidates and best_path.exists():
        candidates.append(best_path)

    checkpoint_dir = out_dir / "checkpoints"
    if checkpoint_dir.is_dir():
        epoch_paths = sorted(checkpoint_dir.glob("epoch_*.pth"), reverse=True)
        for epoch_path in epoch_paths:
            if epoch_path not in candidates:
                candidates.append(epoch_path)
    return candidates


def _load_resume(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    resume_path: str | None,
    device: torch.device,
) -> Tuple[int, float, int]:
    if not resume_path:
        return 0, -float("inf"), -1

    checkpoint = None
    loaded_from: Optional[Path] = None
    last_error: Optional[Exception] = None
    for candidate in _iter_resume_candidates(resume_path):
        try:
            checkpoint = torch.load(candidate, map_location=device)
            loaded_from = candidate
            break
        except Exception as exc:
            last_error = exc
            print(f"[WARN] Failed to load checkpoint: {candidate} ({exc})")

    if checkpoint is None:
        raise RuntimeError(f"Unable to resume from {resume_path}") from last_error

    if loaded_from is not None and loaded_from != Path(resume_path).resolve():
        print(f"[INFO] Resume fallback: using {loaded_from}")

    model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    best_psnr = float(checkpoint.get("best_psnr", checkpoint.get("val_metrics", {}).get("psnr", -float("inf"))))
    best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0)))
    return int(checkpoint.get("epoch", 0)), best_psnr, best_epoch


def _save_recent_epoch_checkpoint(checkpoint: Dict, checkpoint_dir: Path, epoch: int, keep_recent_epochs: int) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"epoch_{epoch:04d}.pth"
    _safe_torch_save(checkpoint, ckpt_path)

    if keep_recent_epochs > 0:
        existing: List[Path] = sorted(checkpoint_dir.glob("epoch_*.pth"))
        stale = existing[:-keep_recent_epochs]
        for old_path in stale:
            old_path.unlink(missing_ok=True)
    return ckpt_path


def _forward_model(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[List]]:
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

    model_output = model(
        hr_msi,
        lr_hsi,
        lr_hsi_approx,
        lr_hsi_details,
        cached_sam_features=cached_sam_features,
        cached_sam_masks=cached_sam_masks,
    )
    if isinstance(model_output, tuple):
        sr, routing_probs = model_output
    else:
        sr = model_output
        routing_probs = None
    return sr, routing_probs


def _routing_entropy_regularizer(routing_probs: Optional[List]) -> Optional[torch.Tensor]:
    if routing_probs is None:
        return None
    total_loss = None
    valid_terms = 0
    for stage_probs in routing_probs:
        candidates = stage_probs if isinstance(stage_probs, list) else [stage_probs]
        for probs in candidates:
            if probs is None:
                continue
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
            total_loss = entropy if total_loss is None else total_loss + entropy
            valid_terms += 1
    if valid_terms == 0:
        return None
    return total_loss / valid_terms


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ratio: int,
) -> Dict[str, float]:
    model.eval()
    l1_values = []
    ssim_values = []
    sam_values = []
    rmse_values = []
    psnr_values = []
    ergas_values = []
    ergas_metric = ERGAS(ratio=ratio).to(device)
    with torch.no_grad():
        for batch in loader:
            hr_hsi = batch["hr_hsi"].to(device).float()
            sr, _ = _forward_model(model, batch, device)
            l1_values.append(torch.mean(torch.abs(sr - hr_hsi)).item())
            ssim_values.append(float(compute_ssim(sr, hr_hsi).item()))
            sam_values.append(float(compute_sam(sr, hr_hsi).item()))
            rmse_values.append(float(compute_rmse(sr, hr_hsi).item()))
            psnr_values.append(float(compute_psnr(sr, hr_hsi).item()))
            ergas_values.append(float(ergas_metric(sr, hr_hsi).item()))

    return {
        "l1": float(np.mean(l1_values)),
        "ssim": float(np.mean(ssim_values)),
        "sam": float(np.mean(sam_values)),
        "rmse": float(np.mean(rmse_values)),
        "psnr": float(np.mean(psnr_values)),
        "ergas": float(np.mean(ergas_values)),
    }


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = _make_output_dir(args)
    if args.resume:
        print(f"[INFO] Resume mode: continuing in existing directory {out_dir}")
    _save_args(args, out_dir)

    use_wavelet_side_inputs = _use_wavelet_side_inputs(args)

    if args.preload_train_sam_cache_to_ram:
        print("[WV3-RAM] train SAM cache preload enabled; this can consume a large amount of RAM.")
    if args.preload_val_data_to_ram or args.preload_val_sam_cache_to_ram:
        print("[WV3-RAM] validation RAM preload enabled.")

    with h5py.File(args.train_h5, "r") as train_handle:
        train_length = int(train_handle["ms"].shape[0])
    train_subset_indices = _build_train_subset_indices(
        dataset_length=train_length,
        subset_size=int(args.train_subset_size),
        subset_seed=int(args.train_subset_seed),
    )
    if train_subset_indices is not None:
        print(
            f"[WV3-TRAIN] using fixed subset: {len(train_subset_indices)}/{train_length} "
            f"(seed={args.train_subset_seed})"
        )
        with (out_dir / "train_subset_indices.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "train_subset_size": int(len(train_subset_indices)),
                    "train_full_size": int(train_length),
                    "train_subset_seed": int(args.train_subset_seed),
                    "indices": train_subset_indices.tolist(),
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

    train_dataset = WV3TrainValDataset(
        args.train_h5,
        use_wavelet=use_wavelet_side_inputs,
        use_offline_sam_cache=args.use_offline_sam_cache,
        sam_cache_path=args.sam_cache_path,
        sam_cache_strict=args.sam_cache_strict,
        preload_to_ram=args.preload_train_data_to_ram,
        preload_sam_cache_to_ram=args.preload_train_sam_cache_to_ram,
        selected_indices=train_subset_indices,
    )
    val_dataset = WV3TrainValDataset(
        args.val_h5,
        use_wavelet=use_wavelet_side_inputs,
        use_offline_sam_cache=args.use_offline_sam_cache,
        sam_cache_path=args.val_sam_cache_path or args.sam_cache_path,
        sam_cache_strict=args.sam_cache_strict,
        preload_to_ram=args.preload_val_data_to_ram,
        preload_sam_cache_to_ram=args.preload_val_sam_cache_to_ram,
    )

    use_persistent_workers = bool(args.num_workers > 0 and not args.disable_persistent_workers)
    pin_memory = bool(not args.disable_pin_memory)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=use_persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=use_persistent_workers,
    )

    model = _build_model(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.L1Loss()

    hf_wavelet_loss_fn = HFWaveletLoss().to(device) if args.use_hf_wavelet_loss else None
    boundary_selective_wavelet_loss_fn = HFWaveletLoss().to(device) if args.use_boundary_selective_wavelet_loss else None

    start_epoch, best_psnr, best_epoch = _load_resume(model, optimizer, scheduler, args.resume, device)

    history_path = out_dir / "history.csv"
    checkpoint_dir = out_dir / "checkpoints"
    global_start_time = time.time()

    if args.use_sam_guided_semantic_scanning and not args.use_offline_sam_cache:
        print("[WV3-SAM] WARNING: semantic scanning is enabled but no offline cache is provided; SAM guidance will be unavailable.")
    if args.use_wavelet_augmented_ss1 and not args.use_offline_sam_cache:
        print("[WV3-WSS] WARNING: wavelet-augmented SS1 is enabled but no offline cache is provided; WSS behavior will be unavailable.")

    history_exists = history_path.exists() and start_epoch > 0
    history_mode = "a" if history_exists else "w"

    with history_path.open(history_mode, encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if not history_exists:
            writer.writerow([
                "epoch",
                "train_l1",
                "val_l1",
                "val_psnr",
                "val_ssim",
                "val_sam",
                "val_rmse",
                "val_ergas",
                "lr",
                "train_sec",
                "val_sec",
                "save_sec",
                "epoch_total_sec",
                "elapsed_sec",
            ])

        for epoch in range(start_epoch, args.epochs):
            epoch_start_time = time.time()
            model.train()
            if hasattr(model, "set_current_epoch"):
                model.set_current_epoch(epoch)

            train_losses = []
            wavelet_aux_values = []
            boundary_wavelet_values = []
            route_reg_values = []

            for batch in train_loader:
                hr_hsi = batch["hr_hsi"].to(device).float()
                cached_sam_masks = batch.get("cached_sam_masks", None)
                if cached_sam_masks is not None:
                    cached_sam_masks = cached_sam_masks.to(device).float()

                optimizer.zero_grad(set_to_none=True)
                sr, routing_probs = _forward_model(model, batch, device)
                loss = criterion(sr, hr_hsi)

                route_reg_loss = None
                if args.route_reg_weight > 0:
                    route_reg_loss = _routing_entropy_regularizer(routing_probs)
                    if route_reg_loss is not None:
                        loss = loss + args.route_reg_weight * route_reg_loss

                hf_wavelet_aux_loss = None
                if hf_wavelet_loss_fn is not None and epoch >= args.hf_wavelet_loss_start_epoch:
                    hf_wavelet_aux_loss = hf_wavelet_loss_fn(sr.float(), hr_hsi.float())
                    loss = loss + args.hf_wavelet_loss_weight * hf_wavelet_aux_loss

                boundary_selective_wavelet_aux_loss = None
                if (
                    boundary_selective_wavelet_loss_fn is not None
                    and epoch >= args.boundary_selective_wavelet_loss_start_epoch
                    and cached_sam_masks is not None
                ):
                    model_ref = _unwrap_model(model)
                    boundary_selective_weight_map = build_boundary_selective_wavelet_weight_map(
                        getattr(model_ref, "current_sam_region_context", None),
                        cached_sam_masks,
                        boundary_boost=args.boundary_selective_wavelet_boundary_boost,
                        frequency_boost=args.boundary_selective_wavelet_frequency_boost,
                    )
                    if boundary_selective_weight_map is not None:
                        boundary_selective_wavelet_aux_loss = boundary_selective_wavelet_loss_fn(
                            sr.float(),
                            hr_hsi.float(),
                            spatial_weight=boundary_selective_weight_map.float(),
                        )
                        loss = loss + args.boundary_selective_wavelet_loss_weight * boundary_selective_wavelet_aux_loss

                loss.backward()
                optimizer.step()

                train_losses.append(float(loss.item()))
                if hf_wavelet_aux_loss is not None:
                    wavelet_aux_values.append(float(hf_wavelet_aux_loss.item()))
                if boundary_selective_wavelet_aux_loss is not None:
                    boundary_wavelet_values.append(float(boundary_selective_wavelet_aux_loss.item()))
                if route_reg_loss is not None:
                    route_reg_values.append(float(route_reg_loss.item()))

            scheduler.step()
            train_l1 = float(np.mean(train_losses))
            train_sec = float(time.time() - epoch_start_time)
            val_sec = 0.0
            save_sec = 0.0
            log_line = (
                f"[Epoch {epoch + 1:03d}/{args.epochs:03d}] "
                f"train_l1={train_l1:.6f} lr={optimizer.param_groups[0]['lr']:.6e} "
                f"train={train_sec:.1f}s"
            )
            if wavelet_aux_values:
                log_line += f" hf_wloss={float(np.mean(wavelet_aux_values)):.6f}"
            if boundary_wavelet_values:
                log_line += f" bound_wloss={float(np.mean(boundary_wavelet_values)):.6f}"
            if route_reg_values:
                log_line += f" route_reg={float(np.mean(route_reg_values)):.6f}"

            if ((epoch + 1) % args.val_freq) == 0 or epoch == args.epochs - 1:
                val_start_time = time.time()
                val_metrics = _validate(model, val_loader, device, args.ratio)
                val_sec = float(time.time() - val_start_time)
                log_line += (
                    f" | val_psnr={val_metrics['psnr']:.4f} val_ssim={val_metrics['ssim']:.4f}"
                    f" val_sam={val_metrics['sam']:.4f} val_rmse={val_metrics['rmse']:.4f}"
                    f" val_ergas={val_metrics['ergas']:.4f}"
                )

                checkpoint = {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "args": vars(args),
                    "val_metrics": val_metrics,
                    "best_psnr": best_psnr,
                    "best_epoch": best_epoch,
                }
                save_start_time = time.time()
                periodic_save = ((epoch + 1) % max(int(args.save_freq), 1)) == 0 or epoch == args.epochs - 1
                recent_ckpt_path = None
                if periodic_save:
                    _safe_torch_save(checkpoint, out_dir / "last.pth")
                    recent_ckpt_path = _save_recent_epoch_checkpoint(
                        checkpoint=checkpoint,
                        checkpoint_dir=checkpoint_dir,
                        epoch=epoch + 1,
                        keep_recent_epochs=args.keep_recent_epochs,
                    )
                if val_metrics["psnr"] > best_psnr:
                    best_psnr = val_metrics["psnr"]
                    best_epoch = epoch + 1
                    checkpoint["best_psnr"] = best_psnr
                    checkpoint["best_epoch"] = best_epoch
                    if not periodic_save:
                        _safe_torch_save(checkpoint, out_dir / "last.pth")
                    _safe_torch_save(checkpoint, out_dir / "best_by_psnr.pth")
                    log_line += " [best_by_psnr updated]"
                save_sec = float(time.time() - save_start_time)
                if recent_ckpt_path is not None:
                    log_line += f" [saved {recent_ckpt_path.name}]"

                epoch_total_sec = float(time.time() - epoch_start_time)
                elapsed_sec = float(time.time() - global_start_time)
                log_line += (
                    f" val={val_sec:.1f}s save={save_sec:.1f}s "
                    f"total={epoch_total_sec:.1f}s elapsed={elapsed_sec/60.0:.1f}m"
                )
                writer.writerow([
                    epoch + 1,
                    train_l1,
                    val_metrics["l1"],
                    val_metrics["psnr"],
                    val_metrics["ssim"],
                    val_metrics["sam"],
                    val_metrics["rmse"],
                    val_metrics["ergas"],
                    optimizer.param_groups[0]["lr"],
                    train_sec,
                    val_sec,
                    save_sec,
                    epoch_total_sec,
                    elapsed_sec,
                ])
                csv_file.flush()
            else:
                epoch_total_sec = float(time.time() - epoch_start_time)
                elapsed_sec = float(time.time() - global_start_time)
                log_line += f" total={epoch_total_sec:.1f}s elapsed={elapsed_sec/60.0:.1f}m"
            print(log_line)

    summary = {
        "best_epoch": best_epoch,
        "best_psnr": best_psnr,
        "keep_recent_epochs": int(args.keep_recent_epochs),
        "total_training_seconds": float(time.time() - global_start_time),
        "output_dir": str(out_dir.resolve()),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"[DONE] WV3 training finished. Best PSNR={best_psnr:.4f} at epoch {best_epoch}.")
    print(f"[OUT] {out_dir}")


if __name__ == "__main__":
    main()
