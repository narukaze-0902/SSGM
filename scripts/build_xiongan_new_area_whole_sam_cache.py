import argparse
import math
import os
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
import torch

from model.sam_ase_mamba import SAMFeatureExtractor
from utils.sam_cache_multi_region import extract_multi_region_raw_sam_outputs


CACHE_VERSION = "v2_xiongan_new_area_large_window_global_norm"
CACHE_VERSION_MULTI_REGION = "v3_xiongan_new_area_large_window_multi_region_global_norm"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Xiongan New Area large-window SAM cache and remap it to patch-level sidecar cache"
    )
    parser.add_argument("--data_h5", type=str, required=True, help="path to xiongan_new_area_train/val/test.h5")
    parser.add_argument(
        "--data_root",
        type=str,
        default="./DATASET_MIGRATION_BUNDLE",
        help="directory containing xantrain.npy and chikuseisrf.mat",
    )
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="path to SAM checkpoint")
    parser.add_argument("--output_cache_path", type=str, default=None, help="output sidecar SAM cache h5 path")
    parser.add_argument("--window_size", type=int, default=512, help="context window size around each patch")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--multi_region", action="store_true", help="extract multiple SAM masks per window instead of a single best mask")
    parser.add_argument("--max_regions", type=int, default=6, help="maximum number of remapped SAM regions kept per patch in multi-region mode")
    parser.add_argument("--point_grid_size", type=int, default=4, help="grid size used to probe multiple SAM point prompts in multi-region mode")
    parser.add_argument("--min_mask_area_ratio", type=float, default=0.01, help="drop masks whose area ratio is smaller than this threshold in multi-region mode")
    parser.add_argument("--max_mask_area_ratio", type=float, default=0.90, help="drop masks whose area ratio is larger than this threshold in multi-region mode")
    parser.add_argument("--mask_iou_dedup_thresh", type=float, default=0.85, help="IoU threshold for deduplicating masks in multi-region mode")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing cache")
    parser.add_argument("--start_index", type=int, default=0, help="resume from sample index")
    return parser.parse_args()


def resolve_output_path(data_h5, output_cache_path, multi_region=False):
    if output_cache_path:
        return output_cache_path
    base_path, _ = os.path.splitext(data_h5)
    if multi_region:
        return base_path + ".whole_sam_cache.multi_region.h5"
    return base_path + ".whole_sam_cache.h5"


def convert_dtype(array, dtype_name):
    if dtype_name == "float16":
        return array.astype(np.float16)
    return array.astype(np.float32)


def load_source_handles(data_root):
    data_root = Path(data_root)
    hyper_path = data_root / "xantrain.npy"
    srf_path = data_root / "chikuseisrf.mat"

    if not hyper_path.exists():
        raise FileNotFoundError(f"Missing Xiongan HSI file: {hyper_path}")
    if not srf_path.exists():
        raise FileNotFoundError(f"Missing Xiongan SRF file: {srf_path}")

    hyper = np.load(hyper_path, mmap_mode="r")
    if hyper.ndim != 3:
        raise ValueError(f"Expected xantrain.npy to be 3D, got shape {hyper.shape}")
    if hyper.shape[0] <= 256 and hyper.shape[1] > 256 and hyper.shape[2] > 256:
        hyper = np.transpose(hyper, (1, 2, 0))

    srf = sio.loadmat(srf_path)["R"]
    srf = np.asarray(srf, dtype=np.float32)
    if srf.shape[0] == hyper.shape[2]:
        band_to_msi = srf
    elif srf.shape[1] == hyper.shape[2]:
        band_to_msi = srf.T
    else:
        raise ValueError(
            f"SRF shape {srf.shape} is incompatible with HSI bands={hyper.shape[2]}. "
            "Expected one dimension of R to equal the HSI band count."
        )
    return hyper, band_to_msi.astype(np.float32), str(hyper_path), str(srf_path)


def load_whole_image_msi(data_root):
    hyper, band_to_msi, hyper_path, srf_path = load_source_handles(data_root)
    hyper = np.asarray(hyper, dtype=np.float32)
    hyper = np.maximum(hyper, 0.0)

    whole_msi = np.tensordot(hyper, band_to_msi, axes=([2], [0])).astype(np.float32)
    whole_msi = np.maximum(whole_msi, 0.0)

    global_max = float(whole_msi.max())
    if global_max > 1e-8:
        whole_msi = whole_msi / global_max
    whole_msi = np.clip(whole_msi, 0.0, 1.0).astype(np.float32)

    return whole_msi, hyper_path, srf_path, global_max


def compute_context_window(top, left, patch_h, patch_w, image_h, image_w, window_size):
    window_h = min(max(window_size, patch_h), image_h)
    window_w = min(max(window_size, patch_w), image_w)

    center_y = top + patch_h // 2
    center_x = left + patch_w // 2

    win_top = max(0, center_y - window_h // 2)
    win_left = max(0, center_x - window_w // 2)

    if win_top + window_h > image_h:
        win_top = image_h - window_h
    if win_left + window_w > image_w:
        win_left = image_w - window_w

    return int(win_top), int(win_left), int(window_h), int(window_w)


def crop_feature_map_by_geometry(features, rel_top, rel_left, patch_h, patch_w, window_h, window_w):
    feat_h, feat_w = features.shape[-2:]

    top_f = int(math.floor(rel_top * feat_h / window_h))
    left_f = int(math.floor(rel_left * feat_w / window_w))
    bottom_f = int(math.ceil((rel_top + patch_h) * feat_h / window_h))
    right_f = int(math.ceil((rel_left + patch_w) * feat_w / window_w))

    top_f = max(0, min(top_f, feat_h - 1))
    left_f = max(0, min(left_f, feat_w - 1))
    bottom_f = max(top_f + 1, min(bottom_f, feat_h))
    right_f = max(left_f + 1, min(right_f, feat_w))

    return features[:, :, top_f:bottom_f, left_f:right_f]


def crop_masks_by_geometry(raw_masks, rel_top, rel_left, patch_h, patch_w):
    if raw_masks.dim() == 3:
        raw_masks = raw_masks.unsqueeze(1)
    if raw_masks.dim() != 4:
        raise ValueError(f"Expected raw_masks to be [B,H,W] or [B,K,H,W], got {tuple(raw_masks.shape)}")

    patch_masks = raw_masks[:, :, rel_top : rel_top + patch_h, rel_left : rel_left + patch_w]
    if patch_masks.shape[0] == 1:
        patch_masks = patch_masks.squeeze(0)
    return patch_masks


def main():
    args = parse_args()
    output_cache_path = resolve_output_path(args.data_h5, args.output_cache_path, args.multi_region)

    if os.path.exists(output_cache_path) and not args.overwrite:
        raise FileExistsError(f"Cache already exists: {output_cache_path}. Use --overwrite to rebuild.")

    os.makedirs(os.path.dirname(output_cache_path) or ".", exist_ok=True)

    whole_msi, hyper_path, srf_path, global_msi_max = load_whole_image_msi(args.data_root)
    image_h, image_w, image_c = whole_msi.shape
    print(
        f"[WHOLE-SAM] Loaded whole-scene MSI: {whole_msi.shape}, "
        f"normalization=global_scene_max, global_msi_max={global_msi_max:.6f}"
    )

    extractor = SAMFeatureExtractor(
        sam_checkpoint_path=args.sam_checkpoint,
        feature_dim=256,
        output_dim=64,
        use_frozen_sam=True,
        use_adapter=False,
        use_learnable_prompts=False,
        use_soft_masks=False,
        device=args.device,
    )

    with h5py.File(args.data_h5, "r") as src, h5py.File(output_cache_path, "w") as dst:
        dst.attrs["cache_version"] = CACHE_VERSION_MULTI_REGION if args.multi_region else CACHE_VERSION
        dst.attrs["source_h5"] = args.data_h5
        dst.attrs["data_root"] = args.data_root
        dst.attrs["hyper_path"] = hyper_path
        dst.attrs["srf_path"] = srf_path
        dst.attrs["sam_checkpoint"] = args.sam_checkpoint
        dst.attrs["cache_dtype"] = args.dtype
        dst.attrs["window_size"] = int(args.window_size)
        dst.attrs["source_image_h"] = int(image_h)
        dst.attrs["source_image_w"] = int(image_w)
        dst.attrs["source_image_c"] = int(image_c)
        dst.attrs["msi_normalization_mode"] = "global_scene_max"
        dst.attrs["global_msi_max"] = float(global_msi_max)
        dst.attrs["prompt_mode"] = "large_window_multi_region" if args.multi_region else "large_window"
        dst.attrs["multi_region"] = bool(args.multi_region)
        if args.multi_region:
            dst.attrs["max_regions"] = int(args.max_regions)
            dst.attrs["point_grid_size"] = int(args.point_grid_size)
            dst.attrs["min_mask_area_ratio"] = float(args.min_mask_area_ratio)
            dst.attrs["max_mask_area_ratio"] = float(args.max_mask_area_ratio)
            dst.attrs["mask_iou_dedup_thresh"] = float(args.mask_iou_dedup_thresh)

        sample_names = list(src.keys())
        total = len(sample_names)
        print(f"[WHOLE-SAM] Building cache for {total} samples -> {output_cache_path}")

        with torch.no_grad():
            for index, sample_name in enumerate(sample_names):
                if index < args.start_index:
                    continue

                grp = src[sample_name]
                required = ["patch_top", "patch_left", "patch_h", "patch_w"]
                if not all(key in grp for key in required):
                    raise KeyError(
                        f"Sample '{sample_name}' is missing patch position metadata. "
                        "Please rebuild H5 with data/preprocess_xiongan_new_area.py."
                    )

                patch_top = int(grp["patch_top"][()])
                patch_left = int(grp["patch_left"][()])
                patch_h = int(grp["patch_h"][()])
                patch_w = int(grp["patch_w"][()])

                win_top, win_left, win_h, win_w = compute_context_window(
                    patch_top, patch_left, patch_h, patch_w, image_h, image_w, args.window_size
                )
                rel_top = patch_top - win_top
                rel_left = patch_left - win_left

                window_msi = whole_msi[win_top : win_top + win_h, win_left : win_left + win_w, :]

                window_tensor = torch.from_numpy(window_msi).permute(2, 0, 1).unsqueeze(0).to(args.device).float()

                if args.multi_region:
                    raw_features, raw_masks = extract_multi_region_raw_sam_outputs(
                        extractor,
                        window_tensor,
                        point_grid_size=args.point_grid_size,
                        max_regions=args.max_regions,
                        min_mask_area_ratio=args.min_mask_area_ratio,
                        max_mask_area_ratio=args.max_mask_area_ratio,
                        mask_iou_dedup_thresh=args.mask_iou_dedup_thresh,
                    )
                else:
                    raw_features, raw_masks = extractor.extract_raw_sam_outputs(window_tensor)
                if raw_features is None or raw_masks is None:
                    raise RuntimeError(f"Failed to extract SAM raw outputs for sample '{sample_name}'")

                patch_mask = crop_masks_by_geometry(
                    raw_masks,
                    rel_top,
                    rel_left,
                    patch_h,
                    patch_w,
                )

                patch_features = crop_feature_map_by_geometry(
                    raw_features, rel_top, rel_left, patch_h, patch_w, win_h, win_w
                )

                out_grp = dst.create_group(sample_name)
                patch_features_np = convert_dtype(patch_features.squeeze(0).detach().cpu().numpy(), args.dtype)
                patch_masks_np = convert_dtype(patch_mask.detach().cpu().numpy(), args.dtype)

                out_grp.create_dataset("sam_features", data=patch_features_np)
                out_grp.create_dataset("sam_masks", data=patch_masks_np)
                out_grp.attrs["hr_msi_shape"] = tuple(int(v) for v in grp["hr_msi"].shape)
                out_grp.attrs["window_top"] = int(win_top)
                out_grp.attrs["window_left"] = int(win_left)
                out_grp.attrs["window_h"] = int(win_h)
                out_grp.attrs["window_w"] = int(win_w)
                out_grp.attrs["patch_top"] = int(patch_top)
                out_grp.attrs["patch_left"] = int(patch_left)
                out_grp.attrs["patch_h"] = int(patch_h)
                out_grp.attrs["patch_w"] = int(patch_w)

                print(
                    f"[WHOLE-SAM] [{index + 1}/{total}] {sample_name}: "
                    f"patch=({patch_top},{patch_left},{patch_h},{patch_w}) "
                    f"window=({win_top},{win_left},{win_h},{win_w}) "
                    f"features={tuple(patch_features_np.shape)} masks={tuple(np.asarray(patch_masks_np).shape)}"
                )

    print(f"[WHOLE-SAM] Done. Saved cache to {output_cache_path}")


if __name__ == "__main__":
    main()
