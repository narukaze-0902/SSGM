from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from model.sam_ase_mamba import SAMFeatureExtractor
from utils.load_wv3_data import WV3_DIVISOR, WV3_RGB_INDICES, WV3_SAM_CACHE_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline SAM cache for WV3 patch datasets.")
    parser.add_argument("--data_h5", type=str, required=True, help="WV3 h5 path")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="SAM checkpoint path")
    parser.add_argument("--output_cache_path", type=str, default=None, help="output cache path")
    parser.add_argument("--source_key", type=str, default="lms", choices=["lms", "pan"], help="which high-resolution view to feed into SAM")
    parser.add_argument("--rgb_indices", type=int, nargs=3, default=list(WV3_RGB_INDICES), help="0-based RGB band indices used when source_key=lms")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--feature_size", type=int, default=32, help="store raw SAM feature map at this spatial size; 64 keeps original size")
    parser.add_argument("--mask_dtype", type=str, default="uint8", choices=["uint8", "float16", "float32"])
    parser.add_argument("--compression", type=str, default="lzf", choices=["none", "lzf", "gzip"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing cache")
    parser.add_argument("--start_index", type=int, default=0, help="resume from sample index")
    return parser.parse_args()


def resolve_output_path(data_h5: str, output_cache_path: str | None) -> str:
    if output_cache_path:
        return output_cache_path
    base_path, _ = os.path.splitext(data_h5)
    return base_path + ".whole_sam_cache.h5"


def convert_dtype(array: np.ndarray, dtype_name: str) -> np.ndarray:
    return array.astype(np.float16 if dtype_name == "float16" else np.float32)


def convert_mask_dtype(array: np.ndarray, dtype_name: str) -> np.ndarray:
    if dtype_name == "uint8":
        return array.astype(np.uint8)
    if dtype_name == "float16":
        return array.astype(np.float16)
    return array.astype(np.float32)


def maybe_resize_features(raw_features: torch.Tensor, feature_size: int) -> torch.Tensor:
    if feature_size <= 0:
        raise ValueError("feature_size must be positive")
    if raw_features.shape[-2:] == (feature_size, feature_size):
        return raw_features
    return F.interpolate(raw_features.float(), size=(feature_size, feature_size), mode="bilinear", align_corners=True)


def resolve_h5_compression(name: str):
    if name == "none":
        return None
    return name


def prepare_sam_input(sample: np.ndarray, source_key: str, rgb_indices: list[int]) -> torch.Tensor:
    sample = np.asarray(sample, dtype=np.float32) / float(WV3_DIVISOR)
    if source_key == "pan":
        if sample.ndim != 3 or sample.shape[0] != 1:
            raise ValueError(f"Expected PAN sample shape [1,H,W], got {sample.shape}")
        tensor = torch.from_numpy(sample)
        return tensor.repeat(3, 1, 1)

    if sample.ndim != 3:
        raise ValueError(f"Expected LMS sample shape [C,H,W], got {sample.shape}")
    if max(rgb_indices) >= sample.shape[0]:
        raise ValueError(f"rgb_indices {rgb_indices} exceed channel count {sample.shape[0]}")
    tensor = torch.from_numpy(sample[rgb_indices, :, :])
    return tensor


def main() -> None:
    args = parse_args()
    output_cache_path = resolve_output_path(args.data_h5, args.output_cache_path)

    if os.path.exists(output_cache_path) and not args.overwrite:
        raise FileExistsError(f"Cache already exists: {output_cache_path}. Use --overwrite to rebuild.")

    os.makedirs(os.path.dirname(output_cache_path) or ".", exist_ok=True)

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
        total = int(src["ms"].shape[0])
        dst.attrs["cache_version"] = WV3_SAM_CACHE_VERSION
        dst.attrs["source_h5"] = args.data_h5
        dst.attrs["sam_checkpoint"] = args.sam_checkpoint
        dst.attrs["source_key"] = args.source_key
        dst.attrs["rgb_indices"] = np.asarray(args.rgb_indices, dtype=np.int32)
        dst.attrs["cache_dtype"] = args.dtype
        dst.attrs["mask_dtype"] = args.mask_dtype
        dst.attrs["feature_size"] = int(args.feature_size)
        dst.attrs["compression"] = args.compression
        dst.attrs["prompt_mode"] = "patch_local"
        dst.attrs["normalization"] = f"/{WV3_DIVISOR}"

        print(f"[WV3-SAM] Building cache for {total} samples -> {output_cache_path}")
        print(f"[WV3-SAM] compact settings: feature_size={args.feature_size}, feature_dtype={args.dtype}, mask_dtype={args.mask_dtype}, compression={args.compression}")

        compression = resolve_h5_compression(args.compression)

        with torch.no_grad():
            for index in range(args.start_index, total):
                if args.source_key == "pan":
                    source = src["pan"][index]
                else:
                    if "lms" not in src:
                        raise KeyError("LMS data not found in source h5; cannot build lms-based SAM cache.")
                    source = src["lms"][index]

                sam_input = prepare_sam_input(source, args.source_key, args.rgb_indices)
                raw_features, raw_masks = extractor.extract_raw_sam_outputs(
                    sam_input.unsqueeze(0).to(args.device).float()
                )
                if raw_features is None or raw_masks is None:
                    raise RuntimeError(f"Failed to extract SAM outputs for sample {index}")

                sample_name = f"sample_{index:06d}"
                grp = dst.create_group(sample_name)
                compact_features = maybe_resize_features(raw_features, args.feature_size)
                sam_features_np = convert_dtype(compact_features.squeeze(0).detach().cpu().numpy(), args.dtype)
                sam_masks_np = convert_mask_dtype(raw_masks.squeeze(0).detach().cpu().numpy(), args.mask_dtype)
                grp.create_dataset("sam_features", data=sam_features_np, compression=compression)
                grp.create_dataset("sam_masks", data=sam_masks_np, compression=compression)

                if (index + 1) % 100 == 0 or (index + 1) == total:
                    print(
                        f"[WV3-SAM] Processed {index + 1}/{total} | "
                        f"features={tuple(sam_features_np.shape)} masks={tuple(sam_masks_np.shape)}"
                    )

    print(f"[WV3-SAM] Done: {output_cache_path}")


if __name__ == "__main__":
    main()
