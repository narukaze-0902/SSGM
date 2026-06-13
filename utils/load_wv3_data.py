from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils.wavelet_utils import build_haar_wavelet_coeffs

WV3_DIVISOR = 2047.0
WV3_RGB_INDICES = (4, 2, 1)
WV3_SAM_CACHE_VERSION = "v2_wv3_patch_sam_cache_compact"
WV3_SAM_CACHE_SUPPORTED_VERSIONS = {
    "v1_wv3_patch_sam_cache",
    "v2_wv3_patch_sam_cache_compact",
}


def _normalize_wv3(array: np.ndarray, divisor: float = WV3_DIVISOR) -> np.ndarray:
    return np.asarray(array, dtype=np.float32) / float(divisor)


class _LazyH5Dataset(Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        preload_to_ram: bool = False,
        preload_keys: Sequence[str] | None = None,
        selected_indices: Sequence[int] | None = None,
    ):
        self.h5_path = str(h5_path)
        self._h5: Optional[h5py.File] = None
        self._ram_arrays: Optional[Dict[str, np.ndarray]] = None
        with h5py.File(self.h5_path, "r") as handle:
            total_length = int(handle["ms"].shape[0])
            self.keys = tuple(handle.keys())
        if selected_indices is not None:
            source_indices = np.asarray(selected_indices, dtype=np.int64)
            if source_indices.ndim != 1:
                raise ValueError("selected_indices must be a 1D sequence")
            if source_indices.size == 0:
                raise ValueError("selected_indices cannot be empty")
            if source_indices.min() < 0 or source_indices.max() >= total_length:
                raise ValueError(f"selected_indices out of range for dataset length {total_length}")
            self._source_indices: Optional[np.ndarray] = source_indices
            self.length = int(source_indices.size)
        else:
            self._source_indices = None
            self.length = total_length
        if preload_to_ram:
            self._preload_h5_to_ram(preload_keys=preload_keys)

    def _preload_h5_to_ram(self, preload_keys: Sequence[str] | None = None) -> None:
        keys = tuple(preload_keys) if preload_keys is not None else self.keys
        arrays: Dict[str, np.ndarray] = {}
        start_time = time.time()
        print(f"[WV3-RAM] preload h5 -> RAM: {self.h5_path}")
        with h5py.File(self.h5_path, "r") as handle:
            for key in keys:
                if key in handle:
                    if self._source_indices is not None:
                        arrays[key] = np.asarray(handle[key][self._source_indices])
                    else:
                        arrays[key] = np.asarray(handle[key][:])
        self._ram_arrays = arrays
        elapsed = time.time() - start_time
        print(f"[WV3-RAM] ready h5: keys={list(arrays.keys())} time={elapsed:.1f}s")

    def _resolve_source_index(self, index: int) -> int:
        if self._source_indices is None:
            return int(index)
        return int(self._source_indices[index])

    def _ensure_open(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def _close_h5(self) -> None:
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
            self._h5 = None

    def _read_array(self, key: str, index: int, retries: int = 3, sleep_sec: float = 0.2) -> np.ndarray:
        if self._ram_arrays is not None and key in self._ram_arrays:
            return np.asarray(self._ram_arrays[key][index])
        source_index = self._resolve_source_index(index)
        last_error: Optional[Exception] = None
        for attempt in range(retries):
            try:
                handle = self._ensure_open()
                return np.asarray(handle[key][source_index])
            except OSError as exc:
                last_error = exc
                self._close_h5()
                if attempt + 1 < retries:
                    time.sleep(sleep_sec * (attempt + 1))
        raise OSError(f"[WV3-H5] failed reading key='{key}' index={source_index} from {self.h5_path}") from last_error

    def __len__(self) -> int:
        return self.length


class _LazyWV3SAMCacheMixin:
    def __init__(
        self,
        use_offline_sam_cache: bool = False,
        sam_cache_path: str | Path | None = None,
        sam_cache_strict: bool = False,
        preload_sam_cache_to_ram: bool = False,
    ):
        self.use_offline_sam_cache = bool(use_offline_sam_cache)
        self.sam_cache_path = str(sam_cache_path) if sam_cache_path else None
        self.sam_cache_strict = bool(sam_cache_strict)
        self.preload_sam_cache_to_ram = bool(preload_sam_cache_to_ram)
        self._sam_cache: Optional[h5py.File] = None
        self._sam_cache_ram: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
        self._sam_cache_ready = False
        self._sam_cache_ready_logged = False

    def _default_sam_cache_path(self) -> str:
        base_path, _ = str(self.h5_path).rsplit(".", 1)
        return base_path + ".whole_sam_cache.h5"

    def _ensure_sam_cache_ready(self) -> None:
        if not self.use_offline_sam_cache or self._sam_cache_ready:
            return

        if self.sam_cache_path is None:
            self.sam_cache_path = self._default_sam_cache_path()

        if not Path(self.sam_cache_path).exists():
            message = f"[WV3-SAM-CACHE] missing cache file: {self.sam_cache_path}"
            if self.sam_cache_strict:
                raise FileNotFoundError(message)
            print(f"{message}; disable offline cache for this dataset.")
            self.use_offline_sam_cache = False
            self._sam_cache_ready = True
            return

        with h5py.File(self.sam_cache_path, "r") as cache_file:
            cache_version = cache_file.attrs.get("cache_version", "")
            source_h5 = cache_file.attrs.get("source_h5", "")
            if isinstance(cache_version, bytes):
                cache_version = cache_version.decode("utf-8")
            if isinstance(source_h5, bytes):
                source_h5 = source_h5.decode("utf-8")

            if cache_version not in WV3_SAM_CACHE_SUPPORTED_VERSIONS:
                message = (
                    f"[WV3-SAM-CACHE] unsupported cache_version={cache_version}, "
                    f"expected one of {sorted(WV3_SAM_CACHE_SUPPORTED_VERSIONS)}"
                )
                if self.sam_cache_strict:
                    raise ValueError(message)
                print(f"{message}; disable offline cache for this dataset.")
                self.use_offline_sam_cache = False
                self._sam_cache_ready = True
                return

            if source_h5 and Path(source_h5).name != Path(self.h5_path).name:
                message = (
                    f"[WV3-SAM-CACHE] source mismatch: cache source={source_h5}, "
                    f"expected={self.h5_path}"
                )
                if self.sam_cache_strict:
                    raise ValueError(message)
                print(f"{message}; disable offline cache for this dataset.")
                self.use_offline_sam_cache = False
                self._sam_cache_ready = True
                return

        self._sam_cache_ready = True
        if not self._sam_cache_ready_logged:
            print(f"[WV3-SAM-CACHE] ready: {self.sam_cache_path}")
            self._sam_cache_ready_logged = True
        if self.preload_sam_cache_to_ram and self._sam_cache_ram is None:
            self._preload_sam_cache_to_ram()

    def _preload_sam_cache_to_ram(self) -> None:
        if not self.sam_cache_path:
            return
        start_time = time.time()
        print(f"[WV3-RAM] preload SAM cache -> RAM: {self.sam_cache_path}")
        cache_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        with h5py.File(self.sam_cache_path, "r") as cache_file:
            if hasattr(self, "_source_indices") and self._source_indices is not None:
                sample_names = [self._sample_group_name(int(idx)) for idx in self._source_indices]
            else:
                sample_names = list(cache_file.keys())
            for sample_name in sample_names:
                if sample_name not in cache_file:
                    continue
                grp = cache_file[sample_name]
                if "sam_features" not in grp or "sam_masks" not in grp:
                    continue
                cache_data[sample_name] = (
                    np.asarray(grp["sam_features"][:], dtype=np.float32),
                    np.asarray(grp["sam_masks"][:], dtype=np.float32),
                )
        self._sam_cache_ram = cache_data
        elapsed = time.time() - start_time
        print(f"[WV3-RAM] ready SAM cache: samples={len(cache_data)} time={elapsed:.1f}s")

    def _ensure_sam_cache_open(self) -> Optional[h5py.File]:
        self._ensure_sam_cache_ready()
        if not self.use_offline_sam_cache or not self.sam_cache_path:
            return None
        if self._sam_cache is None:
            self._sam_cache = h5py.File(self.sam_cache_path, "r")
        return self._sam_cache

    def _close_sam_cache(self) -> None:
        if self._sam_cache is not None:
            try:
                self._sam_cache.close()
            except Exception:
                pass
            self._sam_cache = None

    @staticmethod
    def _sample_group_name(index: int) -> str:
        return f"sample_{index:06d}"

    def _load_cached_sam(self, index: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        source_index = self._resolve_source_index(index) if hasattr(self, "_resolve_source_index") else int(index)
        sample_name = self._sample_group_name(source_index)
        if self._sam_cache_ram is not None:
            sample = self._sam_cache_ram.get(sample_name, None)
            if sample is None:
                if self.sam_cache_strict:
                    raise KeyError(f"[WV3-SAM-CACHE] sample not found: {sample_name}")
                return None, None
            sam_features_np, sam_masks_np = sample
            return torch.from_numpy(np.asarray(sam_features_np, dtype=np.float32)), torch.from_numpy(np.asarray(sam_masks_np, dtype=np.float32))
        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                cache_handle = self._ensure_sam_cache_open()
                if cache_handle is None:
                    return None, None

                if sample_name not in cache_handle:
                    if self.sam_cache_strict:
                        raise KeyError(f"[WV3-SAM-CACHE] sample not found: {sample_name}")
                    return None, None

                grp = cache_handle[sample_name]
                if "sam_features" not in grp or "sam_masks" not in grp:
                    if self.sam_cache_strict:
                        raise KeyError(f"[WV3-SAM-CACHE] invalid sample group: {sample_name}")
                    return None, None

                sam_features = torch.from_numpy(np.asarray(grp["sam_features"][:], dtype=np.float32))
                sam_masks = torch.from_numpy(np.asarray(grp["sam_masks"][:], dtype=np.float32))
                return sam_features, sam_masks
            except OSError as exc:
                last_error = exc
                self._close_sam_cache()
                if attempt < 2:
                    time.sleep(0.2 * (attempt + 1))
        raise OSError(f"[WV3-SAM-CACHE] failed reading sample={sample_name} from {self.sam_cache_path}") from last_error


class WV3TrainValDataset(_LazyWV3SAMCacheMixin, _LazyH5Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        divisor: float = WV3_DIVISOR,
        use_wavelet: bool = False,
        use_offline_sam_cache: bool = False,
        sam_cache_path: str | Path | None = None,
        sam_cache_strict: bool = False,
        preload_to_ram: bool = False,
        preload_sam_cache_to_ram: bool = False,
        selected_indices: Sequence[int] | None = None,
    ):
        _LazyH5Dataset.__init__(self, h5_path, preload_to_ram=preload_to_ram, selected_indices=selected_indices)
        _LazyWV3SAMCacheMixin.__init__(
            self,
            use_offline_sam_cache=use_offline_sam_cache,
            sam_cache_path=sam_cache_path,
            sam_cache_strict=sam_cache_strict,
            preload_sam_cache_to_ram=preload_sam_cache_to_ram,
        )
        self.divisor = divisor
        self.use_wavelet = bool(use_wavelet)
        self.has_gt = "gt" in self.keys
        self.has_lms = "lms" in self.keys

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        source_index = self._resolve_source_index(index)
        sample = {
            "sample_idx": torch.tensor(source_index, dtype=torch.long),
            "lr_hsi": torch.from_numpy(_normalize_wv3(self._read_array("ms", index), self.divisor)),
            "hr_msi": torch.from_numpy(_normalize_wv3(self._read_array("pan", index), self.divisor)),
        }
        if self.has_gt:
            sample["hr_hsi"] = torch.from_numpy(_normalize_wv3(self._read_array("gt", index), self.divisor))
        if self.has_lms:
            sample["lms"] = torch.from_numpy(_normalize_wv3(self._read_array("lms", index), self.divisor))
        if self.use_wavelet:
            lr_hsi_approx, lr_hsi_details = build_haar_wavelet_coeffs(sample["lr_hsi"])
            sample["lr_hsi_approx"] = lr_hsi_approx
            sample["lr_hsi_details"] = lr_hsi_details
            if "hr_hsi" in sample:
                hr_hsi_approx, hr_hsi_details = build_haar_wavelet_coeffs(sample["hr_hsi"])
                sample["hr_hsi_approx"] = hr_hsi_approx
                sample["hr_hsi_details"] = hr_hsi_details
        cached_sam_features, cached_sam_masks = self._load_cached_sam(index)
        if cached_sam_features is not None and cached_sam_masks is not None:
            sample["cached_sam_features"] = cached_sam_features
            sample["cached_sam_masks"] = cached_sam_masks
        return sample


class WV3RealDataset(_LazyWV3SAMCacheMixin, _LazyH5Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        divisor: float = WV3_DIVISOR,
        use_wavelet: bool = False,
        use_offline_sam_cache: bool = False,
        sam_cache_path: str | Path | None = None,
        sam_cache_strict: bool = False,
        preload_to_ram: bool = False,
        preload_sam_cache_to_ram: bool = False,
        selected_indices: Sequence[int] | None = None,
    ):
        _LazyH5Dataset.__init__(self, h5_path, preload_to_ram=preload_to_ram, selected_indices=selected_indices)
        _LazyWV3SAMCacheMixin.__init__(
            self,
            use_offline_sam_cache=use_offline_sam_cache,
            sam_cache_path=sam_cache_path,
            sam_cache_strict=sam_cache_strict,
            preload_sam_cache_to_ram=preload_sam_cache_to_ram,
        )
        self.divisor = divisor
        self.use_wavelet = bool(use_wavelet)
        self.has_lms = "lms" in self.keys

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        source_index = self._resolve_source_index(index)
        sample = {
            "sample_idx": torch.tensor(source_index, dtype=torch.long),
            "lr_hsi": torch.from_numpy(_normalize_wv3(self._read_array("ms", index), self.divisor)),
            "hr_msi": torch.from_numpy(_normalize_wv3(self._read_array("pan", index), self.divisor)),
        }
        if self.has_lms:
            sample["lms"] = torch.from_numpy(_normalize_wv3(self._read_array("lms", index), self.divisor))
        if self.use_wavelet:
            lr_hsi_approx, lr_hsi_details = build_haar_wavelet_coeffs(sample["lr_hsi"])
            sample["lr_hsi_approx"] = lr_hsi_approx
            sample["lr_hsi_details"] = lr_hsi_details
        cached_sam_features, cached_sam_masks = self._load_cached_sam(index)
        if cached_sam_features is not None and cached_sam_masks is not None:
            sample["cached_sam_features"] = cached_sam_features
            sample["cached_sam_masks"] = cached_sam_masks
        return sample


def forward_wv3_model(
    model: torch.nn.Module,
    hr_msi: torch.Tensor,
    lr_hsi: torch.Tensor,
    lr_hsi_approx: torch.Tensor | None = None,
    lr_hsi_details: torch.Tensor | None = None,
    cached_sam_features: torch.Tensor | None = None,
    cached_sam_masks: torch.Tensor | None = None,
) -> torch.Tensor:
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


def bicubic_upsample_ms(ms: torch.Tensor, scale: int = 4) -> torch.Tensor:
    if ms.dim() == 3:
        ms = ms.unsqueeze(0)
    up = F.interpolate(ms, scale_factor=scale, mode="bicubic", align_corners=False)
    return up.squeeze(0)


def get_wv3_rgb_indices() -> Tuple[int, int, int]:
    return WV3_RGB_INDICES


def chw_to_rgb(image: torch.Tensor | np.ndarray, rgb_indices: Sequence[int] = WV3_RGB_INDICES) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().float().numpy()
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image, got shape {tuple(image.shape)}")
    rgb = np.stack([image[idx] for idx in rgb_indices], axis=-1)
    return np.asarray(rgb, dtype=np.float32)


def compute_display_range(reference_rgb: np.ndarray, lower_percentile: float = 1.0, upper_percentile: float = 99.0) -> Tuple[np.ndarray, np.ndarray]:
    flat = reference_rgb.reshape(-1, reference_rgb.shape[-1])
    low = np.percentile(flat, lower_percentile, axis=0).astype(np.float32)
    high = np.percentile(flat, upper_percentile, axis=0).astype(np.float32)
    high = np.maximum(high, low + 1e-6)
    return low, high


def apply_display_range(rgb: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    stretched = (rgb - low.reshape(1, 1, -1)) / (high - low).reshape(1, 1, -1)
    return np.clip(stretched, 0.0, 1.0)


def tensor_to_display_rgb(
    image: torch.Tensor | np.ndarray,
    rgb_indices: Sequence[int] = WV3_RGB_INDICES,
    low: Optional[np.ndarray] = None,
    high: Optional[np.ndarray] = None,
) -> np.ndarray:
    rgb = chw_to_rgb(image, rgb_indices=rgb_indices)
    if low is None or high is None:
        low, high = compute_display_range(rgb)
    return apply_display_range(rgb, low, high)


def select_informative_crop(
    pan: torch.Tensor | np.ndarray,
    crop_size: int = 128,
    stride: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    if isinstance(pan, torch.Tensor):
        pan = pan.detach().cpu().float().numpy()
    if pan.ndim == 3:
        pan = pan[0]
    pan = np.asarray(pan, dtype=np.float32)
    h, w = pan.shape
    crop_size = int(min(crop_size, h, w))
    if crop_size <= 0:
        return 0, 0, h, w
    if h == crop_size and w == crop_size:
        return 0, 0, h, w

    gy, gx = np.gradient(pan)
    energy = gx * gx + gy * gy
    stride = stride or max(16, crop_size // 4)

    best_score = -math.inf
    best_box = (0, 0, crop_size, crop_size)
    for top in range(0, max(h - crop_size, 0) + 1, stride):
        for left in range(0, max(w - crop_size, 0) + 1, stride):
            crop = energy[top:top + crop_size, left:left + crop_size]
            score = float(crop.mean())
            if score > best_score:
                best_score = score
                best_box = (left, top, crop_size, crop_size)

    left, top, width, height = best_box
    return left, top, width, height


def save_wv3_visual_panel(
    pan: torch.Tensor | np.ndarray,
    lms: torch.Tensor | np.ndarray,
    fused: torch.Tensor | np.ndarray,
    save_path: str | Path,
    rgb_indices: Sequence[int] = WV3_RGB_INDICES,
    crop_box: Optional[Tuple[int, int, int, int]] = None,
    title: Optional[str] = None,
) -> Tuple[int, int, int, int]:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(pan, torch.Tensor):
        pan = pan.detach().cpu().float().numpy()
    if pan.ndim == 3:
        pan = pan[0]
    pan = np.asarray(pan, dtype=np.float32)

    lms_rgb_raw = chw_to_rgb(lms, rgb_indices=rgb_indices)
    fused_rgb_raw = chw_to_rgb(fused, rgb_indices=rgb_indices)
    low, high = compute_display_range(lms_rgb_raw)
    lms_rgb = apply_display_range(lms_rgb_raw, low, high)
    fused_rgb = apply_display_range(fused_rgb_raw, low, high)

    if crop_box is None:
        crop_box = select_informative_crop(pan, crop_size=min(128, pan.shape[0], pan.shape[1]))
    left, top, width, height = crop_box

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    if title:
        fig.suptitle(title, fontsize=12)

    axes[0, 0].imshow(pan, cmap="gray")
    axes[0, 0].set_title("PAN")
    axes[0, 1].imshow(lms_rgb)
    axes[0, 1].set_title("LMS")
    axes[0, 2].imshow(fused_rgb)
    axes[0, 2].set_title("Fused")

    for ax in axes[0]:
        rect = plt.Rectangle((left, top), width, height, fill=False, edgecolor="red", linewidth=1.5)
        ax.add_patch(rect)
        ax.axis("off")

    pan_crop = pan[top:top + height, left:left + width]
    lms_crop = lms_rgb[top:top + height, left:left + width]
    fused_crop = fused_rgb[top:top + height, left:left + width]
    axes[1, 0].imshow(pan_crop, cmap="gray")
    axes[1, 0].set_title("PAN Zoom")
    axes[1, 1].imshow(lms_crop)
    axes[1, 1].set_title("LMS Zoom")
    axes[1, 2].imshow(fused_crop)
    axes[1, 2].set_title("Fused Zoom")
    for ax in axes[1]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return crop_box
