import torch
import torch.nn.functional as F
import numpy as np
import os
import h5py
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
from utils.normalization_utils import GlobalNormalizer
from utils.wavelet_utils import build_haar_wavelet_coeffs


_CV2 = None


def _get_cv2():
    global _CV2
    if _CV2 is None:
        import cv2 as _cv2
        try:
            _cv2.setNumThreads(0)
        except Exception:
            pass
        _CV2 = _cv2
    return _CV2


SUPPORTED_SAM_CACHE_SPECS = {
    ('v1_raw_fixed_grid', 'fixed_grid'),
    ('v2_chikusei_large_window', 'large_window'),
    ('v2_pavia_university_large_window', 'large_window'),
    ('v2_xiongan_new_area_large_window', 'large_window'),
    ('v2_xiongan_new_area_large_window_global_norm', 'large_window'),
    ('v3_chikusei_large_window_multi_region', 'large_window_multi_region'),
    ('v3_pavia_university_large_window_multi_region', 'large_window_multi_region'),
    ('v3_xiongan_new_area_large_window_multi_region', 'large_window_multi_region'),
    ('v3_xiongan_new_area_large_window_multi_region_global_norm', 'large_window_multi_region'),
    ('v4_pavia_university_split_scene_region', 'split_scene_region'),
}


class HSIMSI_Dataset(Dataset):
    def __init__(self, root_dir, transform=None, h5_dir=None, is_val=None, dataset_type='cave',
                 use_wavelet=False, use_offline_sam_cache=False, sam_cache_path=None,
                 sam_cache_strict=False, prefer_multi_region_sam_cache=False,
                 prefer_scene_sam_cache=False):


        self.root_dir = root_dir
        self.transform = transform
        self.dataset_type = dataset_type.lower()

        self.is_val = is_val if is_val is not None else 'val' in root_dir.lower()
        self.use_wavelet = use_wavelet
        self.use_offline_sam_cache = use_offline_sam_cache
        self.sam_cache_path = sam_cache_path
        self.sam_cache_strict = sam_cache_strict
        self.prefer_multi_region_sam_cache = prefer_multi_region_sam_cache
        self.prefer_scene_sam_cache = prefer_scene_sam_cache
        self._cache_warning_count = 0
        self._cache_warning_limit = 5


        if h5_dir is None:
            if self.dataset_type == 'cave':

                self.h5_dir = './data/cave'
            else:

                self.h5_dir = root_dir
        else:
            self.h5_dir = h5_dir
        os.makedirs(self.h5_dir, exist_ok=True)


        if self.dataset_type == 'cave':
            cv2 = _get_cv2()
            self.preprocessed_path = os.path.join(
                self.h5_dir,
                'cave_val.h5' if self.is_val else 'cave_train.h5'
            )
        elif self.dataset_type == 'pavia_university':
            self.preprocessed_path = os.path.join(
                self.h5_dir,
                'pavia_university_val.h5' if self.is_val else 'pavia_university_train.h5'
            )
        elif self.dataset_type == 'xiongan_new_area':
            self.preprocessed_path = os.path.join(
                self.h5_dir,
                'xiongan_new_area_val.h5' if self.is_val else 'xiongan_new_area_train.h5'
            )
        else:
            self.preprocessed_path = os.path.join(
                self.h5_dir,
                'chikusei_val.h5' if self.is_val else 'chikusei_train.h5'
            )

        if self.sam_cache_path is None and self.use_offline_sam_cache:
            base_path, _ = os.path.splitext(self.preprocessed_path)
            scene_cache_path = base_path + '.whole_sam_cache.scene_region.h5'
            multi_region_cache_path = base_path + '.whole_sam_cache.multi_region.h5'
            whole_image_cache_path = base_path + '.whole_sam_cache.h5'
            legacy_cache_path = base_path + '.sam_cache.h5'
            if self.prefer_scene_sam_cache and os.path.exists(scene_cache_path):
                self.sam_cache_path = scene_cache_path
            elif self.prefer_multi_region_sam_cache and os.path.exists(multi_region_cache_path):
                self.sam_cache_path = multi_region_cache_path
            else:
                self.sam_cache_path = (
                    whole_image_cache_path
                    if os.path.exists(whole_image_cache_path)
                    else legacy_cache_path
                )


        if os.path.exists(self.preprocessed_path):
            print(f'[INFO] 预处理文件已存在，从 {self.preprocessed_path} 加载数据')
            self._load_preprocessed()
            print(f'[SUCCESS] 成功加载 {len(self.samples)} 个样本')
        else:
            if self.dataset_type == 'cave':
                print(f'[INFO] 正在生成新的预处理文件到 {self.preprocessed_path}')
                self._preprocess_and_save()
                print('[SUCCESS] 预处理完成并成功保存H5文件')
            elif self.dataset_type == 'pavia_university':
                raise FileNotFoundError(
                    "Pavia University预处理文件不存在。请先运行 "
                    "data/preprocess_pavia_university.py 生成 pavia_university_train/val/test.h5。"
                )
            elif self.dataset_type == 'xiongan_new_area':
                raise FileNotFoundError(
                    "Xiongan New Area预处理文件不存在。请先运行 "
                    "data/preprocess_xiongan_new_area.py 生成 xiongan_new_area_train/val/test.h5。"
                )
            else:

                raise FileNotFoundError(f"CHIKUSEI数据集预处理文件不存在。请先运行preprocess_chikusei.py生成数据集。")

        self._initialize_cache_state()

    def _preprocess_and_save(self):

        self.samples = [d for d in os.listdir(self.root_dir)
                    if os.path.isdir(os.path.join(self.root_dir, d))]
        print(f'[PROCESS] 开始预处理 {len(self.samples)} 个样本')


        for sample in self.samples:
            sample_path = os.path.join(self.root_dir, sample)
            for i in range(1, 32):
                band_path = os.path.join(sample_path, f"{sample}_{i:02d}.png")
                if not os.path.exists(band_path):
                    raise FileNotFoundError(f"Missing band file: {band_path}")
            rgb_path = os.path.join(sample_path, f"{sample}_RGB.bmp")
            if not os.path.exists(rgb_path):
                raise FileNotFoundError(f"Missing RGB file: {rgb_path}")


        with h5py.File(self.preprocessed_path, 'w') as f:
            sample_count = 0
            for sample in self.samples:
                sample_path = os.path.join(self.root_dir, sample)
                hr_hsi, hr_msi, lr_hsi = self._process_sample(sample_path)


                if self.dataset_type == 'cave':

                    h_patches, m_patches, l_patches = self._patchify(hr_hsi, hr_msi, lr_hsi)


                    for i in range(h_patches.shape[0]):
                        grp_name = f"{sample}_patch_{i:04d}"
                        grp = f.create_group(grp_name)
                        grp.create_dataset('hr_hsi', data=h_patches[i].numpy())
                        grp.create_dataset('hr_msi', data=m_patches[i].numpy())
                        grp.create_dataset('lr_hsi', data=l_patches[i].numpy())
                        sample_count += 1
                else:

                    grp_name = f"{sample}"
                    grp = f.create_group(grp_name)
                    grp.create_dataset('hr_hsi', data=hr_hsi.numpy())
                    grp.create_dataset('hr_msi', data=hr_msi.numpy())
                    grp.create_dataset('lr_hsi', data=lr_hsi.numpy())
                    sample_count += 1

        print(f'[SUCCESS] 所有样本预处理完成，已保存到 {self.preprocessed_path}')
        if self.dataset_type == 'cave':
            print(f'[INFO] 总共生成了 {sample_count} 个64x64的小块')
        else:
            print(f'[INFO] 总共生成了 {sample_count} 个样本')
        self._load_preprocessed()

    def _load_preprocessed(self):
        with h5py.File(self.preprocessed_path, 'r') as f:

            self.samples = list(f.keys())

    def _initialize_cache_state(self):
        if not self.use_offline_sam_cache:
            return

        if not self.sam_cache_path:
            if self.sam_cache_strict:
                raise ValueError("[SAM-CACHE] sam_cache_path is required when offline SAM cache is enabled")
            self.use_offline_sam_cache = False
            return

        if not os.path.exists(self.sam_cache_path):
            message = f"[SAM-CACHE] Cache file not found: {self.sam_cache_path}. Falling back to online SAM."
            if self.sam_cache_strict:
                raise FileNotFoundError(message)
            self._warn_cache_issue(message)
            self.use_offline_sam_cache = False
            return

        with h5py.File(self.sam_cache_path, 'r') as cache_file:
            cache_version = cache_file.attrs.get('cache_version', '')
            prompt_mode = cache_file.attrs.get('prompt_mode', '')
            source_h5 = cache_file.attrs.get('source_h5', '')
            if isinstance(cache_version, bytes):
                cache_version = cache_version.decode('utf-8')
            if isinstance(prompt_mode, bytes):
                prompt_mode = prompt_mode.decode('utf-8')
            if isinstance(source_h5, bytes):
                source_h5 = source_h5.decode('utf-8')
            if (cache_version, prompt_mode) not in SUPPORTED_SAM_CACHE_SPECS:
                supported_desc = ", ".join(
                    [f"(version={version}, prompt_mode={mode})" for version, mode in sorted(SUPPORTED_SAM_CACHE_SPECS)]
                )
                message = (
                    f"[SAM-CACHE] Cache metadata mismatch: version={cache_version}, "
                    f"prompt_mode={prompt_mode}. Supported specs: {supported_desc}. "
                    f"Falling back to online SAM."
                )
                if self.sam_cache_strict:
                    raise ValueError(message)
                self._warn_cache_issue(message)
                self.use_offline_sam_cache = False
                return

            if source_h5 and os.path.basename(source_h5) != os.path.basename(self.preprocessed_path):
                message = (
                    f"[SAM-CACHE] Cache source mismatch: source_h5={source_h5}, "
                    f"expected={self.preprocessed_path}. Falling back to online SAM."
                )
                if self.sam_cache_strict:
                    raise ValueError(message)
                self._warn_cache_issue(message)
                self.use_offline_sam_cache = False
                return

            cache_keys = set(cache_file.keys())
            missing_samples = [sample for sample in self.samples if sample not in cache_keys]
            if missing_samples:
                message = (
                    f"[SAM-CACHE] Cache is incomplete ({len(missing_samples)} missing samples). "
                    f"Example: {missing_samples[0]}. Falling back to online SAM."
                )
                if self.sam_cache_strict:
                    raise KeyError(message)
                self._warn_cache_issue(message)
                self.use_offline_sam_cache = False
                return

        print(f"[SAM-CACHE] Cache ready: {self.sam_cache_path}")

    def _warn_cache_issue(self, message):
        if self._cache_warning_count < self._cache_warning_limit:
            print(message)
            self._cache_warning_count += 1

    def _load_cached_sam(self, sample_name):
        if not self.use_offline_sam_cache or not self.sam_cache_path:
            return None, None

        if not os.path.exists(self.sam_cache_path):
            message = f"[SAM-CACHE] Cache file not found: {self.sam_cache_path}"
            if self.sam_cache_strict:
                raise FileNotFoundError(message)
            self._warn_cache_issue(message)
            return None, None

        with h5py.File(self.sam_cache_path, 'r') as cache_file:
            if sample_name not in cache_file:
                message = f"[SAM-CACHE] Sample '{sample_name}' not found in cache: {self.sam_cache_path}"
                if self.sam_cache_strict:
                    raise KeyError(message)
                self._warn_cache_issue(message)
                return None, None

            grp = cache_file[sample_name]
            if 'sam_features' not in grp or 'sam_masks' not in grp:
                message = f"[SAM-CACHE] Sample '{sample_name}' is missing sam_features or sam_masks"
                if self.sam_cache_strict:
                    raise KeyError(message)
                self._warn_cache_issue(message)
                return None, None

            cached_sam_features = torch.from_numpy(grp['sam_features'][:])
            cached_sam_masks = torch.from_numpy(grp['sam_masks'][:])
            if cached_sam_features.dim() == 4 and cached_sam_features.shape[0] == 1:
                cached_sam_features = cached_sam_features.squeeze(0)
            if cached_sam_masks.dim() == 3 and cached_sam_masks.shape[0] == 1:
                cached_sam_masks = cached_sam_masks.squeeze(0)


            if cached_sam_features.dim() == 3:
                patch_h = None
                patch_w = None
                if 'patch_h' in grp and 'patch_w' in grp:
                    patch_h = int(grp['patch_h'][()])
                    patch_w = int(grp['patch_w'][()])
                elif 'hr_msi_shape' in grp.attrs:
                    hr_shape = tuple(int(v) for v in grp.attrs['hr_msi_shape'])
                    if len(hr_shape) >= 3:
                        patch_h = hr_shape[-2]
                        patch_w = hr_shape[-1]

                if patch_h is not None and patch_w is not None:
                    target_feat_h = max(1, int(round(patch_h / 8.0)))
                    target_feat_w = max(1, int(round(patch_w / 8.0)))
                    if cached_sam_features.shape[-2:] != (target_feat_h, target_feat_w):
                        cached_sam_features = F.interpolate(
                            cached_sam_features.unsqueeze(0).float(),
                            size=(target_feat_h, target_feat_w),
                            mode='bilinear',
                            align_corners=False,
                        ).squeeze(0).to(dtype=cached_sam_features.dtype)
            return cached_sam_features, cached_sam_masks

    def _patchify(self, hr_hsi, hr_msi, lr_hsi, patch_size=64):

        if not hasattr(self, 'normalizer_initialized'):
            self._setup_global_normalization()
            self.normalizer_initialized = True


        _, h_hsi, w_hsi = hr_hsi.shape
        _, h_msi, w_msi = hr_msi.shape
        _, h_lr, w_lr = lr_hsi.shape


        assert h_hsi == h_msi and w_hsi == w_msi, "HR HSI和HR MSI尺寸不匹配"
        assert h_hsi % patch_size == 0 and w_hsi % patch_size == 0, "图像尺寸不能被patch_size整除"
        assert h_lr == h_hsi // 4 and w_lr == w_hsi // 4, "LR HSI尺寸不是HR HSI的1/4"


        n_h = h_hsi // patch_size
        n_w = w_hsi // patch_size
        n_patches = n_h * n_w


        c_hsi = hr_hsi.shape[0]
        c_msi = hr_msi.shape[0]
        h_patches = torch.zeros(n_patches, c_hsi, patch_size, patch_size)
        m_patches = torch.zeros(n_patches, c_msi, patch_size, patch_size)
        l_patches = torch.zeros(n_patches, c_hsi, patch_size//4, patch_size//4)


        for i in range(n_h):
            for j in range(n_w):
                idx = i * n_w + j

                h_start = i * patch_size
                h_end = h_start + patch_size
                w_start = j * patch_size
                w_end = w_start + patch_size


                h_patches[idx] = hr_hsi[:, h_start:h_end, w_start:w_end]
                m_patches[idx] = hr_msi[:, h_start:h_end, w_start:w_end]


                lr_h_start = h_start // 4
                lr_h_end = lr_h_start + patch_size // 4
                lr_w_start = w_start // 4
                lr_w_end = lr_w_start + patch_size // 4
                l_patches[idx] = lr_hsi[:, lr_h_start:lr_h_end, lr_w_start:lr_w_end]


        if not self.is_val and self.transform is not None:

            augmented_h_patches = torch.zeros_like(h_patches)
            augmented_m_patches = torch.zeros_like(m_patches)
            augmented_l_patches = torch.zeros_like(l_patches)


            if hasattr(self, 'needs_stats_calculation') and self.needs_stats_calculation:
                if not hasattr(self, 'stats_data'):
                    self.stats_data = {
                        'hr_hsi': [],
                        'hr_msi': [],
                        'lr_hsi': []
                    }

            for i in range(h_patches.shape[0]):

                sample = {
                    'hr_hsi': h_patches[i],
                    'hr_msi': m_patches[i],
                    'lr_hsi': l_patches[i]
                }


                augmented_sample = self.transform(sample)


                if hasattr(self, 'needs_stats_calculation') and self.needs_stats_calculation:
                    self.stats_data['hr_hsi'].append(augmented_sample['hr_hsi'].numpy())
                    self.stats_data['hr_msi'].append(augmented_sample['hr_msi'].numpy())
                    self.stats_data['lr_hsi'].append(augmented_sample['lr_hsi'].numpy())


                if hasattr(self, 'normalizer') and self.normalizer is not None:

                    augmented_h_patches[i] = self.normalizer.transform(
                        augmented_sample['hr_hsi'].numpy(), f"{self.dataset_type}_hr_hsi"
                    )
                    augmented_m_patches[i] = self.normalizer.transform(
                        augmented_sample['hr_msi'].numpy(), f"{self.dataset_type}_hr_msi"
                    )
                    augmented_l_patches[i] = self.normalizer.transform(
                        augmented_sample['lr_hsi'].numpy(), f"{self.dataset_type}_lr_hsi"
                    )
                else:

                    augmented_h_patches[i] = augmented_sample['hr_hsi'] / 255.0
                    augmented_m_patches[i] = augmented_sample['hr_msi'] / 255.0
                    augmented_l_patches[i] = augmented_sample['lr_hsi'] / 255.0


            if hasattr(self, 'needs_stats_calculation') and self.needs_stats_calculation:
                if hasattr(self, 'stats_data') and len(self.stats_data['hr_hsi']) > 0:
                    print("[INFO] 计算全局标准化参数...")


                    all_h = np.concatenate(self.stats_data['hr_hsi'], axis=0)
                    all_m = np.concatenate(self.stats_data['hr_msi'], axis=0)
                    all_l = np.concatenate(self.stats_data['lr_hsi'], axis=0)


                    self.normalizer.fit(all_h, f"{self.dataset_type}_hr_hsi")
                    self.normalizer.fit(all_m, f"{self.dataset_type}_hr_msi")
                    self.normalizer.fit(all_l, f"{self.dataset_type}_lr_hsi")


                    stats_path = os.path.join(self.h5_dir, f'{self.dataset_type}_normalization_stats.npz')
                    self.normalizer.save_stats(stats_path)
                    print(f"[INFO] 全局标准化参数已保存: {stats_path}")


                    del self.stats_data
                    self.needs_stats_calculation = False

            return augmented_h_patches, augmented_m_patches, augmented_l_patches
        else:

            if hasattr(self, 'normalizer') and self.normalizer is not None:

                normalized_h_patches = self.normalizer.transform(
                    h_patches.numpy(), f"{self.dataset_type}_hr_hsi"
                )
                normalized_m_patches = self.normalizer.transform(
                    m_patches.numpy(), f"{self.dataset_type}_hr_msi"
                )
                normalized_l_patches = self.normalizer.transform(
                    l_patches.numpy(), f"{self.dataset_type}_lr_hsi"
                )
                return torch.from_numpy(normalized_h_patches), torch.from_numpy(normalized_m_patches), torch.from_numpy(normalized_l_patches)
            else:

                return h_patches / 255.0, m_patches / 255.0, l_patches / 255.0


    def _setup_global_normalization(self):
        if not hasattr(self, 'normalizer') or self.normalizer is None:
            self.normalizer = GlobalNormalizer()


            stats_path = os.path.join(self.h5_dir, f'{self.dataset_type}_normalization_stats.npz')

            if os.path.exists(stats_path):

                self.normalizer.load_stats(stats_path)
                print(f"[INFO] 已加载全局标准化参数: {stats_path}")
            else:

                if self.is_val:

                    train_path = self.preprocessed_path.replace('val.h5', 'train.h5')
                    if os.path.exists(train_path):

                        print("[INFO] 从训练集计算全局标准化参数...")
                        with h5py.File(train_path, 'r') as f:
                            train_samples = list(f.keys())
                            if train_samples:

                                all_h_data = []
                                all_m_data = []
                                all_l_data = []

                                for sample_name in train_samples[:10]:
                                    sample = f[sample_name]
                                    all_h_data.append(sample['hr_hsi'][:])
                                    all_m_data.append(sample['hr_msi'][:])
                                    all_l_data.append(sample['lr_hsi'][:])


                                all_h = np.concatenate([d for d in all_h_data], axis=0)
                                all_m = np.concatenate([d for d in all_m_data], axis=0)
                                all_l = np.concatenate([d for d in all_l_data], axis=0)


                                self.normalizer.fit(all_h, f"{self.dataset_type}_hr_hsi")
                                self.normalizer.fit(all_m, f"{self.dataset_type}_hr_msi")
                                self.normalizer.fit(all_l, f"{self.dataset_type}_lr_hsi")


                                self.normalizer.save_stats(stats_path)
                                print(f"[INFO] 全局标准化参数已保存: {stats_path}")
                    else:
                        print("[WARNING] 未找到训练集文件，将使用简单的255.0归一化")
                        return False
                else:

                    print("[INFO] 训练集将计算并保存全局标准化参数...")

                    self.needs_stats_calculation = True
                    return True

    def _normalize_with_global_stats(self, data, data_type='hr_hsi'):
        if not hasattr(self, 'normalizer') or self.normalizer is None:

            return data / 255.0


        return data / 255.0

    def _process_sample(self, sample_path):
        folder_name = os.path.basename(sample_path)

        if self.dataset_type == 'cave':


            hyperspectral_bands = []
            for i in range(1, 32):
                band_path = os.path.join(sample_path, f"{folder_name}_{i:02d}.png")
                band_image = cv2.imread(band_path, cv2.IMREAD_GRAYSCALE)
                hyperspectral_bands.append(band_image)
            hr_hsi = np.stack(hyperspectral_bands, axis=0)


            rgb_path = os.path.join(sample_path, f"{folder_name}_RGB.bmp")
            hr_msi = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            hr_msi = cv2.cvtColor(hr_msi, cv2.COLOR_BGR2RGB)
            hr_msi = np.transpose(hr_msi, (2,0,1))


            blurred_hsi = np.zeros_like(hr_hsi)
            for i in range(hr_hsi.shape[0]):
                blurred_hsi[i] = gaussian_filter(hr_hsi[i], sigma=1.5)
            lr_hsi = blurred_hsi[:, ::4, ::4]
        else:


            raise NotImplementedError("Chikusei数据集应该使用专门的预处理脚本preprocess_chikusei.py")


        hr_hsi = torch.from_numpy(hr_hsi.astype(np.float32))
        hr_msi = torch.from_numpy(hr_msi.astype(np.float32))
        lr_hsi = torch.from_numpy(lr_hsi.astype(np.float32))

        return hr_hsi, hr_msi, lr_hsi


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        with h5py.File(self.preprocessed_path, 'r') as f:

            sample = self.samples[idx]
            grp = f[sample]


            lr_hsi = torch.from_numpy(grp['lr_hsi'][:])
            hr_msi = torch.from_numpy(grp['hr_msi'][:])
            hr_hsi = torch.from_numpy(grp['hr_hsi'][:])
            cached_sam_features, cached_sam_masks = self._load_cached_sam(sample)


            if self.use_wavelet:
                hr_hsi_approx, hr_hsi_details = build_haar_wavelet_coeffs(hr_hsi)
                lr_hsi_approx, lr_hsi_details = build_haar_wavelet_coeffs(lr_hsi)

                result = {
                    'lr_hsi': lr_hsi,
                    'lr_hsi_approx': lr_hsi_approx,
                    'lr_hsi_details': lr_hsi_details,
                    'hr_msi': hr_msi,
                    'hr_hsi': hr_hsi,
                    'hr_hsi_approx': hr_hsi_approx,
                    'hr_hsi_details': hr_hsi_details,
                    'sample_id': sample
                }
                if cached_sam_features is not None and cached_sam_masks is not None:
                    result['cached_sam_features'] = cached_sam_features
                    result['cached_sam_masks'] = cached_sam_masks
                if 'patch_top' in grp and 'patch_left' in grp:
                    result['patch_top'] = int(grp['patch_top'][()])
                    result['patch_left'] = int(grp['patch_left'][()])
                if 'patch_h' in grp and 'patch_w' in grp:
                    result['patch_h'] = int(grp['patch_h'][()])
                    result['patch_w'] = int(grp['patch_w'][()])
                if 'downsample_ratio' in grp:
                    result['downsample_ratio'] = int(grp['downsample_ratio'][()])
                if 'source_image_h' in f.attrs and 'source_image_w' in f.attrs:
                    result['source_image_h'] = int(f.attrs['source_image_h'])
                    result['source_image_w'] = int(f.attrs['source_image_w'])
                return result

            if self.use_wavelet and False:
                def apply_2d_haar_wavelet(hsi_data):
                    hsi_np = hsi_data.numpy()

                    approx_list = []
                    detail_lh = []
                    detail_hl = []
                    detail_hh = []

                    for channel_idx in range(hsi_np.shape[0]):
                        approx, (lh, hl, hh) = pywt.dwt2(hsi_np[channel_idx], wavelet='haar', mode='symmetric')
                        approx_list.append(torch.from_numpy(approx))
                        detail_lh.append(torch.from_numpy(lh))
                        detail_hl.append(torch.from_numpy(hl))
                        detail_hh.append(torch.from_numpy(hh))

                    approx = torch.stack(approx_list, dim=0)
                    details = torch.stack([
                        torch.stack(detail_lh, dim=0),
                        torch.stack(detail_hl, dim=0),
                        torch.stack(detail_hh, dim=0),
                    ], dim=0)
                    return approx, details

                def apply_3d_wavelet(hsi_data):

                    hsi_np = hsi_data.numpy()

                    wavelet = pywt.Wavelet('sym3')

                    max_level = pywt.dwtn_max_level(hsi_np.shape, wavelet)
                    if max_level < 1:
                        approx = torch.from_numpy(hsi_np.copy())
                        details = torch.zeros(
                            (3,) + tuple(hsi_np.shape),
                            dtype=approx.dtype,
                        )
                        return approx, details
                    level = min(2, max_level)


                    coeffs = pywt.wavedecn(hsi_np, wavelet=wavelet, level=level, mode='symmetric')


                    approx = coeffs[0]
                    details = coeffs[1]


                    approx = torch.from_numpy(approx)


                    detail_list = []
                    for key in sorted(details.keys()):
                        detail_list.append(torch.from_numpy(details[key]))
                    details = torch.stack(detail_list, dim=0)

                    return approx, details


                hr_hsi_approx, hr_hsi_details = apply_2d_haar_wavelet(hr_hsi)
                lr_hsi_approx, lr_hsi_details = apply_2d_haar_wavelet(lr_hsi)

                result = {
                    'lr_hsi': lr_hsi,
                    'lr_hsi_approx': lr_hsi_approx,
                    'lr_hsi_details': lr_hsi_details,
                    'hr_msi': hr_msi,
                    'hr_hsi': hr_hsi,
                    'hr_hsi_approx': hr_hsi_approx,
                    'hr_hsi_details': hr_hsi_details,
                    'sample_id': sample
                }
                if cached_sam_features is not None and cached_sam_masks is not None:
                    result['cached_sam_features'] = cached_sam_features
                    result['cached_sam_masks'] = cached_sam_masks
                if 'patch_top' in grp and 'patch_left' in grp:
                    result['patch_top'] = int(grp['patch_top'][()])
                    result['patch_left'] = int(grp['patch_left'][()])
                if 'patch_h' in grp and 'patch_w' in grp:
                    result['patch_h'] = int(grp['patch_h'][()])
                    result['patch_w'] = int(grp['patch_w'][()])
                if 'downsample_ratio' in grp:
                    result['downsample_ratio'] = int(grp['downsample_ratio'][()])
                if 'source_image_h' in f.attrs and 'source_image_w' in f.attrs:
                    result['source_image_h'] = int(f.attrs['source_image_h'])
                    result['source_image_w'] = int(f.attrs['source_image_w'])
                return result
            else:

                result = {
                    'lr_hsi': lr_hsi,
                    'hr_msi': hr_msi,
                    'hr_hsi': hr_hsi,
                    'sample_id': sample
                }
                if cached_sam_features is not None and cached_sam_masks is not None:
                    result['cached_sam_features'] = cached_sam_features
                    result['cached_sam_masks'] = cached_sam_masks
                if 'patch_top' in grp and 'patch_left' in grp:
                    result['patch_top'] = int(grp['patch_top'][()])
                    result['patch_left'] = int(grp['patch_left'][()])
                if 'patch_h' in grp and 'patch_w' in grp:
                    result['patch_h'] = int(grp['patch_h'][()])
                    result['patch_w'] = int(grp['patch_w'][()])
                if 'downsample_ratio' in grp:
                    result['downsample_ratio'] = int(grp['downsample_ratio'][()])
                if 'source_image_h' in f.attrs and 'source_image_w' in f.attrs:
                    result['source_image_h'] = int(f.attrs['source_image_h'])
                    result['source_image_w'] = int(f.attrs['source_image_w'])
                return result
