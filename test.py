import os
import csv
import json
import sys
import time
import platform
import torch
import argparse
import numpy as np
import scipy.io as sio
import h5py
import cv2
from model.u2net import U2Net as Net
from utils.load_hsimsi_data import HSIMSI_Dataset
from utils.tools import SSIM, SAM, RMSE, PSNR, ERGAS, compute_ssim, compute_sam, compute_rmse, compute_psnr
from utils.tools import save_matv73
from utils.wavelet_utils import build_haar_wavelet_coeffs, should_use_wavelet_priors
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from pathlib import Path


try:
    from model.enhanced_fusion_mamba import EnhancedFusionMamba
    ENHANCED_FUSION_AVAILABLE = True
except ImportError:
    ENHANCED_FUSION_AVAILABLE = False
    print("Warning: Enhanced FusionMamba not available, using original FusionMamba")
def resolve_sam_cache_path_for_h5(h5_path, explicit_cache_path=None, prefer_scene_region=False, prefer_multi_region=False):
    if explicit_cache_path:
        return explicit_cache_path
    base_path, _ = os.path.splitext(h5_path)
    scene_region_cache_path = base_path + '.whole_sam_cache.scene_region.h5'
    multi_region_cache_path = base_path + '.whole_sam_cache.multi_region.h5'
    whole_image_cache_path = base_path + '.whole_sam_cache.h5'
    legacy_cache_path = base_path + '.sam_cache.h5'
    if prefer_scene_region and os.path.exists(scene_region_cache_path):
        return scene_region_cache_path
    if prefer_multi_region and os.path.exists(multi_region_cache_path):
        return multi_region_cache_path
    if os.path.exists(whole_image_cache_path):
        return whole_image_cache_path
    return legacy_cache_path


def load_test_sam_cache(cache_path, sample_names, device):
    if not cache_path or not os.path.exists(cache_path):
        return None, None

    cached_features = []
    cached_masks = []
    with h5py.File(cache_path, 'r') as cache_file:
        for sample_name in sample_names:
            if sample_name not in cache_file:
                return None, None
            grp = cache_file[sample_name]
            if 'sam_features' not in grp or 'sam_masks' not in grp:
                return None, None
            sam_features = torch.from_numpy(grp['sam_features'][:]).float().to(device)
            sam_masks = torch.from_numpy(grp['sam_masks'][:]).float().to(device)
            if sam_features.dim() == 4 and sam_features.shape[0] == 1:
                sam_features = sam_features.squeeze(0)
            if sam_masks.dim() == 3 and sam_masks.shape[0] == 1:
                sam_masks = sam_masks.squeeze(0)
            cached_features.append(sam_features)
            cached_masks.append(sam_masks)
    return cached_features, cached_masks

def create_h5_dataset(data_dir, h5_path):

    os.makedirs(os.path.dirname(h5_path), exist_ok=True)


    samples = [d for d in os.listdir(data_dir)
              if os.path.isdir(os.path.join(data_dir, d))]

    if not samples:
        raise FileNotFoundError(f"No subdirectories found in {data_dir}")


    for sample in samples:
        sample_path = os.path.join(data_dir, sample)
        for i in range(1, 32):
            band_path = os.path.join(sample_path, f"{sample}_{i:02d}.png")
            if not os.path.exists(band_path):
                raise FileNotFoundError(f"Missing band file: {band_path}")
        rgb_path = os.path.join(sample_path, f"{sample}_RGB.bmp")
        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"Missing RGB file: {rgb_path}")


    with h5py.File(h5_path, 'w') as f:
        for sample in samples:
            sample_path = os.path.join(data_dir, sample)


            hsi_bands = []
            for i in range(1, 32):
                band_path = os.path.join(sample_path, f"{sample}_{i:02d}.png")
                band_img = cv2.imread(band_path, cv2.IMREAD_GRAYSCALE)
                hsi_bands.append(band_img)
            hr_hsi = np.stack(hsi_bands, axis=0)


            rgb_path = os.path.join(sample_path, f"{sample}_RGB.bmp")
            hr_msi = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            hr_msi = cv2.cvtColor(hr_msi, cv2.COLOR_BGR2RGB)
            hr_msi = np.transpose(hr_msi, (2,0,1))


            blurred_hsi = np.zeros_like(hr_hsi)
            for i in range(hr_hsi.shape[0]):
                blurred_hsi[i] = gaussian_filter(hr_hsi[i], sigma=0.5)
            lr_hsi = blurred_hsi[:, ::4, ::4]


            grp = f.create_group(sample)
            grp.create_dataset('hr_hsi', data=hr_hsi.astype(np.float32)/255.0)
            grp.create_dataset('hr_msi', data=hr_msi.astype(np.float32)/255.0)
            grp.create_dataset('lr_hsi', data=lr_hsi.astype(np.float32)/255.0)

    print(f'H5 dataset created successfully at {h5_path}')


def visualize_results(output, gt, sample_idx, save_dir, dataset_type='cave'):

    viz_dir = os.path.join(save_dir, 'visualization')
    os.makedirs(viz_dir, exist_ok=True)


    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()


    if dataset_type.lower() == 'cave':


        rgb_indices = [24, 14, 4]
    else:


        total_channels = gt.shape[0]
        step = total_channels // 3
        rgb_indices = [step-1, 2*step-1, 3*step-1]


    def extract_rgb(image, indices):

        valid_indices = [min(i, image.shape[0]-1) for i in indices]
        rgb = image[valid_indices, :, :]

        rgb = np.transpose(rgb, (1, 2, 0))

        rgb_min = rgb.min()
        rgb_max = rgb.max()
        if rgb_max > rgb_min:
            rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
        return rgb


    output_rgb = extract_rgb(output, rgb_indices)
    gt_rgb = extract_rgb(gt, rgb_indices)


    error = np.abs(output - gt)
    error_rgb = extract_rgb(error, rgb_indices)

    plt.figure(figsize=(15, 5))


    plt.subplot(131)
    plt.imshow(gt_rgb)
    plt.title('Ground Truth')
    plt.axis('off')


    plt.subplot(132)
    plt.imshow(output_rgb)
    plt.title('Fusion Result')
    plt.axis('off')


    plt.subplot(133)
    plt.imshow(error_rgb, cmap='jet')
    plt.title('Absolute Error')
    plt.axis('off')

    plt.tight_layout()


    viz_path = os.path.join(viz_dir, f'comparison_{sample_idx}.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"可视化结果已保存到: {viz_path}")
    return viz_path


def _to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return value


def _sync_device(device):
    if isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()


def _count_parameters(model):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'non_trainable_parameters': int(total_params - trainable_params),
        'total_parameters_m': float(total_params / 1e6),
        'trainable_parameters_m': float(trainable_params / 1e6),
    }


def _compute_block_positions(length, cut_size, stride):
    if length <= cut_size:
        return [0]
    positions = list(range(0, max(length - cut_size, 0) + 1, stride))
    last = length - cut_size
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


def _build_blend_window(size, mode, device, dtype):
    if mode == 'uniform':
        window_2d = torch.ones((size, size), dtype=dtype, device=device)
    else:
        one_d = torch.hann_window(size, periodic=False)
        one_d = one_d.to(device=device, dtype=dtype).clamp(min=1e-3)
        window_2d = torch.outer(one_d, one_d)
    return window_2d.clamp(min=1e-3).unsqueeze(0).unsqueeze(0)


def _build_enabled_feature_flags(args):
    feature_flag_names = [
        'use_ase',
        'use_sam_ase',
        'use_offline_sam_cache',
        'prefer_scene_sam_cache',
        'prefer_multi_region_sam_cache',
        'use_wavelet',
        'use_wavelet_priors',
        'use_wavelet_local_bias',
        'use_wavelet_local_gate',
        'use_sam_local_gate',
        'use_sam_semantic_prompt_bank',
        'use_sam_region_prototype_bank',
        'use_wavelet_guided_sam_prototype_scaling',
        'use_sam_region_prompt_mixture',
        'use_sam_guided_semantic_scanning',
        'use_sam_feature_cluster_scanning',
        'use_wavelet_augmented_ss1',
        'use_sam_boundary_aware_state_propagation',
        'use_sam_state_reset_stronger',
        'use_sam_state_organizer_v1',
        'use_sam_region_prompt_subspace',
        'use_wavelet_guided_semantic_state_organization',
        'use_joint_spatial_spectral_wavelet_prior',
        'use_dual_prototype_bank',
        'use_semantic_frequency_state_modulation',
        'use_fass',
        'use_sam_prior_bank',
        'use_sam_prior_c_refiner',
        'use_semantic_frequency_adaptive_scanning',
        'use_structure_guided_sam_ase',
    ]
    return sorted([name for name in feature_flag_names if getattr(args, name, False)])


def _serialize_routing_probs(routing_probs):
    if routing_probs is None:
        return None
    if isinstance(routing_probs, torch.Tensor):
        tensor = routing_probs.detach().float().cpu()
        return {
            'shape': list(tensor.shape),
            'mean': float(tensor.mean().item()),
            'std': float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0,
            'min': float(tensor.min().item()),
            'max': float(tensor.max().item()),
        }
    return _to_jsonable(routing_probs)


def _export_comparison_artifacts(
    args,
    sample_names,
    per_sample_records,
    summary_payload,
    model_param_stats,
    dataset_info,
):
    if getattr(args, 'comparison_artifacts_dir', None):
        artifact_root = Path(args.comparison_artifacts_dir)
    else:
        artifact_root = Path(args.save_dir) / getattr(args, 'comparison_artifacts_subdir', 'comparison_artifacts')
    artifact_root.mkdir(parents=True, exist_ok=True)

    metrics_summary_path = artifact_root / 'metrics_summary.json'
    metrics_per_sample_json_path = artifact_root / 'metrics_per_sample.json'
    metrics_per_sample_csv_path = artifact_root / 'metrics_per_sample.csv'
    model_info_path = artifact_root / 'model_info.json'
    runtime_info_path = artifact_root / 'runtime_info.json'
    dataset_info_path = artifact_root / 'dataset_info.json'
    artifact_index_path = artifact_root / 'artifact_index.json'
    paper_row_json_path = artifact_root / 'paper_table_row.json'
    paper_row_csv_path = artifact_root / 'paper_table_row.csv'

    metrics_summary_path.write_text(
        json.dumps(_to_jsonable(summary_payload), ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    metrics_per_sample_json_path.write_text(
        json.dumps(_to_jsonable(per_sample_records), ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    with metrics_per_sample_csv_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            'sample_index',
            'sample_name',
            'ssim',
            'sam_deg',
            'rmse',
            'psnr_db',
            'ergas',
            'inference_time_sec',
            'result_mat_path',
            'visualization_path',
            'hr_shape',
            'lr_shape',
            'eval_mode',
        ])
        for record in per_sample_records:
            writer.writerow([
                record['sample_index'],
                record['sample_name'],
                record['metrics']['ssim'],
                record['metrics']['sam_deg'],
                record['metrics']['rmse'],
                record['metrics']['psnr_db'],
                record['metrics']['ergas'],
                record['runtime']['inference_time_sec'],
                record['paths']['result_mat'],
                record['paths']['visualization'],
                'x'.join(map(str, record['shapes']['hr_hsi'])),
                'x'.join(map(str, record['shapes']['lr_hsi'])),
                record['runtime']['eval_mode'],
            ])

    model_info = {
        'dataset': args.dataset,
        'ratio': int(args.ratio),
        'weight_path': str(Path(args.weight).resolve()),
        'save_dir': str(Path(args.save_dir).resolve()),
        'device': args.device,
        'channels': int(args.channels),
        'cut_size': int(args.cut_size),
        'pad': int(args.pad),
        'force_small_image_block_eval': bool(getattr(args, 'force_small_image_block_eval', False)),
        'use_overlap_blend': bool(getattr(args, 'use_overlap_blend', False)),
        'block_stride': None if getattr(args, 'block_stride', None) is None else int(getattr(args, 'block_stride', None)),
        'blend_window': getattr(args, 'blend_window', 'hann'),
        'enabled_feature_flags': _build_enabled_feature_flags(args),
        'parameter_stats': model_param_stats,
        'command': ' '.join(sys.argv),
        'platform': {
            'python_version': sys.version,
            'platform': platform.platform(),
            'torch_version': torch.__version__,
            'cuda_available': bool(torch.cuda.is_available()),
            'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    model_info_path.write_text(
        json.dumps(_to_jsonable(model_info), ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    inference_times = [record['runtime']['inference_time_sec'] for record in per_sample_records]
    runtime_info = {
        'sample_count': len(per_sample_records),
        'total_inference_time_sec': float(sum(inference_times)),
        'average_inference_time_sec': float(np.mean(inference_times)) if inference_times else None,
        'std_inference_time_sec': float(np.std(inference_times)) if inference_times else None,
        'min_inference_time_sec': float(np.min(inference_times)) if inference_times else None,
        'max_inference_time_sec': float(np.max(inference_times)) if inference_times else None,
        'eval_modes': sorted(list({record['runtime']['eval_mode'] for record in per_sample_records})),
        'note': 'Inference time is measured per sample around model forward / block processing only; metric calculation and disk I/O are excluded.',
    }
    runtime_info_path.write_text(
        json.dumps(_to_jsonable(runtime_info), ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    dataset_info_path.write_text(
        json.dumps(_to_jsonable(dataset_info), ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    artifact_index = {
        'artifact_root': str(artifact_root.resolve()),
        'average_results_txt': str((Path(args.save_dir) / 'average_results.txt').resolve()),
        'metrics_summary_json': str(metrics_summary_path.resolve()),
        'metrics_per_sample_json': str(metrics_per_sample_json_path.resolve()),
        'metrics_per_sample_csv': str(metrics_per_sample_csv_path.resolve()),
        'model_info_json': str(model_info_path.resolve()),
        'runtime_info_json': str(runtime_info_path.resolve()),
        'dataset_info_json': str(dataset_info_path.resolve()),
        'paper_table_row_json': str(paper_row_json_path.resolve()),
        'paper_table_row_csv': str(paper_row_csv_path.resolve()),
        'per_sample_outputs': [
            {
                'sample_index': record['sample_index'],
                'sample_name': record['sample_name'],
                'result_mat': record['paths']['result_mat'],
                'visualization': record['paths']['visualization'],
            }
            for record in per_sample_records
        ],
    }
    artifact_index_path.write_text(
        json.dumps(_to_jsonable(artifact_index), ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    paper_table_row = {
        'method_name': Path(args.weight).parent.name,
        'dataset': args.dataset,
        'ratio': int(args.ratio),
        'metrics': {
            'ssim_mean': summary_payload['metrics']['ssim']['mean'],
            'ssim_std': summary_payload['metrics']['ssim']['std'],
            'sam_mean': summary_payload['metrics']['sam_deg']['mean'],
            'sam_std': summary_payload['metrics']['sam_deg']['std'],
            'rmse_mean': summary_payload['metrics']['rmse']['mean'],
            'rmse_std': summary_payload['metrics']['rmse']['std'],
            'psnr_mean': summary_payload['metrics']['psnr_db']['mean'],
            'psnr_std': summary_payload['metrics']['psnr_db']['std'],
            'ergas_mean': summary_payload['metrics']['ergas']['mean'],
            'ergas_std': summary_payload['metrics']['ergas']['std'],
        },
        'parameter_stats': model_param_stats,
        'runtime': runtime_info,
    }
    paper_row_json_path.write_text(
        json.dumps(_to_jsonable(paper_table_row), ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    with paper_row_csv_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            'method_name', 'dataset', 'ratio',
            'ssim_mean', 'ssim_std',
            'sam_mean', 'sam_std',
            'rmse_mean', 'rmse_std',
            'psnr_mean', 'psnr_std',
            'ergas_mean', 'ergas_std',
            'total_parameters_m', 'trainable_parameters_m',
            'average_inference_time_sec',
        ])
        writer.writerow([
            paper_table_row['method_name'],
            paper_table_row['dataset'],
            paper_table_row['ratio'],
            paper_table_row['metrics']['ssim_mean'],
            paper_table_row['metrics']['ssim_std'],
            paper_table_row['metrics']['sam_mean'],
            paper_table_row['metrics']['sam_std'],
            paper_table_row['metrics']['rmse_mean'],
            paper_table_row['metrics']['rmse_std'],
            paper_table_row['metrics']['psnr_mean'],
            paper_table_row['metrics']['psnr_std'],
            paper_table_row['metrics']['ergas_mean'],
            paper_table_row['metrics']['ergas_std'],
            model_param_stats['total_parameters_m'],
            model_param_stats['trainable_parameters_m'],
            runtime_info['average_inference_time_sec'],
        ])

    return artifact_root

def test(args):


    weight_dir = os.path.dirname(args.weight)


    if args.save_dir is None:
        test_results_dir = os.path.join(weight_dir, 'test_results')
        args.save_dir = test_results_dir


    os.makedirs(args.save_dir, exist_ok=True)


    if args.dataset.lower() == 'chikusei':

        print(f'使用现有的Chikusei h5文件: {args.h5_path}')
    elif args.dataset.lower() == 'pavia_university':
        args.h5_path = args.h5_path or './data/pavia_university_x4/pavia_university_test.h5'
        if not os.path.exists(args.h5_path):
            raise FileNotFoundError(
                f"Pavia University测试H5文件不存在: {args.h5_path}。"
                "请先运行 data/preprocess_pavia_university.py。"
            )
        print(f'使用现有的Pavia University h5文件: {args.h5_path}')
    elif args.dataset.lower() == 'xiongan_new_area':
        args.h5_path = args.h5_path or './data/xiongan_new_area_x4/xiongan_new_area_test.h5'
        if not os.path.exists(args.h5_path):
            raise FileNotFoundError(
                f"Xiongan New Area测试H5文件不存在: {args.h5_path}。"
                "请先运行 data/preprocess_xiongan_new_area.py。"
            )
        print(f'使用现有的Xiongan New Area h5文件: {args.h5_path}')
    elif args.dataset.lower() == 'cave':

        args.h5_path = args.h5_path or './data/cave/cave_test.h5'

        if not os.path.exists(args.h5_path):
            print(f'CAVE测试H5文件不存在，正在从{args.test_data_path}生成...')
            if not os.path.exists(args.test_data_path):
                raise FileNotFoundError(f"测试数据目录 {args.test_data_path} 不存在")
            create_h5_dataset(args.test_data_path, args.h5_path)
        else:
            print(f'使用现有的CAVE h5文件: {args.h5_path}')
    else:

        if not os.path.exists(args.h5_path):
            print('h5文件不存在，正在创建...')
            if not os.path.exists(args.test_data_path):
                raise FileNotFoundError(f"数据目录 {args.test_data_path} 不存在")
            create_h5_dataset(args.test_data_path, args.h5_path)
        else:
            print(f'使用现有的h5文件: {args.h5_path}')


    sample_names = None
    with h5py.File(args.h5_path, 'r') as f:

        dataset_type = getattr(args, 'dataset', 'cave').lower()

        try:

            sample_names = [name for name in f.keys() if isinstance(f[name], h5py.Group)]
            if sample_names:
                all_lr_hsi = [f[sample]['lr_hsi'][:] for sample in sample_names]
                all_hr_hsi = [f[sample]['hr_hsi'][:] for sample in sample_names]
                all_hr_msi = [f[sample]['hr_msi'][:] for sample in sample_names]
            else:

                all_lr_hsi = [f['lr_hsi'][:]]
                all_hr_hsi = [f['hr_hsi'][:]] if 'hr_hsi' in f else None
                all_hr_msi = [f['hr_msi'][:]]
        except Exception as e:
            print(f"读取H5文件错误: {e}")
            raise


    lr_hsi = torch.from_numpy(np.stack(all_lr_hsi, axis=0)).float().to(args.device)
    hr_hsi = torch.from_numpy(np.stack(all_hr_hsi, axis=0)).float().to(args.device)
    hr_msi = torch.from_numpy(np.stack(all_hr_msi, axis=0)).float().to(args.device)
    cached_sam_feature_list, cached_sam_mask_list = None, None
    if getattr(args, 'use_offline_sam_cache', False):
        cache_path = resolve_sam_cache_path_for_h5(
            args.h5_path,
            getattr(args, 'sam_cache_path', None),
            getattr(args, 'prefer_scene_sam_cache', False),
            getattr(args, 'prefer_multi_region_sam_cache', False),
        )
        cached_sam_feature_list, cached_sam_mask_list = load_test_sam_cache(cache_path, sample_names or [], args.device)
        if cached_sam_feature_list is None or cached_sam_mask_list is None:
            print(f"[SAM-CACHE] WARNING: test cache unavailable or incompatible: {cache_path}")
        else:
            print(f"[SAM-CACHE] Test cache ready: {cache_path}")


    _, _, H_first, W_first = hr_msi.shape
    if hasattr(args, 'cut_size'):
        force_full_image_eval = False
        if (
            max(H_first, W_first) <= 256
            and args.cut_size < max(H_first, W_first)
            and not getattr(args, 'force_small_image_block_eval', False)
        ):
            full_image_cut = max(H_first, W_first)
            print(
                f"[INFO] small test image {H_first}x{W_first}: "
                f"override cut_size from {args.cut_size} to {full_image_cut} to avoid block stitching seams"
            )
            args.cut_size = full_image_cut
            args.pad = 0
            force_full_image_eval = True
        elif max(H_first, W_first) <= 256 and args.cut_size < max(H_first, W_first):
            print(
                f"[INFO] small test image {H_first}x{W_first}: keep explicit block eval "
                f"(cut_size={args.cut_size}, pad={args.pad}) because --force_small_image_block_eval is enabled"
            )

        if max(H_first, W_first) > 1024:
            recommended_cut_size = min(512, max(H_first, W_first) // 4)
            if args.cut_size > recommended_cut_size:
                print(f"检测到大尺寸图像 ({H_first}x{W_first})，自动调整cut_size从{args.cut_size}到{recommended_cut_size}")
                args.cut_size = recommended_cut_size

                args.pad = args.cut_size // 8


        if args.pad == 0 and not force_full_image_eval:
            args.pad = args.cut_size // 8

        stride_info = getattr(args, 'block_stride', None)
        print(
            f"使用分块参数: cut_size={args.cut_size}, pad={args.pad}, "
            f"overlap_blend={getattr(args, 'use_overlap_blend', False)}, "
            f"block_stride={stride_info}, blend_window={getattr(args, 'blend_window', 'hann')}"
        )


    if dataset_type == 'chikusei':
        lr_hsi_dim = 128
        hr_msi_dim = 4
    else:
        lr_hsi_dim = 31
        hr_msi_dim = 3


    if dataset_type == 'pavia_university':
        lr_hsi_dim = 103
        hr_msi_dim = 4
    elif dataset_type == 'xiongan_new_area':
        lr_hsi_dim = 93
        hr_msi_dim = 4

    if sample_names and len(all_lr_hsi) > 0 and len(all_hr_msi) > 0:
        lr_hsi_dim = int(np.asarray(all_lr_hsi[0]).shape[0])
        hr_msi_dim = int(np.asarray(all_hr_msi[0]).shape[0])

    checkpoint = torch.load(args.weight, map_location='cpu')


    has_ase_params = False
    has_sam_ase_params = False
    has_sam_local_gate_params = False
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        for key in checkpoint['state_dict'].keys():
            if 'ase_module' in key or 'route_fusion' in key or 'ase' in key.lower():
                has_ase_params = True
            if '.sam_ase_module.' in key:
                has_sam_ase_params = True
            if 'sam_local_gate_proj_stage' in key:
                has_sam_local_gate_params = True


    if has_sam_ase_params:
        print(f"✓ 检测到SAM-ASE权重参数")
        if not hasattr(args, 'use_sam_ase') or not args.use_sam_ase:
            args.use_sam_ase = True
            print(f"  自动启用use_sam_ase=True")


    if hasattr(args, 'use_ase'):
        use_ase = args.use_ase
        print(f"使用命令行参数设置use_ase={use_ase}")
    else:

        use_ase = has_ase_params
        print(f"根据权重文件内容自动设置use_ase={use_ase}")


    if has_sam_local_gate_params:
        print("✅ 检测到SAM Local Gate权重参数")
        if not hasattr(args, 'use_sam_local_gate') or not args.use_sam_local_gate:
            args.use_sam_local_gate = True
            print("  自动启用use_sam_local_gate=True")
    has_sam_semantic_prompt_bank_params = False
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        for key in checkpoint['state_dict'].keys():
            if 'sam_semantic_prompt_bank_refiner' in key:
                has_sam_semantic_prompt_bank_params = True
                break
    if has_sam_semantic_prompt_bank_params:
        print("检测到SAM Semantic Prompt Bank权重参数")
        if not hasattr(args, 'use_sam_semantic_prompt_bank') or not args.use_sam_semantic_prompt_bank:
            args.use_sam_semantic_prompt_bank = True
            print("  自动启用use_sam_semantic_prompt_bank=True")
    use_sam_ase = getattr(args, 'use_sam_ase', False)
    sam_checkpoint = getattr(args, 'sam_checkpoint', None)
    sam_prompt_dim = getattr(args, 'sam_prompt_dim', 64)


    use_learnable_prompts = getattr(args, 'use_learnable_prompts', False)
    num_learnable_prompts = getattr(args, 'num_learnable_prompts', 16)
    use_soft_masks = getattr(args, 'use_soft_masks', False)
    num_soft_regions = getattr(args, 'num_soft_regions', 8)


    use_fass = getattr(args, 'use_fass', False)
    fass_compression_ratio = getattr(args, 'fass_compression_ratio', 2)
    fass_threshold = getattr(args, 'fass_threshold', 0.5)
    fass_sparsity_target = getattr(args, 'fass_sparsity_target', 0.3)
    fass_ll_sparsity = getattr(args, 'fass_ll_sparsity', 1.0)
    fass_hf_sparsity = getattr(args, 'fass_hf_sparsity', 0.20)
    fass_d_state = getattr(args, 'fass_d_state', 16)
    train_mode = getattr(args, 'train_mode', 'sparse')
    dense_epochs = getattr(args, 'dense_epochs', 100)
    gating_loss_weight = getattr(args, 'gating_loss_weight', 1.0)
    gating_input_mode = getattr(args, 'gating_input_mode', 'energy')
    gating_use_semantic_mask = getattr(args, 'gating_use_semantic_mask', True)
    gating_use_prompt_strength = getattr(args, 'gating_use_prompt_strength', True)
    gating_use_local_contrast = getattr(args, 'gating_use_local_contrast', True)
    use_sam_prior_bank = getattr(args, 'use_sam_prior_bank', False)
    sam_prior_use_boundary = getattr(args, 'sam_prior_use_boundary', True)
    sam_prior_use_confidence = getattr(args, 'sam_prior_use_confidence', True)
    use_sam_prior_c_refiner = getattr(args, 'use_sam_prior_c_refiner', False)
    sam_prior_c_refiner_scale = float(getattr(args, 'sam_prior_c_refiner_scale', 0.1))
    sam_prior_c_refiner_scope = getattr(args, 'sam_prior_c_refiner_scope', 'fusion')
    use_structure_guided_sam_ase = getattr(args, 'use_structure_guided_sam_ase', False)
    structure_texture_weight = getattr(args, 'structure_texture_weight', 0.25)

    def configure_fass_gating_modules(net):
        configured_gating_modules = 0
        for module in net.modules():
            if hasattr(module, 'gating_input_mode') and hasattr(module, 'gating_net_ll_hybrid'):
                module.gating_input_mode = 'semantic_frequency_v1' if getattr(args, 'use_semantic_frequency_adaptive_scanning', False) else gating_input_mode
                module.gating_use_semantic_mask = gating_use_semantic_mask
                module.gating_use_prompt_strength = gating_use_prompt_strength
                module.gating_use_local_contrast = gating_use_local_contrast
                module.use_sam_prior_bank = use_sam_prior_bank
                module.sam_prior_use_boundary = sam_prior_use_boundary
                module.sam_prior_use_confidence = sam_prior_use_confidence
                module.use_semantic_frequency_adaptive_scanning = getattr(args, 'use_semantic_frequency_adaptive_scanning', False)
                module.semantic_frequency_semantic_weight = float(getattr(args, 'semantic_frequency_semantic_weight', 1.0))
                module.semantic_frequency_wavelet_weight = float(getattr(args, 'semantic_frequency_wavelet_weight', 1.0))
                module.semantic_frequency_boundary_weight = float(getattr(args, 'semantic_frequency_boundary_weight', 0.5))
                module.semantic_frequency_confidence_weight = float(getattr(args, 'semantic_frequency_confidence_weight', 0.25))
                module.semantic_frequency_prompt_weight = float(getattr(args, 'semantic_frequency_prompt_weight', 0.5))
                configured_gating_modules += 1
        return configured_gating_modules

    def configure_prior_c_refiner_modules(net):
        configured_prior_c_refiner_modules = 0
        for module in net.modules():
            if hasattr(module, 'use_sam_prior_c_refiner') and hasattr(module, 'prior_refiner'):
                module_role = getattr(module, 'prior_c_refiner_role', 'unknown')
                enable_module = use_sam_prior_c_refiner and (
                    sam_prior_c_refiner_scope == 'all' or module_role == 'fusion'
                )
                module.use_sam_prior_c_refiner = enable_module
                module.sam_prior_c_refiner_scale = sam_prior_c_refiner_scale
                if enable_module:
                    configured_prior_c_refiner_modules += 1
        return configured_prior_c_refiner_modules


    if use_sam_ase:
        print(f"[SAM-ASE] 测试模式已启用")
        if sam_checkpoint:
            print(f"[SAM-ASE] SAM权重路径: {sam_checkpoint}")
        else:
            print(f"[WARNING] 未指定SAM权重路径，将使用默认值")


    if use_fass:
        print(f"[FASS] 测试模式已启用")
        print(f"[FASS] 训练模式: {train_mode}")
        print(f"[FASS] LL分支稀疏度: {fass_ll_sparsity} (1.0=Dense)")
        print(f"[FASS] HF分支稀疏度: {fass_hf_sparsity}")


    if use_fass:
        effective_gating_mode = 'semantic_frequency_v1' if getattr(args, 'use_semantic_frequency_adaptive_scanning', False) else gating_input_mode
        print(f"[FASS-GATING] mode={effective_gating_mode}, semantic_mask={gating_use_semantic_mask}, prompt_strength={gating_use_prompt_strength}, local_contrast={gating_use_local_contrast}, sam_prior_bank={use_sam_prior_bank}, prior_boundary={sam_prior_use_boundary}, prior_confidence={sam_prior_use_confidence}")
        if use_sam_prior_bank and (not getattr(args, 'use_semantic_frequency_adaptive_scanning', False)) and gating_input_mode != 'hybrid_v2':
            print(f"[SAM-PRIOR] WARNING: use_sam_prior_bank only affects hybrid_v2/semantic_frequency_v1 gating, current mode={gating_input_mode}; prior bank will be ignored")
    if use_sam_prior_c_refiner:
        if not use_sam_ase:
            print("[SAM-PRIOR-C] WARNING: use_sam_prior_c_refiner requires --use_sam_ase; current run will ignore it")
        elif use_fass:
            print("[SAM-PRIOR-C] WARNING: Prior C Refiner V1 currently only affects the non-FASS SAMASE path; it is ignored when --use_fass is enabled")
    if use_sam_ase and use_structure_guided_sam_ase:
        print(f"[STRUCT-SAM-ASE] enabled: texture_weight={structure_texture_weight}")
    if use_ase and not use_sam_ase and not use_fass:
        print(
            f"[ASE-HISR] prompt_mode={getattr(args, 'ase_prompt_mode', 'hard')}, "
            f"route_temperature={float(getattr(args, 'ase_route_temperature', 1.0))}, "
            f"prompt_soft_mix={float(getattr(args, 'ase_prompt_soft_mix', 0.5))}, "
            f"scope={getattr(args, 'ase_scope', 'all')}, "
            f"stage_scope={getattr(args, 'ase_stage_scope', 'all_stages')}, "
            f"fusion_residual={getattr(args, 'use_ase_fusion_residual', False)}, "
            f"fusion_res_scale={float(getattr(args, 'ase_fusion_res_scale', 0.3))}, "
            f"stage_res_scales={getattr(args, 'ase_stage_res_scales', None)}, "
            f"wavelet_local_bias={getattr(args, 'use_wavelet_local_bias', False)}, "
            f"wavelet_local_bias_scale={float(getattr(args, 'wavelet_local_bias_scale', 0.1))}, "
            f"wavelet_local_gate={getattr(args, 'use_wavelet_local_gate', False)}, "
            f"wavelet_local_gate_scale={float(getattr(args, 'wavelet_local_gate_scale', 0.1))}, "
            f"sam_local_gate={getattr(args, 'use_sam_local_gate', False)}, "
            f"sam_local_gate_scale={float(getattr(args, 'sam_local_gate_scale', 0.1))}, "
            f"sam_semantic_prompt_bank={getattr(args, 'use_sam_semantic_prompt_bank', False)}, "
            f"sam_semantic_prompt_bank_scale={float(getattr(args, 'sam_semantic_prompt_bank_scale', 0.1))}, "
            f"sam_region_prototype_bank={getattr(args, 'use_sam_region_prototype_bank', False)}, "
            f"sam_region_prototype_bank_scale={float(getattr(args, 'sam_region_prototype_bank_scale', 0.1))}, "
            f"sam_region_prototype_count={int(getattr(args, 'sam_region_prototype_count', 8))}, "
            f"wavelet_guided_sam_prototype_scaling={getattr(args, 'use_wavelet_guided_sam_prototype_scaling', False)}, "
            f"wavelet_guided_sam_prototype_scale={float(getattr(args, 'wavelet_guided_sam_prototype_scale', 0.1))}, "
            f"sam_region_prompt_mixture={getattr(args, 'use_sam_region_prompt_mixture', False)}, "
            f"sam_region_prompt_mixture_scale={float(getattr(args, 'sam_region_prompt_mixture_scale', 0.05))}, "
            f"sam_region_prompt_mixture_count={int(getattr(args, 'sam_region_prompt_mixture_count', 8))}, "
            f"sam_guided_semantic_scanning={getattr(args, 'use_sam_guided_semantic_scanning', False)}, "
            f"sam_semantic_scanning_count={int(getattr(args, 'sam_semantic_scanning_count', 6))}, "
            f"sam_feature_cluster_scanning={getattr(args, 'use_sam_feature_cluster_scanning', False)}, "
            f"sam_feature_cluster_count={int(getattr(args, 'sam_feature_cluster_count', 6))}, "
            f"sam_feature_cluster_iters={int(getattr(args, 'sam_feature_cluster_iters', 2))}, "
            f"sam_feature_cluster_spatial_weight={float(getattr(args, 'sam_feature_cluster_spatial_weight', 0.05))}, "
            f"wavelet_augmented_ss1={getattr(args, 'use_wavelet_augmented_ss1', False)}, "
            f"wavelet_augmented_ss1_count={int(getattr(args, 'wavelet_augmented_ss1_count', 6))}, "
            f"wavelet_augmented_ss1_topk_ratio={float(getattr(args, 'wavelet_augmented_ss1_topk_ratio', 0.25))}, "
            f"wavelet_augmented_ss1_strength={float(getattr(args, 'wavelet_augmented_ss1_strength', 0.5))}, "
            f"wavelet_augmented_ss1_mode={getattr(args, 'wavelet_augmented_ss1_mode', 'stable_intra_region')}, "
            f"sam_boundary_aware_state_propagation={getattr(args, 'use_sam_boundary_aware_state_propagation', False)}, "
            f"sam_boundary_aware_state_scale={float(getattr(args, 'sam_boundary_aware_state_scale', 0.2))}, "
            f"sam_state_reset_stronger={getattr(args, 'use_sam_state_reset_stronger', False)}, "
            f"sam_state_reset_scale={float(getattr(args, 'sam_state_reset_scale', 0.35))}, "
            f"sam_state_organizer_v1={getattr(args, 'use_sam_state_organizer_v1', False)}, "
            f"sam_state_organizer_count={int(getattr(args, 'sam_state_organizer_count', 6))}, "
            f"sam_region_prompt_subspace={getattr(args, 'use_sam_region_prompt_subspace', False)}, "
            f"sam_region_prompt_subspace_scale={float(getattr(args, 'sam_region_prompt_subspace_scale', 0.05))}, "
            f"sam_region_prompt_subspace_count={int(getattr(args, 'sam_region_prompt_subspace_count', 6))}, "
            f"wavelet_guided_semantic_state_organization={getattr(args, 'use_wavelet_guided_semantic_state_organization', False)}, "
            f"wavelet_guided_semantic_state_count={int(getattr(args, 'wavelet_guided_semantic_state_count', 6))}, "
            f"wavelet_guided_semantic_state_scale={float(getattr(args, 'wavelet_guided_semantic_state_scale', 0.05))}, "
            f"joint_spatial_spectral_wavelet_prior={getattr(args, 'use_joint_spatial_spectral_wavelet_prior', False)}, "
            f"joint_wavelet_spatial_weight={float(getattr(args, 'joint_wavelet_spatial_weight', 1.0))}, "
            f"joint_wavelet_spectral_weight={float(getattr(args, 'joint_wavelet_spectral_weight', 1.0))}, "
            f"dual_prototype_bank={getattr(args, 'use_dual_prototype_bank', False)}, "
            f"dual_prototype_semantic_scale={float(getattr(args, 'dual_prototype_semantic_scale', 0.05))}, "
            f"dual_prototype_frequency_scale={float(getattr(args, 'dual_prototype_frequency_scale', 0.05))}, "
            f"dual_prototype_count={int(getattr(args, 'dual_prototype_count', 6))}, "
            f"semantic_frequency_state_modulation={getattr(args, 'use_semantic_frequency_state_modulation', False)}, "
            f"semantic_frequency_state_count={int(getattr(args, 'semantic_frequency_state_count', 6))}, "
            f"semantic_frequency_state_write_scale={float(getattr(args, 'semantic_frequency_state_write_scale', 0.08))}, "
            f"semantic_frequency_state_read_scale={float(getattr(args, 'semantic_frequency_state_read_scale', 0.08))}, "
            f"semantic_frequency_state_delta_scale={float(getattr(args, 'semantic_frequency_state_delta_scale', 0.05))}, "
            f"semantic_frequency_adaptive_scanning={getattr(args, 'use_semantic_frequency_adaptive_scanning', False)}"
        )
    if getattr(args, 'use_sam_local_gate', False) and not (use_ase and not use_sam_ase and not use_fass and getattr(args, 'use_ase_fusion_residual', False) and getattr(args, 'ase_scope', 'all') == 'fusion_only'):
        print("[SAM-LOCAL-GATE] WARNING: this option is designed for ASE + fusion_only + fusion_residual; current run may ignore it")
    if getattr(args, 'use_sam_local_gate', False) and not sam_checkpoint:
        print("[SAM-LOCAL-GATE] WARNING: no --sam_checkpoint provided; SAM local gate may fail to initialize")
    if getattr(args, 'use_sam_semantic_prompt_bank', False) and not (use_ase and not use_sam_ase and not use_fass and getattr(args, 'use_ase_fusion_residual', False) and getattr(args, 'ase_scope', 'all') == 'fusion_only'):
        print("[SAM-PBANK] WARNING: this option is designed for ASE + fusion_only + fusion_residual; current run may ignore it")
    if getattr(args, 'use_sam_semantic_prompt_bank', False) and not sam_checkpoint:
        print("[SAM-PBANK] WARNING: no --sam_checkpoint provided; semantic prompt bank may fail to initialize")
    if getattr(args, 'use_sam_region_prototype_bank', False) and not (use_ase and not use_sam_ase and not use_fass and getattr(args, 'use_ase_fusion_residual', False) and getattr(args, 'ase_scope', 'all') == 'fusion_only'):
        print("[SAM-RPROTO] WARNING: this option is designed for ASE + fusion_only + fusion_residual; current run may ignore it")
    if getattr(args, 'use_sam_region_prototype_bank', False) and not getattr(args, 'use_offline_sam_cache', False):
        print("[SAM-RPROTO] WARNING: this option relies on remapped whole-image SAM cache; without --use_offline_sam_cache prototype conditioning will be unavailable")
    if getattr(args, 'use_sam_region_prompt_mixture', False) and not (use_ase and not use_sam_ase and not use_fass and getattr(args, 'use_ase_fusion_residual', False) and getattr(args, 'ase_scope', 'all') == 'fusion_only'):
        print("[SAM-RPMIX] WARNING: this option is designed for ASE + fusion_only + fusion_residual; current run may ignore it")
    if getattr(args, 'use_sam_region_prompt_mixture', False) and not getattr(args, 'use_offline_sam_cache', False):
        print("[SAM-RPMIX] WARNING: this option relies on remapped whole-image SAM cache; without --use_offline_sam_cache prompt mixture will be unavailable")
    if getattr(args, 'use_wavelet_guided_sam_prototype_scaling', False) and not getattr(args, 'use_sam_region_prototype_bank', False):
        print("[WAVE-RPROTO] WARNING: this option requires --use_sam_region_prototype_bank; otherwise it will have no effect")
    if getattr(args, 'use_sam_guided_semantic_scanning', False) and not getattr(args, 'use_offline_sam_cache', False):
        print("[SAM-SCAN] WARNING: --use_sam_guided_semantic_scanning is enabled but no --use_offline_sam_cache was provided; test-time SAM-guided reordering will be unavailable and the run will follow train-time-SAM-only inference.")
    if getattr(args, 'use_sam_feature_cluster_scanning', False) and not getattr(args, 'use_sam_guided_semantic_scanning', False):
        raise ValueError("--use_sam_feature_cluster_scanning requires --use_sam_guided_semantic_scanning")
    if getattr(args, 'use_sam_feature_cluster_scanning', False) and not getattr(args, 'use_offline_sam_cache', False):
        print("[SAM-FCLUSTER] WARNING: --use_sam_feature_cluster_scanning is enabled but no --use_offline_sam_cache was provided; feature-cluster scanning will be unavailable at test time.")
    if getattr(args, 'use_wavelet_augmented_ss1', False) and not getattr(args, 'use_sam_guided_semantic_scanning', False):
        raise ValueError("--use_wavelet_augmented_ss1 requires --use_sam_guided_semantic_scanning")
    if getattr(args, 'use_wavelet_augmented_ss1', False) and not getattr(args, 'use_offline_sam_cache', False):
        print("[WAVE-SS1] WARNING: --use_wavelet_augmented_ss1 is enabled but no --use_offline_sam_cache was provided; wavelet-augmented semantic scanning will be unavailable at test time.")
    if getattr(args, 'use_wavelet_augmented_ss1', False) and not getattr(args, 'use_joint_spatial_spectral_wavelet_prior', False):
        raise ValueError("--use_wavelet_augmented_ss1 requires --use_joint_spatial_spectral_wavelet_prior")
    if getattr(args, 'use_sam_boundary_aware_state_propagation', False) and not getattr(args, 'use_offline_sam_cache', False):
        print("[SAM-BOUNDARY] WARNING: --use_sam_boundary_aware_state_propagation is enabled but no --use_offline_sam_cache was provided; boundary-aware state propagation will be unavailable at test time.")
    if getattr(args, 'use_sam_state_reset_stronger', False) and not getattr(args, 'use_offline_sam_cache', False):
        print("[SAM-RESET] WARNING: --use_sam_state_reset_stronger is enabled but no --use_offline_sam_cache was provided; stronger state reset will be unavailable at test time.")
    if getattr(args, 'use_sam_state_organizer_v1', False) and not getattr(args, 'use_offline_sam_cache', False):
        print("[SAM-ORG] WARNING: --use_sam_state_organizer_v1 is enabled but no --use_offline_sam_cache was provided; SAM state organizer will be unavailable at test time.")
    if getattr(args, 'use_sam_region_prompt_subspace', False) and not getattr(args, 'use_offline_sam_cache', False):
        print("[SAM-RSUB] WARNING: --use_sam_region_prompt_subspace is enabled but no --use_offline_sam_cache was provided; region prompt subspace will be unavailable at test time.")
    if getattr(args, 'use_wavelet_guided_semantic_state_organization', False) and not getattr(args, 'use_offline_sam_cache', False):
        print("[WAVE-SORG] WARNING: --use_wavelet_guided_semantic_state_organization is enabled but no --use_offline_sam_cache was provided; wavelet-guided semantic state organization will be unavailable at test time.")
    if getattr(args, 'use_semantic_frequency_adaptive_scanning', False) and not use_fass:
        raise ValueError("--use_semantic_frequency_adaptive_scanning requires --use_fass")
    fusion_only_ase_mainline = use_ase and not use_sam_ase and not use_fass and getattr(args, 'use_ase_fusion_residual', False) and getattr(args, 'ase_scope', 'all') == 'fusion_only'
    if getattr(args, 'use_sam_state_organizer_v1', False) and not fusion_only_ase_mainline:
        raise ValueError("--use_sam_state_organizer_v1 requires ASE + fusion_only + fusion_residual (non-SAMASE, non-FASS)")
    if getattr(args, 'use_sam_region_prompt_subspace', False) and not fusion_only_ase_mainline:
        raise ValueError("--use_sam_region_prompt_subspace requires ASE + fusion_only + fusion_residual (non-SAMASE, non-FASS)")
    if getattr(args, 'use_wavelet_guided_semantic_state_organization', False) and not fusion_only_ase_mainline:
        raise ValueError("--use_wavelet_guided_semantic_state_organization requires ASE + fusion_only + fusion_residual (non-SAMASE, non-FASS)")
    if getattr(args, 'use_sam_feature_cluster_scanning', False) and not fusion_only_ase_mainline:
        raise ValueError("--use_sam_feature_cluster_scanning requires ASE + fusion_only + fusion_residual (non-SAMASE, non-FASS)")
    if getattr(args, 'use_wavelet_augmented_ss1', False) and not fusion_only_ase_mainline:
        raise ValueError("--use_wavelet_augmented_ss1 requires ASE + fusion_only + fusion_residual (non-SAMASE, non-FASS)")
    if getattr(args, 'use_sam_feature_cluster_scanning', False) and getattr(args, 'use_wavelet_augmented_ss1', False):
        raise ValueError("--use_sam_feature_cluster_scanning cannot be combined with --use_wavelet_augmented_ss1")
    if getattr(args, 'use_sam_state_organizer_v1', False) and (
        getattr(args, 'use_sam_guided_semantic_scanning', False)
        or getattr(args, 'use_sam_boundary_aware_state_propagation', False)
        or getattr(args, 'use_sam_state_reset_stronger', False)
    ):
        raise ValueError("--use_sam_state_organizer_v1 cannot be combined with old standalone SAM scan/boundary/reset controls")
    if getattr(args, 'use_wavelet_augmented_ss1', False) and (
        getattr(args, 'use_sam_boundary_aware_state_propagation', False)
        or getattr(args, 'use_sam_state_reset_stronger', False)
        or getattr(args, 'use_sam_state_organizer_v1', False)
        or getattr(args, 'use_wavelet_guided_semantic_state_organization', False)
    ):
        raise ValueError("--use_wavelet_augmented_ss1 is a standalone SS1 enhancement and cannot be combined with other SAM state organizers/boundary-reset controls")
    if getattr(args, 'use_wavelet_guided_semantic_state_organization', False) and (
        getattr(args, 'use_sam_guided_semantic_scanning', False)
        or getattr(args, 'use_sam_boundary_aware_state_propagation', False)
        or getattr(args, 'use_sam_state_reset_stronger', False)
        or getattr(args, 'use_sam_state_organizer_v1', False)
    ):
        raise ValueError("--use_wavelet_guided_semantic_state_organization cannot be combined with old standalone SAM state organizers")
    if getattr(args, 'use_dual_prototype_bank', False) and not fusion_only_ase_mainline:
        raise ValueError("--use_dual_prototype_bank requires ASE + fusion_only + fusion_residual (non-SAMASE, non-FASS)")
    if getattr(args, 'use_semantic_frequency_state_modulation', False) and not fusion_only_ase_mainline:
        raise ValueError("--use_semantic_frequency_state_modulation requires ASE + fusion_only + fusion_residual (non-SAMASE, non-FASS)")
    if getattr(args, 'use_dual_prototype_bank', False) and not getattr(args, 'use_offline_sam_cache', False):
        print("[DUAL-RPROTO] WARNING: --use_dual_prototype_bank is enabled but no --use_offline_sam_cache was provided; dual prototype conditioning will be unavailable at test time.")
    if getattr(args, 'use_semantic_frequency_state_modulation', False) and not getattr(args, 'use_offline_sam_cache', False):
        print("[SFM] WARNING: --use_semantic_frequency_state_modulation is enabled but no --use_offline_sam_cache was provided; semantic-frequency state modulation will be unavailable at test time.")
    if getattr(args, 'use_dual_prototype_bank', False) and not getattr(args, 'use_joint_spatial_spectral_wavelet_prior', False):
        raise ValueError("--use_dual_prototype_bank requires --use_joint_spatial_spectral_wavelet_prior")
    if getattr(args, 'use_semantic_frequency_state_modulation', False) and not getattr(args, 'use_joint_spatial_spectral_wavelet_prior', False):
        raise ValueError("--use_semantic_frequency_state_modulation requires --use_joint_spatial_spectral_wavelet_prior")

    use_wavelet_side_inputs = should_use_wavelet_priors(
        use_wavelet_legacy=getattr(args, 'use_wavelet', False),
        use_wavelet_priors=getattr(args, 'use_wavelet_priors', False),
        use_joint_spatial_spectral_wavelet_prior=getattr(args, 'use_joint_spatial_spectral_wavelet_prior', False),
        use_structure_guided_sam_ase=getattr(args, 'use_structure_guided_sam_ase', False),
        use_wavelet_local_bias=getattr(args, 'use_wavelet_local_bias', False),
        use_wavelet_local_gate=getattr(args, 'use_wavelet_local_gate', False),
        use_wavelet_guided_sam_prototype_scaling=getattr(args, 'use_wavelet_guided_sam_prototype_scaling', False),
        use_wavelet_guided_semantic_state_organization=getattr(args, 'use_wavelet_guided_semantic_state_organization', False),
        use_dual_prototype_bank=getattr(args, 'use_dual_prototype_bank', False),
        use_semantic_frequency_state_modulation=getattr(args, 'use_semantic_frequency_state_modulation', False),
        use_hf_wavelet_loss=getattr(args, 'use_hf_wavelet_loss', False),
        use_semantic_region_weighted_hf_wavelet_loss=getattr(args, 'use_semantic_region_weighted_hf_wavelet_loss', False),
        use_semantic_frequency_adaptive_scanning=getattr(args, 'use_semantic_frequency_adaptive_scanning', False),
    )

    model = Net(dim=args.channels,
                lr_hsi_dim=lr_hsi_dim,
                hr_msi_dim=hr_msi_dim,
                scale=args.ratio,
                use_ase=use_ase,
                ase_prompt_mode=getattr(args, 'ase_prompt_mode', 'hard'),
                ase_route_temperature=float(getattr(args, 'ase_route_temperature', 1.0)),
                ase_prompt_soft_mix=float(getattr(args, 'ase_prompt_soft_mix', 0.5)),
                ase_scope=getattr(args, 'ase_scope', 'all'),
                ase_stage_scope=getattr(args, 'ase_stage_scope', 'all_stages'),
                use_ase_fusion_residual=getattr(args, 'use_ase_fusion_residual', False),
                ase_fusion_res_scale=float(getattr(args, 'ase_fusion_res_scale', 0.3)),
                ase_stage_res_scales=getattr(args, 'ase_stage_res_scales', None),
                use_learnable_ase_fusion_res_scale=getattr(args, 'use_learnable_ase_fusion_res_scale', False),
                use_sam_ase=use_sam_ase,
                sam_checkpoint=sam_checkpoint,
                sam_prompt_dim=sam_prompt_dim,
                use_learnable_prompts=use_learnable_prompts,
                num_learnable_prompts=num_learnable_prompts,
                use_soft_masks=use_soft_masks,
                num_soft_regions=num_soft_regions,
                use_wavelet=args.use_wavelet,
                use_wavelet_priors=getattr(args, 'use_wavelet_priors', False),
                use_wavelet_local_bias=getattr(args, 'use_wavelet_local_bias', False),
                wavelet_local_bias_scale=float(getattr(args, 'wavelet_local_bias_scale', 0.1)),
                use_wavelet_local_gate=getattr(args, 'use_wavelet_local_gate', False),
                wavelet_local_gate_scale=float(getattr(args, 'wavelet_local_gate_scale', 0.1)),
                use_sam_local_gate=getattr(args, 'use_sam_local_gate', False),
                sam_local_gate_scale=float(getattr(args, 'sam_local_gate_scale', 0.1)),
                use_sam_semantic_prompt_bank=getattr(args, 'use_sam_semantic_prompt_bank', False),
                sam_semantic_prompt_bank_scale=float(getattr(args, 'sam_semantic_prompt_bank_scale', 0.1)),
                use_sam_region_prototype_bank=getattr(args, 'use_sam_region_prototype_bank', False),
                sam_region_prototype_bank_scale=float(getattr(args, 'sam_region_prototype_bank_scale', 0.1)),
                sam_region_prototype_count=int(getattr(args, 'sam_region_prototype_count', 8)),
                use_wavelet_guided_sam_prototype_scaling=getattr(args, 'use_wavelet_guided_sam_prototype_scaling', False),
                wavelet_guided_sam_prototype_scale=float(getattr(args, 'wavelet_guided_sam_prototype_scale', 0.1)),
                use_sam_region_prompt_mixture=getattr(args, 'use_sam_region_prompt_mixture', False),
                sam_region_prompt_mixture_scale=float(getattr(args, 'sam_region_prompt_mixture_scale', 0.05)),
                sam_region_prompt_mixture_count=int(getattr(args, 'sam_region_prompt_mixture_count', 8)),
                use_sam_guided_semantic_scanning=getattr(args, 'use_sam_guided_semantic_scanning', False),
                sam_semantic_scanning_count=int(getattr(args, 'sam_semantic_scanning_count', 6)),
                use_sam_feature_cluster_scanning=getattr(args, 'use_sam_feature_cluster_scanning', False),
                sam_feature_cluster_count=int(getattr(args, 'sam_feature_cluster_count', 6)),
                sam_feature_cluster_iters=int(getattr(args, 'sam_feature_cluster_iters', 2)),
                sam_feature_cluster_spatial_weight=float(getattr(args, 'sam_feature_cluster_spatial_weight', 0.05)),
                use_wavelet_augmented_ss1=getattr(args, 'use_wavelet_augmented_ss1', False),
                wavelet_augmented_ss1_count=int(getattr(args, 'wavelet_augmented_ss1_count', 6)),
                wavelet_augmented_ss1_topk_ratio=float(getattr(args, 'wavelet_augmented_ss1_topk_ratio', 0.25)),
                wavelet_augmented_ss1_strength=float(getattr(args, 'wavelet_augmented_ss1_strength', 0.5)),
                wavelet_augmented_ss1_mode=getattr(args, 'wavelet_augmented_ss1_mode', 'stable_intra_region'),
                use_sam_boundary_aware_state_propagation=getattr(args, 'use_sam_boundary_aware_state_propagation', False),
                sam_boundary_aware_state_scale=float(getattr(args, 'sam_boundary_aware_state_scale', 0.2)),
                use_sam_state_reset_stronger=getattr(args, 'use_sam_state_reset_stronger', False),
                sam_state_reset_scale=float(getattr(args, 'sam_state_reset_scale', 0.35)),
                use_sam_state_organizer_v1=getattr(args, 'use_sam_state_organizer_v1', False),
                sam_state_organizer_count=int(getattr(args, 'sam_state_organizer_count', 6)),
                sam_state_organizer_boundary_scale=float(getattr(args, 'sam_state_organizer_boundary_scale', 0.1)),
                sam_state_organizer_reset_scale=float(getattr(args, 'sam_state_organizer_reset_scale', 0.15)),
                use_sam_region_prompt_subspace=getattr(args, 'use_sam_region_prompt_subspace', False),
                sam_region_prompt_subspace_scale=float(getattr(args, 'sam_region_prompt_subspace_scale', 0.05)),
                sam_region_prompt_subspace_count=int(getattr(args, 'sam_region_prompt_subspace_count', 6)),
                use_wavelet_guided_semantic_state_organization=getattr(args, 'use_wavelet_guided_semantic_state_organization', False),
                wavelet_guided_semantic_state_count=int(getattr(args, 'wavelet_guided_semantic_state_count', 6)),
                wavelet_guided_semantic_state_scale=float(getattr(args, 'wavelet_guided_semantic_state_scale', 0.05)),
                wavelet_guided_semantic_boundary_scale=float(getattr(args, 'wavelet_guided_semantic_boundary_scale', 0.1)),
                wavelet_guided_semantic_reset_scale=float(getattr(args, 'wavelet_guided_semantic_reset_scale', 0.15)),
                use_joint_spatial_spectral_wavelet_prior=getattr(args, 'use_joint_spatial_spectral_wavelet_prior', False),
                joint_wavelet_spatial_weight=float(getattr(args, 'joint_wavelet_spatial_weight', 1.0)),
                joint_wavelet_spectral_weight=float(getattr(args, 'joint_wavelet_spectral_weight', 1.0)),
                use_dual_prototype_bank=getattr(args, 'use_dual_prototype_bank', False),
                dual_prototype_semantic_scale=float(getattr(args, 'dual_prototype_semantic_scale', 0.05)),
                dual_prototype_frequency_scale=float(getattr(args, 'dual_prototype_frequency_scale', 0.05)),
                dual_prototype_count=int(getattr(args, 'dual_prototype_count', 6)),
                use_semantic_frequency_state_modulation=getattr(args, 'use_semantic_frequency_state_modulation', False),
                semantic_frequency_state_count=int(getattr(args, 'semantic_frequency_state_count', 6)),
                semantic_frequency_state_write_scale=float(getattr(args, 'semantic_frequency_state_write_scale', 0.08)),
                semantic_frequency_state_read_scale=float(getattr(args, 'semantic_frequency_state_read_scale', 0.08)),
                semantic_frequency_state_delta_scale=float(getattr(args, 'semantic_frequency_state_delta_scale', 0.05)),
                use_structure_guided_sam_ase=use_structure_guided_sam_ase,
                structure_texture_weight=structure_texture_weight,
                use_fass=use_fass,
                fass_compression_ratio=fass_compression_ratio,
                fass_threshold=fass_threshold,
                fass_sparsity_target=fass_sparsity_target,
                fass_ll_sparsity=fass_ll_sparsity,
                fass_hf_sparsity=fass_hf_sparsity,
                fass_d_state=fass_d_state,
                train_mode=train_mode,
                dense_epochs=dense_epochs,
                gating_loss_weight=gating_loss_weight).to(args.device)
    configured_prior_c_refiner_modules = configure_prior_c_refiner_modules(model)
    if use_sam_prior_c_refiner and use_sam_ase and not use_fass and configured_prior_c_refiner_modules > 0:
        print(
            f"[SAM-PRIOR-C] enabled: region + prompt_strength -> prior_bias -> C, "
            f"scope={sam_prior_c_refiner_scope}, scale={sam_prior_c_refiner_scale}, "
            f"modules={configured_prior_c_refiner_modules}"
        )
    configure_fass_gating_modules(model)


    with torch.no_grad():


        if args.dataset.lower() == 'chikusei':

            sizes = [(64, 64), (128, 128)]
        else:

            sizes = [(64, 64), (128, 128), (256, 256)]

        print(f"使用以下尺寸初始化模型: {sizes}")

        for h, w in sizes:

            h_rounded = int(h // args.ratio * args.ratio)
            w_rounded = int(w // args.ratio * args.ratio)


            lr_h = int(h_rounded // args.ratio)
            lr_w = int(w_rounded // args.ratio)

            dummy_hr_msi = torch.randn(1, hr_msi_dim, h_rounded, w_rounded).to(args.device)
            dummy_lr_hsi = torch.randn(1, lr_hsi_dim, lr_h, lr_w).to(args.device)


            try:
                _ = model(dummy_hr_msi, dummy_lr_hsi)
                print(f"成功使用尺寸 {h_rounded}x{w_rounded} 初始化模型")
                break
            except Exception as e:
                print(f"使用尺寸 {h_rounded}x{w_rounded} 初始化模型时出错: {e}")


        print("模型初始化完成，Stage模块将在实际推理时自动初始化")


    if has_sam_ase_params and not use_sam_ase:
        print("警告：权重文件包含SAM-ASE参数，但模型未启用SAM-ASE。自动启用SAM-ASE以匹配权重文件")
        use_sam_ase = True
        use_ase = False
        model = Net(dim=args.channels,
                    lr_hsi_dim=lr_hsi_dim,
                    hr_msi_dim=hr_msi_dim,
                    scale=args.ratio,
                    use_ase=use_ase,
                    ase_prompt_mode=getattr(args, 'ase_prompt_mode', 'hard'),
                    ase_route_temperature=float(getattr(args, 'ase_route_temperature', 1.0)),
                    ase_prompt_soft_mix=float(getattr(args, 'ase_prompt_soft_mix', 0.5)),
                    ase_scope=getattr(args, 'ase_scope', 'all'),
                    ase_stage_scope=getattr(args, 'ase_stage_scope', 'all_stages'),
                    use_ase_fusion_residual=getattr(args, 'use_ase_fusion_residual', False),
                    ase_fusion_res_scale=float(getattr(args, 'ase_fusion_res_scale', 0.3)),
                    ase_stage_res_scales=getattr(args, 'ase_stage_res_scales', None),
                    use_learnable_ase_fusion_res_scale=getattr(args, 'use_learnable_ase_fusion_res_scale', False),
                    use_sam_ase=use_sam_ase,
                    sam_checkpoint=sam_checkpoint,
                    sam_prompt_dim=sam_prompt_dim,
                    use_learnable_prompts=use_learnable_prompts,
                    num_learnable_prompts=num_learnable_prompts,
                    use_soft_masks=use_soft_masks,
                    num_soft_regions=num_soft_regions,
                use_wavelet=args.use_wavelet,
                use_wavelet_priors=getattr(args, 'use_wavelet_priors', False),
                use_wavelet_local_bias=getattr(args, 'use_wavelet_local_bias', False),
                wavelet_local_bias_scale=float(getattr(args, 'wavelet_local_bias_scale', 0.1)),
                use_wavelet_local_gate=getattr(args, 'use_wavelet_local_gate', False),
                wavelet_local_gate_scale=float(getattr(args, 'wavelet_local_gate_scale', 0.1)),
                use_sam_local_gate=getattr(args, 'use_sam_local_gate', False),
                sam_local_gate_scale=float(getattr(args, 'sam_local_gate_scale', 0.1)),
                use_sam_semantic_prompt_bank=getattr(args, 'use_sam_semantic_prompt_bank', False),
                sam_semantic_prompt_bank_scale=float(getattr(args, 'sam_semantic_prompt_bank_scale', 0.1)),
                use_sam_region_prototype_bank=getattr(args, 'use_sam_region_prototype_bank', False),
                sam_region_prototype_bank_scale=float(getattr(args, 'sam_region_prototype_bank_scale', 0.1)),
                sam_region_prototype_count=int(getattr(args, 'sam_region_prototype_count', 8)),
                use_wavelet_guided_sam_prototype_scaling=getattr(args, 'use_wavelet_guided_sam_prototype_scaling', False),
                wavelet_guided_sam_prototype_scale=float(getattr(args, 'wavelet_guided_sam_prototype_scale', 0.1)),
                use_sam_region_prompt_mixture=getattr(args, 'use_sam_region_prompt_mixture', False),
                sam_region_prompt_mixture_scale=float(getattr(args, 'sam_region_prompt_mixture_scale', 0.05)),
                sam_region_prompt_mixture_count=int(getattr(args, 'sam_region_prompt_mixture_count', 8)),
                use_sam_guided_semantic_scanning=getattr(args, 'use_sam_guided_semantic_scanning', False),
                sam_semantic_scanning_count=int(getattr(args, 'sam_semantic_scanning_count', 6)),
                use_sam_feature_cluster_scanning=getattr(args, 'use_sam_feature_cluster_scanning', False),
                sam_feature_cluster_count=int(getattr(args, 'sam_feature_cluster_count', 6)),
                sam_feature_cluster_iters=int(getattr(args, 'sam_feature_cluster_iters', 2)),
                sam_feature_cluster_spatial_weight=float(getattr(args, 'sam_feature_cluster_spatial_weight', 0.05)),
                use_sam_boundary_aware_state_propagation=getattr(args, 'use_sam_boundary_aware_state_propagation', False),
                sam_boundary_aware_state_scale=float(getattr(args, 'sam_boundary_aware_state_scale', 0.2)),
                use_sam_state_reset_stronger=getattr(args, 'use_sam_state_reset_stronger', False),
                sam_state_reset_scale=float(getattr(args, 'sam_state_reset_scale', 0.35)),
                use_sam_state_organizer_v1=getattr(args, 'use_sam_state_organizer_v1', False),
                sam_state_organizer_count=int(getattr(args, 'sam_state_organizer_count', 6)),
                sam_state_organizer_boundary_scale=float(getattr(args, 'sam_state_organizer_boundary_scale', 0.1)),
                sam_state_organizer_reset_scale=float(getattr(args, 'sam_state_organizer_reset_scale', 0.15)),
                use_sam_region_prompt_subspace=getattr(args, 'use_sam_region_prompt_subspace', False),
                sam_region_prompt_subspace_scale=float(getattr(args, 'sam_region_prompt_subspace_scale', 0.05)),
                sam_region_prompt_subspace_count=int(getattr(args, 'sam_region_prompt_subspace_count', 6)),
                use_wavelet_guided_semantic_state_organization=getattr(args, 'use_wavelet_guided_semantic_state_organization', False),
                wavelet_guided_semantic_state_count=int(getattr(args, 'wavelet_guided_semantic_state_count', 6)),
                wavelet_guided_semantic_state_scale=float(getattr(args, 'wavelet_guided_semantic_state_scale', 0.05)),
                wavelet_guided_semantic_boundary_scale=float(getattr(args, 'wavelet_guided_semantic_boundary_scale', 0.1)),
                wavelet_guided_semantic_reset_scale=float(getattr(args, 'wavelet_guided_semantic_reset_scale', 0.15)),
                use_joint_spatial_spectral_wavelet_prior=getattr(args, 'use_joint_spatial_spectral_wavelet_prior', False),
                joint_wavelet_spatial_weight=float(getattr(args, 'joint_wavelet_spatial_weight', 1.0)),
                joint_wavelet_spectral_weight=float(getattr(args, 'joint_wavelet_spectral_weight', 1.0)),
                use_dual_prototype_bank=getattr(args, 'use_dual_prototype_bank', False),
                dual_prototype_semantic_scale=float(getattr(args, 'dual_prototype_semantic_scale', 0.05)),
                dual_prototype_frequency_scale=float(getattr(args, 'dual_prototype_frequency_scale', 0.05)),
                dual_prototype_count=int(getattr(args, 'dual_prototype_count', 6)),
                use_semantic_frequency_state_modulation=getattr(args, 'use_semantic_frequency_state_modulation', False),
                semantic_frequency_state_count=int(getattr(args, 'semantic_frequency_state_count', 6)),
                semantic_frequency_state_write_scale=float(getattr(args, 'semantic_frequency_state_write_scale', 0.08)),
                semantic_frequency_state_read_scale=float(getattr(args, 'semantic_frequency_state_read_scale', 0.08)),
                semantic_frequency_state_delta_scale=float(getattr(args, 'semantic_frequency_state_delta_scale', 0.05)),
                use_structure_guided_sam_ase=use_structure_guided_sam_ase,
                structure_texture_weight=structure_texture_weight,
                    use_fass=use_fass,
                    fass_compression_ratio=fass_compression_ratio,
                    fass_threshold=fass_threshold,
                    fass_sparsity_target=fass_sparsity_target,
                    fass_ll_sparsity=fass_ll_sparsity,
                    fass_hf_sparsity=fass_hf_sparsity,
                    fass_d_state=fass_d_state,
                    train_mode=train_mode,
                    dense_epochs=dense_epochs,
                    gating_loss_weight=gating_loss_weight).to(args.device)
        configure_fass_gating_modules(model)


    elif has_ase_params and not use_ase and not has_sam_ase_params:
        print("警告：权重文件包含ASE参数，但模型未启用ASE。自动启用ASE以匹配权重文件")
        use_ase = True
        model = Net(dim=args.channels,
                    lr_hsi_dim=lr_hsi_dim,
                    hr_msi_dim=hr_msi_dim,
                    scale=args.ratio,
                    use_ase=use_ase,
                    ase_prompt_mode=getattr(args, 'ase_prompt_mode', 'hard'),
                    ase_route_temperature=float(getattr(args, 'ase_route_temperature', 1.0)),
                    ase_prompt_soft_mix=float(getattr(args, 'ase_prompt_soft_mix', 0.5)),
                    ase_scope=getattr(args, 'ase_scope', 'all'),
                    ase_stage_scope=getattr(args, 'ase_stage_scope', 'all_stages'),
                    use_ase_fusion_residual=getattr(args, 'use_ase_fusion_residual', False),
                    ase_fusion_res_scale=float(getattr(args, 'ase_fusion_res_scale', 0.3)),
                    ase_stage_res_scales=getattr(args, 'ase_stage_res_scales', None),
                    use_learnable_ase_fusion_res_scale=getattr(args, 'use_learnable_ase_fusion_res_scale', False),
                    use_learnable_prompts=use_learnable_prompts,
                    num_learnable_prompts=num_learnable_prompts,
                    use_soft_masks=use_soft_masks,
                    num_soft_regions=num_soft_regions,
                    use_wavelet=args.use_wavelet,
                    use_wavelet_priors=getattr(args, 'use_wavelet_priors', False),
                    use_wavelet_local_bias=getattr(args, 'use_wavelet_local_bias', False),
                    wavelet_local_bias_scale=float(getattr(args, 'wavelet_local_bias_scale', 0.1)),
                    use_wavelet_local_gate=getattr(args, 'use_wavelet_local_gate', False),
                    wavelet_local_gate_scale=float(getattr(args, 'wavelet_local_gate_scale', 0.1)),
                    use_sam_local_gate=getattr(args, 'use_sam_local_gate', False),
                    sam_local_gate_scale=float(getattr(args, 'sam_local_gate_scale', 0.1)),
                    use_sam_semantic_prompt_bank=getattr(args, 'use_sam_semantic_prompt_bank', False),
                    sam_semantic_prompt_bank_scale=float(getattr(args, 'sam_semantic_prompt_bank_scale', 0.1)),
                    use_sam_region_prototype_bank=getattr(args, 'use_sam_region_prototype_bank', False),
                    sam_region_prototype_bank_scale=float(getattr(args, 'sam_region_prototype_bank_scale', 0.1)),
                    sam_region_prototype_count=int(getattr(args, 'sam_region_prototype_count', 8)),
                    use_wavelet_guided_sam_prototype_scaling=getattr(args, 'use_wavelet_guided_sam_prototype_scaling', False),
                    wavelet_guided_sam_prototype_scale=float(getattr(args, 'wavelet_guided_sam_prototype_scale', 0.1)),
                    use_sam_region_prompt_mixture=getattr(args, 'use_sam_region_prompt_mixture', False),
                    sam_region_prompt_mixture_scale=float(getattr(args, 'sam_region_prompt_mixture_scale', 0.05)),
                    sam_region_prompt_mixture_count=int(getattr(args, 'sam_region_prompt_mixture_count', 8)),
                    use_sam_guided_semantic_scanning=getattr(args, 'use_sam_guided_semantic_scanning', False),
                    sam_semantic_scanning_count=int(getattr(args, 'sam_semantic_scanning_count', 6)),
                    use_sam_feature_cluster_scanning=getattr(args, 'use_sam_feature_cluster_scanning', False),
                    sam_feature_cluster_count=int(getattr(args, 'sam_feature_cluster_count', 6)),
                    sam_feature_cluster_iters=int(getattr(args, 'sam_feature_cluster_iters', 2)),
                    sam_feature_cluster_spatial_weight=float(getattr(args, 'sam_feature_cluster_spatial_weight', 0.05)),
                    use_sam_boundary_aware_state_propagation=getattr(args, 'use_sam_boundary_aware_state_propagation', False),
                    sam_boundary_aware_state_scale=float(getattr(args, 'sam_boundary_aware_state_scale', 0.2)),
                    use_sam_state_reset_stronger=getattr(args, 'use_sam_state_reset_stronger', False),
                    sam_state_reset_scale=float(getattr(args, 'sam_state_reset_scale', 0.35)),
                    use_sam_state_organizer_v1=getattr(args, 'use_sam_state_organizer_v1', False),
                    sam_state_organizer_count=int(getattr(args, 'sam_state_organizer_count', 6)),
                    sam_state_organizer_boundary_scale=float(getattr(args, 'sam_state_organizer_boundary_scale', 0.1)),
                    sam_state_organizer_reset_scale=float(getattr(args, 'sam_state_organizer_reset_scale', 0.15)),
                    use_sam_region_prompt_subspace=getattr(args, 'use_sam_region_prompt_subspace', False),
                    sam_region_prompt_subspace_scale=float(getattr(args, 'sam_region_prompt_subspace_scale', 0.05)),
                    sam_region_prompt_subspace_count=int(getattr(args, 'sam_region_prompt_subspace_count', 6)),
                use_wavelet_guided_semantic_state_organization=getattr(args, 'use_wavelet_guided_semantic_state_organization', False),
                wavelet_guided_semantic_state_count=int(getattr(args, 'wavelet_guided_semantic_state_count', 6)),
                wavelet_guided_semantic_state_scale=float(getattr(args, 'wavelet_guided_semantic_state_scale', 0.05)),
                wavelet_guided_semantic_boundary_scale=float(getattr(args, 'wavelet_guided_semantic_boundary_scale', 0.1)),
                wavelet_guided_semantic_reset_scale=float(getattr(args, 'wavelet_guided_semantic_reset_scale', 0.15)),
                use_joint_spatial_spectral_wavelet_prior=getattr(args, 'use_joint_spatial_spectral_wavelet_prior', False),
                joint_wavelet_spatial_weight=float(getattr(args, 'joint_wavelet_spatial_weight', 1.0)),
                joint_wavelet_spectral_weight=float(getattr(args, 'joint_wavelet_spectral_weight', 1.0)),
                use_dual_prototype_bank=getattr(args, 'use_dual_prototype_bank', False),
                dual_prototype_semantic_scale=float(getattr(args, 'dual_prototype_semantic_scale', 0.05)),
                dual_prototype_frequency_scale=float(getattr(args, 'dual_prototype_frequency_scale', 0.05)),
                dual_prototype_count=int(getattr(args, 'dual_prototype_count', 6)),
                use_semantic_frequency_state_modulation=getattr(args, 'use_semantic_frequency_state_modulation', False),
                semantic_frequency_state_count=int(getattr(args, 'semantic_frequency_state_count', 6)),
                semantic_frequency_state_write_scale=float(getattr(args, 'semantic_frequency_state_write_scale', 0.08)),
                semantic_frequency_state_read_scale=float(getattr(args, 'semantic_frequency_state_read_scale', 0.08)),
                semantic_frequency_state_delta_scale=float(getattr(args, 'semantic_frequency_state_delta_scale', 0.05)),
                use_structure_guided_sam_ase=use_structure_guided_sam_ase,
                    structure_texture_weight=structure_texture_weight,
                    use_fass=use_fass,
                    fass_compression_ratio=fass_compression_ratio,
                    fass_threshold=fass_threshold,
                    fass_sparsity_target=fass_sparsity_target,
                    fass_ll_sparsity=fass_ll_sparsity,
                    fass_hf_sparsity=fass_hf_sparsity,
                    fass_d_state=fass_d_state,
                    train_mode=train_mode,
                    dense_epochs=dense_epochs,
                    gating_loss_weight=gating_loss_weight).to(args.device)
        configure_fass_gating_modules(model)


    elif not has_ase_params and not has_sam_ase_params and (use_ase or use_sam_ase):
        print("警告：权重文件不包含ASE/SAM-ASE参数，但模型启用了ASE/SAM-ASE。禁用ASE/SAM-ASE以匹配权重文件")
        use_ase = False
        use_sam_ase = False
        model = Net(dim=args.channels,
                    lr_hsi_dim=lr_hsi_dim,
                    hr_msi_dim=hr_msi_dim,
                    scale=args.ratio,
                    use_ase=use_ase,
                    ase_prompt_mode=getattr(args, 'ase_prompt_mode', 'hard'),
                    ase_route_temperature=float(getattr(args, 'ase_route_temperature', 1.0)),
                    ase_prompt_soft_mix=float(getattr(args, 'ase_prompt_soft_mix', 0.5)),
                    ase_scope=getattr(args, 'ase_scope', 'all'),
                    ase_stage_scope=getattr(args, 'ase_stage_scope', 'all_stages'),
                    use_ase_fusion_residual=getattr(args, 'use_ase_fusion_residual', False),
                    ase_fusion_res_scale=float(getattr(args, 'ase_fusion_res_scale', 0.3)),
                    ase_stage_res_scales=getattr(args, 'ase_stage_res_scales', None),
                    use_learnable_ase_fusion_res_scale=getattr(args, 'use_learnable_ase_fusion_res_scale', False),
                    use_sam_ase=use_sam_ase,
                    sam_checkpoint=sam_checkpoint,
                    sam_prompt_dim=sam_prompt_dim,
                    use_learnable_prompts=use_learnable_prompts,
                    num_learnable_prompts=num_learnable_prompts,
                    use_soft_masks=use_soft_masks,
                    num_soft_regions=num_soft_regions,
                    use_wavelet=args.use_wavelet,
                    use_wavelet_priors=getattr(args, 'use_wavelet_priors', False),
                    use_wavelet_local_bias=getattr(args, 'use_wavelet_local_bias', False),
                    wavelet_local_bias_scale=float(getattr(args, 'wavelet_local_bias_scale', 0.1)),
                    use_wavelet_local_gate=getattr(args, 'use_wavelet_local_gate', False),
                    wavelet_local_gate_scale=float(getattr(args, 'wavelet_local_gate_scale', 0.1)),
                    use_sam_local_gate=getattr(args, 'use_sam_local_gate', False),
                    sam_local_gate_scale=float(getattr(args, 'sam_local_gate_scale', 0.1)),
                    use_sam_semantic_prompt_bank=getattr(args, 'use_sam_semantic_prompt_bank', False),
                    sam_semantic_prompt_bank_scale=float(getattr(args, 'sam_semantic_prompt_bank_scale', 0.1)),
                    use_sam_region_prototype_bank=getattr(args, 'use_sam_region_prototype_bank', False),
                    sam_region_prototype_bank_scale=float(getattr(args, 'sam_region_prototype_bank_scale', 0.1)),
                    sam_region_prototype_count=int(getattr(args, 'sam_region_prototype_count', 8)),
                    use_wavelet_guided_sam_prototype_scaling=getattr(args, 'use_wavelet_guided_sam_prototype_scaling', False),
                    wavelet_guided_sam_prototype_scale=float(getattr(args, 'wavelet_guided_sam_prototype_scale', 0.1)),
                    use_sam_region_prompt_mixture=getattr(args, 'use_sam_region_prompt_mixture', False),
                    sam_region_prompt_mixture_scale=float(getattr(args, 'sam_region_prompt_mixture_scale', 0.05)),
                    sam_region_prompt_mixture_count=int(getattr(args, 'sam_region_prompt_mixture_count', 8)),
                    use_sam_guided_semantic_scanning=getattr(args, 'use_sam_guided_semantic_scanning', False),
                    sam_semantic_scanning_count=int(getattr(args, 'sam_semantic_scanning_count', 6)),
                    use_sam_feature_cluster_scanning=getattr(args, 'use_sam_feature_cluster_scanning', False),
                    sam_feature_cluster_count=int(getattr(args, 'sam_feature_cluster_count', 6)),
                    sam_feature_cluster_iters=int(getattr(args, 'sam_feature_cluster_iters', 2)),
                    sam_feature_cluster_spatial_weight=float(getattr(args, 'sam_feature_cluster_spatial_weight', 0.05)),
                    use_sam_boundary_aware_state_propagation=getattr(args, 'use_sam_boundary_aware_state_propagation', False),
                    sam_boundary_aware_state_scale=float(getattr(args, 'sam_boundary_aware_state_scale', 0.2)),
                    use_sam_state_reset_stronger=getattr(args, 'use_sam_state_reset_stronger', False),
                    sam_state_reset_scale=float(getattr(args, 'sam_state_reset_scale', 0.35)),
                    use_sam_state_organizer_v1=getattr(args, 'use_sam_state_organizer_v1', False),
                    sam_state_organizer_count=int(getattr(args, 'sam_state_organizer_count', 6)),
                    sam_state_organizer_boundary_scale=float(getattr(args, 'sam_state_organizer_boundary_scale', 0.1)),
                    sam_state_organizer_reset_scale=float(getattr(args, 'sam_state_organizer_reset_scale', 0.15)),
                    use_sam_region_prompt_subspace=getattr(args, 'use_sam_region_prompt_subspace', False),
                    sam_region_prompt_subspace_scale=float(getattr(args, 'sam_region_prompt_subspace_scale', 0.05)),
                    sam_region_prompt_subspace_count=int(getattr(args, 'sam_region_prompt_subspace_count', 6)),
                use_wavelet_guided_semantic_state_organization=getattr(args, 'use_wavelet_guided_semantic_state_organization', False),
                wavelet_guided_semantic_state_count=int(getattr(args, 'wavelet_guided_semantic_state_count', 6)),
                wavelet_guided_semantic_state_scale=float(getattr(args, 'wavelet_guided_semantic_state_scale', 0.05)),
                wavelet_guided_semantic_boundary_scale=float(getattr(args, 'wavelet_guided_semantic_boundary_scale', 0.1)),
                wavelet_guided_semantic_reset_scale=float(getattr(args, 'wavelet_guided_semantic_reset_scale', 0.15)),
                use_joint_spatial_spectral_wavelet_prior=getattr(args, 'use_joint_spatial_spectral_wavelet_prior', False),
                joint_wavelet_spatial_weight=float(getattr(args, 'joint_wavelet_spatial_weight', 1.0)),
                joint_wavelet_spectral_weight=float(getattr(args, 'joint_wavelet_spectral_weight', 1.0)),
                use_dual_prototype_bank=getattr(args, 'use_dual_prototype_bank', False),
                dual_prototype_semantic_scale=float(getattr(args, 'dual_prototype_semantic_scale', 0.05)),
                dual_prototype_frequency_scale=float(getattr(args, 'dual_prototype_frequency_scale', 0.05)),
                dual_prototype_count=int(getattr(args, 'dual_prototype_count', 6)),
                use_semantic_frequency_state_modulation=getattr(args, 'use_semantic_frequency_state_modulation', False),
                semantic_frequency_state_count=int(getattr(args, 'semantic_frequency_state_count', 6)),
                semantic_frequency_state_write_scale=float(getattr(args, 'semantic_frequency_state_write_scale', 0.08)),
                semantic_frequency_state_read_scale=float(getattr(args, 'semantic_frequency_state_read_scale', 0.08)),
                semantic_frequency_state_delta_scale=float(getattr(args, 'semantic_frequency_state_delta_scale', 0.05)),
                use_structure_guided_sam_ase=use_structure_guided_sam_ase,
                    structure_texture_weight=structure_texture_weight,
                    use_fass=use_fass,
                    fass_compression_ratio=fass_compression_ratio,
                    fass_threshold=fass_threshold,
                    fass_sparsity_target=fass_sparsity_target,
                    fass_ll_sparsity=fass_ll_sparsity,
                    fass_hf_sparsity=fass_hf_sparsity,
                    fass_d_state=fass_d_state,
                    train_mode=train_mode,
                    dense_epochs=dense_epochs,
                    gating_loss_weight=gating_loss_weight).to(args.device)
        configure_fass_gating_modules(model)


    try:

        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint


        filtered_state_dict = {}
        sam_filtered_count = 0
        for key, value in state_dict.items():

            if 'sam_model' in key or 'sam_extractor.sam_model' in key:
                sam_filtered_count += 1
                continue

            filtered_state_dict[key] = value

        if sam_filtered_count > 0:
            print(f"[INFO] 过滤掉 {sam_filtered_count} 个SAM模型参数（SAM应该使用本地权重）")


        model.load_state_dict(filtered_state_dict, strict=False)
        print(f"[SUCCESS] 模型权重加载成功")

    except Exception as e:
        print(f"Error loading model: {e}")

        model_dict = model.state_dict()
        pretrained_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        model_keys = set(model_dict.keys())
        pretrained_keys = set(pretrained_dict.keys())

        missing_keys = model_keys - pretrained_keys
        unexpected_keys = pretrained_keys - model_keys

        print(f"Missing keys in checkpoint: {len(missing_keys)}")
        if missing_keys:
            print("Sample missing keys:", list(missing_keys)[:5])

        print(f"Unexpected keys in checkpoint: {len(unexpected_keys)}")
        if unexpected_keys:
            print("Sample unexpected keys:", list(unexpected_keys)[:5])


        shape_mismatch_keys = []
        for k in model_keys & pretrained_keys:
            if model_dict[k].shape != pretrained_dict[k].shape:
                shape_mismatch_keys.append((k, model_dict[k].shape, pretrained_dict[k].shape))

        print(f"Shape mismatch keys: {len(shape_mismatch_keys)}")
        if shape_mismatch_keys:
            print("Sample shape mismatch keys:", shape_mismatch_keys[:3])


        print("Attempting partial load...")
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Partial load successful: {len(pretrained_dict)}/{len(model.state_dict())} parameters loaded")


        if len(unexpected_keys) > 100:
            print("Attempting to load with strict=False due to many unexpected keys...")
            try:
                model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint, strict=False)
                print("Model loaded with strict=False (some parameters ignored)")


                loaded_params = len([k for k in model.state_dict().keys() if any(k.startswith(uk.split('.')[0]) for uk in pretrained_dict.keys())])
                print(f"Approximately {loaded_params} parameters may have been loaded")


                if loaded_params < 100:
                    print("Detailed parameter analysis:")
                    model_param_prefixes = set(['.'.join(k.split('.')[:2]) for k in model_dict.keys()])
                    checkpoint_param_prefixes = set(['.'.join(k.split('.')[:2]) for k in pretrained_dict.keys()])

                    print("Model parameter prefixes:", sorted(list(model_param_prefixes))[:10])
                    print("Checkpoint parameter prefixes:", sorted(list(checkpoint_param_prefixes))[:10])


                    common_prefixes = model_param_prefixes & checkpoint_param_prefixes
                    print(f"Common parameter prefixes: {len(common_prefixes)}")
                    if common_prefixes:
                        print("Sample common prefixes:", sorted(list(common_prefixes))[:5])

                    model_only_prefixes = model_param_prefixes - checkpoint_param_prefixes
                    checkpoint_only_prefixes = checkpoint_param_prefixes - model_param_prefixes
                    print(f"Model-only prefixes: {len(model_only_prefixes)}")
                    print(f"Checkpoint-only prefixes: {len(checkpoint_only_prefixes)}")

                    if checkpoint_only_prefixes:
                        print("Sample checkpoint-only prefixes:", sorted(list(checkpoint_only_prefixes))[:5])
            except Exception as e2:
                print(f"Failed to load with strict=False: {e2}")
                print("Using partial load result")

    model.eval()
    model_param_stats = _count_parameters(model)
    resolved_sample_names = sample_names or [f'sample_{idx:04d}' for idx in range(len(lr_hsi))]
    per_sample_records = []

    dataset_info = {
        'dataset': args.dataset,
        'ratio': int(args.ratio),
        'h5_path': str(Path(args.h5_path).resolve()) if args.h5_path else None,
        'sam_cache_path': str(Path(cache_path).resolve()) if getattr(args, 'use_offline_sam_cache', False) and cache_path else None,
        'sample_count': int(len(lr_hsi)),
        'sample_names': resolved_sample_names,
        'tensor_shapes': {
            'lr_hsi_batch': list(lr_hsi.shape),
            'hr_msi_batch': list(hr_msi.shape),
            'hr_hsi_batch': list(hr_hsi.shape),
        },
        'channel_config': {
            'lr_hsi_dim': int(lr_hsi_dim),
            'hr_msi_dim': int(hr_msi_dim),
        },
        'block_processing': {
            'cut_size': int(args.cut_size),
            'pad': int(args.pad),
        },
    }


    ssim_values = []
    sam_values = []
    rmse_values = []
    psnr_values = []
    ergas_values = []

    with torch.no_grad():
        for i in range(len(lr_hsi)):
            print(f"Sample {i+1}/{len(lr_hsi)}")
            sample_name = resolved_sample_names[i]
            cur_cached_sam_features = None if cached_sam_feature_list is None else cached_sam_feature_list[i].unsqueeze(0)
            cur_cached_sam_masks = None if cached_sam_mask_list is None else cached_sam_mask_list[i].unsqueeze(0)


            cur_lr = lr_hsi[i].unsqueeze(0)
            cur_hr = hr_msi[i].unsqueeze(0)
            gt = hr_hsi[i].unsqueeze(0)


            _, _, H_lr, W_lr = cur_lr.shape
            _, _, H, W = cur_hr.shape


            cut_size = args.cut_size
            pad = args.pad
            eval_mode = 'block' if (H > cut_size or W > cut_size) else 'full'
            use_overlap_blend = bool(getattr(args, 'use_overlap_blend', False))
            _sync_device(args.device)
            inference_start_time = time.perf_counter()

            if use_overlap_blend and (H > cut_size or W > cut_size):
                print(f"  Processing large image ({H}x{W}) with overlap-blend blocks of {cut_size}x{cut_size}")

                if pad % args.ratio != 0:
                    raise ValueError(f"pad ({pad}) must be divisible by ratio ({args.ratio}) for block inference")

                lr_msi_size = cut_size // args.ratio
                block_stride = args.block_stride if args.block_stride is not None else max(1, cut_size - 2 * pad)
                if block_stride <= 0 or block_stride > cut_size:
                    raise ValueError(f"invalid block_stride={block_stride}; expected 1..{cut_size}")
                if block_stride % args.ratio != 0:
                    raise ValueError(
                        f"block_stride ({block_stride}) must be divisible by ratio ({args.ratio}) for overlap blending"
                    )

                h_positions = _compute_block_positions(H, cut_size, block_stride)
                w_positions = _compute_block_positions(W, cut_size, block_stride)
                total_blocks = len(h_positions) * len(w_positions)
                processed_blocks = 0

                cur_hr_pad = torch.zeros(1, hr_msi_dim, H + 2 * pad, W + 2 * pad, device=args.device)
                cur_lr_pad = torch.zeros(
                    1,
                    lr_hsi_dim,
                    (H + 2 * pad) // args.ratio,
                    (W + 2 * pad) // args.ratio,
                    device=args.device,
                )
                cur_hr_pad[:, :, pad:pad + H, pad:pad + W] = cur_hr
                cur_lr_pad[
                    :,
                    :,
                    pad // args.ratio:(pad + H) // args.ratio,
                    pad // args.ratio:(pad + W) // args.ratio,
                ] = cur_lr

                output_sum = torch.zeros(1, lr_hsi_dim, H, W, device=args.device)
                weight_sum = torch.zeros(1, 1, H, W, device=args.device)
                blend_weight = _build_blend_window(
                    cut_size,
                    getattr(args, 'blend_window', 'hann'),
                    device=args.device,
                    dtype=cur_hr.dtype,
                )
                routing_probs = None

                for hr_start_h in h_positions:
                    for hr_start_w in w_positions:
                        processed_blocks += 1
                        if total_blocks > 10:
                            progress = (processed_blocks / total_blocks) * 100
                            print(f"  Processing block {processed_blocks}/{total_blocks} ({progress:.1f}%)", end='\r')

                        try:
                            hr_end_h = hr_start_h + cut_size + 2 * pad
                            hr_end_w = hr_start_w + cut_size + 2 * pad
                            lr_start_h = hr_start_h // args.ratio
                            lr_start_w = hr_start_w // args.ratio
                            lr_end_h = lr_start_h + lr_msi_size + 2 * (pad // args.ratio)
                            lr_end_w = lr_start_w + lr_msi_size + 2 * (pad // args.ratio)

                            hr_block = cur_hr_pad[:, :, hr_start_h:hr_end_h, hr_start_w:hr_end_w]
                            lr_block = cur_lr_pad[:, :, lr_start_h:lr_end_h, lr_start_w:lr_end_w]

                            lr_block_approx = None
                            lr_block_details = None
                            if use_wavelet_side_inputs:
                                lr_block_approx, lr_block_details = build_haar_wavelet_coeffs(lr_block)
                            block_output = model(
                                hr_block,
                                lr_block,
                                lr_hsi_approx=lr_block_approx,
                                lr_hsi_details=lr_block_details,
                                cached_sam_features=cur_cached_sam_features,
                                cached_sam_masks=cur_cached_sam_masks,
                            )

                            if isinstance(block_output, tuple):
                                block_output = block_output[0]
                            block_output = torch.clamp(block_output, 0, 1)
                            block_result = block_output[:, :, pad:pad + cut_size, pad:pad + cut_size]

                            valid_h = min(cut_size, H - hr_start_h)
                            valid_w = min(cut_size, W - hr_start_w)
                            block_result = block_result[:, :, :valid_h, :valid_w]
                            block_weight = blend_weight[:, :, :valid_h, :valid_w]

                            output_sum[:, :, hr_start_h:hr_start_h + valid_h, hr_start_w:hr_start_w + valid_w] += (
                                block_result * block_weight
                            )
                            weight_sum[:, :, hr_start_h:hr_start_h + valid_h, hr_start_w:hr_start_w + valid_w] += (
                                block_weight
                            )
                        except Exception as block_e:
                            print(f"\n  Error processing block ({hr_start_h}, {hr_start_w}): {block_e}")
                            print(f"    Block dimensions - HR: {hr_block.shape}, LR: {lr_block.shape}")

                if total_blocks > 10:
                    print()

                output = output_sum / weight_sum.clamp(min=1e-6)
                _sync_device(args.device)
                inference_time_sec = time.perf_counter() - inference_start_time

                ssim = compute_ssim(output, gt)
                sam = compute_sam(output, gt)
                rmse = compute_rmse(output, gt)
                psnr = compute_psnr(output, gt)
                ergas = ERGAS(ratio=args.ratio)(output, gt)

                ssim_values.append(ssim.item())
                sam_values.append(sam.item())
                rmse_values.append(rmse.item())
                psnr_values.append(psnr.item())
                ergas_values.append(ergas.item())

                print(f'SSIM: {ssim.item():.4f}, SAM: {sam.item():.4f} deg, RMSE: {rmse.item():.4f}, PSNR: {psnr.item():.4f} dB, ERGAS: {ergas.item():.4f}')

                save_path = os.path.join(args.save_dir, f'result_{i}.mat')
                save_matv73(save_path, 'result', output.squeeze(0).cpu().numpy())
                viz_path = visualize_results(output.squeeze(0), gt.squeeze(0), i, args.save_dir, args.dataset)

                per_sample_records.append({
                    'sample_index': int(i),
                    'sample_name': sample_name,
                    'metrics': {
                        'ssim': float(ssim.item()),
                        'sam_deg': float(sam.item()),
                        'rmse': float(rmse.item()),
                        'psnr_db': float(psnr.item()),
                        'ergas': float(ergas.item()),
                    },
                    'runtime': {
                        'inference_time_sec': float(inference_time_sec),
                        'eval_mode': 'block_overlap_blend',
                    },
                    'shapes': {
                        'hr_hsi': list(gt.squeeze(0).shape),
                        'lr_hsi': list(cur_lr.squeeze(0).shape),
                        'hr_msi': list(cur_hr.squeeze(0).shape),
                        'prediction': list(output.squeeze(0).shape),
                    },
                    'paths': {
                        'result_mat': str(Path(save_path).resolve()),
                        'visualization': str(Path(viz_path).resolve()) if viz_path else None,
                    },
                    'routing': _serialize_routing_probs(routing_probs),
                })
                continue


            if H > cut_size or W > cut_size:
                print(f"  Processing large image ({H}x{W}) in blocks of {cut_size}x{cut_size}")


                lr_msi_size = cut_size // args.ratio


                scale_H = (H - 1) // cut_size + 1
                scale_W = (W - 1) // cut_size + 1


                new_H = scale_H * cut_size
                new_W = scale_W * cut_size


                cur_hr_pad = torch.zeros(1, hr_msi_dim, new_H + 2 * pad, new_W + 2 * pad).to(args.device)
                cur_lr_pad = torch.zeros(1, lr_hsi_dim, (new_H + 2 * pad) // args.ratio, (new_W + 2 * pad) // args.ratio).to(args.device)


                cur_hr_pad[:, :, pad:pad + H, pad:pad + W] = cur_hr
                cur_lr_pad[:, :, pad // args.ratio:(pad + H) // args.ratio, pad // args.ratio:(pad + W) // args.ratio] = cur_lr


                output = torch.zeros(1, lr_hsi_dim, new_H, new_W).to(args.device)
                routing_probs = None


                total_blocks = scale_H * scale_W
                processed_blocks = 0

                for h_idx in range(scale_H):
                    for w_idx in range(scale_W):
                        processed_blocks += 1


                        if total_blocks > 10:
                            progress = (processed_blocks / total_blocks) * 100
                            print(f"  Processing block {processed_blocks}/{total_blocks} ({progress:.1f}%)", end='\r')

                        try:


                            hr_start_h = h_idx * cut_size
                            hr_end_h = (h_idx + 1) * cut_size + 2 * pad
                            hr_start_w = w_idx * cut_size
                            hr_end_w = (w_idx + 1) * cut_size + 2 * pad


                            lr_start_h = h_idx * lr_msi_size
                            lr_end_h = (h_idx + 1) * lr_msi_size + 2 * (pad // args.ratio)
                            lr_start_w = w_idx * lr_msi_size
                            lr_end_w = (w_idx + 1) * lr_msi_size + 2 * (pad // args.ratio)


                            hr_block = cur_hr_pad[:, :, hr_start_h:hr_end_h, hr_start_w:hr_end_w]
                            lr_block = cur_lr_pad[:, :, lr_start_h:lr_end_h, lr_start_w:lr_end_w]


                            lr_block_approx = None
                            lr_block_details = None
                            if use_wavelet_side_inputs:
                                lr_block_approx, lr_block_details = build_haar_wavelet_coeffs(lr_block)
                            block_output = model(
                                hr_block,
                                lr_block,
                                lr_hsi_approx=lr_block_approx,
                                lr_hsi_details=lr_block_details,
                                cached_sam_features=cur_cached_sam_features,
                                cached_sam_masks=cur_cached_sam_masks,
                            )


                            if isinstance(block_output, tuple):
                                block_output = block_output[0]


                            block_output = torch.clamp(block_output, 0, 1)


                            output_start_h = h_idx * cut_size
                            output_end_h = (h_idx + 1) * cut_size
                            output_start_w = w_idx * cut_size
                            output_end_w = (w_idx + 1) * cut_size


                            block_result = block_output[:, :, pad:pad + cut_size, pad:pad + cut_size]


                            output[:, :, output_start_h:output_end_h, output_start_w:output_end_w] = block_result
                        except Exception as block_e:
                            print(f"\n  Error processing block ({h_idx}, {w_idx}): {block_e}")
                            print(f"    Block dimensions - HR: {hr_block.shape}, LR: {lr_block.shape}")

                            output_start_h = h_idx * cut_size
                            output_end_h = min((h_idx + 1) * cut_size, new_H)
                            output_start_w = w_idx * cut_size
                            output_end_w = min((w_idx + 1) * cut_size, new_W)
                            output[:, :, output_start_h:output_end_h, output_start_w:output_end_w] = 0


                if total_blocks > 10:
                    print()


                output = output[:, :, :H, :W]
            else:

                cur_lr_approx = None
                cur_lr_details = None
                if use_wavelet_side_inputs:
                    cur_lr_approx, cur_lr_details = build_haar_wavelet_coeffs(cur_lr)
                model_output = model(
                    cur_hr,
                    cur_lr,
                    lr_hsi_approx=cur_lr_approx,
                    lr_hsi_details=cur_lr_details,
                    cached_sam_features=cur_cached_sam_features,
                    cached_sam_masks=cur_cached_sam_masks,
                )


                if isinstance(model_output, tuple):

                    output, routing_probs = model_output
                else:

                    output = model_output
                    routing_probs = None


                output = torch.clamp(output, 0, 1)

            _sync_device(args.device)
            inference_time_sec = time.perf_counter() - inference_start_time


            ssim = compute_ssim(output, gt)
            sam = compute_sam(output, gt)
            rmse = compute_rmse(output, gt)
            psnr = compute_psnr(output, gt)
            ergas = ERGAS(ratio=args.ratio)(output, gt)


            ssim_values.append(ssim.item())
            sam_values.append(sam.item())
            rmse_values.append(rmse.item())
            psnr_values.append(psnr.item())
            ergas_values.append(ergas.item())


            print(f'SSIM: {ssim.item():.4f}, SAM: {sam.item():.4f} deg, RMSE: {rmse.item():.4f}, PSNR: {psnr.item():.4f} dB, ERGAS: {ergas.item():.4f}')


            save_path = os.path.join(args.save_dir, f'result_{i}.mat')
            save_matv73(save_path, 'result', output.squeeze(0).cpu().numpy())


            viz_path = visualize_results(output.squeeze(0), gt.squeeze(0), i, args.save_dir, args.dataset)

            per_sample_records.append({
                'sample_index': int(i),
                'sample_name': sample_name,
                'metrics': {
                    'ssim': float(ssim.item()),
                    'sam_deg': float(sam.item()),
                    'rmse': float(rmse.item()),
                    'psnr_db': float(psnr.item()),
                    'ergas': float(ergas.item()),
                },
                'runtime': {
                    'inference_time_sec': float(inference_time_sec),
                    'eval_mode': eval_mode,
                },
                'shapes': {
                    'hr_hsi': list(gt.squeeze(0).shape),
                    'lr_hsi': list(cur_lr.squeeze(0).shape),
                    'hr_msi': list(cur_hr.squeeze(0).shape),
                    'prediction': list(output.squeeze(0).shape),
                },
                'paths': {
                    'result_mat': str(Path(save_path).resolve()),
                    'visualization': str(Path(viz_path).resolve()) if viz_path else None,
                },
            })


        print("\n" + "="*50)
        print("平均测试结果 (Average Test Results)")
        print("="*50)
        print(f'平均 SSIM: {np.mean(ssim_values):.4f} +/- {np.std(ssim_values):.4f}')
        print(f'平均 SAM: {np.mean(sam_values):.4f} deg +/- {np.std(sam_values):.4f} deg')
        print(f'平均 RMSE: {np.mean(rmse_values):.4f} +/- {np.std(rmse_values):.4f}')
        print(f'平均 PSNR: {np.mean(psnr_values):.4f} dB +/- {np.std(psnr_values):.4f} dB')
        print(f'平均 ERGAS: {np.mean(ergas_values):.4f} +/- {np.std(ergas_values):.4f}')
        print("="*50)

        summary_payload = {
            'method_name': Path(args.weight).parent.name,
            'dataset': args.dataset,
            'ratio': int(args.ratio),
            'sample_count': int(len(lr_hsi)),
            'metrics': {
                'ssim': {'mean': float(np.mean(ssim_values)), 'std': float(np.std(ssim_values))},
                'sam_deg': {'mean': float(np.mean(sam_values)), 'std': float(np.std(sam_values))},
                'rmse': {'mean': float(np.mean(rmse_values)), 'std': float(np.std(rmse_values))},
                'psnr_db': {'mean': float(np.mean(psnr_values)), 'std': float(np.std(psnr_values))},
                'ergas': {'mean': float(np.mean(ergas_values)), 'std': float(np.std(ergas_values))},
            },
        }

        artifact_root = None
        if not getattr(args, 'disable_comparison_artifacts', False):
            artifact_root = _export_comparison_artifacts(
                args=args,
                sample_names=resolved_sample_names,
                per_sample_records=per_sample_records,
                summary_payload=summary_payload,
                model_param_stats=model_param_stats,
                dataset_info=dataset_info,
            )


        avg_results_path = os.path.join(args.save_dir, 'average_results.txt')
        with open(avg_results_path, 'w') as f:
            f.write("==================================================\n")
            f.write("平均测试结果 (Average Test Results)\n")
            f.write("==================================================\n")
            f.write(f"数据集类型: {args.dataset}\n")
            f.write(f"权重文件路径: {args.weight}\n")
            if use_sam_ase:
                f.write(f"SAM-ASE模块状态: 启用\n")
                if sam_checkpoint:
                    f.write(f"SAM权重路径: {sam_checkpoint}\n")
                f.write(f"SAM提示维度: {sam_prompt_dim}\n")
            elif use_ase:
                f.write(f"ASE模块状态: 启用\n")
            else:
                f.write(f"ASE/SAM-ASE模块状态: 禁用\n")
            f.write(f"小波变换: {'启用' if args.use_wavelet else '禁用'}\n")
            f.write(f"测试样本数量: {len(lr_hsi)}\n")
            f.write("==================================================\n")
            f.write(f"平均 SSIM: {np.mean(ssim_values):.4f} +/- {np.std(ssim_values):.4f}\n")
            f.write(f"平均 SAM: {np.mean(sam_values):.4f} deg +/- {np.std(sam_values):.4f} deg\n")
            f.write(f"平均 RMSE: {np.mean(rmse_values):.4f} +/- {np.std(rmse_values):.4f}\n")
            f.write(f"平均 PSNR: {np.mean(psnr_values):.4f} dB +/- {np.std(psnr_values):.4f} dB\n")
            f.write(f"平均 ERGAS: {np.mean(ergas_values):.4f} +/- {np.std(ergas_values):.4f}\n")
            if artifact_root is not None:
                f.write(f"结构化导出目录: {artifact_root}\n")
            f.write("==================================================\n")

        print(f"平均结果已保存到: {avg_results_path}")
        if artifact_root is not None:
            print(f"Structured comparison artifacts saved to: {artifact_root}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='chikusei', choices=['cave', 'chikusei', 'pavia_university', 'xiongan_new_area'],
                       help='Dataset type to use: cave, chikusei, pavia_university or xiongan_new_area')
    parser.add_argument('--test_data_path', type=str, default=None, help='test data directory')
    parser.add_argument('--weight', type=str, required=True, help='path to model weights file')
    parser.add_argument('--h5_path', type=str, default=None, help='preprocessed H5 file path')
    parser.add_argument('--save_dir', type=str, default=None, help='save directory')
    parser.add_argument('--comparison_artifacts_dir', type=str, default=None,
                       help='optional explicit output directory for structured comparison artifacts')
    parser.add_argument('--comparison_artifacts_subdir', type=str, default='comparison_artifacts',
                       help='subdirectory name under save_dir for structured comparison artifacts')
    parser.add_argument('--disable_comparison_artifacts', action='store_true',
                       help='disable structured comparison artifact export')
    parser.add_argument('--channels', type=int, default=64, help='Feature channels')
    parser.add_argument('--spa_channels', type=int, default=31, help='HSI channels')
    parser.add_argument('--spe_channels', type=int, default=3, help='MSI channels')
    parser.add_argument('--H', type=int, default=256, help='Height of the image')
    parser.add_argument('--W', type=int, default=256, help='Width of the image')
    parser.add_argument('--ratio', type=int, default=4, help='Ratio of the image')
    parser.add_argument('--cut_size', type=int, default=64, help='Cut size for block processing')
    parser.add_argument('--pad', type=int, default=0, help='Padding size for block processing')
    parser.add_argument('--force_small_image_block_eval', action='store_true',
                       help='Force block-wise inference even for small test images')
    parser.add_argument('--use_overlap_blend', action='store_true',
                       help='Use overlapped block inference with weighted blending')
    parser.add_argument('--block_stride', type=int, default=None,
                       help='Stride for overlapped block inference; defaults to cut_size - 2*pad')
    parser.add_argument('--blend_window', type=str, default='hann', choices=['hann', 'uniform'],
                       help='Blend window used for overlapped block fusion')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')


    parser.add_argument('--use_ase', action='store_true', help='Use ASE (Adaptive Semantic Enhancement) module')
    parser.add_argument('--ase_prompt_mode', type=str, default='soft', choices=['hard', 'soft', 'hybrid'],
                       help='ASE prompt generation mode: hard=original hard prompt, soft=soft prompt, hybrid=hard sort + soft prompt blend')
    parser.add_argument('--ase_route_temperature', type=float, default=1.2,
                       help='Temperature for ASE routing / prompt generation (default: 1.0)')
    parser.add_argument('--ase_prompt_soft_mix', type=float, default=0.5,
                       help='Soft prompt weight in hybrid ASE prompt mode (default: 0.5)')
    parser.add_argument('--ase_scope', type=str, default='fusion_only', choices=['all', 'fusion_only'],
                       help='ASE application scope: all branches or fusion branch only')
    parser.add_argument('--ase_stage_scope', type=str, default='all_stages',
                       choices=['all_stages', 'deep34', 'deep234', 'stage4_only'],
                       help='Stage scope for ASE: all_stages, deep34, deep234, or stage4_only')
    parser.add_argument('--use_ase_fusion_residual', action='store_true',
                       help='use residual correction form for ASE fusion output')
    parser.add_argument('--ase_fusion_res_scale', type=float, default=0.4,
                       help='residual scale for ASE fusion correction (default: 0.3)')
    parser.add_argument('--use_learnable_ase_fusion_res_scale', action='store_true',
                       help='make ASE fusion residual scale learnable, initialized from --ase_fusion_res_scale')
    parser.add_argument('--ase_stage_res_scales', type=str, default=None,
                       help='Optional 5 comma-separated residual scales for stage0~stage4, e.g. 0.2,0.25,0.35,0.4,0.4')
    parser.add_argument('--use_sam_ase', action='store_true', help='Use SAM-guided ASE module')
    parser.add_argument('--sam_checkpoint', type=str, default=None, help='SAM model checkpoint path')
    parser.add_argument('--sam_prompt_dim', type=int, default=64, help='SAM prompt feature dimension')
    parser.add_argument('--use_offline_sam_cache', action='store_true', help='use offline SAM cache during testing')
    parser.add_argument('--sam_cache_path', type=str, default=None, help='path to test-time sidecar SAM cache h5 file')
    parser.add_argument('--sam_cache_strict', action='store_true', help='fail if SAM cache is missing or incompatible')
    parser.add_argument('--prefer_scene_sam_cache', action='store_true',
                       help='when offline SAM cache is enabled and no explicit cache path is provided, prefer *.whole_sam_cache.scene_region.h5 over older cache variants')
    parser.add_argument('--prefer_multi_region_sam_cache', action='store_true',
                       help='when offline SAM cache is enabled and no explicit cache path is provided, prefer *.whole_sam_cache.multi_region.h5 over the default single-mask cache')


    parser.add_argument('--use_learnable_prompts', action='store_true', help='use learnable prompt generator for SAM')
    parser.add_argument('--num_learnable_prompts', type=int, default=16, help='number of learnable prompts for SAM')
    parser.add_argument('--use_soft_masks', action='store_true', help='use soft multi-region masks instead of binary masks')
    parser.add_argument('--num_soft_regions', type=int, default=8, help='number of soft mask regions (8-16 recommended)')
    parser.add_argument('--use_structure_guided_sam_ase', action='store_true',
                       help='enable structure-guided SAM-ASE step-1: semantic coarse grouping + wavelet texture fine ranking')
    parser.add_argument('--structure_texture_weight', type=float, default=0.25,
                       help='texture weight for structure-guided SAM-ASE sorting and gating (default: 0.25)')


    parser.add_argument('--use_wavelet', action='store_true',
                       help='legacy wavelet backbone branch for feature enhancement (kept for backward compatibility)')
    parser.add_argument('--use_wavelet_priors', action='store_true',
                       help='use unified wavelet prior pipeline without enabling the legacy wavelet backbone branch')
    parser.add_argument('--use_wavelet_local_bias', action='store_true',
                       help='use stage4 fusion-only wavelet local bias on top of ASE fusion residual')
    parser.add_argument('--wavelet_local_bias_scale', type=float, default=0.1,
                       help='scale for stage4 fusion-only wavelet local bias (default: 0.1)')
    parser.add_argument('--use_wavelet_local_gate', action='store_true',
                       help='use stage4 fusion-local wavelet gate to modulate ASE fusion residual')
    parser.add_argument('--wavelet_local_gate_scale', type=float, default=0.1,
                       help='scale for stage4 fusion-local wavelet gate (default: 0.1)')


    parser.add_argument('--use_sam_local_gate', action='store_true',
                       help='use fusion-local SAM gate to modulate ASE fusion residual')
    parser.add_argument('--sam_local_gate_scale', type=float, default=0.1,
                       help='scale for fusion-local SAM gate (default: 0.1)')
    parser.add_argument('--use_sam_semantic_prompt_bank', action='store_true',
                       help='use SAM semantic prompt bank to refine fusion-only ASE prompt bank')
    parser.add_argument('--sam_semantic_prompt_bank_scale', type=float, default=0.1,
                       help='scale for SAM semantic prompt bank bias (default: 0.1)')
    parser.add_argument('--use_sam_region_prototype_bank', action='store_true',
                       help='use remapped whole-image SAM region prototypes to condition fusion-only ASE prompt bank')
    parser.add_argument('--sam_region_prototype_bank_scale', type=float, default=0.1,
                       help='scale for SAM region prototype prompt-bank bias (default: 0.1)')
    parser.add_argument('--sam_region_prototype_count', type=int, default=8,
                       help='number of SAM region prototypes used for prompt-bank conditioning (default: 8)')
    parser.add_argument('--use_wavelet_guided_sam_prototype_scaling', action='store_true',
                       help='use wavelet high-frequency complexity to rescale SAM region prototypes before prompt-bank conditioning')
    parser.add_argument('--wavelet_guided_sam_prototype_scale', type=float, default=0.1,
                       help='scale for wavelet-guided SAM prototype rescaling (default: 0.1)')
    parser.add_argument('--use_sam_region_prompt_mixture', action='store_true',
                       help='use remapped whole-image SAM regions to provide a prompt-mixture prior for fusion-only ASE')
    parser.add_argument('--sam_region_prompt_mixture_scale', type=float, default=0.05,
                       help='scale for SAM region prompt-mixture prior (default: 0.05)')
    parser.add_argument('--sam_region_prompt_mixture_count', type=int, default=8,
                       help='top-k SAM regions used by prompt-mixture prior (default: 8)')
    parser.add_argument('--use_sam_guided_semantic_scanning', action='store_true',
                       help='use remapped whole-image SAM regions to reorder fusion tokens before ASE scanning')
    parser.add_argument('--sam_semantic_scanning_count', type=int, default=6,
                       help='top-k SAM regions used to build semantic scan order (default: 6)')
    parser.add_argument('--use_sam_feature_cluster_scanning', action='store_true',
                       help='replace mask-driven SS1 ordering with SAM feature clustering to build semantic scan groups')
    parser.add_argument('--sam_feature_cluster_count', type=int, default=6,
                       help='number of SAM-feature clusters used by feature-clustering semantic scanning (default: 6)')
    parser.add_argument('--sam_feature_cluster_iters', type=int, default=2,
                       help='number of prototype refinement iterations used by feature-clustering semantic scanning (default: 2)')
    parser.add_argument('--sam_feature_cluster_spatial_weight', type=float, default=0.05,
                       help='spatial coordinate weight blended into SAM features during feature clustering (default: 0.05)')
    parser.add_argument('--use_wavelet_augmented_ss1', action='store_true',
                       help='enhance SS1 by using joint wavelet prior to refine token priority inside each SAM region')
    parser.add_argument('--wavelet_augmented_ss1_count', type=int, default=6,
                       help='top-k SAM regions used by wavelet-augmented SS1 scanner (default: 6)')
    parser.add_argument('--wavelet_augmented_ss1_topk_ratio', type=float, default=0.25,
                       help='top-k ratio of high-frequency tokens promoted inside each SAM region for WSS1 (default: 0.25)')
    parser.add_argument('--wavelet_augmented_ss1_strength', type=float, default=0.5,
                       help='blend strength between wavelet priority and original stable order in WSS1 (default: 0.5)')
    parser.add_argument('--wavelet_augmented_ss1_mode', type=str, default='stable_intra_region', choices=['stable_intra_region'],
                       help='scan refinement mode for WSS1 (default: stable_intra_region)')
    parser.add_argument('--use_sam_boundary_aware_state_propagation', action='store_true',
                       help='use remapped whole-image SAM regions to attenuate ASE state propagation across semantic boundaries')
    parser.add_argument('--sam_boundary_aware_state_scale', type=float, default=0.2,
                       help='attenuation scale for semantic boundary-aware ASE state propagation (default: 0.2)')
    parser.add_argument('--use_sam_state_reset_stronger', action='store_true',
                       help='use a stronger SAM-guided state reset surrogate at semantic boundaries on top of boundary-aware propagation')
    parser.add_argument('--sam_state_reset_scale', type=float, default=0.35,
                       help='strong reset scale applied at semantic boundaries (default: 0.35)')
    parser.add_argument('--use_sam_state_organizer_v1', action='store_true',
                       help='use a unified SAM-guided state organizer to jointly build semantic scan, boundary gate, and reset gate')
    parser.add_argument('--sam_state_organizer_count', type=int, default=6,
                       help='top-k SAM regions used by the unified state organizer (default: 6)')
    parser.add_argument('--sam_state_organizer_boundary_scale', type=float, default=0.1,
                       help='boundary attenuation strength used by SAM-guided state organizer v1 (default: 0.1)')
    parser.add_argument('--sam_state_organizer_reset_scale', type=float, default=0.15,
                       help='state reset strength used by SAM-guided state organizer v1 (default: 0.15)')
    parser.add_argument('--use_sam_region_prompt_subspace', action='store_true',
                       help='use SAM region-specific prompt subspace to generate token-wise ASE prompt residuals')
    parser.add_argument('--sam_region_prompt_subspace_scale', type=float, default=0.05,
                       help='scale for SAM region prompt subspace token residual (default: 0.05)')
    parser.add_argument('--sam_region_prompt_subspace_count', type=int, default=6,
                       help='top-k SAM regions used by region-specific prompt subspace (default: 6)')
    parser.add_argument('--use_wavelet_guided_semantic_state_organization', action='store_true',
                       help='use wavelet-guided SAM state organizer for semantic scan, boundary gate, and reset gate')
    parser.add_argument('--wavelet_guided_semantic_state_count', type=int, default=6,
                       help='top-k SAM regions used by wavelet-guided semantic state organization (default: 6)')
    parser.add_argument('--wavelet_guided_semantic_state_scale', type=float, default=0.05,
                       help='wavelet complexity scale used inside wavelet-guided semantic state organization (default: 0.05)')
    parser.add_argument('--wavelet_guided_semantic_boundary_scale', type=float, default=0.1,
                       help='boundary attenuation strength for wavelet-guided semantic state organization (default: 0.1)')
    parser.add_argument('--wavelet_guided_semantic_reset_scale', type=float, default=0.15,
                       help='state reset strength for wavelet-guided semantic state organization (default: 0.15)')
    parser.add_argument('--use_joint_spatial_spectral_wavelet_prior', action='store_true',
                       help='use joint spatial-spectral wavelet prior map as unified semantic-frequency prior')
    parser.add_argument('--joint_wavelet_spatial_weight', type=float, default=1.0,
                       help='spatial wavelet prior weight inside joint spatial-spectral prior (default: 1.0)')
    parser.add_argument('--joint_wavelet_spectral_weight', type=float, default=0.7,
                       help='spectral variation prior weight inside joint spatial-spectral prior (default: 1.0)')
    parser.add_argument('--use_dual_prototype_bank', action='store_true',
                       help='use semantic-frequency dual prototype bank to refine fusion-only ASE prompt bank')
    parser.add_argument('--dual_prototype_semantic_scale', type=float, default=0.05,
                       help='semantic prototype contribution in dual prototype bank (default: 0.05)')
    parser.add_argument('--dual_prototype_frequency_scale', type=float, default=0.05,
                       help='frequency prototype contribution in dual prototype bank (default: 0.05)')
    parser.add_argument('--dual_prototype_count', type=int, default=6,
                       help='top-k SAM regions used by semantic-frequency dual prototype bank (default: 6)')
    parser.add_argument('--use_semantic_frequency_state_modulation', action='store_true',
                       help='use semantic-frequency prior to modulate ASE state write/read/delta gates')
    parser.add_argument('--semantic_frequency_state_count', type=int, default=6,
                       help='top-k SAM regions used by semantic-frequency state modulation (default: 6)')
    parser.add_argument('--semantic_frequency_state_write_scale', type=float, default=0.08,
                       help='write-gate modulation scale for semantic-frequency state modulation (default: 0.08)')
    parser.add_argument('--semantic_frequency_state_read_scale', type=float, default=0.08,
                       help='read-gate modulation scale for semantic-frequency state modulation (default: 0.08)')
    parser.add_argument('--semantic_frequency_state_delta_scale', type=float, default=0.05,
                       help='delta-gate modulation scale for semantic-frequency state modulation (default: 0.05)')
    parser.add_argument('--use_fass', action='store_true', help='use FASS (Frequency-Adaptive Sparse Scanning) module')
    parser.add_argument('--fass_compression_ratio', type=int, default=2, help='FASS high-frequency compression ratio (2=compress to 1/2)')
    parser.add_argument('--fass_threshold', type=float, default=0.5, help='[DEPRECATED] FASS gating network threshold (use Top-K instead)')
    parser.add_argument('--fass_sparsity_target', type=float, default=0.3, help='[DEPRECATED] FASS target sparsity (use fass_ll_sparsity and fass_hf_sparsity instead)')
    parser.add_argument('--fass_ll_sparsity', type=float, default=1.0, help='FASS LL branch keep ratio (1.0=Dense mode, 0.25=keep 25%% tokens). Default: 1.0 (LL use Dense)')
    parser.add_argument('--fass_hf_sparsity', type=float, default=0.20, help='FASS HF branch keep ratio (0.20=keep 20%% tokens, i.e., 80%% sparse). Default: 0.20')
    parser.add_argument('--fass_d_state', type=int, default=16, help='FASS Mamba state dimension')


    parser.add_argument('--train_mode', type=str, default='sparse', choices=['dense', 'sparse', 'auto'],
                       help='FASS training mode: dense=full training, sparse=sparse training, auto=auto switch (default: sparse for testing)')
    parser.add_argument('--dense_epochs', type=int, default=100, help='Number of epochs for dense training before switching to sparse')
    parser.add_argument('--gating_loss_weight', type=float, default=1.0, help='Weight for gating network training loss')
    parser.add_argument('--gating_input_mode', type=str, default='energy', choices=['energy', 'hybrid_v2', 'semantic_frequency_v1'],
                       help='Gating input mode: energy=original energy-only gate, hybrid_v2=energy+semantic hybrid gate')
    parser.add_argument('--disable_gating_semantic_mask', action='store_false', dest='gating_use_semantic_mask',
                       help='Disable semantic mask channel in hybrid_v2 gating input')
    parser.add_argument('--disable_gating_prompt_strength', action='store_false', dest='gating_use_prompt_strength',
                       help='Disable SAM prompt strength channel in hybrid_v2 gating input')
    parser.add_argument('--disable_gating_local_contrast', action='store_false', dest='gating_use_local_contrast',
                       help='Disable local contrast channel in hybrid_v2 HF gating input')
    parser.add_argument('--use_sam_prior_bank', action='store_true',
                       help='Use SAM Prior Bank V1 to refine hybrid_v2 semantic/prompt channels')
    parser.add_argument('--disable_sam_prior_boundary', action='store_false', dest='sam_prior_use_boundary',
                       help='Disable boundary_map contribution when SAM prior bank is enabled')
    parser.add_argument('--disable_sam_prior_confidence', action='store_false', dest='sam_prior_use_confidence',
                       help='Disable confidence_map modulation when SAM prior bank is enabled')
    parser.add_argument('--use_sam_prior_c_refiner', action='store_true',
                       help='Use Prior Refiner V1: region + prompt strength -> prior bias -> C (non-FASS SAMASE path)')
    parser.add_argument('--sam_prior_c_refiner_scope', type=str, default='fusion', choices=['fusion', 'all'],
                       help='Scope for Prior C Refiner V1: fusion=only fusion branch, all=spa/spe/fusion branches')
    parser.add_argument('--sam_prior_c_refiner_scale', type=float, default=0.1,
                       help='Scale for Prior Refiner V1 bias before adding to prompt_proj(sam_prompt)')
    parser.add_argument('--use_semantic_frequency_adaptive_scanning', action='store_true',
                       help='enable Semantic-Frequency Adaptive Scanning (FASS 2.0) with SAM regions and wavelet guidance')
    parser.add_argument('--semantic_frequency_semantic_weight', type=float, default=1.0,
                       help='weight for semantic-region component in FASS 2.0 gating inputs (default: 1.0)')
    parser.add_argument('--semantic_frequency_wavelet_weight', type=float, default=1.0,
                       help='weight for wavelet-frequency component in FASS 2.0 gating inputs (default: 1.0)')
    parser.add_argument('--semantic_frequency_boundary_weight', type=float, default=0.5,
                       help='weight for semantic-boundary component in FASS 2.0 gating inputs (default: 0.5)')
    parser.add_argument('--semantic_frequency_confidence_weight', type=float, default=0.25,
                       help='weight for semantic-confidence component in FASS 2.0 gating inputs (default: 0.25)')
    parser.add_argument('--semantic_frequency_prompt_weight', type=float, default=0.5,
                       help='weight for prompt-strength component in FASS 2.0 gating inputs (default: 0.5)')
    parser.set_defaults(
        gating_use_semantic_mask=True,
        gating_use_prompt_strength=True,
        gating_use_local_contrast=True,
        sam_prior_use_boundary=True,
        sam_prior_use_confidence=True,
        use_ase=True,
        use_ase_fusion_residual=True,
        use_wavelet_priors=True,
        use_joint_spatial_spectral_wavelet_prior=True,
        use_sam_state_organizer_v1=True,
    )

    args = parser.parse_args()


    if not hasattr(args, 'use_ase'):
        args.use_ase = False
    if not hasattr(args, 'use_sam_ase'):
        args.use_sam_ase = False
    if not hasattr(args, 'sam_checkpoint'):
        args.sam_checkpoint = None
    if not hasattr(args, 'sam_prompt_dim'):
        args.sam_prompt_dim = 64
    if args.dataset.lower() == 'pavia_university':
        if args.h5_path is None:
            args.h5_path = './data/pavia_university_x4/pavia_university_test.h5'
        if args.test_data_path is None:
            args.test_data_path = os.path.dirname(args.h5_path) if args.h5_path.endswith('.h5') else './data/pavia_university_x4'
        print(
            f"[配置] Pavia University数据集: dim={args.channels}, "
            f"expected HSI通道≈103, MSI通道=4, H5={args.h5_path}"
        )
        test(args)
        sys.exit(0)

    if args.dataset.lower() == 'xiongan_new_area':
        if args.h5_path is None:
            args.h5_path = './data/xiongan_new_area_x4/xiongan_new_area_test.h5'
        if args.test_data_path is None:
            args.test_data_path = os.path.dirname(args.h5_path) if args.h5_path.endswith('.h5') else './data/xiongan_new_area_x4'
        print(
            f"[配置] Xiongan New Area数据集: dim={args.channels}, "
            f"HSI通道=93, MSI通道=4, H5={args.h5_path}"
        )
        test(args)
        sys.exit(0)


    if args.dataset.lower() == 'chikusei':

        if not hasattr(args, 'channels') or args.channels == 64:

            pass

        if args.h5_path is None:
            args.h5_path = './data/chikusei/chikusei_test.h5'
        if args.test_data_path is None:
            args.test_data_path = os.path.dirname(args.h5_path) if args.h5_path.endswith('.h5') else './data/chikusei'
        print(f"[配置] Chikusei数据集: dim={args.channels}, HSI通道=128, MSI通道=4, H5={args.h5_path}")
    else:

        if 'channels' not in args or (hasattr(args, 'channels') and args.channels == 64):

            args.channels = 32
            print(f"[配置] CAVE数据集自动调整: dim从64改为32")

        if args.h5_path is None:
            args.h5_path = './data/cave/cave_test.h5'
        if args.test_data_path is None:
            args.test_data_path = './data/train/CAVE/test'
        print(f"[配置] CAVE数据集: dim={args.channels}, HSI通道=31, MSI通道=3, H5={args.h5_path}")

    test(args)
