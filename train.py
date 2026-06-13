import os
import torch
import time
import math
import re
import shlex
import array
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from model.u2net import U2Net as Net
from model.haar_dwt import SimpleDWT
from torch.utils.data import DataLoader
from utils.load_hsimsi_data import HSIMSI_Dataset
from utils.wavelet_utils import should_use_wavelet_priors
from torch.cuda.amp import autocast, GradScaler
from utils.tools import SSIM, PSNR, ERGAS
import sys
try:
    import cv2
except ImportError:
    cv2 = None

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class Logger:
    def __init__(self, log_file=None):
        self.terminal = sys.stdout
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        if self.log_file is not None:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(message)

    def flush(self):
        self.terminal.flush()

    def close(self):
        pass


def _stable_dataloader_worker_init(worker_id):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    if cv2 is not None:
        try:
            cv2.setNumThreads(0)
        except Exception:
            pass


def _coerce_optimizer_step_tensor(step_value):
    if torch.is_tensor(step_value):
        return step_value
    if isinstance(step_value, np.ndarray):
        if step_value.size <= 0:
            step_value = 0.0
        else:
            step_value = step_value.reshape(-1)[0].item()
    elif isinstance(step_value, array.array):
        step_value = step_value[0] if len(step_value) > 0 else 0.0
    elif isinstance(step_value, (list, tuple)):
        step_value = step_value[0] if len(step_value) > 0 else 0.0
    elif hasattr(step_value, 'item'):
        try:
            step_value = step_value.item()
        except Exception:
            pass
    if step_value is None:
        step_value = 0.0
    try:
        step_value = float(step_value)
    except Exception:
        step_value = 0.0
    return torch.tensor(step_value, dtype=torch.float32)


def _sanitize_optimizer_state_dict(optimizer_state_dict):
    if not isinstance(optimizer_state_dict, dict):
        return 0
    state = optimizer_state_dict.get('state', None)
    if not isinstance(state, dict):
        return 0
    fixed_count = 0
    for _, param_state in state.items():
        if not isinstance(param_state, dict):
            continue
        if 'step' in param_state and not torch.is_tensor(param_state['step']):
            param_state['step'] = _coerce_optimizer_step_tensor(param_state['step'])
            fixed_count += 1
    return fixed_count

class HFWaveletLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.dwt = SimpleDWT()
        self.criterion = nn.L1Loss(reduction='mean')

    def _crop_to_even_size(self, x):
        h = x.shape[-2] - (x.shape[-2] % 2)
        w = x.shape[-1] - (x.shape[-1] % 2)
        return x[..., :h, :w]

    def _weighted_l1(self, pred_band, target_band, spatial_weight=None):
        diff = torch.abs(pred_band - target_band)
        if spatial_weight is None:
            return diff.mean()

        if spatial_weight.dim() == 3:
            spatial_weight = spatial_weight.unsqueeze(1)
        if spatial_weight.dim() != 4:
            raise ValueError(
                f"spatial_weight must be [B, 1, H, W] or [B, H, W], got {tuple(spatial_weight.shape)}"
            )

        if spatial_weight.shape[-2:] != diff.shape[-2:]:
            spatial_weight = F.interpolate(
                spatial_weight,
                size=diff.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

        spatial_weight = spatial_weight.to(device=diff.device, dtype=diff.dtype)
        weighted = diff * spatial_weight
        norm = spatial_weight.sum() * diff.shape[1] + 1e-6
        return weighted.sum() / norm

    def forward(self, pred, target, spatial_weight=None):
        if pred.shape != target.shape:
            raise ValueError(
                f"HF wavelet loss expects matched shapes, got pred={tuple(pred.shape)} vs target={tuple(target.shape)}"
            )

        pred = self._crop_to_even_size(pred)
        target = self._crop_to_even_size(target)
        if spatial_weight is not None:
            spatial_weight = self._crop_to_even_size(spatial_weight)

        _, pred_hf_list = self.dwt(pred)
        _, target_hf_list = self.dwt(target)

        hf_loss = 0.0
        for pred_band, target_band in zip(pred_hf_list, target_hf_list):
            hf_loss = hf_loss + self._weighted_l1(pred_band, target_band, spatial_weight=spatial_weight)
        return hf_loss / len(pred_hf_list)


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def _normalize_spatial_prior(map_tensor):
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


def _build_boundary_map_from_regions(region_map):
    if region_map is None:
        return None
    dx = torch.abs(region_map[:, :, :, 1:] - region_map[:, :, :, :-1])
    dy = torch.abs(region_map[:, :, 1:, :] - region_map[:, :, :-1, :])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return _normalize_spatial_prior(dx + dy)


def build_sam_distillation_priors(cached_sam_masks):
    if cached_sam_masks is None:
        return None

    region_map = _normalize_spatial_prior(cached_sam_masks.float())
    if region_map is None:
        return None

    boundary_map = _build_boundary_map_from_regions(region_map)
    return {
        'region_map': region_map,
        'boundary_map': boundary_map,
    }


def compute_sam_route_consistency_loss(routing_probs, prior_bank):
    if routing_probs is None or prior_bank is None:
        return None

    boundary_map = prior_bank.get('boundary_map')
    if boundary_map is None:
        return None

    if boundary_map.dim() == 3:
        boundary_map = boundary_map.unsqueeze(1)

    total_loss = 0.0
    valid_terms = 0

    def _compute_single_route_loss(route_prob):
        if route_prob is None or route_prob.dim() != 3:
            return None
        bsz, seq_len, num_tokens = route_prob.shape
        side = int(round(math.sqrt(seq_len)))
        if side * side != seq_len:
            return None

        route_map = route_prob.permute(0, 2, 1).reshape(bsz, num_tokens, side, side)
        boundary = F.interpolate(boundary_map, size=(side, side), mode='bilinear', align_corners=False)
        interior = (1.0 - boundary).clamp(0.0, 1.0)

        weight_x = 0.5 * (interior[:, :, :, 1:] + interior[:, :, :, :-1])
        weight_y = 0.5 * (interior[:, :, 1:, :] + interior[:, :, :-1, :])

        diff_x = torch.abs(route_map[:, :, :, 1:] - route_map[:, :, :, :-1]).mean(dim=1, keepdim=True)
        diff_y = torch.abs(route_map[:, :, 1:, :] - route_map[:, :, :-1, :]).mean(dim=1, keepdim=True)

        loss_x = (diff_x * weight_x).sum() / (weight_x.sum() + 1e-6)
        loss_y = (diff_y * weight_y).sum() / (weight_y.sum() + 1e-6)
        return 0.5 * (loss_x + loss_y)

    for stage_probs in routing_probs:
        if isinstance(stage_probs, list):
            for probs in stage_probs:
                single_loss = _compute_single_route_loss(probs)
                if single_loss is not None:
                    total_loss = total_loss + single_loss
                    valid_terms += 1
        else:
            single_loss = _compute_single_route_loss(stage_probs)
            if single_loss is not None:
                total_loss = total_loss + single_loss
                valid_terms += 1

    if valid_terms == 0:
        return None
    return total_loss / valid_terms


def compute_sam_boundary_recon_loss(pred, target, prior_bank):
    if pred.shape != target.shape or prior_bank is None:
        return None

    boundary_map = prior_bank.get('boundary_map')
    if boundary_map is None:
        return None

    if boundary_map.dim() == 3:
        boundary_map = boundary_map.unsqueeze(1)

    boundary = F.interpolate(boundary_map, size=pred.shape[-2:], mode='bilinear', align_corners=False)
    boundary = boundary.clamp(0.0, 1.0)

    pixel_l1 = torch.abs(pred - target).mean(dim=1, keepdim=True)
    return (pixel_l1 * boundary).sum() / (boundary.sum() + 1e-6)


def build_semantic_region_wavelet_weight_map(cached_sam_masks, boundary_boost=0.5):
    if cached_sam_masks is None:
        return None

    if cached_sam_masks.dim() == 3:
        cached_sam_masks = cached_sam_masks.unsqueeze(1)
    if cached_sam_masks.dim() != 4:
        return None

    sam_masks = cached_sam_masks.float()
    if sam_masks.shape[1] <= 0:
        return None

    confidence_map, region_index = sam_masks.max(dim=1, keepdim=True)
    region_ids = region_index.float() + 1.0
    region_ids = torch.where(
        confidence_map > 1e-5,
        region_ids,
        torch.zeros_like(region_ids),
    )
    boundary_map = _build_boundary_map_from_regions(region_ids)

    weight_map = confidence_map
    if boundary_map is not None:
        weight_map = weight_map + float(boundary_boost) * boundary_map.to(weight_map.dtype)
    return _normalize_spatial_prior(weight_map).clamp(0.0, 1.0)


def build_boundary_selective_wavelet_weight_map(
    sam_region_context,
    cached_sam_masks,
    boundary_boost=0.75,
    frequency_boost=0.5,
):
    if sam_region_context is None and cached_sam_masks is None:
        return None

    prior_map = None
    if sam_region_context is not None:
        prior_map = sam_region_context.get('semantic_frequency_prior_map')
        if prior_map is None:
            prior_map = sam_region_context.get('wavelet_prior_map')
        if prior_map is None:
            prior_map = sam_region_context.get('wavelet_guidance')

    confidence_map = None
    boundary_map = None
    if cached_sam_masks is not None:
        if cached_sam_masks.dim() == 3:
            cached_sam_masks = cached_sam_masks.unsqueeze(1)
        if cached_sam_masks.dim() == 4:
            sam_masks = cached_sam_masks.float()
            confidence_map, region_index = sam_masks.max(dim=1, keepdim=True)
            region_ids = region_index.float() + 1.0
            region_ids = torch.where(
                confidence_map > 1e-5,
                region_ids,
                torch.zeros_like(region_ids),
            )
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


def compute_sam_region_relation_loss(feature_map, cached_sam_masks, num_regions=4, separation_weight=0.1):
    if feature_map is None or cached_sam_masks is None:
        return None
    if feature_map.dim() != 4:
        return None
    if cached_sam_masks.dim() == 3:
        cached_sam_masks = cached_sam_masks.unsqueeze(1)
    if cached_sam_masks.dim() != 4:
        return None

    bsz, channels, height, width = feature_map.shape
    masks = F.interpolate(cached_sam_masks.float(), size=(height, width), mode='bilinear', align_corners=False)
    total_loss = feature_map.new_tensor(0.0)
    valid = 0

    for b in range(bsz):
        sample_masks = masks[b]
        area = sample_masks.flatten(-2).mean(dim=-1)
        valid_mask = area > 1e-5
        if valid_mask.sum() == 0:
            continue

        sample_masks = sample_masks[valid_mask]
        area = area[valid_mask]
        region_count = min(int(num_regions), int(sample_masks.shape[0]))
        if region_count <= 0:
            continue

        topk_idx = torch.topk(area, k=region_count, dim=0).indices
        selected_masks = sample_masks[topk_idx]
        sample_feat = feature_map[b]
        prototypes = []
        consistency_loss = feature_map.new_tensor(0.0)

        for region_mask in selected_masks:
            weight = region_mask.unsqueeze(0)
            norm = weight.sum().clamp(min=1e-6)
            proto = (sample_feat * weight).flatten(1).sum(dim=-1) / norm
            prototypes.append(proto)

            sq_error = (sample_feat - proto[:, None, None]).pow(2).mean(dim=0, keepdim=True)
            consistency_loss = consistency_loss + (sq_error * weight).sum() / norm

        consistency_loss = consistency_loss / float(region_count)
        proto_tensor = torch.stack(prototypes, dim=0)
        proto_tensor = F.normalize(proto_tensor, dim=-1)
        sim = torch.matmul(proto_tensor, proto_tensor.transpose(0, 1))
        off_diag = sim - torch.eye(region_count, device=sim.device, dtype=sim.dtype)
        separation_loss = F.relu(off_diag).sum() / max(region_count * (region_count - 1), 1)

        total_loss = total_loss + consistency_loss + separation_weight * separation_loss
        valid += 1

    if valid == 0:
        return None
    return total_loss / valid


def should_skip_checkpoint_key(key, args):
    if '.sam_model' in key or 'predictor.' in key:
        return True
    if (
        getattr(args, 'use_sam_distillation', False)
        and not getattr(args, 'use_sam_ase', False)
        and not getattr(args, 'use_sam_local_gate', False)
        and 'sam_extractor.' in key
    ):
        return True
    return False


def filter_checkpoint_state_dict(state_dict, args):
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if should_skip_checkpoint_key(key, args):
            continue
        filtered_state_dict[key] = value
    return filtered_state_dict

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.benchmark = True


def save_checkpoint(args, model, optimizer, epoch, custom_weight_dir=None):

    weight_dir = custom_weight_dir if custom_weight_dir else args.weight_dir

    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    model_out_path = os.path.join(weight_dir, "{}.pth".format(epoch))


    state_dict = model.state_dict()
    filtered_state_dict = {}


    for key, value in state_dict.items():


        if '.sam_model' in key or 'predictor.' in key:
            continue
        filtered_state_dict[key] = value

    torch.save({
        'epoch': epoch,
        'state_dict': filtered_state_dict,
        'optimizer': optimizer.state_dict()
    }, model_out_path)

def _get_weight_root(args):
    weight_root = getattr(args, 'weight_dir', 'weights/')
    weight_root = weight_root.rstrip('/\\')
    return weight_root if weight_root else 'weights'


def _get_exp_index_path(weight_root):
    return os.path.join(weight_root, "EXPERIMENT_INDEX.md")


def _get_next_experiment_code(weight_root):
    pattern = re.compile(r'^E(\d+)(?:_|$)', re.IGNORECASE)
    max_id = 0
    if os.path.exists(weight_root):
        for name in os.listdir(weight_root):
            full_path = os.path.join(weight_root, name)
            if not os.path.isdir(full_path):
                continue
            match = pattern.match(name)
            if match:
                max_id = max(max_id, int(match.group(1)))
    return f"E{max_id + 1}"


def _get_or_create_experiment_code(args, weight_root):
    exp_code = getattr(args, 'exp_code', None)
    if exp_code:
        exp_code = exp_code.strip()
        if not re.match(r'^[A-Za-z][A-Za-z0-9_-]*$', exp_code):
            raise ValueError(f"Invalid --exp_code '{exp_code}'. Use letters, numbers, '_' or '-'.")
        return exp_code
    return _get_next_experiment_code(weight_root)


def _normalize_dataset_slug(dataset_name):
    return str(dataset_name).strip().lower().replace(" ", "_").replace("-", "_")


def _parse_run_folder_metadata(run_folder_name):
    match = re.match(r'^(?P<prefix>.+)_x(?P<ratio>\d+)_(?P<time>\d{8}_\d{6})$', run_folder_name)
    if not match:
        return None

    prefix = match.group('prefix')
    dataset_slug = None
    exp_code = prefix

    for candidate_slug in sorted({'cave', 'chikusei', 'pavia_university', 'xiongan_new_area'}, key=len, reverse=True):
        suffix = f"_{candidate_slug}"
        if prefix.endswith(suffix) and len(prefix) > len(suffix):
            exp_code = prefix[:-len(suffix)]
            dataset_slug = candidate_slug
            break

    return {
        'exp_code': exp_code,
        'dataset_slug': dataset_slug,
        'ratio': int(match.group('ratio')),
        'time': match.group('time'),
    }


def _parse_exp_code_from_run_folder(run_folder_name):
    metadata = _parse_run_folder_metadata(run_folder_name)
    if metadata:
        return metadata.get('exp_code')
    match = re.match(r'^(?P<code>.+?)_x\d+_\d{8}_\d{6}$', run_folder_name)
    if match:
        return match.group('code')
    return None


def _append_flag(cmd_parts, enabled, flag):
    if enabled:
        cmd_parts.append(flag)


def _append_option(cmd_parts, flag, value):
    if value is None:
        return
    cmd_parts.extend([flag, str(value)])


def _build_train_command():
    argv = [os.path.basename(sys.argv[0])] + sys.argv[1:]
    return "python " + " ".join(shlex.quote(arg) for arg in argv)


def _detect_route_profiles(args):
    fusion_only_ase_mainline = (
        getattr(args, 'use_ase', False)
        and not getattr(args, 'use_sam_ase', False)
        and not getattr(args, 'use_fass', False)
        and getattr(args, 'use_ase_fusion_residual', False)
        and getattr(args, 'ase_scope', 'all') == 'fusion_only'
    )
    sam_region_proto = getattr(args, 'use_sam_region_prototype_bank', False)
    sam_scan = getattr(args, 'use_sam_guided_semantic_scanning', False)
    sam_feature_cluster = getattr(args, 'use_sam_feature_cluster_scanning', False)
    wavelet_augmented_ss1 = getattr(args, 'use_wavelet_augmented_ss1', False)
    wavelet_priors = getattr(args, 'use_wavelet_priors', False)
    joint_wavelet = getattr(args, 'use_joint_spatial_spectral_wavelet_prior', False)
    bsw_loss = getattr(args, 'use_boundary_selective_wavelet_loss', False)
    offline_sam_cache = getattr(args, 'use_offline_sam_cache', False)
    wavelet_augmented_ss1_count = int(getattr(args, 'wavelet_augmented_ss1_count', 6))
    wavelet_augmented_ss1_topk_ratio = float(getattr(args, 'wavelet_augmented_ss1_topk_ratio', 0.25))
    wavelet_augmented_ss1_strength = float(getattr(args, 'wavelet_augmented_ss1_strength', 0.5))
    boundary_selective_wavelet_loss_weight = float(getattr(args, 'boundary_selective_wavelet_loss_weight', 0.003))
    boundary_selective_wavelet_loss_start_epoch = int(getattr(args, 'boundary_selective_wavelet_loss_start_epoch', 80))
    boundary_selective_wavelet_boundary_boost = float(getattr(args, 'boundary_selective_wavelet_boundary_boost', 0.75))
    boundary_selective_wavelet_frequency_boost = float(getattr(args, 'boundary_selective_wavelet_frequency_boost', 0.5))

    official_ss1 = (
        fusion_only_ase_mainline
        and offline_sam_cache
        and sam_region_proto
        and sam_scan
        and not sam_feature_cluster
    )
    official_fcs1 = fusion_only_ase_mainline and offline_sam_cache and sam_scan and sam_feature_cluster
    base_wss1 = official_ss1 and wavelet_augmented_ss1 and joint_wavelet
    official_wss1 = base_wss1 and not bsw_loss
    official_bsw1 = fusion_only_ase_mainline and offline_sam_cache and sam_region_proto and wavelet_priors and joint_wavelet and bsw_loss
    official_ssb1 = official_ss1 and official_bsw1 and not wavelet_augmented_ss1
    official_wssb1 = base_wss1 and official_bsw1
    recommended_wssb1 = (
        official_wssb1
        and wavelet_augmented_ss1_count <= 4
        and wavelet_augmented_ss1_topk_ratio <= 0.12
        and wavelet_augmented_ss1_strength <= 0.30
        and boundary_selective_wavelet_loss_weight <= 0.0025
        and boundary_selective_wavelet_loss_start_epoch >= 100
        and boundary_selective_wavelet_boundary_boost <= 0.65
        and boundary_selective_wavelet_frequency_boost <= 0.45
    )
    partial_ssb1 = sam_scan and bsw_loss and not wavelet_augmented_ss1
    partial_wssb1 = wavelet_augmented_ss1 and bsw_loss

    enabled_profiles = []
    if official_wssb1:
        enabled_profiles.append('WSSB1')
    elif official_ssb1:
        enabled_profiles.append('SSB1')
    else:
        if official_wss1:
            enabled_profiles.append('WSS1')
        if official_ss1:
            enabled_profiles.append('SS1-Core')
        if official_fcs1:
            enabled_profiles.append('FCS1')
        if official_bsw1:
            enabled_profiles.append('BSW1-Core')

    return {
        'fusion_only_ase_mainline': fusion_only_ase_mainline,
        'official_ss1': official_ss1,
        'official_fcs1': official_fcs1,
        'official_wss1': official_wss1,
        'official_bsw1': official_bsw1,
        'official_ssb1': official_ssb1,
        'official_wssb1': official_wssb1,
        'recommended_wssb1': recommended_wssb1,
        'partial_ssb1': partial_ssb1,
        'partial_wssb1': partial_wssb1,
        'enabled_profiles': enabled_profiles,
    }


def _infer_test_sam_cache_path(args):
    dataset = getattr(args, 'dataset', 'cave').lower()
    if dataset == 'chikusei':
        return "./data/chikusei/chikusei_test.whole_sam_cache.h5"
    if dataset == 'pavia_university':
        if getattr(args, 'prefer_scene_sam_cache', False):
            return "./data/pavia_university_x4/pavia_university_test.whole_sam_cache.scene_region.h5"
        if getattr(args, 'prefer_multi_region_sam_cache', False):
            return "./data/pavia_university_x4/pavia_university_test.whole_sam_cache.multi_region.h5"
        return "./data/pavia_university_x4/pavia_university_test.whole_sam_cache.h5"
    if dataset == 'xiongan_new_area':
        if getattr(args, 'prefer_scene_sam_cache', False):
            return "./data/xiongan_new_area_x4/xiongan_new_area_test.whole_sam_cache.scene_region.h5"
        if getattr(args, 'prefer_multi_region_sam_cache', False):
            return "./data/xiongan_new_area_x4/xiongan_new_area_test.whole_sam_cache.multi_region.h5"
        return "./data/xiongan_new_area_x4/xiongan_new_area_test.whole_sam_cache.h5"
    if dataset == 'cave':
        return "./data/cave/cave_test.whole_sam_cache.h5"
    return None


def _infer_test_h5_path(args):
    explicit_h5_path = getattr(args, 'test_h5_path', None)
    if explicit_h5_path:
        return explicit_h5_path

    dataset = getattr(args, 'dataset', 'cave').lower()
    if dataset == 'chikusei':
        base_dir = getattr(args, 'train_data_path', None) or './data/chikusei'
        return os.path.join(base_dir, 'chikusei_test.h5').replace("\\", "/")
    if dataset == 'pavia_university':
        base_dir = getattr(args, 'train_data_path', None) or './data/pavia_university_x4'
        return os.path.join(base_dir, 'pavia_university_test.h5').replace("\\", "/")
    if dataset == 'xiongan_new_area':
        base_dir = getattr(args, 'train_data_path', None) or './data/xiongan_new_area_x4'
        return os.path.join(base_dir, 'xiongan_new_area_test.h5').replace("\\", "/")
    if dataset == 'cave':
        return './data/cave/cave_test.h5'
    return None


def _infer_test_sam_cache_path_for_run(args):
    explicit_cache_path = getattr(args, 'test_sam_cache_path', None)
    if explicit_cache_path:
        return explicit_cache_path

    test_h5_path = _infer_test_h5_path(args)
    if not test_h5_path:
        return _infer_test_sam_cache_path(args)

    base_path, _ = os.path.splitext(test_h5_path)
    if getattr(args, 'prefer_scene_sam_cache', False):
        return (base_path + '.whole_sam_cache.scene_region.h5').replace("\\", "/")
    if getattr(args, 'prefer_multi_region_sam_cache', False):
        return (base_path + '.whole_sam_cache.multi_region.h5').replace("\\", "/")
    return (base_path + '.whole_sam_cache.h5').replace("\\", "/")


def _build_test_command(args, custom_weight_dir):
    cmd_parts = [
        "python", "test.py",
        "--dataset", str(args.dataset),
        "--weight", os.path.join(custom_weight_dir, "best_by_psnr.pth").replace("\\", "/"),
        "--device", str(args.device),
        "--channels", str(args.channels),
    ]

    _append_option(cmd_parts, "--ratio", getattr(args, 'ratio', None))
    _append_option(cmd_parts, "--h5_path", _infer_test_h5_path(args))
    _append_option(cmd_parts, "--ase_prompt_mode", getattr(args, 'ase_prompt_mode', None))
    _append_option(cmd_parts, "--ase_route_temperature", getattr(args, 'ase_route_temperature', None))
    _append_option(cmd_parts, "--ase_prompt_soft_mix", getattr(args, 'ase_prompt_soft_mix', None))
    _append_option(cmd_parts, "--ase_scope", getattr(args, 'ase_scope', None))
    _append_option(cmd_parts, "--ase_stage_scope", getattr(args, 'ase_stage_scope', None))
    _append_option(cmd_parts, "--ase_fusion_res_scale", getattr(args, 'ase_fusion_res_scale', None))
    _append_option(cmd_parts, "--ase_stage_res_scales", getattr(args, 'ase_stage_res_scales', None))

    _append_flag(cmd_parts, getattr(args, 'use_ase', False), "--use_ase")
    _append_flag(cmd_parts, getattr(args, 'use_sam_ase', False), "--use_sam_ase")
    _append_flag(cmd_parts, getattr(args, 'use_ase_fusion_residual', False), "--use_ase_fusion_residual")
    _append_flag(cmd_parts, getattr(args, 'use_learnable_ase_fusion_res_scale', False), "--use_learnable_ase_fusion_res_scale")
    _append_flag(cmd_parts, getattr(args, 'use_offline_sam_cache', False), "--use_offline_sam_cache")
    _append_flag(cmd_parts, getattr(args, 'prefer_scene_sam_cache', False), "--prefer_scene_sam_cache")
    _append_flag(cmd_parts, getattr(args, 'prefer_multi_region_sam_cache', False), "--prefer_multi_region_sam_cache")
    _append_flag(cmd_parts, getattr(args, 'use_learnable_prompts', False), "--use_learnable_prompts")
    _append_flag(cmd_parts, getattr(args, 'use_soft_masks', False), "--use_soft_masks")
    _append_flag(cmd_parts, getattr(args, 'use_structure_guided_sam_ase', False), "--use_structure_guided_sam_ase")
    _append_flag(cmd_parts, getattr(args, 'use_wavelet', False), "--use_wavelet")
    _append_flag(cmd_parts, getattr(args, 'use_wavelet_priors', False), "--use_wavelet_priors")
    _append_flag(cmd_parts, getattr(args, 'use_wavelet_local_bias', False), "--use_wavelet_local_bias")
    _append_flag(cmd_parts, getattr(args, 'use_wavelet_local_gate', False), "--use_wavelet_local_gate")
    _append_flag(cmd_parts, getattr(args, 'use_sam_local_gate', False), "--use_sam_local_gate")
    _append_flag(cmd_parts, getattr(args, 'use_sam_semantic_prompt_bank', False), "--use_sam_semantic_prompt_bank")
    _append_flag(cmd_parts, getattr(args, 'use_sam_region_prototype_bank', False), "--use_sam_region_prototype_bank")
    _append_flag(cmd_parts, getattr(args, 'use_sam_region_prompt_mixture', False), "--use_sam_region_prompt_mixture")
    _append_flag(cmd_parts, getattr(args, 'use_wavelet_guided_sam_prototype_scaling', False), "--use_wavelet_guided_sam_prototype_scaling")
    _append_flag(cmd_parts, getattr(args, 'use_sam_guided_semantic_scanning', False), "--use_sam_guided_semantic_scanning")
    _append_flag(cmd_parts, getattr(args, 'use_sam_feature_cluster_scanning', False), "--use_sam_feature_cluster_scanning")
    _append_flag(cmd_parts, getattr(args, 'use_wavelet_augmented_ss1', False), "--use_wavelet_augmented_ss1")
    _append_flag(cmd_parts, getattr(args, 'use_sam_boundary_aware_state_propagation', False), "--use_sam_boundary_aware_state_propagation")
    _append_flag(cmd_parts, getattr(args, 'use_sam_state_reset_stronger', False), "--use_sam_state_reset_stronger")
    _append_flag(cmd_parts, getattr(args, 'use_sam_state_organizer_v1', False), "--use_sam_state_organizer_v1")
    _append_flag(cmd_parts, getattr(args, 'use_sam_region_prompt_subspace', False), "--use_sam_region_prompt_subspace")
    _append_flag(cmd_parts, getattr(args, 'use_wavelet_guided_semantic_state_organization', False), "--use_wavelet_guided_semantic_state_organization")
    _append_flag(cmd_parts, getattr(args, 'use_joint_spatial_spectral_wavelet_prior', False), "--use_joint_spatial_spectral_wavelet_prior")
    _append_flag(cmd_parts, getattr(args, 'use_dual_prototype_bank', False), "--use_dual_prototype_bank")
    _append_flag(cmd_parts, getattr(args, 'use_semantic_frequency_state_modulation', False), "--use_semantic_frequency_state_modulation")
    _append_flag(cmd_parts, getattr(args, 'use_fass', False), "--use_fass")
    _append_flag(cmd_parts, getattr(args, 'use_semantic_frequency_adaptive_scanning', False), "--use_semantic_frequency_adaptive_scanning")
    _append_flag(cmd_parts, getattr(args, 'use_sam_prior_bank', False), "--use_sam_prior_bank")
    _append_flag(cmd_parts, getattr(args, 'use_sam_prior_c_refiner', False), "--use_sam_prior_c_refiner")

    _append_option(cmd_parts, "--sam_checkpoint", getattr(args, 'sam_checkpoint', None))
    if getattr(args, 'use_offline_sam_cache', False):
        _append_option(cmd_parts, "--sam_cache_path", _infer_test_sam_cache_path_for_run(args))
    if getattr(args, 'use_wavelet_local_bias', False):
        _append_option(cmd_parts, "--wavelet_local_bias_scale", getattr(args, 'wavelet_local_bias_scale', None))
    if getattr(args, 'use_wavelet_local_gate', False):
        _append_option(cmd_parts, "--wavelet_local_gate_scale", getattr(args, 'wavelet_local_gate_scale', None))
    if getattr(args, 'use_sam_local_gate', False):
        _append_option(cmd_parts, "--sam_local_gate_scale", getattr(args, 'sam_local_gate_scale', None))
    if getattr(args, 'use_sam_semantic_prompt_bank', False):
        _append_option(cmd_parts, "--sam_semantic_prompt_bank_scale", getattr(args, 'sam_semantic_prompt_bank_scale', None))
    if getattr(args, 'use_sam_region_prototype_bank', False):
        _append_option(cmd_parts, "--sam_region_prototype_bank_scale", getattr(args, 'sam_region_prototype_bank_scale', None))
        _append_option(cmd_parts, "--sam_region_prototype_count", getattr(args, 'sam_region_prototype_count', None))
    if getattr(args, 'use_wavelet_guided_sam_prototype_scaling', False):
        _append_option(cmd_parts, "--wavelet_guided_sam_prototype_scale", getattr(args, 'wavelet_guided_sam_prototype_scale', None))
    if getattr(args, 'use_sam_region_prompt_mixture', False):
        _append_option(cmd_parts, "--sam_region_prompt_mixture_scale", getattr(args, 'sam_region_prompt_mixture_scale', None))
        _append_option(cmd_parts, "--sam_region_prompt_mixture_count", getattr(args, 'sam_region_prompt_mixture_count', None))
    if getattr(args, 'use_sam_guided_semantic_scanning', False):
        _append_option(cmd_parts, "--sam_semantic_scanning_count", getattr(args, 'sam_semantic_scanning_count', None))
    if getattr(args, 'use_sam_feature_cluster_scanning', False):
        _append_option(cmd_parts, "--sam_feature_cluster_count", getattr(args, 'sam_feature_cluster_count', None))
        _append_option(cmd_parts, "--sam_feature_cluster_iters", getattr(args, 'sam_feature_cluster_iters', None))
        _append_option(cmd_parts, "--sam_feature_cluster_spatial_weight", getattr(args, 'sam_feature_cluster_spatial_weight', None))
    if getattr(args, 'use_wavelet_augmented_ss1', False):
        _append_option(cmd_parts, "--wavelet_augmented_ss1_count", getattr(args, 'wavelet_augmented_ss1_count', None))
        _append_option(cmd_parts, "--wavelet_augmented_ss1_topk_ratio", getattr(args, 'wavelet_augmented_ss1_topk_ratio', None))
        _append_option(cmd_parts, "--wavelet_augmented_ss1_strength", getattr(args, 'wavelet_augmented_ss1_strength', None))
        _append_option(cmd_parts, "--wavelet_augmented_ss1_mode", getattr(args, 'wavelet_augmented_ss1_mode', None))
    if getattr(args, 'use_sam_boundary_aware_state_propagation', False):
        _append_option(cmd_parts, "--sam_boundary_aware_state_scale", getattr(args, 'sam_boundary_aware_state_scale', None))
    if getattr(args, 'use_sam_state_reset_stronger', False):
        _append_option(cmd_parts, "--sam_state_reset_scale", getattr(args, 'sam_state_reset_scale', None))
    if getattr(args, 'use_sam_state_organizer_v1', False):
        _append_option(cmd_parts, "--sam_state_organizer_count", getattr(args, 'sam_state_organizer_count', None))
        _append_option(cmd_parts, "--sam_state_organizer_boundary_scale", getattr(args, 'sam_state_organizer_boundary_scale', None))
        _append_option(cmd_parts, "--sam_state_organizer_reset_scale", getattr(args, 'sam_state_organizer_reset_scale', None))
    if getattr(args, 'use_sam_region_prompt_subspace', False):
        _append_option(cmd_parts, "--sam_region_prompt_subspace_scale", getattr(args, 'sam_region_prompt_subspace_scale', None))
        _append_option(cmd_parts, "--sam_region_prompt_subspace_count", getattr(args, 'sam_region_prompt_subspace_count', None))
    if getattr(args, 'use_wavelet_guided_semantic_state_organization', False):
        _append_option(cmd_parts, "--wavelet_guided_semantic_state_count", getattr(args, 'wavelet_guided_semantic_state_count', None))
        _append_option(cmd_parts, "--wavelet_guided_semantic_state_scale", getattr(args, 'wavelet_guided_semantic_state_scale', None))
        _append_option(cmd_parts, "--wavelet_guided_semantic_boundary_scale", getattr(args, 'wavelet_guided_semantic_boundary_scale', None))
        _append_option(cmd_parts, "--wavelet_guided_semantic_reset_scale", getattr(args, 'wavelet_guided_semantic_reset_scale', None))
    if getattr(args, 'use_joint_spatial_spectral_wavelet_prior', False):
        _append_option(cmd_parts, "--joint_wavelet_spatial_weight", getattr(args, 'joint_wavelet_spatial_weight', None))
        _append_option(cmd_parts, "--joint_wavelet_spectral_weight", getattr(args, 'joint_wavelet_spectral_weight', None))
    if getattr(args, 'use_dual_prototype_bank', False):
        _append_option(cmd_parts, "--dual_prototype_semantic_scale", getattr(args, 'dual_prototype_semantic_scale', None))
        _append_option(cmd_parts, "--dual_prototype_frequency_scale", getattr(args, 'dual_prototype_frequency_scale', None))
        _append_option(cmd_parts, "--dual_prototype_count", getattr(args, 'dual_prototype_count', None))
    if getattr(args, 'use_semantic_frequency_state_modulation', False):
        _append_option(cmd_parts, "--semantic_frequency_state_count", getattr(args, 'semantic_frequency_state_count', None))
        _append_option(cmd_parts, "--semantic_frequency_state_write_scale", getattr(args, 'semantic_frequency_state_write_scale', None))
        _append_option(cmd_parts, "--semantic_frequency_state_read_scale", getattr(args, 'semantic_frequency_state_read_scale', None))
        _append_option(cmd_parts, "--semantic_frequency_state_delta_scale", getattr(args, 'semantic_frequency_state_delta_scale', None))
    if getattr(args, 'use_structure_guided_sam_ase', False):
        _append_option(cmd_parts, "--structure_texture_weight", getattr(args, 'structure_texture_weight', None))
    if getattr(args, 'use_learnable_prompts', False):
        _append_option(cmd_parts, "--num_learnable_prompts", getattr(args, 'num_learnable_prompts', None))
    if getattr(args, 'use_soft_masks', False):
        _append_option(cmd_parts, "--num_soft_regions", getattr(args, 'num_soft_regions', None))
    if getattr(args, 'use_fass', False):
        _append_option(cmd_parts, "--fass_compression_ratio", getattr(args, 'fass_compression_ratio', None))
        _append_option(cmd_parts, "--fass_ll_sparsity", getattr(args, 'fass_ll_sparsity', None))
        _append_option(cmd_parts, "--fass_hf_sparsity", getattr(args, 'fass_hf_sparsity', None))
        _append_option(cmd_parts, "--fass_d_state", getattr(args, 'fass_d_state', None))
        _append_option(cmd_parts, "--gating_loss_weight", getattr(args, 'gating_loss_weight', None))
        _append_option(cmd_parts, "--gating_input_mode", getattr(args, 'gating_input_mode', None))
        if getattr(args, 'use_semantic_frequency_adaptive_scanning', False):
            _append_option(cmd_parts, "--semantic_frequency_semantic_weight", getattr(args, 'semantic_frequency_semantic_weight', None))
            _append_option(cmd_parts, "--semantic_frequency_wavelet_weight", getattr(args, 'semantic_frequency_wavelet_weight', None))
            _append_option(cmd_parts, "--semantic_frequency_boundary_weight", getattr(args, 'semantic_frequency_boundary_weight', None))
            _append_option(cmd_parts, "--semantic_frequency_confidence_weight", getattr(args, 'semantic_frequency_confidence_weight', None))
            _append_option(cmd_parts, "--semantic_frequency_prompt_weight", getattr(args, 'semantic_frequency_prompt_weight', None))
    if getattr(args, 'use_sam_prior_c_refiner', False):
        _append_option(cmd_parts, "--sam_prior_c_refiner_scope", getattr(args, 'sam_prior_c_refiner_scope', None))
        _append_option(cmd_parts, "--sam_prior_c_refiner_scale", getattr(args, 'sam_prior_c_refiner_scale', None))

    return " ".join(shlex.quote(str(part)) for part in cmd_parts)


def _build_experiment_summary(args):
    pieces = []
    pieces.append(f"x{getattr(args, 'ratio', 4)}")
    route_profiles = _detect_route_profiles(args)
    if route_profiles.get('official_wssb1', False):
        pieces.append("WSSB1")
    elif route_profiles['official_ssb1']:
        pieces.append("SSB1")
    elif route_profiles['official_wss1']:
        pieces.append("WSS1")
    if getattr(args, 'use_ase', False):
        ase_piece = f"ASE[{getattr(args, 'ase_prompt_mode', 'hard')},{getattr(args, 'ase_scope', 'all')},{getattr(args, 'ase_stage_scope', 'all_stages')}]"
        if getattr(args, 'use_ase_fusion_residual', False):
            ase_piece += f"+FusionRes({getattr(args, 'ase_fusion_res_scale', 0.3)})"
        pieces.append(ase_piece)
    else:
        pieces.append("Backbone")

    if getattr(args, 'use_sam_region_prototype_bank', False):
        pieces.append(f"SAMRProto(scale={getattr(args, 'sam_region_prototype_bank_scale', 0.1)},count={getattr(args, 'sam_region_prototype_count', 8)})")
    if getattr(args, 'use_wavelet_guided_sam_prototype_scaling', False):
        pieces.append(f"WaveRProto(scale={getattr(args, 'wavelet_guided_sam_prototype_scale', 0.1)})")
    if getattr(args, 'use_sam_region_relation_loss', False):
        pieces.append(f"SAMRRel(w={getattr(args, 'sam_region_relation_weight', 0.003)},start={getattr(args, 'sam_region_relation_start_epoch', 20)},count={getattr(args, 'sam_region_relation_count', 4)})")
    if getattr(args, 'use_sam_region_prompt_mixture', False):
        pieces.append(f"SAMRPMix(scale={getattr(args, 'sam_region_prompt_mixture_scale', 0.05)},count={getattr(args, 'sam_region_prompt_mixture_count', 8)})")
    if getattr(args, 'use_sam_guided_semantic_scanning', False):
        if getattr(args, 'use_sam_feature_cluster_scanning', False):
            pieces.append(
                "SAMFeatClusterScan("
                f"clusters={getattr(args, 'sam_feature_cluster_count', 6)},"
                f"iters={getattr(args, 'sam_feature_cluster_iters', 2)},"
                f"spatial_w={getattr(args, 'sam_feature_cluster_spatial_weight', 0.05)}"
                ")"
            )
        else:
            pieces.append(f"SAMScan(count={getattr(args, 'sam_semantic_scanning_count', 6)})")
    if getattr(args, 'use_wavelet_augmented_ss1', False):
        pieces.append(
            "WaveletScan("
            f"count={getattr(args, 'wavelet_augmented_ss1_count', 6)},"
            f"topk={getattr(args, 'wavelet_augmented_ss1_topk_ratio', 0.25)},"
            f"strength={getattr(args, 'wavelet_augmented_ss1_strength', 0.5)},"
            f"mode={getattr(args, 'wavelet_augmented_ss1_mode', 'stable_intra_region')}"
            ")"
        )
    if getattr(args, 'use_sam_boundary_aware_state_propagation', False):
        pieces.append(f"SAMBoundary(scale={getattr(args, 'sam_boundary_aware_state_scale', 0.2)})")
    if getattr(args, 'use_sam_state_reset_stronger', False):
        pieces.append(f"SAMReset(scale={getattr(args, 'sam_state_reset_scale', 0.35)})")
    if getattr(args, 'use_sam_state_organizer_v1', False):
        pieces.append(
            f"SAMStateOrg(count={getattr(args, 'sam_state_organizer_count', 6)},"
            f"b={getattr(args, 'sam_state_organizer_boundary_scale', 0.1)},"
            f"r={getattr(args, 'sam_state_organizer_reset_scale', 0.15)})"
        )
    if getattr(args, 'use_sam_region_prompt_subspace', False):
        pieces.append(
            f"SAMPromptSub(scale={getattr(args, 'sam_region_prompt_subspace_scale', 0.05)},"
            f"count={getattr(args, 'sam_region_prompt_subspace_count', 6)})"
        )
    if getattr(args, 'use_wavelet_guided_semantic_state_organization', False):
        pieces.append(
            f"WaveSemState(count={getattr(args, 'wavelet_guided_semantic_state_count', 6)},"
            f"w={getattr(args, 'wavelet_guided_semantic_state_scale', 0.05)})"
        )
    if getattr(args, 'use_joint_spatial_spectral_wavelet_prior', False):
        pieces.append(
            f"JointWave(sw={getattr(args, 'joint_wavelet_spatial_weight', 1.0)},"
            f"spw={getattr(args, 'joint_wavelet_spectral_weight', 1.0)})"
        )
    if getattr(args, 'use_dual_prototype_bank', False):
        pieces.append(
            f"DualProto(s={getattr(args, 'dual_prototype_semantic_scale', 0.05)},"
            f"f={getattr(args, 'dual_prototype_frequency_scale', 0.05)},"
            f"count={getattr(args, 'dual_prototype_count', 6)})"
        )
    if getattr(args, 'use_semantic_frequency_state_modulation', False):
        pieces.append(
            f"SemFreqState(count={getattr(args, 'semantic_frequency_state_count', 6)},"
            f"w={getattr(args, 'semantic_frequency_state_write_scale', 0.08)},"
            f"r={getattr(args, 'semantic_frequency_state_read_scale', 0.08)},"
            f"d={getattr(args, 'semantic_frequency_state_delta_scale', 0.05)})"
        )
    if getattr(args, 'use_sam_semantic_prompt_bank', False):
        pieces.append(f"SAMPBank(scale={getattr(args, 'sam_semantic_prompt_bank_scale', 0.1)})")
    if getattr(args, 'use_sam_local_gate', False):
        pieces.append(f"SAMGate(scale={getattr(args, 'sam_local_gate_scale', 0.1)})")
    if getattr(args, 'use_sam_distillation', False):
        pieces.append("SAMDistill")
    if getattr(args, 'use_wavelet', False):
        pieces.append("LegacyWave")
    if getattr(args, 'use_wavelet_priors', False):
        pieces.append("WavePrior")
    if getattr(args, 'use_wavelet_local_bias', False):
        pieces.append(f"WaveBias(scale={getattr(args, 'wavelet_local_bias_scale', 0.1)})")
    if getattr(args, 'use_wavelet_local_gate', False):
        pieces.append(f"WaveGate(scale={getattr(args, 'wavelet_local_gate_scale', 0.1)})")
    if getattr(args, 'use_hf_wavelet_loss', False):
        pieces.append(f"HFWLoss(w={getattr(args, 'hf_wavelet_loss_weight', 0.01)},start={getattr(args, 'hf_wavelet_loss_start_epoch', 5)})")
    if getattr(args, 'use_semantic_region_weighted_hf_wavelet_loss', False):
        pieces.append(
            f"SemRegHFWLoss(w={getattr(args, 'semantic_region_hf_wavelet_loss_weight', 0.003)},"
            f"start={getattr(args, 'semantic_region_hf_wavelet_loss_start_epoch', 100)})"
        )
    if getattr(args, 'use_boundary_selective_wavelet_loss', False):
        pieces.append(
            f"BoundSelHFWLoss(w={getattr(args, 'boundary_selective_wavelet_loss_weight', 0.003)},"
            f"start={getattr(args, 'boundary_selective_wavelet_loss_start_epoch', 80)})"
        )
    if getattr(args, 'use_semantic_frequency_adaptive_scanning', False):
        pieces.append(
            f"FASS2(sw={getattr(args, 'semantic_frequency_semantic_weight', 1.0)},"
            f"ww={getattr(args, 'semantic_frequency_wavelet_weight', 1.0)})"
        )
    if getattr(args, 'use_offline_sam_cache', False):
        pieces.append("OfflineSAMCache")
    pieces.append(f"lr={args.lr}")
    pieces.append(f"epoch={args.epoch}")
    pieces.append(f"step={args.step}")
    pieces.append(f"bs={args.batch_size}")
    return "; ".join(pieces)


def _write_experiment_index(weight_root, exp_code, current_time, args, custom_weight_dir):
    index_path = _get_exp_index_path(weight_root)
    summary = _build_experiment_summary(args)
    rel_dir = os.path.relpath(custom_weight_dir, weight_root).replace("\\", "/")
    row = f"| {exp_code} | {current_time} | {args.dataset} | {summary} | `{rel_dir}` |\n"

    if not os.path.exists(index_path):
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write("# Experiment Index\n\n")
            f.write("<!-- AUTO_INDEX_ROWS_START -->\n")
            f.write("| Code | Time | Dataset | Summary | Folder |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            f.write("<!-- AUTO_INDEX_ROWS_END -->\n")

    with open(index_path, 'r', encoding='utf-8') as f:
        existing_text = f.read()
    if f"`{rel_dir}`" in existing_text:
        return

    if "<!-- AUTO_INDEX_ROWS_END -->" in existing_text:
        updated_text = existing_text.replace("<!-- AUTO_INDEX_ROWS_END -->", row + "<!-- AUTO_INDEX_ROWS_END -->")
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(updated_text)
    else:
        with open(index_path, 'a', encoding='utf-8') as f:
            f.write(row)


def _write_experiment_record(args, custom_weight_dir, exp_code, current_time):
    train_command = _build_train_command()
    test_command = _build_test_command(args, custom_weight_dir)
    summary = _build_experiment_summary(args)
    args_items = sorted(vars(args).items(), key=lambda item: item[0])

    lines = [
        "=" * 72,
        "Experiment Record",
        "=" * 72,
        f"Experiment Code: {exp_code}",
        f"Created Time: {current_time}",
        f"Dataset: {args.dataset}",
        f"Run Folder Name: {os.path.basename(custom_weight_dir)}",
        f"Weight Directory: {custom_weight_dir}",
        f"Summary: {summary}",
        "",
        "[Train Command]",
        train_command,
        "",
        "[Test Command]",
        test_command,
        "",
        "[Key Paths]",
        f"Best PSNR Weight: {os.path.join(custom_weight_dir, 'best_by_psnr.pth')}",
        f"Training Log: {os.path.join(custom_weight_dir, 'training_log.txt')}",
        "",
        "[All Args]",
    ]

    for key, value in args_items:
        lines.append(f"{key}: {value}")
    content = "\n".join(lines) + "\n"

    for filename in ["experiment_record.txt", "training_params.txt"]:
        with open(os.path.join(custom_weight_dir, filename), 'w', encoding='utf-8') as f:
            f.write(content)


def create_custom_weight_dir(args):
    from datetime import datetime


    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    weight_root = _get_weight_root(args)
    os.makedirs(weight_root, exist_ok=True)
    resume_path = getattr(args, 'resume', None)

    if resume_path:
        resume_path = os.path.abspath(resume_path)
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        custom_weight_dir = os.path.dirname(resume_path)
        if not os.path.isdir(custom_weight_dir):
            raise NotADirectoryError(f"Resume directory not found: {custom_weight_dir}")

        inferred_exp_code = _parse_exp_code_from_run_folder(os.path.basename(custom_weight_dir))
        if inferred_exp_code:
            args.exp_code = inferred_exp_code
        elif getattr(args, 'exp_code', None):
            args.exp_code = args.exp_code.strip()

        if getattr(args, 'dataset', None):
            args.dataset = _normalize_dataset_slug(getattr(args, 'dataset'))

        index_path = _get_exp_index_path(weight_root)
        if os.path.exists(index_path):
            _write_experiment_index(weight_root, getattr(args, 'exp_code', 'RESUME'), current_time, args, custom_weight_dir)
        return custom_weight_dir

    exp_code = _get_or_create_experiment_code(args, weight_root)
    args.exp_code = exp_code
    dataset_slug = _normalize_dataset_slug(getattr(args, 'dataset', 'dataset'))
    run_folder_name = f"{exp_code}_{dataset_slug}_x{getattr(args, 'ratio', 4)}_{current_time}"
    custom_weight_dir = os.path.join(weight_root, run_folder_name)

    if os.path.exists(custom_weight_dir) and os.listdir(custom_weight_dir):
        raise FileExistsError(
            f"Experiment directory already exists and is not empty: {custom_weight_dir}. "
            f"Please choose another --exp_code or rerun at a different time."
        )

    os.makedirs(custom_weight_dir, exist_ok=True)
    _write_experiment_record(args, custom_weight_dir, exp_code, current_time)
    _write_experiment_index(weight_root, exp_code, current_time, args, custom_weight_dir)
    return custom_weight_dir


    use_fass = getattr(args, 'use_fass', False)
    use_sam_ase = getattr(args, 'use_sam_ase', False)
    use_ase = getattr(args, 'use_ase', False)

    if use_fass and use_sam_ase:

        sam_checkpoint = getattr(args, 'sam_checkpoint', None)
        if sam_checkpoint:
            if 'vit_h' in sam_checkpoint or 'huge' in sam_checkpoint:
                sam_tag = "FASS-SAM-ASE-H"
            elif 'vit_l' in sam_checkpoint or 'large' in sam_checkpoint:
                sam_tag = "FASS-SAM-ASE-L"
            elif 'vit_b' in sam_checkpoint or 'base' in sam_checkpoint:
                sam_tag = "FASS-SAM-ASE-B"
            else:
                sam_tag = "FASS-SAM-ASE"
        else:
            sam_tag = "FASS-SAM-ASE-Adapter"


        train_mode = getattr(args, 'train_mode', 'auto')
        if train_mode == 'auto':
            dense_epochs = getattr(args, 'dense_epochs', 100)
            mode_tag = f"_auto_ep{dense_epochs}"
        elif train_mode == 'dense':
            mode_tag = "_dense"
        else:
            mode_tag = "_sparse"

        ase_tag = f"{sam_tag}{mode_tag}"
    elif use_sam_ase:

        sam_checkpoint = getattr(args, 'sam_checkpoint', None)
        if sam_checkpoint:
            if 'vit_h' in sam_checkpoint or 'huge' in sam_checkpoint:
                sam_tag = "SAM-ASE-H"
            elif 'vit_l' in sam_checkpoint or 'large' in sam_checkpoint:
                sam_tag = "SAM-ASE-L"
            elif 'vit_b' in sam_checkpoint or 'base' in sam_checkpoint:
                sam_tag = "SAM-ASE-B"
            else:
                sam_tag = "SAM-ASE"
        else:
            sam_tag = "SAM-ASE-Adapter"
        ase_tag = sam_tag
    elif use_ase:
        ase_prompt_mode = getattr(args, 'ase_prompt_mode', 'hard')
        ase_scope = getattr(args, 'ase_scope', 'all')
        ase_stage_scope = getattr(args, 'ase_stage_scope', 'all_stages')
        ase_stage_res_scales = getattr(args, 'ase_stage_res_scales', None)
        use_ase_fusion_residual = getattr(args, 'use_ase_fusion_residual', False)
        if ase_prompt_mode == 'hard':
            ase_tag = "ASE"
        else:
            ase_tag = f"ASE-{ase_prompt_mode}"
        if ase_scope == 'fusion_only':
            ase_tag += "-FusionOnly"
        if ase_stage_scope == 'deep34':
            ase_tag += "-Deep34"
        elif ase_stage_scope == 'deep234':
            ase_tag += "-Deep234"
        elif ase_stage_scope == 'stage4_only':
            ase_tag += "-Stage4Only"
        if use_ase_fusion_residual:
            ase_tag += "-FusionRes"
        if getattr(args, 'use_learnable_ase_fusion_res_scale', False):
            ase_tag += "-LearnableRes"
        if ase_stage_res_scales:
            scale_tag = str(ase_stage_res_scales).replace(' ', '').replace('.', 'p').replace(',', '-')
            ase_tag += f"-StageScale-{scale_tag}"
    else:
        ase_tag = "NoASE"

    wavelet_tag = "LegacyWave" if getattr(args, 'use_wavelet', False) else "NoLegacyWave"
    wavelet_prior_tag = "WavePrior" if getattr(args, 'use_wavelet_priors', False) else "NoWavePrior"
    wavelet_local_bias_tag = "WaveBias" if getattr(args, 'use_wavelet_local_bias', False) else "NoWaveBias"
    wavelet_local_gate_tag = "WaveGate" if getattr(args, 'use_wavelet_local_gate', False) else "NoWaveGate"
    sam_local_gate_tag = "SAMGate" if getattr(args, 'use_sam_local_gate', False) else "NoSAMGate"
    sam_semantic_prompt_bank_tag = "SAMPBank" if getattr(args, 'use_sam_semantic_prompt_bank', False) else "NoSAMPBank"
    sam_region_prototype_bank_tag = "SAMRProto" if getattr(args, 'use_sam_region_prototype_bank', False) else "NoSAMRProto"
    sam_region_relation_tag = "SAMRRel" if getattr(args, 'use_sam_region_relation_loss', False) else "NoSAMRRel"
    sam_region_prompt_mixture_tag = "SAMRPMix" if getattr(args, 'use_sam_region_prompt_mixture', False) else "NoSAMRPMix"
    sam_distill_tag = "SAMDistill" if getattr(args, 'use_sam_distillation', False) else "NoSAMDistill"
    sam_cache_tag = "SAMCache" if getattr(args, 'use_offline_sam_cache', False) else "OnlineSAM"
    hf_wavelet_loss_tag = "HFWLoss" if getattr(args, 'use_hf_wavelet_loss', False) else "NoHFWLoss"
    sam_prior_bank_tag = "PriorBank" if getattr(args, 'use_sam_prior_bank', False) else "NoPriorBank"
    effective_prior_c_refiner = (
        getattr(args, 'use_sam_prior_c_refiner', False)
        and getattr(args, 'use_sam_ase', False)
        and not getattr(args, 'use_fass', False)
    )
    sam_prior_c_refiner_scope = getattr(args, 'sam_prior_c_refiner_scope', 'fusion')
    if effective_prior_c_refiner:
        sam_prior_c_refiner_tag = (
            "PriorCRefFusion" if sam_prior_c_refiner_scope == 'fusion' else "PriorCRefAll"
        )
    else:
        sam_prior_c_refiner_tag = "NoPriorCRef"


    lr_str = f"{args.lr:.0e}".replace('.', '').replace('e', 'p')
    dir_name = f"{args.dataset}_{ase_tag}_{wavelet_tag}_{wavelet_prior_tag}_{wavelet_local_bias_tag}_{wavelet_local_gate_tag}_{sam_local_gate_tag}_{sam_semantic_prompt_bank_tag}_{sam_region_prototype_bank_tag}_{sam_region_relation_tag}_{sam_region_prompt_mixture_tag}_{sam_distill_tag}_{sam_cache_tag}_{hf_wavelet_loss_tag}_{sam_prior_bank_tag}_{sam_prior_c_refiner_tag}_lr{lr_str}_{current_time}"


    custom_weight_dir = os.path.join("./weights", dir_name)
    os.makedirs(custom_weight_dir, exist_ok=True)


    save_training_params(args, custom_weight_dir, current_time)

    return custom_weight_dir

def save_training_params(args, weight_dir, current_time):
    params_file = os.path.join(weight_dir, "training_params.txt")
    route_profiles = _detect_route_profiles(args)

    enabled_modules = []
    if getattr(args, 'use_ase', False):
        enabled_modules.append('ASE')
    if getattr(args, 'use_sam_ase', False):
        enabled_modules.append('SAM-ASE')
    if getattr(args, 'use_fass', False):
        enabled_modules.append('FASS')
    if getattr(args, 'use_offline_sam_cache', False):
        enabled_modules.append('Offline SAM Cache')
    if getattr(args, 'use_wavelet', False):
        enabled_modules.append('Legacy Wavelet Backbone')
    if getattr(args, 'use_wavelet_priors', False):
        enabled_modules.append('Wavelet Priors')
    if getattr(args, 'use_joint_spatial_spectral_wavelet_prior', False):
        enabled_modules.append('Joint Spatial-Spectral Prior')
    if getattr(args, 'use_sam_region_prototype_bank', False):
        enabled_modules.append('SAM Region Prototype Bank')
    if getattr(args, 'use_sam_guided_semantic_scanning', False):
        if getattr(args, 'use_sam_feature_cluster_scanning', False):
            enabled_modules.append('SAM Feature Clustering Scan')
        else:
            enabled_modules.append('SAM Semantic Scanning')
    if getattr(args, 'use_dual_prototype_bank', False):
        enabled_modules.append('Dual Prototype Bank')
    if getattr(args, 'use_semantic_frequency_state_modulation', False):
        enabled_modules.append('Semantic-Frequency State Modulation')
    if getattr(args, 'use_boundary_selective_wavelet_loss', False):
        enabled_modules.append('Boundary-Selective Wavelet Loss')
    if getattr(args, 'use_semantic_region_weighted_hf_wavelet_loss', False):
        enabled_modules.append('Semantic-Region HF Wavelet Loss')
    if getattr(args, 'use_hf_wavelet_loss', False):
        enabled_modules.append('HF Wavelet Loss')
    if getattr(args, 'use_semantic_frequency_adaptive_scanning', False):
        enabled_modules.append('Semantic-Frequency Adaptive Scanning')
    if route_profiles.get('official_wssb1', False):
        enabled_modules.append('WSS1C+LightBSW1 Official Combined Route')
    elif route_profiles['official_ssb1']:
        enabled_modules.append('SS1+BSW1 Official Combined Route')

    notes = []
    if route_profiles['enabled_profiles']:
        notes.append("Route profiles: " + ", ".join(route_profiles['enabled_profiles']))
    if route_profiles['partial_ssb1'] and not route_profiles['official_ssb1']:
        notes.append(
            "Partial SS1+BSW1 combo detected; official SSB1 expects "
            "ASE fusion_only residual + offline SAM cache + SAM region prototype bank "
            "+ joint spatial-spectral wavelet prior"
        )
    if route_profiles.get('partial_wssb1', False) and not route_profiles.get('official_wssb1', False):
        notes.append(
            "Partial WSS1+BSW1 combo detected; official WSSB1 expects "
            "WSS1 mainline + wavelet priors + SAM region prototype bank + boundary-selective wavelet loss"
        )
    if route_profiles.get('official_wssb1', False) and not route_profiles.get('recommended_wssb1', False):
        notes.append(
            "Current WSSB1 is heavier than the recommended light profile "
            "(scan count<=4, topk<=0.12, strength<=0.30, loss<=0.0025, start>=100)."
        )
    if getattr(args, 'use_wavelet_guided_sam_prototype_scaling', False):
        notes.append(f"Wavelet-guided prototype scale={getattr(args, 'wavelet_guided_sam_prototype_scale', 0.1)}")
    if getattr(args, 'use_wavelet_guided_semantic_state_organization', False):
        notes.append(
            "Wavelet-guided semantic state "
            f"count={getattr(args, 'wavelet_guided_semantic_state_count', 6)}, "
            f"scale={getattr(args, 'wavelet_guided_semantic_state_scale', 0.05)}"
        )
    if getattr(args, 'use_dual_prototype_bank', False):
        notes.append(
            "Dual prototype bank "
            f"semantic_scale={getattr(args, 'dual_prototype_semantic_scale', 0.05)}, "
            f"frequency_scale={getattr(args, 'dual_prototype_frequency_scale', 0.05)}, "
            f"count={getattr(args, 'dual_prototype_count', 6)}"
        )
    if getattr(args, 'use_semantic_frequency_state_modulation', False):
        notes.append(
            "Semantic-frequency state modulation "
            f"count={getattr(args, 'semantic_frequency_state_count', 6)}, "
            f"write={getattr(args, 'semantic_frequency_state_write_scale', 0.08)}, "
            f"read={getattr(args, 'semantic_frequency_state_read_scale', 0.08)}, "
            f"delta={getattr(args, 'semantic_frequency_state_delta_scale', 0.05)}"
        )
    if getattr(args, 'use_sam_feature_cluster_scanning', False):
        notes.append(
            "SAM feature clustering scan "
            f"clusters={getattr(args, 'sam_feature_cluster_count', 6)}, "
            f"iters={getattr(args, 'sam_feature_cluster_iters', 2)}, "
            f"spatial_weight={getattr(args, 'sam_feature_cluster_spatial_weight', 0.05)}"
        )
    if getattr(args, 'use_boundary_selective_wavelet_loss', False):
        notes.append(
            "Boundary-selective wavelet loss "
            f"weight={getattr(args, 'boundary_selective_wavelet_loss_weight', 0.003)}, "
            f"start_epoch={getattr(args, 'boundary_selective_wavelet_loss_start_epoch', 80)}, "
            f"boundary_boost={getattr(args, 'boundary_selective_wavelet_boundary_boost', 0.75)}, "
            f"frequency_boost={getattr(args, 'boundary_selective_wavelet_frequency_boost', 0.5)}"
        )

    full_command = "python train.py " + " ".join(shlex.quote(arg) for arg in sys.argv[1:])

    with open(params_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Training Parameters\n")
        f.write("=" * 70 + "\n")
        f.write(f"Start Time: {current_time}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Experiment Code: {getattr(args, 'exp_code', None)}\n")
        f.write(f"Weight Directory: {weight_dir}\n\n")

        f.write("[Model]\n")
        f.write(f"Channels: {args.channels}\n")
        f.write(f"Scale Ratio: {args.ratio}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Route Profiles: {', '.join(route_profiles['enabled_profiles']) if route_profiles['enabled_profiles'] else 'None'}\n")
        f.write(f"ASE Enabled: {getattr(args, 'use_ase', False)}\n")
        f.write(f"ASE Scope: {getattr(args, 'ase_scope', None)}\n")
        f.write(f"ASE Stage Scope: {getattr(args, 'ase_stage_scope', None)}\n")
        f.write(f"ASE Fusion Residual: {getattr(args, 'use_ase_fusion_residual', False)}\n")
        f.write(f"ASE Fusion Residual Scale: {getattr(args, 'ase_fusion_res_scale', None)}\n")
        f.write(f"ASE Prompt Mode: {getattr(args, 'ase_prompt_mode', None)}\n")
        f.write(f"ASE Route Temperature: {getattr(args, 'ase_route_temperature', None)}\n")
        f.write(f"SAM-ASE Enabled: {getattr(args, 'use_sam_ase', False)}\n")
        f.write(f"SAM Checkpoint: {getattr(args, 'sam_checkpoint', None)}\n")
        f.write(f"Offline SAM Cache: {getattr(args, 'use_offline_sam_cache', False)}\n")
        f.write(f"SAM Cache Path: {getattr(args, 'sam_cache_path', None)}\n")
        f.write(f"Prefer Scene SAM Cache: {getattr(args, 'prefer_scene_sam_cache', False)}\n")
        f.write(f"Prefer Multi-Region SAM Cache: {getattr(args, 'prefer_multi_region_sam_cache', False)}\n")
        f.write(f"SAM Guided Semantic Scanning: {getattr(args, 'use_sam_guided_semantic_scanning', False)}\n")
        f.write(f"SAM Semantic Scanning Count: {getattr(args, 'sam_semantic_scanning_count', 6)}\n")
        f.write(f"SAM Feature Cluster Scanning: {getattr(args, 'use_sam_feature_cluster_scanning', False)}\n")
        f.write(f"SAM Feature Cluster Count: {getattr(args, 'sam_feature_cluster_count', 6)}\n")
        f.write(f"SAM Feature Cluster Iters: {getattr(args, 'sam_feature_cluster_iters', 2)}\n")
        f.write(f"SAM Feature Cluster Spatial Weight: {getattr(args, 'sam_feature_cluster_spatial_weight', 0.05)}\n")
        f.write(f"Use Wavelet Priors: {getattr(args, 'use_wavelet_priors', False)}\n")
        f.write(f"Use Legacy Wavelet Backbone: {getattr(args, 'use_wavelet', False)}\n")
        f.write(f"Use Joint Spatial-Spectral Prior: {getattr(args, 'use_joint_spatial_spectral_wavelet_prior', False)}\n")
        f.write(f"Joint Spatial Weight: {getattr(args, 'joint_wavelet_spatial_weight', None)}\n")
        f.write(f"Joint Spectral Weight: {getattr(args, 'joint_wavelet_spectral_weight', None)}\n\n")

        f.write("[Optimization]\n")
        f.write(f"Epochs: {args.epoch}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"LR Step: {args.step}\n")
        f.write(f"LR Decay: {args.decay}\n")
        f.write(f"Checkpoint Interval: {args.ckpt}\n")
        f.write(f"Validation Frequency: {args.val_freq}\n")
        f.write(f"Mixed Precision: {getattr(args, 'mixed_precision', True)}\n")
        f.write(f"Num Workers: {getattr(args, 'num_workers', None)}\n")
        f.write(f"Resume Checkpoint: {getattr(args, 'resume', None)}\n")
        f.write(f"Route Regularization Weight: {getattr(args, 'route_reg_weight', None)}\n\n")

        f.write("[Data]\n")
        f.write(f"Train Data Path: {args.train_data_path}\n")
        f.write(f"Validation Data Path: {args.val_data_path}\n")
        f.write(f"Weight Root: {getattr(args, 'weight_dir', 'weights/')}\n\n")

        f.write("[Enabled Modules]\n")
        if enabled_modules:
            for item in enabled_modules:
                f.write(f"- {item}\n")
        else:
            f.write("- Baseline only\n")
        f.write("\n")

        f.write("[Notes]\n")
        if notes:
            for item in notes:
                f.write(f"- {item}\n")
        else:
            f.write("- No extra notes\n")
        f.write("\n")

        f.write("[Reproduce Command]\n")
        f.write(full_command + "\n")
        f.write("=" * 70 + "\n")
def prepare_training_data(args):


    dataset_type = args.dataset.lower()
    if getattr(args, 'use_offline_sam_cache', False):
        print(f"[SAM-CACHE] Offline SAM cache enabled: {getattr(args, 'sam_cache_path', None)}")
        if getattr(args, 'use_learnable_prompts', False):
            print("[SAM-CACHE] WARNING: learnable prompts are enabled; cache will fall back to online SAM")

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
        use_boundary_selective_wavelet_loss=getattr(args, 'use_boundary_selective_wavelet_loss', False),
        use_semantic_frequency_adaptive_scanning=getattr(args, 'use_semantic_frequency_adaptive_scanning', False),
    )

    training_data = HSIMSI_Dataset(
        args.train_data_path,
        is_val=False,
        dataset_type=dataset_type,
        use_wavelet=use_wavelet_side_inputs,
        use_offline_sam_cache=getattr(args, 'use_offline_sam_cache', False),
        sam_cache_path=getattr(args, 'sam_cache_path', None),
        sam_cache_strict=getattr(args, 'sam_cache_strict', False),
        prefer_multi_region_sam_cache=getattr(args, 'prefer_multi_region_sam_cache', False),
        prefer_scene_sam_cache=getattr(args, 'prefer_scene_sam_cache', False),
    )
    if getattr(training_data, 'use_offline_sam_cache', False):
        print(f"[SAM-CACHE] Training cache path: {training_data.sam_cache_path}")

    validate_data = HSIMSI_Dataset(
        args.val_data_path,
        is_val=True,
        dataset_type=dataset_type,
        use_wavelet=use_wavelet_side_inputs,
        use_offline_sam_cache=getattr(args, 'use_offline_sam_cache', False),
        sam_cache_path=getattr(args, 'sam_cache_path', None),
        sam_cache_strict=getattr(args, 'sam_cache_strict', False),
        prefer_multi_region_sam_cache=getattr(args, 'prefer_multi_region_sam_cache', False),
        prefer_scene_sam_cache=getattr(args, 'prefer_scene_sam_cache', False),
    )
    if getattr(validate_data, 'use_offline_sam_cache', False):
        print(f"[SAM-CACHE] Validation cache path: {validate_data.sam_cache_path}")


    if getattr(args, 'num_workers', None) is not None:
        num_workers = max(0, int(args.num_workers))
    else:
        num_workers = min(12, max(4, args.batch_size // 2))
    print(f"[INFO] DataLoader: num_workers={num_workers}, batch_size={args.batch_size}, pin_memory=True")
    val_dataset_size = len(validate_data)
    val_batch_size = min(args.batch_size, max(1, val_dataset_size))
    if val_dataset_size < args.batch_size:
        print(
            f"[INFO] Validation dataset is smaller than batch_size "
            f"({val_dataset_size} < {args.batch_size}); using val_batch_size={val_batch_size}, drop_last=False"
        )

    training_data_loader = DataLoader(dataset=training_data, num_workers=num_workers, batch_size=args.batch_size,
                                      shuffle=True, pin_memory=True, drop_last=True,
                                      prefetch_factor=2 if num_workers > 0 else None,
                                      persistent_workers=True if num_workers > 0 else False)
    validate_data_loader = DataLoader(dataset=validate_data, num_workers=min(4, num_workers), batch_size=val_batch_size,
                                      shuffle=False, pin_memory=True, drop_last=False)

    return training_data_loader, validate_data_loader


def _format_tensor_shape(shape):
    return "x".join(str(int(v)) for v in shape)


def _infer_dataset_io_spec(data_loader):
    try:
        sample = data_loader.dataset[0]
    except Exception:
        return None

    if not isinstance(sample, dict):
        return None

    hr_hsi = sample.get('hr_hsi')
    hr_msi = sample.get('hr_msi')
    lr_hsi = sample.get('lr_hsi')
    if hr_hsi is None or hr_msi is None or lr_hsi is None:
        return None

    if hasattr(hr_hsi, 'shape') and hasattr(hr_msi, 'shape') and hasattr(lr_hsi, 'shape'):
        return {
            'hr_hsi': tuple(int(v) for v in hr_hsi.shape),
            'hr_msi': tuple(int(v) for v in hr_msi.shape),
            'lr_hsi': tuple(int(v) for v in lr_hsi.shape),
            'ratio': sample.get('downsample_ratio', getattr(data_loader.dataset, 'test_ratio', None)),
        }
    return None


def train(args, training_data_loader, validate_data_loader, custom_weight_dir=None):

    if custom_weight_dir is None:
        custom_weight_dir = create_custom_weight_dir(args)


    log_file = os.path.join(custom_weight_dir, 'training_log.txt')


    sys.stdout = Logger(log_file=log_file)

    print(f"[INFO] 权重将保存到: {custom_weight_dir}")
    print(f"[INFO] 训练日志将保存到: {log_file}")
    if getattr(args, 'resume', None):
        print(f"[INFO] Resume mode: continuing in existing directory {custom_weight_dir}")
    print(f"[INFO] 模型参数: dim={args.channels}, scale={args.ratio}")
    io_spec = _infer_dataset_io_spec(training_data_loader)
    if io_spec is not None:
        print("[INFO] 当前训练数据输入规格:")
        print(f"  - HR-HSI: [C, H, W] = {_format_tensor_shape(io_spec['hr_hsi'])}")
        print(f"  - HR-MSI: [C, H, W] = {_format_tensor_shape(io_spec['hr_msi'])}")
        print(f"  - LR-HSI: [C, H, W] = {_format_tensor_shape(io_spec['lr_hsi'])}")
        if io_spec.get('ratio') is not None:
            print(f"  - Data Downsample Ratio: x{io_spec['ratio']}")
    else:
        print("[INFO] 当前训练数据输入规格将在首个 batch 中动态确定")

    if io_spec is not None:
        lr_hsi_dim = int(io_spec['lr_hsi'][0])
        hr_msi_dim = int(io_spec['hr_msi'][0])
    elif args.dataset.lower() == 'cave':
        lr_hsi_dim = 31
        hr_msi_dim = 3
    elif args.dataset.lower() == 'pavia_university':
        lr_hsi_dim = 103
        hr_msi_dim = 4
    elif args.dataset.lower() == 'xiongan_new_area':
        lr_hsi_dim = 93
        hr_msi_dim = 4
    else:
        lr_hsi_dim = 128
        hr_msi_dim = 4

    print(f"[INFO] {args.dataset}数据集配置: lr_hsi_dim={lr_hsi_dim}, hr_msi_dim={hr_msi_dim}")


    use_ase = getattr(args, 'use_ase', False)
    ase_prompt_mode = getattr(args, 'ase_prompt_mode', 'hard')
    ase_route_temperature = float(getattr(args, 'ase_route_temperature', 1.0))
    ase_prompt_soft_mix = float(getattr(args, 'ase_prompt_soft_mix', 0.5))
    ase_scope = getattr(args, 'ase_scope', 'all')
    ase_stage_scope = getattr(args, 'ase_stage_scope', 'all_stages')
    ase_stage_res_scales = getattr(args, 'ase_stage_res_scales', None)
    use_ase_fusion_residual = getattr(args, 'use_ase_fusion_residual', False)
    ase_fusion_res_scale = float(getattr(args, 'ase_fusion_res_scale', 0.3))
    use_learnable_ase_fusion_res_scale = getattr(args, 'use_learnable_ase_fusion_res_scale', False)
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
    fass_ll_sparsity = getattr(args, 'fass_ll_sparsity', 0.25)
    fass_hf_sparsity = getattr(args, 'fass_hf_sparsity', 0.08)
    fass_d_state = getattr(args, 'fass_d_state', 16)
    gating_input_mode = getattr(args, 'gating_input_mode', 'energy')
    gating_loss_weight = getattr(args, 'gating_loss_weight', 1.0)
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
    use_wavelet_local_bias = getattr(args, 'use_wavelet_local_bias', False)
    wavelet_local_bias_scale = float(getattr(args, 'wavelet_local_bias_scale', 0.1))
    use_wavelet_local_gate = getattr(args, 'use_wavelet_local_gate', False)
    wavelet_local_gate_scale = float(getattr(args, 'wavelet_local_gate_scale', 0.1))
    use_sam_local_gate = getattr(args, 'use_sam_local_gate', False)
    sam_local_gate_scale = float(getattr(args, 'sam_local_gate_scale', 0.1))
    use_sam_semantic_prompt_bank = getattr(args, 'use_sam_semantic_prompt_bank', False)
    sam_semantic_prompt_bank_scale = float(getattr(args, 'sam_semantic_prompt_bank_scale', 0.1))
    use_sam_region_prototype_bank = getattr(args, 'use_sam_region_prototype_bank', False)
    sam_region_prototype_bank_scale = float(getattr(args, 'sam_region_prototype_bank_scale', 0.1))
    sam_region_prototype_count = int(getattr(args, 'sam_region_prototype_count', 8))
    use_wavelet_guided_sam_prototype_scaling = getattr(args, 'use_wavelet_guided_sam_prototype_scaling', False)
    wavelet_guided_sam_prototype_scale = float(getattr(args, 'wavelet_guided_sam_prototype_scale', 0.1))
    use_sam_region_relation_loss = getattr(args, 'use_sam_region_relation_loss', False)
    sam_region_relation_weight = float(getattr(args, 'sam_region_relation_weight', 0.003))
    sam_region_relation_start_epoch = int(getattr(args, 'sam_region_relation_start_epoch', 20))
    sam_region_relation_count = int(getattr(args, 'sam_region_relation_count', 4))
    use_sam_region_prompt_mixture = getattr(args, 'use_sam_region_prompt_mixture', False)
    sam_region_prompt_mixture_scale = float(getattr(args, 'sam_region_prompt_mixture_scale', 0.05))
    sam_region_prompt_mixture_count = int(getattr(args, 'sam_region_prompt_mixture_count', 8))
    use_sam_guided_semantic_scanning = getattr(args, 'use_sam_guided_semantic_scanning', False)
    sam_semantic_scanning_count = int(getattr(args, 'sam_semantic_scanning_count', 6))
    use_sam_feature_cluster_scanning = getattr(args, 'use_sam_feature_cluster_scanning', False)
    sam_feature_cluster_count = int(getattr(args, 'sam_feature_cluster_count', 6))
    sam_feature_cluster_iters = int(getattr(args, 'sam_feature_cluster_iters', 2))
    sam_feature_cluster_spatial_weight = float(getattr(args, 'sam_feature_cluster_spatial_weight', 0.05))
    use_wavelet_augmented_ss1 = getattr(args, 'use_wavelet_augmented_ss1', False)
    wavelet_augmented_ss1_count = int(getattr(args, 'wavelet_augmented_ss1_count', 6))
    wavelet_augmented_ss1_topk_ratio = float(getattr(args, 'wavelet_augmented_ss1_topk_ratio', 0.25))
    wavelet_augmented_ss1_strength = float(getattr(args, 'wavelet_augmented_ss1_strength', 0.5))
    wavelet_augmented_ss1_mode = getattr(args, 'wavelet_augmented_ss1_mode', 'stable_intra_region')
    use_sam_boundary_aware_state_propagation = getattr(args, 'use_sam_boundary_aware_state_propagation', False)
    sam_boundary_aware_state_scale = float(getattr(args, 'sam_boundary_aware_state_scale', 0.2))
    use_sam_state_reset_stronger = getattr(args, 'use_sam_state_reset_stronger', False)
    sam_state_reset_scale = float(getattr(args, 'sam_state_reset_scale', 0.35))
    use_sam_state_organizer_v1 = getattr(args, 'use_sam_state_organizer_v1', False)
    sam_state_organizer_count = int(getattr(args, 'sam_state_organizer_count', 6))
    sam_state_organizer_boundary_scale = float(getattr(args, 'sam_state_organizer_boundary_scale', 0.1))
    sam_state_organizer_reset_scale = float(getattr(args, 'sam_state_organizer_reset_scale', 0.15))
    use_sam_region_prompt_subspace = getattr(args, 'use_sam_region_prompt_subspace', False)
    sam_region_prompt_subspace_scale = float(getattr(args, 'sam_region_prompt_subspace_scale', 0.05))
    sam_region_prompt_subspace_count = int(getattr(args, 'sam_region_prompt_subspace_count', 6))
    use_wavelet_guided_semantic_state_organization = getattr(args, 'use_wavelet_guided_semantic_state_organization', False)
    wavelet_guided_semantic_state_count = int(getattr(args, 'wavelet_guided_semantic_state_count', 6))
    wavelet_guided_semantic_state_scale = float(getattr(args, 'wavelet_guided_semantic_state_scale', 0.05))
    wavelet_guided_semantic_boundary_scale = float(getattr(args, 'wavelet_guided_semantic_boundary_scale', 0.1))
    wavelet_guided_semantic_reset_scale = float(getattr(args, 'wavelet_guided_semantic_reset_scale', 0.15))
    use_joint_spatial_spectral_wavelet_prior = getattr(args, 'use_joint_spatial_spectral_wavelet_prior', False)
    joint_wavelet_spatial_weight = float(getattr(args, 'joint_wavelet_spatial_weight', 1.0))
    joint_wavelet_spectral_weight = float(getattr(args, 'joint_wavelet_spectral_weight', 1.0))
    use_dual_prototype_bank = getattr(args, 'use_dual_prototype_bank', False)
    dual_prototype_semantic_scale = float(getattr(args, 'dual_prototype_semantic_scale', 0.05))
    dual_prototype_frequency_scale = float(getattr(args, 'dual_prototype_frequency_scale', 0.05))
    dual_prototype_count = int(getattr(args, 'dual_prototype_count', 6))
    use_semantic_frequency_state_modulation = getattr(args, 'use_semantic_frequency_state_modulation', False)
    semantic_frequency_state_count = int(getattr(args, 'semantic_frequency_state_count', 6))
    semantic_frequency_state_write_scale = float(getattr(args, 'semantic_frequency_state_write_scale', 0.08))
    semantic_frequency_state_read_scale = float(getattr(args, 'semantic_frequency_state_read_scale', 0.08))
    semantic_frequency_state_delta_scale = float(getattr(args, 'semantic_frequency_state_delta_scale', 0.05))
    use_sam_distillation = getattr(args, 'use_sam_distillation', False)
    sam_distill_route_weight = float(getattr(args, 'sam_distill_route_weight', 0.005))
    sam_distill_boundary_recon_weight = float(getattr(args, 'sam_distill_boundary_recon_weight', 0.01))
    sam_distill_start_epoch = int(getattr(args, 'sam_distill_start_epoch', 20))
    use_hf_wavelet_loss = getattr(args, 'use_hf_wavelet_loss', False)
    hf_wavelet_loss_weight = float(getattr(args, 'hf_wavelet_loss_weight', 0.01))
    hf_wavelet_loss_start_epoch = int(getattr(args, 'hf_wavelet_loss_start_epoch', 5))
    use_semantic_region_weighted_hf_wavelet_loss = getattr(args, 'use_semantic_region_weighted_hf_wavelet_loss', False)
    semantic_region_hf_wavelet_loss_weight = float(getattr(args, 'semantic_region_hf_wavelet_loss_weight', 0.003))
    semantic_region_hf_wavelet_loss_start_epoch = int(getattr(args, 'semantic_region_hf_wavelet_loss_start_epoch', 100))
    semantic_region_hf_wavelet_boundary_boost = float(getattr(args, 'semantic_region_hf_wavelet_boundary_boost', 0.5))
    use_boundary_selective_wavelet_loss = getattr(args, 'use_boundary_selective_wavelet_loss', False)
    boundary_selective_wavelet_loss_weight = float(getattr(args, 'boundary_selective_wavelet_loss_weight', 0.003))
    boundary_selective_wavelet_loss_start_epoch = int(getattr(args, 'boundary_selective_wavelet_loss_start_epoch', 80))
    boundary_selective_wavelet_boundary_boost = float(getattr(args, 'boundary_selective_wavelet_boundary_boost', 0.75))
    boundary_selective_wavelet_frequency_boost = float(getattr(args, 'boundary_selective_wavelet_frequency_boost', 0.5))
    use_semantic_frequency_adaptive_scanning = getattr(args, 'use_semantic_frequency_adaptive_scanning', False)
    semantic_frequency_semantic_weight = float(getattr(args, 'semantic_frequency_semantic_weight', 1.0))
    semantic_frequency_wavelet_weight = float(getattr(args, 'semantic_frequency_wavelet_weight', 1.0))
    semantic_frequency_boundary_weight = float(getattr(args, 'semantic_frequency_boundary_weight', 0.5))
    semantic_frequency_confidence_weight = float(getattr(args, 'semantic_frequency_confidence_weight', 0.25))
    semantic_frequency_prompt_weight = float(getattr(args, 'semantic_frequency_prompt_weight', 0.5))
    train_mode = getattr(args, 'train_mode', 'auto')

    dense_epochs = getattr(args, 'dense_epochs', 100)

    model = Net(dim=args.channels,
               lr_hsi_dim=lr_hsi_dim,
               hr_msi_dim=hr_msi_dim,
               H=64,
               W=64,
               scale=args.ratio,
               use_ase=use_ase,
               ase_prompt_mode=ase_prompt_mode,
               ase_route_temperature=ase_route_temperature,
               ase_prompt_soft_mix=ase_prompt_soft_mix,
               ase_scope=ase_scope,
               ase_stage_scope=ase_stage_scope,
               use_ase_fusion_residual=use_ase_fusion_residual,
               ase_fusion_res_scale=ase_fusion_res_scale,
               ase_stage_res_scales=ase_stage_res_scales,
               use_learnable_ase_fusion_res_scale=use_learnable_ase_fusion_res_scale,
               use_sam_ase=use_sam_ase,
               sam_checkpoint=sam_checkpoint,
               sam_prompt_dim=sam_prompt_dim,
               use_learnable_prompts=use_learnable_prompts,
               num_learnable_prompts=num_learnable_prompts,
               use_soft_masks=use_soft_masks,
               num_soft_regions=num_soft_regions,
                use_wavelet=args.use_wavelet,
                use_wavelet_priors=getattr(args, 'use_wavelet_priors', False),
               use_wavelet_local_bias=use_wavelet_local_bias,
               wavelet_local_bias_scale=wavelet_local_bias_scale,
               use_wavelet_local_gate=use_wavelet_local_gate,
               wavelet_local_gate_scale=wavelet_local_gate_scale,
               use_sam_local_gate=use_sam_local_gate,
               sam_local_gate_scale=sam_local_gate_scale,
               use_sam_semantic_prompt_bank=use_sam_semantic_prompt_bank,
               sam_semantic_prompt_bank_scale=sam_semantic_prompt_bank_scale,
               use_sam_region_prototype_bank=use_sam_region_prototype_bank,
               sam_region_prototype_bank_scale=sam_region_prototype_bank_scale,
               sam_region_prototype_count=sam_region_prototype_count,
               use_wavelet_guided_sam_prototype_scaling=use_wavelet_guided_sam_prototype_scaling,
               wavelet_guided_sam_prototype_scale=wavelet_guided_sam_prototype_scale,
               use_sam_region_prompt_mixture=use_sam_region_prompt_mixture,
               sam_region_prompt_mixture_scale=sam_region_prompt_mixture_scale,
               sam_region_prompt_mixture_count=sam_region_prompt_mixture_count,
               use_sam_guided_semantic_scanning=use_sam_guided_semantic_scanning,
               sam_semantic_scanning_count=sam_semantic_scanning_count,
               use_sam_feature_cluster_scanning=use_sam_feature_cluster_scanning,
               sam_feature_cluster_count=sam_feature_cluster_count,
               sam_feature_cluster_iters=sam_feature_cluster_iters,
               sam_feature_cluster_spatial_weight=sam_feature_cluster_spatial_weight,
               use_wavelet_augmented_ss1=use_wavelet_augmented_ss1,
               wavelet_augmented_ss1_count=wavelet_augmented_ss1_count,
               wavelet_augmented_ss1_topk_ratio=wavelet_augmented_ss1_topk_ratio,
               wavelet_augmented_ss1_strength=wavelet_augmented_ss1_strength,
               wavelet_augmented_ss1_mode=wavelet_augmented_ss1_mode,
               use_sam_boundary_aware_state_propagation=use_sam_boundary_aware_state_propagation,
               sam_boundary_aware_state_scale=sam_boundary_aware_state_scale,
               use_sam_state_reset_stronger=use_sam_state_reset_stronger,
               sam_state_reset_scale=sam_state_reset_scale,
               use_sam_state_organizer_v1=use_sam_state_organizer_v1,
               sam_state_organizer_count=sam_state_organizer_count,
               sam_state_organizer_boundary_scale=sam_state_organizer_boundary_scale,
               sam_state_organizer_reset_scale=sam_state_organizer_reset_scale,
               use_sam_region_prompt_subspace=use_sam_region_prompt_subspace,
               sam_region_prompt_subspace_scale=sam_region_prompt_subspace_scale,
               sam_region_prompt_subspace_count=sam_region_prompt_subspace_count,
               use_wavelet_guided_semantic_state_organization=use_wavelet_guided_semantic_state_organization,
               wavelet_guided_semantic_state_count=wavelet_guided_semantic_state_count,
               wavelet_guided_semantic_state_scale=wavelet_guided_semantic_state_scale,
               wavelet_guided_semantic_boundary_scale=wavelet_guided_semantic_boundary_scale,
               wavelet_guided_semantic_reset_scale=wavelet_guided_semantic_reset_scale,
               use_joint_spatial_spectral_wavelet_prior=use_joint_spatial_spectral_wavelet_prior,
               joint_wavelet_spatial_weight=joint_wavelet_spatial_weight,
               joint_wavelet_spectral_weight=joint_wavelet_spectral_weight,
               use_dual_prototype_bank=use_dual_prototype_bank,
               dual_prototype_semantic_scale=dual_prototype_semantic_scale,
               dual_prototype_frequency_scale=dual_prototype_frequency_scale,
               dual_prototype_count=dual_prototype_count,
               use_semantic_frequency_state_modulation=use_semantic_frequency_state_modulation,
               semantic_frequency_state_count=semantic_frequency_state_count,
               semantic_frequency_state_write_scale=semantic_frequency_state_write_scale,
               semantic_frequency_state_read_scale=semantic_frequency_state_read_scale,
               semantic_frequency_state_delta_scale=semantic_frequency_state_delta_scale,
               use_structure_guided_sam_ase=use_structure_guided_sam_ase,
               structure_texture_weight=structure_texture_weight,
               use_fass=use_fass,
               fass_compression_ratio=fass_compression_ratio,
               fass_threshold=fass_threshold,
               fass_sparsity_target=fass_sparsity_target,
               fass_ll_sparsity=fass_ll_sparsity,
               fass_hf_sparsity=fass_hf_sparsity,
               fass_d_state=fass_d_state,
               gating_loss_weight=gating_loss_weight,
               train_mode=train_mode,
               dense_epochs=dense_epochs).to(args.device)
    if use_ase and not use_sam_ase and not use_fass:
        print(
            f"[ASE-HISR] prompt_mode={ase_prompt_mode}, "
            f"route_temperature={ase_route_temperature}, "
            f"prompt_soft_mix={ase_prompt_soft_mix}, "
            f"scope={ase_scope}, "
            f"stage_scope={ase_stage_scope}, "
            f"fusion_residual={use_ase_fusion_residual}, "
            f"fusion_res_scale={ase_fusion_res_scale}, "
            f"learnable_fusion_res_scale={use_learnable_ase_fusion_res_scale}, "
            f"stage_res_scales={ase_stage_res_scales}, "
            f"wavelet_local_bias={use_wavelet_local_bias}, "
            f"wavelet_local_bias_scale={wavelet_local_bias_scale}, "
            f"wavelet_local_gate={use_wavelet_local_gate}, "
            f"wavelet_local_gate_scale={wavelet_local_gate_scale}, "
            f"sam_local_gate={use_sam_local_gate}, "
            f"sam_local_gate_scale={sam_local_gate_scale}, "
            f"sam_semantic_prompt_bank={use_sam_semantic_prompt_bank}, "
            f"sam_semantic_prompt_bank_scale={sam_semantic_prompt_bank_scale}, "
            f"sam_region_prototype_bank={use_sam_region_prototype_bank}, "
            f"sam_region_prototype_bank_scale={sam_region_prototype_bank_scale}, "
            f"sam_region_prototype_count={sam_region_prototype_count}, "
            f"wavelet_guided_sam_prototype_scaling={use_wavelet_guided_sam_prototype_scaling}, "
            f"wavelet_guided_sam_prototype_scale={wavelet_guided_sam_prototype_scale}, "
            f"sam_region_relation_loss={use_sam_region_relation_loss}, "
            f"sam_region_relation_weight={sam_region_relation_weight}, "
            f"sam_region_relation_start_epoch={sam_region_relation_start_epoch}, "
            f"sam_region_relation_count={sam_region_relation_count}, "
            f"sam_region_prompt_mixture={use_sam_region_prompt_mixture}, "
            f"sam_region_prompt_mixture_scale={sam_region_prompt_mixture_scale}, "
            f"sam_region_prompt_mixture_count={sam_region_prompt_mixture_count}, "
            f"sam_guided_semantic_scanning={use_sam_guided_semantic_scanning}, "
            f"sam_semantic_scanning_count={sam_semantic_scanning_count}, "
            f"sam_feature_cluster_scanning={use_sam_feature_cluster_scanning}, "
            f"sam_feature_cluster_count={sam_feature_cluster_count}, "
            f"sam_feature_cluster_iters={sam_feature_cluster_iters}, "
            f"sam_feature_cluster_spatial_weight={sam_feature_cluster_spatial_weight}, "
            f"wavelet_augmented_ss1={use_wavelet_augmented_ss1}, "
            f"wavelet_augmented_ss1_count={wavelet_augmented_ss1_count}, "
            f"wavelet_augmented_ss1_topk_ratio={wavelet_augmented_ss1_topk_ratio}, "
            f"wavelet_augmented_ss1_strength={wavelet_augmented_ss1_strength}, "
            f"wavelet_augmented_ss1_mode={wavelet_augmented_ss1_mode}, "
            f"sam_boundary_aware_state_propagation={use_sam_boundary_aware_state_propagation}, "
            f"sam_boundary_aware_state_scale={sam_boundary_aware_state_scale}, "
            f"sam_state_reset_stronger={use_sam_state_reset_stronger}, "
            f"sam_state_reset_scale={sam_state_reset_scale}, "
            f"sam_state_organizer_v1={use_sam_state_organizer_v1}, "
            f"sam_state_organizer_count={sam_state_organizer_count}, "
            f"sam_region_prompt_subspace={use_sam_region_prompt_subspace}, "
            f"sam_region_prompt_subspace_scale={sam_region_prompt_subspace_scale}, "
            f"sam_region_prompt_subspace_count={sam_region_prompt_subspace_count}, "
            f"wavelet_guided_semantic_state_organization={use_wavelet_guided_semantic_state_organization}, "
            f"wavelet_guided_semantic_state_count={wavelet_guided_semantic_state_count}, "
            f"wavelet_guided_semantic_state_scale={wavelet_guided_semantic_state_scale}, "
            f"joint_spatial_spectral_wavelet_prior={use_joint_spatial_spectral_wavelet_prior}, "
            f"joint_wavelet_spatial_weight={joint_wavelet_spatial_weight}, "
            f"joint_wavelet_spectral_weight={joint_wavelet_spectral_weight}, "
            f"dual_prototype_bank={use_dual_prototype_bank}, "
            f"dual_prototype_semantic_scale={dual_prototype_semantic_scale}, "
            f"dual_prototype_frequency_scale={dual_prototype_frequency_scale}, "
            f"dual_prototype_count={dual_prototype_count}, "
            f"semantic_frequency_state_modulation={use_semantic_frequency_state_modulation}, "
            f"semantic_frequency_state_count={semantic_frequency_state_count}, "
            f"semantic_frequency_state_write_scale={semantic_frequency_state_write_scale}, "
            f"semantic_frequency_state_read_scale={semantic_frequency_state_read_scale}, "
            f"semantic_frequency_state_delta_scale={semantic_frequency_state_delta_scale}, "
            f"sam_distillation={use_sam_distillation}, "
            f"sam_distill_route_weight={sam_distill_route_weight}, "
            f"sam_distill_boundary_recon_weight={sam_distill_boundary_recon_weight}, "
            f"sam_distill_start_epoch={sam_distill_start_epoch}, "
            f"route_reg_weight={getattr(args, 'route_reg_weight', 0.01)}"
        )
    if use_sam_local_gate and not (use_ase and not use_sam_ase and not use_fass and use_ase_fusion_residual and ase_scope == 'fusion_only'):
        print("[SAM-LOCAL-GATE] WARNING: this option is designed for ASE + fusion_only + fusion_residual; current run may ignore it")
    if use_sam_local_gate and not sam_checkpoint:
        print("[SAM-LOCAL-GATE] WARNING: no --sam_checkpoint provided; SAM local gate may fail to initialize")
    if use_sam_semantic_prompt_bank and not (use_ase and not use_sam_ase and not use_fass and use_ase_fusion_residual and ase_scope == 'fusion_only'):
        print("[SAM-PBANK] WARNING: this option is designed for ASE + fusion_only + fusion_residual; current run may ignore it")
    if use_sam_semantic_prompt_bank and not sam_checkpoint:
        print("[SAM-PBANK] WARNING: no --sam_checkpoint provided; semantic prompt bank may fail to initialize")
    if use_sam_semantic_prompt_bank:
        print(
            f"[SAM-PBANK] enabled: SAM region prototypes -> prompt-bank bias -> fusion ASE, "
            f"scale={sam_semantic_prompt_bank_scale}"
        )
    if use_sam_region_prototype_bank and not (use_ase and not use_sam_ase and not use_fass and use_ase_fusion_residual and ase_scope == 'fusion_only'):
        print("[SAM-RPROTO] WARNING: this option is designed for ASE + fusion_only + fusion_residual; current run may ignore it")
    if use_sam_region_prototype_bank and not getattr(args, 'use_offline_sam_cache', False):
        print("[SAM-RPROTO] WARNING: this option relies on remapped whole-image SAM cache; without --use_offline_sam_cache prototype conditioning will be unavailable")
    if use_sam_region_prototype_bank:
        print(
            f"[SAM-RPROTO] enabled: remapped SAM region prototypes -> fusion ASE prompt-bank bias, "
            f"scale={sam_region_prototype_bank_scale}, count={sam_region_prototype_count}"
        )
    if use_wavelet_guided_sam_prototype_scaling and not use_sam_region_prototype_bank:
        print("[WAVE-RPROTO] WARNING: this option requires --use_sam_region_prototype_bank; otherwise it will have no effect")
    if use_wavelet_guided_sam_prototype_scaling:
        print(
            f"[WAVE-RPROTO] enabled: wavelet region complexity -> SAM prototypes -> prompt-bank bias, "
            f"scale={wavelet_guided_sam_prototype_scale}"
        )
    if use_sam_region_relation_loss and not getattr(args, 'use_offline_sam_cache', False):
        print("[SAM-RREL] WARNING: this loss relies on remapped whole-image SAM cache; without --use_offline_sam_cache it will be unavailable")
    if use_sam_region_relation_loss:
        print(
            f"[SAM-RREL] enabled: region-level feature consistency loss, "
            f"weight={sam_region_relation_weight}, start_epoch={sam_region_relation_start_epoch}, count={sam_region_relation_count}"
        )
    if use_sam_region_prompt_mixture and not (use_ase and not use_sam_ase and not use_fass and use_ase_fusion_residual and ase_scope == 'fusion_only'):
        print("[SAM-RPMIX] WARNING: this option is designed for ASE + fusion_only + fusion_residual; current run may ignore it")
    if use_sam_region_prompt_mixture and not getattr(args, 'use_offline_sam_cache', False):
        print("[SAM-RPMIX] WARNING: this option relies on remapped whole-image SAM cache; without --use_offline_sam_cache prompt mixture will be unavailable")
    if use_sam_region_prompt_mixture:
        print(
            f"[SAM-RPMIX] enabled: SAM region prototypes -> fusion ASE route prior, "
            f"scale={sam_region_prompt_mixture_scale}, count={sam_region_prompt_mixture_count}"
        )
    if use_sam_guided_semantic_scanning and not getattr(args, 'use_offline_sam_cache', False):
        print("[SAM-SCAN] WARNING: this option relies on remapped whole-image SAM cache; without --use_offline_sam_cache semantic scanning will be unavailable")
    if use_sam_guided_semantic_scanning:
        if use_sam_feature_cluster_scanning:
            print(
                f"[SAM-FCLUSTER] enabled: SAM features -> clustering-based semantic scan ordering before ASE, "
                f"clusters={sam_feature_cluster_count}, "
                f"iters={sam_feature_cluster_iters}, "
                f"spatial_weight={sam_feature_cluster_spatial_weight}"
            )
        else:
            print(
                f"[SAM-SCAN] enabled: SAM regions -> semantic scan ordering before ASE, "
                f"count={sam_semantic_scanning_count}"
            )
    if use_sam_feature_cluster_scanning and not use_sam_guided_semantic_scanning:
        raise ValueError("--use_sam_feature_cluster_scanning requires --use_sam_guided_semantic_scanning")
    if use_sam_feature_cluster_scanning and not getattr(args, 'use_offline_sam_cache', False):
        raise ValueError("--use_sam_feature_cluster_scanning requires --use_offline_sam_cache")
    if use_wavelet_augmented_ss1 and not use_sam_guided_semantic_scanning:
        raise ValueError("--use_wavelet_augmented_ss1 requires --use_sam_guided_semantic_scanning")
    if use_wavelet_augmented_ss1 and not getattr(args, 'use_offline_sam_cache', False):
        raise ValueError("--use_wavelet_augmented_ss1 requires --use_offline_sam_cache")
    if use_wavelet_augmented_ss1 and not use_joint_spatial_spectral_wavelet_prior:
        raise ValueError("--use_wavelet_augmented_ss1 requires --use_joint_spatial_spectral_wavelet_prior")
    if use_sam_boundary_aware_state_propagation and not getattr(args, 'use_offline_sam_cache', False):
        print("[SAM-BOUNDARY] WARNING: this option relies on remapped whole-image SAM cache; without --use_offline_sam_cache boundary-aware state propagation will be unavailable")
    if use_sam_boundary_aware_state_propagation:
        print(
            f"[SAM-BOUNDARY] enabled: semantic boundary attenuation on ASE state updates, "
            f"scale={sam_boundary_aware_state_scale}"
        )
    if use_sam_state_reset_stronger and not getattr(args, 'use_offline_sam_cache', False):
        raise ValueError("--use_sam_state_reset_stronger requires --use_offline_sam_cache")
    if use_sam_state_organizer_v1 and not getattr(args, 'use_offline_sam_cache', False):
        raise ValueError("--use_sam_state_organizer_v1 requires --use_offline_sam_cache")
    if use_sam_region_prompt_subspace and not getattr(args, 'use_offline_sam_cache', False):
        raise ValueError("--use_sam_region_prompt_subspace requires --use_offline_sam_cache")
    if use_wavelet_guided_semantic_state_organization and not getattr(args, 'use_offline_sam_cache', False):
        raise ValueError("--use_wavelet_guided_semantic_state_organization requires --use_offline_sam_cache")
    fusion_only_ase_mainline = use_ase and not use_sam_ase and not use_fass and use_ase_fusion_residual and ase_scope == 'fusion_only'
    route_profiles = _detect_route_profiles(args)
    if use_sam_state_organizer_v1 and not fusion_only_ase_mainline:
        raise ValueError("--use_sam_state_organizer_v1 requires ASE + fusion_only + fusion_residual (non-SAMASE, non-FASS)")
    if use_sam_region_prompt_subspace and not fusion_only_ase_mainline:
        raise ValueError("--use_sam_region_prompt_subspace requires ASE + fusion_only + fusion_residual (non-SAMASE, non-FASS)")
    if use_wavelet_guided_semantic_state_organization and not fusion_only_ase_mainline:
        raise ValueError("--use_wavelet_guided_semantic_state_organization requires ASE + fusion_only + fusion_residual (non-SAMASE, non-FASS)")
    if use_sam_feature_cluster_scanning and not fusion_only_ase_mainline:
        raise ValueError("--use_sam_feature_cluster_scanning requires ASE + fusion_only + fusion_residual (non-SAMASE, non-FASS)")
    if use_wavelet_augmented_ss1 and not fusion_only_ase_mainline:
        raise ValueError("--use_wavelet_augmented_ss1 requires ASE + fusion_only + fusion_residual (non-SAMASE, non-FASS)")
    if use_sam_feature_cluster_scanning and use_wavelet_augmented_ss1:
        raise ValueError("--use_sam_feature_cluster_scanning cannot be combined with --use_wavelet_augmented_ss1")
    if use_sam_state_organizer_v1 and (
        use_sam_guided_semantic_scanning or use_sam_boundary_aware_state_propagation or use_sam_state_reset_stronger
    ):
        raise ValueError("--use_sam_state_organizer_v1 cannot be combined with old standalone SAM scan/boundary/reset controls")
    if use_wavelet_augmented_ss1 and (
        use_sam_boundary_aware_state_propagation
        or use_sam_state_reset_stronger
        or use_sam_state_organizer_v1
        or use_wavelet_guided_semantic_state_organization
    ):
        raise ValueError("--use_wavelet_augmented_ss1 is a standalone SS1 enhancement and cannot be combined with other SAM state organizers/boundary-reset controls")
    if use_wavelet_guided_semantic_state_organization and (
        use_sam_guided_semantic_scanning or use_sam_boundary_aware_state_propagation or use_sam_state_reset_stronger or use_sam_state_organizer_v1
    ):
        raise ValueError("--use_wavelet_guided_semantic_state_organization cannot be combined with old standalone SAM state organizers")
    if use_dual_prototype_bank and not fusion_only_ase_mainline:
        raise ValueError("--use_dual_prototype_bank requires ASE + fusion_only + fusion_residual (non-SAMASE, non-FASS)")
    if use_semantic_frequency_state_modulation and not fusion_only_ase_mainline:
        raise ValueError("--use_semantic_frequency_state_modulation requires ASE + fusion_only + fusion_residual (non-SAMASE, non-FASS)")
    if use_dual_prototype_bank and not getattr(args, 'use_offline_sam_cache', False):
        raise ValueError("--use_dual_prototype_bank requires --use_offline_sam_cache")
    if use_semantic_frequency_state_modulation and not getattr(args, 'use_offline_sam_cache', False):
        raise ValueError("--use_semantic_frequency_state_modulation requires --use_offline_sam_cache")
    if use_dual_prototype_bank and not use_joint_spatial_spectral_wavelet_prior:
        raise ValueError("--use_dual_prototype_bank requires --use_joint_spatial_spectral_wavelet_prior")
    if use_semantic_frequency_state_modulation and not use_joint_spatial_spectral_wavelet_prior:
        raise ValueError("--use_semantic_frequency_state_modulation requires --use_joint_spatial_spectral_wavelet_prior")
    if use_boundary_selective_wavelet_loss and not getattr(args, 'use_offline_sam_cache', False):
        raise ValueError("--use_boundary_selective_wavelet_loss requires --use_offline_sam_cache")
    if route_profiles['partial_ssb1'] and not route_profiles['official_ssb1']:
        missing_requirements = []
        if not fusion_only_ase_mainline:
            missing_requirements.append("ASE+fusion_only+fusion_residual")
        if not getattr(args, 'use_sam_region_prototype_bank', False):
            missing_requirements.append("--use_sam_region_prototype_bank")
        if not getattr(args, 'use_wavelet_priors', False):
            missing_requirements.append("--use_wavelet_priors")
        if not use_joint_spatial_spectral_wavelet_prior:
            missing_requirements.append("--use_joint_spatial_spectral_wavelet_prior")
        print(
            "[SS1-BSW1] WARNING: partial combo detected. "
            "Official SSB1 expects "
            + ", ".join(missing_requirements if missing_requirements else ["official mainline settings"])
        )
    if route_profiles.get('partial_wssb1', False) and not route_profiles.get('official_wssb1', False):
        missing_requirements = []
        if not fusion_only_ase_mainline:
            missing_requirements.append("ASE+fusion_only+fusion_residual")
        if not getattr(args, 'use_sam_region_prototype_bank', False):
            missing_requirements.append("--use_sam_region_prototype_bank")
        if not getattr(args, 'use_wavelet_priors', False):
            missing_requirements.append("--use_wavelet_priors")
        if not use_joint_spatial_spectral_wavelet_prior:
            missing_requirements.append("--use_joint_spatial_spectral_wavelet_prior")
        print(
            "[WSS1-BSW1] WARNING: partial combo detected. "
            "Official WSSB1 expects "
            + ", ".join(missing_requirements if missing_requirements else ["official mainline settings"])
        )
    if route_profiles.get('official_wssb1', False):
        print(
            f"[WSSB1] enabled: WSS1C-style semantic scanning + light boundary-selective wavelet supervision, "
            f"scan_count={wavelet_augmented_ss1_count}, "
            f"topk_ratio={wavelet_augmented_ss1_topk_ratio}, "
            f"strength={wavelet_augmented_ss1_strength}, "
            f"loss_w={boundary_selective_wavelet_loss_weight}, "
            f"start={boundary_selective_wavelet_loss_start_epoch}, "
            f"boundary_boost={boundary_selective_wavelet_boundary_boost}, "
            f"frequency_boost={boundary_selective_wavelet_frequency_boost}"
        )
        if not route_profiles.get('recommended_wssb1', False):
            print(
                "[WSSB1] WARNING: current combo is heavier than the recommended light profile "
                "(count<=4, topk<=0.12, strength<=0.30, loss<=0.0025, start>=100, "
                "boundary_boost<=0.65, frequency_boost<=0.45)"
            )
    elif route_profiles['official_ssb1']:
        print(
            f"[SS1-BSW1] enabled: semantic scanning + boundary-selective wavelet supervision, "
            f"scan_count={sam_semantic_scanning_count}, "
            f"loss_w={boundary_selective_wavelet_loss_weight}, "
            f"start={boundary_selective_wavelet_loss_start_epoch}, "
            f"boundary_boost={boundary_selective_wavelet_boundary_boost}, "
            f"frequency_boost={boundary_selective_wavelet_frequency_boost}"
        )
    if route_profiles.get('official_fcs1', False):
        print(
            f"[FCS1] enabled: SAM feature clustering -> semantic scan ordering before ASE, "
            f"clusters={sam_feature_cluster_count}, "
            f"iters={sam_feature_cluster_iters}, "
            f"spatial_weight={sam_feature_cluster_spatial_weight}"
        )
    if route_profiles['official_wss1']:
        print(
            f"[WSS1] enabled: wavelet-augmented intra-region semantic scanning, "
            f"scan_count={wavelet_augmented_ss1_count}, "
            f"topk_ratio={wavelet_augmented_ss1_topk_ratio}, "
            f"strength={wavelet_augmented_ss1_strength}, "
            f"mode={wavelet_augmented_ss1_mode}"
        )
    if use_sam_state_reset_stronger:
        print(
            f"[SAM-RESET] enabled: stronger boundary-conditioned state reset surrogate, "
            f"scale={sam_state_reset_scale}"
        )
    if use_sam_state_organizer_v1:
        print(
            f"[SAM-STATE-ORG] enabled: semantic scan + boundary gate + reset gate, "
            f"count={sam_state_organizer_count}, "
            f"boundary_scale={sam_state_organizer_boundary_scale}, "
            f"reset_scale={sam_state_organizer_reset_scale}"
        )
    if use_sam_region_prompt_subspace:
        print(
            f"[SAM-PROMPT-SUBSPACE] enabled: region-specific token prompt residual, "
            f"scale={sam_region_prompt_subspace_scale}, count={sam_region_prompt_subspace_count}"
        )
    if use_wavelet_guided_semantic_state_organization:
        print(
            f"[WAVE-SAM-STATE] enabled: wavelet-guided semantic scan/boundary/reset organizer, "
            f"count={wavelet_guided_semantic_state_count}, "
            f"wavelet_scale={wavelet_guided_semantic_state_scale}, "
            f"boundary_scale={wavelet_guided_semantic_boundary_scale}, "
            f"reset_scale={wavelet_guided_semantic_reset_scale}"
        )
    if use_joint_spatial_spectral_wavelet_prior:
        print(
            f"[JOINT-WAVELET-PRIOR] enabled: spatial_weight={joint_wavelet_spatial_weight}, "
            f"spectral_weight={joint_wavelet_spectral_weight}"
        )
    if use_dual_prototype_bank:
        print(
            f"[DUAL-PROTOTYPE] enabled: semantic_scale={dual_prototype_semantic_scale}, "
            f"frequency_scale={dual_prototype_frequency_scale}, count={dual_prototype_count}"
        )
    if use_semantic_frequency_state_modulation:
        print(
            f"[SEM-FREQ-STATE] enabled: count={semantic_frequency_state_count}, "
            f"write_scale={semantic_frequency_state_write_scale}, "
            f"read_scale={semantic_frequency_state_read_scale}, "
            f"delta_scale={semantic_frequency_state_delta_scale}"
        )
    if getattr(args, 'use_wavelet_priors', False) and not getattr(args, 'use_wavelet', False):
        print("[WAVELET-PRIORS] enabled: using unified wavelet prior pipeline without legacy wavelet backbone branch")
    if use_semantic_region_weighted_hf_wavelet_loss and not getattr(args, 'use_offline_sam_cache', False):
        raise ValueError("--use_semantic_region_weighted_hf_wavelet_loss requires --use_offline_sam_cache")
    if use_semantic_region_weighted_hf_wavelet_loss:
        print(
            f"[SEM-HF-WAVELET] enabled: weight={semantic_region_hf_wavelet_loss_weight}, "
            f"start_epoch={semantic_region_hf_wavelet_loss_start_epoch}, "
            f"boundary_boost={semantic_region_hf_wavelet_boundary_boost}"
        )
    if use_boundary_selective_wavelet_loss:
        print(
            f"[BOUNDARY-SEL-WAVELET] enabled: weight={boundary_selective_wavelet_loss_weight}, "
            f"start_epoch={boundary_selective_wavelet_loss_start_epoch}, "
            f"boundary_boost={boundary_selective_wavelet_boundary_boost}, "
            f"frequency_boost={boundary_selective_wavelet_frequency_boost}"
        )
    if use_semantic_frequency_adaptive_scanning and not use_fass:
        raise ValueError("--use_semantic_frequency_adaptive_scanning requires --use_fass")
    if use_semantic_frequency_adaptive_scanning:
        print(
            f"[FASS2] enabled: semantic_weight={semantic_frequency_semantic_weight}, "
            f"wavelet_weight={semantic_frequency_wavelet_weight}, "
            f"boundary_weight={semantic_frequency_boundary_weight}, "
            f"confidence_weight={semantic_frequency_confidence_weight}, "
            f"prompt_weight={semantic_frequency_prompt_weight}"
        )
    if use_sam_distillation:
        print(
            f"[SAM-DISTILL] enabled: route_weight={sam_distill_route_weight}, "
            f"boundary_recon_weight={sam_distill_boundary_recon_weight}, "
            f"start_epoch={sam_distill_start_epoch}"
        )
    if use_sam_distillation and not getattr(args, 'use_offline_sam_cache', False):
        print("[SAM-DISTILL] WARNING: V1 is designed around offline SAM cache; without --use_offline_sam_cache the semantic teacher may be unavailable")
    if use_sam_distillation and not (use_ase and not use_sam_ase and not use_fass and use_ase_fusion_residual and ase_scope == 'fusion_only'):
        print("[SAM-DISTILL] WARNING: current best use case is ASE + fusion_only + fusion_residual; this run may weaken the distillation effect")

    configured_prior_c_refiner_modules = 0
    for module in model.modules():
        if hasattr(module, 'use_sam_prior_c_refiner') and hasattr(module, 'prior_refiner'):
            module_role = getattr(module, 'prior_c_refiner_role', 'unknown')
            enable_module = use_sam_prior_c_refiner and (
                sam_prior_c_refiner_scope == 'all' or module_role == 'fusion'
            )
            module.use_sam_prior_c_refiner = enable_module
            module.sam_prior_c_refiner_scale = sam_prior_c_refiner_scale
            if enable_module:
                configured_prior_c_refiner_modules += 1
    if use_fass:
        configured_gating_modules = 0
        for module in model.modules():
            if hasattr(module, 'gating_input_mode') and hasattr(module, 'gating_net_ll_hybrid'):
                module.gating_input_mode = 'semantic_frequency_v1' if use_semantic_frequency_adaptive_scanning else gating_input_mode
                module.gating_use_semantic_mask = gating_use_semantic_mask
                module.gating_use_prompt_strength = gating_use_prompt_strength
                module.gating_use_local_contrast = gating_use_local_contrast
                module.use_sam_prior_bank = use_sam_prior_bank
                module.sam_prior_use_boundary = sam_prior_use_boundary
                module.sam_prior_use_confidence = sam_prior_use_confidence
                module.use_semantic_frequency_adaptive_scanning = use_semantic_frequency_adaptive_scanning
                module.semantic_frequency_semantic_weight = semantic_frequency_semantic_weight
                module.semantic_frequency_wavelet_weight = semantic_frequency_wavelet_weight
                module.semantic_frequency_boundary_weight = semantic_frequency_boundary_weight
                module.semantic_frequency_confidence_weight = semantic_frequency_confidence_weight
                module.semantic_frequency_prompt_weight = semantic_frequency_prompt_weight
                configured_gating_modules += 1
        if configured_gating_modules > 0:
            print(
                f"[FASS-GATING] mode={'semantic_frequency_v1' if use_semantic_frequency_adaptive_scanning else gating_input_mode}, "
                f"semantic_mask={gating_use_semantic_mask}, "
                f"prompt_strength={gating_use_prompt_strength}, "
                f"local_contrast={gating_use_local_contrast}, "
                f"sam_prior_bank={use_sam_prior_bank}, "
                f"prior_boundary={sam_prior_use_boundary}, "
                f"prior_confidence={sam_prior_use_confidence}, "
                f"modules={configured_gating_modules}"
            )
        if use_sam_prior_bank and (not use_semantic_frequency_adaptive_scanning) and gating_input_mode != 'hybrid_v2':
            print(f"[SAM-PRIOR] WARNING: use_sam_prior_bank only affects hybrid_v2/semantic_frequency_v1 gating, current mode={gating_input_mode}; prior bank will be ignored")
    if use_sam_prior_c_refiner:
        if not use_sam_ase:
            print("[SAM-PRIOR-C] WARNING: use_sam_prior_c_refiner requires --use_sam_ase; current run will ignore it")
        elif use_fass:
            print("[SAM-PRIOR-C] WARNING: Prior C Refiner V1 currently only affects the non-FASS SAMASE path; it is ignored when --use_fass is enabled")
        elif configured_prior_c_refiner_modules > 0:
            print(
                f"[SAM-PRIOR-C] enabled: region + prompt_strength -> prior_bias -> C, "
                f"scope={sam_prior_c_refiner_scope}, scale={sam_prior_c_refiner_scale}, "
                f"modules={configured_prior_c_refiner_modules}"
            )
        else:
            print("[SAM-PRIOR-C] WARNING: no SAMASE modules were configured for Prior C Refiner V1")
    if use_sam_ase and use_structure_guided_sam_ase:
        print(
            f"[STRUCT-SAM-ASE] enabled: semantic coarse grouping + wavelet texture fine ranking, "
            f"texture_weight={structure_texture_weight}"
        )
    hf_wavelet_loss_fn = None
    if use_hf_wavelet_loss and hf_wavelet_loss_weight > 0:
        hf_wavelet_loss_fn = HFWaveletLoss().to(args.device)
        print(
            f"[HF-WAVELET-LOSS] enabled: weight={hf_wavelet_loss_weight}, "
            f"start_epoch={hf_wavelet_loss_start_epoch}"
        )
    semantic_region_hf_wavelet_loss_fn = None
    if use_semantic_region_weighted_hf_wavelet_loss and semantic_region_hf_wavelet_loss_weight > 0:
        semantic_region_hf_wavelet_loss_fn = HFWaveletLoss().to(args.device)
    boundary_selective_wavelet_loss_fn = None
    if use_boundary_selective_wavelet_loss and boundary_selective_wavelet_loss_weight > 0:
        boundary_selective_wavelet_loss_fn = HFWaveletLoss().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.decay)

    best_val_loss = float('inf')
    best_epoch = 0
    best_psnr = float('-inf')
    best_psnr_epoch = 0
    start_epoch = 1




    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')


        model.load_state_dict(checkpoint['state_dict'], strict=False)


        try:
            optimizer_state = checkpoint.get('optimizer', None)
            repaired_steps = _sanitize_optimizer_state_dict(optimizer_state)
            if repaired_steps > 0:
                print(f"[INFO] Sanitized {repaired_steps} optimizer step entries from resume checkpoint")
            optimizer.load_state_dict(optimizer_state)
            print(f"[INFO] Optimizer state loaded successfully")
        except (ValueError, KeyError) as e:
            print(f"[WARNING] Failed to load optimizer state (model structure may have changed): {e}")
            print(f"[INFO] Continuing with freshly initialized optimizer")

        start_epoch = checkpoint['epoch'] + 1
        print(f"[INFO] Resumed from epoch {start_epoch - 1}, checkpoint: {args.resume}")
    else:
        start_epoch = 1

    if args.use_ergas:
        criterion0 = nn.L1Loss(reduction='mean').to(args.device)
        criterion1 = ERGAS(args.ratio).to(args.device)
    else:
        criterion = nn.L1Loss(reduction='mean').to(args.device)


    scaler = GradScaler()


    start_time = time.time()
    print('Start training...')

    def get_stage0_sam_ase_module(net):
        if not hasattr(net, 'stage0'):
            return None
        stage0 = net.stage0
        if not hasattr(stage0, 'fm'):
            return None
        fusion_mamba = stage0.fm
        if not hasattr(fusion_mamba, 'spa_ase_mamba'):
            return None
        spa_ase_mamba = fusion_mamba.spa_ase_mamba
        if not hasattr(spa_ase_mamba, 'sam_ase_module'):
            return None
        return spa_ase_mamba.sam_ase_module


    def compute_gating_regularization(module):
        if (
            not args.use_fass
            or module is None
            or getattr(module, 'is_gating_frozen', False)
        ):
            return None

        mask_ll = getattr(module, '_last_mask_ll', None)
        mask_hf = getattr(module, '_last_mask_hf', None)
        if mask_ll is None or mask_hf is None:
            return None

        reg_mode = getattr(args, 'gating_reg_mode', 'legacy')
        reg_weight = float(getattr(args, 'gating_reg_weight', 0.01))
        if reg_weight <= 0:
            return None

        if reg_mode == 'budget':
            ll_target = getattr(args, 'll_target_keep', None)
            hf_target = getattr(args, 'hf_target_keep', None)
            if ll_target is None:
                ll_target = getattr(args, 'fass_ll_sparsity', 1.0)
            if hf_target is None:
                hf_target = getattr(args, 'fass_hf_sparsity', 0.2)

            budget_loss = None
            if ll_target < 1.0:
                ll_loss = (mask_ll.mean() - ll_target) ** 2
                budget_loss = ll_loss if budget_loss is None else budget_loss + ll_loss
            if hf_target < 1.0:
                hf_loss = (mask_hf.mean() - hf_target) ** 2
                budget_loss = hf_loss if budget_loss is None else budget_loss + hf_loss

            if budget_loss is None:
                return None
            return reg_weight * budget_loss

        return reg_weight * (mask_ll.mean() + mask_hf.mean())

    for epoch in range(start_epoch, args.epoch + 1):
        epoch_start_time = time.time()
        model.train()
        sam_ase_module = get_stage0_sam_ase_module(model)
        sam_extractor = getattr(model, 'sam_extractor', None)
        if sam_extractor is not None and hasattr(sam_extractor, 'set_timing_epoch'):
            sam_extractor.set_timing_epoch(epoch)


        if hasattr(model, 'set_current_epoch'):
            model.set_current_epoch(epoch)

        epoch_train_loss = []
        epoch_train_loss0 = []
        epoch_train_loss1 = []
        epoch_hf_wavelet_loss = []
        epoch_semantic_region_hf_wavelet_loss = []
        epoch_boundary_selective_wavelet_loss = []
        epoch_sam_distill_route_loss = []
        epoch_sam_distill_boundary_recon_loss = []


        data_load_times = []
        forward_times = []
        backward_times = []
        sam_extract_times = []


        for iteration, batch in enumerate(training_data_loader, 1):

            data_load_start = time.time()


            lr_hsi = batch['lr_hsi'].to(args.device, non_blocking=True).float()
            hr_msi = batch['hr_msi'].to(args.device, non_blocking=True).float()
            hr_hsi = batch['hr_hsi'].to(args.device, non_blocking=True).float()


            lr_hsi_approx = batch.get('lr_hsi_approx', None)
            lr_hsi_details = batch.get('lr_hsi_details', None)

            if lr_hsi_approx is not None:
                lr_hsi_approx = lr_hsi_approx.to(args.device, non_blocking=True).float()
            if lr_hsi_details is not None:
                lr_hsi_details = lr_hsi_details.to(args.device, non_blocking=True).float()


            if args.device == 'cuda':
                torch.cuda.synchronize()
            data_load_time = time.time() - data_load_start
            data_load_times.append(data_load_time)


            if lr_hsi.shape[1] != lr_hsi_dim:
                raise ValueError(f"LR-HSI channel mismatch: data={lr_hsi.shape[1]} vs model={lr_hsi_dim}")
            if hr_msi.shape[1] != hr_msi_dim:
                raise ValueError(f"HR-MSI channel mismatch: data={hr_msi.shape[1]} vs model={hr_msi_dim}")
            if hr_hsi.shape[1] != lr_hsi_dim:
                raise ValueError(f"HR-HSI/LR-HSI channel mismatch: hr_hsi={hr_hsi.shape[1]} vs lr_hsi_dim={lr_hsi_dim}")

            optimizer.zero_grad()


            forward_start = time.time()


            with autocast():

                cached_sam_features = batch.get('cached_sam_features', None)
                cached_sam_masks = batch.get('cached_sam_masks', None)
                if cached_sam_features is not None:
                    cached_sam_features = cached_sam_features.to(args.device, non_blocking=True).float()
                if cached_sam_masks is not None:
                    cached_sam_masks = cached_sam_masks.to(args.device, non_blocking=True).float()

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

                sam_distill_prior_bank = None
                if use_sam_distillation and cached_sam_masks is not None:
                    sam_distill_prior_bank = build_sam_distillation_priors(cached_sam_masks)

                if args.use_ergas:
                    loss = criterion0(sr, hr_hsi) + args.ergas_hp * criterion1(sr, hr_hsi)
                else:
                    loss = criterion(sr, hr_hsi)

                sam_route_distill_loss = None
                sam_boundary_recon_loss = None
                sam_region_relation_loss = None
                if use_sam_distillation and epoch >= sam_distill_start_epoch and sam_distill_prior_bank is not None:
                    sam_route_distill_loss = compute_sam_route_consistency_loss(routing_probs, sam_distill_prior_bank)
                    sam_boundary_recon_loss = compute_sam_boundary_recon_loss(sr, hr_hsi, sam_distill_prior_bank)
                    if sam_route_distill_loss is not None and sam_distill_route_weight > 0:
                        loss = loss + sam_distill_route_weight * sam_route_distill_loss
                    if sam_boundary_recon_loss is not None and sam_distill_boundary_recon_weight > 0:
                        loss = loss + sam_distill_boundary_recon_weight * sam_boundary_recon_loss

                if use_sam_region_relation_loss and epoch >= sam_region_relation_start_epoch and cached_sam_masks is not None:
                    model_ref = _unwrap_model(model)
                    fusion_feature_map = getattr(model_ref, 'current_fusion_feature_map', None)
                    sam_region_relation_loss = compute_sam_region_relation_loss(
                        fusion_feature_map,
                        cached_sam_masks,
                        num_regions=sam_region_relation_count,
                    )
                    if sam_region_relation_loss is not None and sam_region_relation_weight > 0:
                        loss = loss + sam_region_relation_weight * sam_region_relation_loss


                if routing_probs is not None and hasattr(args, 'route_reg_weight') and args.route_reg_weight > 0:

                    route_entropy_loss = 0
                    for stage_probs in routing_probs:
                        if isinstance(stage_probs, list):
                            for probs in stage_probs:
                                if probs is not None:

                                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                                    route_entropy_loss += entropy
                        elif stage_probs is not None:
                            entropy = -torch.sum(stage_probs * torch.log(stage_probs + 1e-8), dim=-1).mean()
                            route_entropy_loss += entropy


                    loss = loss + args.route_reg_weight * route_entropy_loss

                hf_wavelet_aux_loss = None
                if hf_wavelet_loss_fn is not None and epoch >= hf_wavelet_loss_start_epoch:
                    with autocast(enabled=False):
                        hf_wavelet_aux_loss = hf_wavelet_loss_fn(sr.float(), hr_hsi.float())
                    loss = loss + hf_wavelet_loss_weight * hf_wavelet_aux_loss
                semantic_region_hf_wavelet_aux_loss = None
                if (
                    semantic_region_hf_wavelet_loss_fn is not None
                    and epoch >= semantic_region_hf_wavelet_loss_start_epoch
                    and cached_sam_masks is not None
                ):
                    semantic_wavelet_weight_map = build_semantic_region_wavelet_weight_map(
                        cached_sam_masks,
                        boundary_boost=semantic_region_hf_wavelet_boundary_boost,
                    )
                    if semantic_wavelet_weight_map is not None:
                        with autocast(enabled=False):
                            semantic_region_hf_wavelet_aux_loss = semantic_region_hf_wavelet_loss_fn(
                                sr.float(),
                                hr_hsi.float(),
                                spatial_weight=semantic_wavelet_weight_map.float(),
                            )
                        loss = loss + semantic_region_hf_wavelet_loss_weight * semantic_region_hf_wavelet_aux_loss
                boundary_selective_wavelet_aux_loss = None
                if (
                    boundary_selective_wavelet_loss_fn is not None
                    and epoch >= boundary_selective_wavelet_loss_start_epoch
                    and cached_sam_masks is not None
                ):
                    model_ref = _unwrap_model(model)
                    boundary_selective_weight_map = build_boundary_selective_wavelet_weight_map(
                        getattr(model_ref, 'current_sam_region_context', None),
                        cached_sam_masks,
                        boundary_boost=boundary_selective_wavelet_boundary_boost,
                        frequency_boost=boundary_selective_wavelet_frequency_boost,
                    )
                    if boundary_selective_weight_map is not None:
                        with autocast(enabled=False):
                            boundary_selective_wavelet_aux_loss = boundary_selective_wavelet_loss_fn(
                                sr.float(),
                                hr_hsi.float(),
                                spatial_weight=boundary_selective_weight_map.float(),
                            )
                        loss = loss + boundary_selective_wavelet_loss_weight * boundary_selective_wavelet_aux_loss


            if args.device == 'cuda':
                torch.cuda.synchronize()
            forward_time = time.time() - forward_start
            forward_times.append(forward_time)


            if sam_extractor is not None and hasattr(sam_extractor, '_last_extract_time'):
                sam_t = float(sam_extractor._last_extract_time)
                sam_extract_times.append(sam_t)


            if (
                args.use_fass
                and sam_ase_module is not None
                and not getattr(sam_ase_module, 'is_gating_frozen', False)
            ):
                try:

                    mask_ll = sam_ase_module._last_mask_ll
                    mask_hf = sam_ase_module._last_mask_hf


                    sparsity_loss = compute_gating_regularization(sam_ase_module)
                    if sparsity_loss is not None:
                        loss = loss + sparsity_loss
                except Exception:

                    pass


            backward_start = time.time()
            scaler.scale(loss).backward()


            if args.device == 'cuda':
                torch.cuda.synchronize()
            backward_time = time.time() - backward_start
            backward_times.append(backward_time)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()


            current_loss = loss.item()
            epoch_train_loss.append(current_loss)
            if hf_wavelet_aux_loss is not None:
                epoch_hf_wavelet_loss.append(float(hf_wavelet_aux_loss.item()))
            if semantic_region_hf_wavelet_aux_loss is not None:
                epoch_semantic_region_hf_wavelet_loss.append(float(semantic_region_hf_wavelet_aux_loss.item()))
            if boundary_selective_wavelet_aux_loss is not None:
                epoch_boundary_selective_wavelet_loss.append(float(boundary_selective_wavelet_aux_loss.item()))
            if sam_route_distill_loss is not None:
                epoch_sam_distill_route_loss.append(float(sam_route_distill_loss.item()))
            if sam_boundary_recon_loss is not None:
                epoch_sam_distill_boundary_recon_loss.append(float(sam_boundary_recon_loss.item()))

            if args.use_ergas:

                loss0 = criterion0(sr, hr_hsi).item()
                loss1 = criterion1(sr, hr_hsi).item()
                epoch_train_loss0.append(loss0)
                epoch_train_loss1.append(loss1)


            del sr, loss
            if routing_probs is not None:
                del routing_probs


            if iteration % 50 == 0:
                torch.cuda.empty_cache()


        if epoch % args.val_freq == 0:
            model.eval()
            val_loss = []
            val_psnr = []
            val_ssim = []

            with torch.no_grad():
                for batch in validate_data_loader:

                    lr_hsi = batch['lr_hsi'].to(args.device).float()
                    hr_msi = batch['hr_msi'].to(args.device).float()


                    lr_hsi_approx = batch.get('lr_hsi_approx', None)
                    lr_hsi_details = batch.get('lr_hsi_details', None)

                    if lr_hsi_approx is not None:
                        lr_hsi_approx = lr_hsi_approx.to(args.device).float()
                    if lr_hsi_details is not None:
                        lr_hsi_details = lr_hsi_details.to(args.device).float()
                    hr_hsi = batch['hr_hsi'].to(args.device).float()


                    if lr_hsi.shape[1] != lr_hsi_dim:
                        raise ValueError(f"Validation LR-HSI channel mismatch: data={lr_hsi.shape[1]} vs model={lr_hsi_dim}")
                    if hr_msi.shape[1] != hr_msi_dim:
                        raise ValueError(f"Validation HR-MSI channel mismatch: data={hr_msi.shape[1]} vs model={hr_msi_dim}")
                    if hr_hsi.shape[1] != lr_hsi_dim:
                        raise ValueError(f"Validation HR-HSI/LR-HSI channel mismatch: hr_hsi={hr_hsi.shape[1]} vs lr_hsi_dim={lr_hsi_dim}")


                    with autocast(enabled=args.mixed_precision):

                        cached_sam_features = batch.get('cached_sam_features', None)
                        cached_sam_masks = batch.get('cached_sam_masks', None)
                        if cached_sam_features is not None:
                            cached_sam_features = cached_sam_features.to(args.device).float()
                        if cached_sam_masks is not None:
                            cached_sam_masks = cached_sam_masks.to(args.device).float()

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


                        if torch.isnan(sr).any() or torch.isinf(sr).any():
                            print("Warning: NaN/Inf detected in validation output")
                            continue

                        if args.use_ergas:
                            loss = criterion0(sr, hr_hsi) + args.ergas_hp * criterion1(sr, hr_hsi)
                        else:
                            loss = criterion(sr, hr_hsi)


                    sr_np = (sr.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    hr_np = (hr_hsi.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


                    psnr_value = PSNR(sr, hr_hsi)

                    ssim_channels = lr_hsi_dim
                    ssim_calculator = SSIM(channels=ssim_channels)
                    ssim_value = ssim_calculator(sr.float(), hr_hsi.float())


                    val_loss.append(loss.item())
                    val_psnr.append(psnr_value.item())
                    val_ssim.append(ssim_value.item())


            if len(val_loss) == 0:
                print("[WARNING] Validation loader produced no valid batches; metrics will be recorded as NaN for this epoch.")
                current_val_loss = float('nan')
                avg_psnr = float('nan')
                avg_ssim = float('nan')
            else:
                current_val_loss = np.mean(val_loss)
                avg_psnr = np.mean(val_psnr)
                avg_ssim = np.mean(val_ssim)
            checkpoint_state_dict = model.state_dict()
            checkpoint_filtered_state_dict = {}
            for key, value in checkpoint_state_dict.items():
                if '.sam_model' in key or 'predictor.' in key:
                    continue
                checkpoint_filtered_state_dict[key] = value

            checkpoint_payload_clean = {
                'epoch': epoch,
                'state_dict': checkpoint_filtered_state_dict,
                'optimizer': optimizer.state_dict(),
                'val_loss': current_val_loss,
                'psnr': avg_psnr,
                'ssim': avg_ssim
            }

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_epoch = epoch
                torch.save(checkpoint_payload_clean, os.path.join(custom_weight_dir, 'best_model.pth'))
                torch.save(checkpoint_payload_clean, os.path.join(custom_weight_dir, 'best_by_loss.pth'))

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_psnr_epoch = epoch
                torch.save(checkpoint_payload_clean, os.path.join(custom_weight_dir, 'best_by_psnr.pth'))

            state_dict = checkpoint_state_dict


            state_dict = model.state_dict()
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if '.sam_model' in key or 'predictor.' in key:
                    continue

            checkpoint_payload = {
                'epoch': epoch,
                'state_dict': filtered_state_dict,
                'optimizer': optimizer.state_dict(),
                'val_loss': current_val_loss,
                'psnr': avg_psnr,
                'ssim': avg_ssim
            }

            current_is_best_loss = current_val_loss < best_val_loss
            current_is_best_psnr = avg_psnr > best_psnr

            if current_is_best_loss:
                best_val_loss = current_val_loss
                best_epoch = epoch
                best_model_path = os.path.join(custom_weight_dir, 'best_model.pth')
                best_by_loss_path = os.path.join(custom_weight_dir, 'best_by_loss.pth')
                torch.save(checkpoint_payload, best_model_path)
                torch.save(checkpoint_payload, best_by_loss_path)

            if current_is_best_psnr:
                best_psnr = avg_psnr
                best_psnr_epoch = epoch
                best_by_psnr_path = os.path.join(custom_weight_dir, 'best_by_psnr.pth')
                torch.save(checkpoint_payload, best_by_psnr_path)


            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_epoch = epoch
                best_model_path = os.path.join(custom_weight_dir, 'best_model.pth')


                state_dict = model.state_dict()
                filtered_state_dict = {}
                for key, value in state_dict.items():


                    if '.sam_model' in key or 'predictor.' in key:
                        continue
                    filtered_state_dict[key] = value

                torch.save({
                    'epoch': epoch,
                    'state_dict': filtered_state_dict,
                    'optimizer': optimizer.state_dict(),
                    'val_loss': current_val_loss,
                    'psnr': avg_psnr,
                    'ssim': avg_ssim
                }, best_model_path)

            model.train()


        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time


        lr_scheduler.step()


        t_loss = np.nanmean(np.array(epoch_train_loss))
        avg_data_load = float(np.mean(data_load_times)) if len(data_load_times) > 0 else 0.0
        avg_forward = float(np.mean(forward_times)) if len(forward_times) > 0 else 0.0
        avg_backward = float(np.mean(backward_times)) if len(backward_times) > 0 else 0.0
        avg_sam_extract = float(np.mean(sam_extract_times)) if len(sam_extract_times) > 0 else 0.0
        timing_info = f" | Time/batch: data={avg_data_load*1000:.1f}ms, fwd={avg_forward*1000:.1f}ms, bwd={avg_backward*1000:.1f}ms, sam={avg_sam_extract*1000:.1f}ms"
        hf_wavelet_loss_info = ""
        if len(epoch_hf_wavelet_loss) > 0:
            hf_wavelet_loss_info = f" | HF-WLoss: {float(np.mean(epoch_hf_wavelet_loss)):.6f}"
        semantic_region_hf_wavelet_loss_info = ""
        if len(epoch_semantic_region_hf_wavelet_loss) > 0:
            semantic_region_hf_wavelet_loss_info = f" | SemHF-WLoss: {float(np.mean(epoch_semantic_region_hf_wavelet_loss)):.6f}"
        boundary_selective_wavelet_loss_info = ""
        if len(epoch_boundary_selective_wavelet_loss) > 0:
            boundary_selective_wavelet_loss_info = f" | BoundSel-WLoss: {float(np.mean(epoch_boundary_selective_wavelet_loss)):.6f}"
        sam_distill_info = ""
        if len(epoch_sam_distill_route_loss) > 0 or len(epoch_sam_distill_boundary_recon_loss) > 0:
            route_mean = float(np.mean(epoch_sam_distill_route_loss)) if len(epoch_sam_distill_route_loss) > 0 else 0.0
            recon_mean = float(np.mean(epoch_sam_distill_boundary_recon_loss)) if len(epoch_sam_distill_boundary_recon_loss) > 0 else 0.0
            sam_distill_info = f" | SAM-Distill: Route={route_mean:.6f}, BRecon={recon_mean:.6f}"


        gating_info = ""
        mode_str = "N/A"


        sam_ase_module = get_stage0_sam_ase_module(model)


        if sam_ase_module is not None and hasattr(sam_ase_module, '_current_epoch_gating_stats'):
            stats = sam_ase_module._current_epoch_gating_stats
            if stats is not None:
                if stats['mode'] == 'dense':
                    mode_str = "Dense"

                    ll_mean = stats['ll_mean']
                    ll_std = stats['ll_std']
                    hf_mean = stats['hf_mean']
                    hf_std = stats['hf_std']
                    ll_keep_ratio = stats.get('ll_keep_ratio', 0.0)
                    hf_keep_ratio = stats.get('hf_keep_ratio', 0.0)
                    gating_info = f" | Gating: LL={ll_mean:.3f}+/-{ll_std:.3f}, HF={hf_mean:.3f}+/-{hf_std:.3f}"
                    gating_info += f" | Keep>0.5: LL={ll_keep_ratio*100:.1f}%, HF={hf_keep_ratio*100:.1f}%"
                elif stats['mode'] == 'sparse':
                    mode_str = "Sparse"

                    ll_active = stats['ll_active']
                    ll_total = stats['ll_total']
                    ll_sparsity = stats['ll_sparsity']
                    hf_active = stats['hf_active']
                    hf_total = stats['hf_total']
                    hf_sparsity = stats['hf_sparsity']
                    ll_soft_mean = stats.get('ll_mean', 0.0)
                    hf_soft_mean = stats.get('hf_mean', 0.0)
                    gating_info = f" | Sparsity: LL={ll_active}/{ll_total} ({ll_sparsity:.1f}%), HF={hf_active}/{hf_total} ({hf_sparsity:.1f}%)"
                    gating_info += f" | SoftGate: LL={ll_soft_mean:.3f}, HF={hf_soft_mean:.3f}"


        if epoch % args.val_freq == 0:

            if args.use_ergas is True:
                    print(f'Epoch {epoch}/{args.epoch} ({mode_str}) ({epoch_time:.1f}s, {total_time/3600:.1f}h) | '
                          f'Loss: {t_loss:.6f}{hf_wavelet_loss_info}{semantic_region_hf_wavelet_loss_info}{boundary_selective_wavelet_loss_info}{sam_distill_info}{gating_info}{timing_info} | '
                          f'Val: {current_val_loss:.4f}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}')
            else:
                    print(f'Epoch {epoch}/{args.epoch} ({mode_str}) ({epoch_time:.1f}s, {total_time/3600:.1f}h) | '
                          f'Loss: {t_loss:.6f}{hf_wavelet_loss_info}{semantic_region_hf_wavelet_loss_info}{boundary_selective_wavelet_loss_info}{sam_distill_info}{gating_info}{timing_info} | '
                          f'Val: {current_val_loss:.4f}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}')
        else:

                if args.use_ergas is True:
                    print(f'Epoch {epoch}/{args.epoch} ({mode_str}) ({epoch_time:.1f}s, {total_time/3600:.1f}h) | '
                          f'Loss: {t_loss:.6f} (L1: {np.nanmean(epoch_train_loss0):.6f}, ERGAS: {np.nanmean(epoch_train_loss1):.6f}){hf_wavelet_loss_info}{semantic_region_hf_wavelet_loss_info}{boundary_selective_wavelet_loss_info}{sam_distill_info}{gating_info}{timing_info}')
                else:
                    print(f'Epoch {epoch}/{args.epoch} ({mode_str}) ({epoch_time:.1f}s, {total_time/3600:.1f}h) | '
                          f'Loss: {t_loss:.6f}{hf_wavelet_loss_info}{semantic_region_hf_wavelet_loss_info}{boundary_selective_wavelet_loss_info}{sam_distill_info}{gating_info}{timing_info}')


        if epoch % args.ckpt == 0:
            save_checkpoint(args, model, optimizer, epoch, custom_weight_dir)


        if args.use_fass and args.train_mode == 'auto':
            dense_epochs = getattr(args, 'dense_epochs', 100)

            if epoch == dense_epochs - 1:
                special_name = f"last_dense_epoch_{epoch}.pth"
                special_path = os.path.join(custom_weight_dir, special_name)

                state_dict = model.state_dict()
                filtered_state_dict = {}
                for key, value in state_dict.items():


                    if '.sam_model' in key or 'predictor.' in key:
                        continue
                    filtered_state_dict[key] = value

                torch.save({
                    'epoch': epoch,
                    'state_dict': filtered_state_dict,
                    'optimizer': optimizer.state_dict(),
                    'mode': 'dense',
                    'description': f'Last Dense epoch (epoch {epoch}) before switching to Sparse at epoch {dense_epochs}'
                }, special_path)
                print(f"[INFO] 已保存特殊 checkpoint: {special_name} (Dense 模式结束，下一个 epoch 切换到 Sparse)")


    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"总训练时间: {total_time/3600:.2f} 小时")
    print(f"最佳模型: Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
    print(f"权重保存路径: {custom_weight_dir}")
    print(f"日志已保存到: {log_file}")
    print(f"Best PSNR checkpoint: Epoch {best_psnr_epoch}, PSNR: {best_psnr:.4f}")
    print(f"{'='*60}\n")


    if isinstance(sys.stdout, Logger):
        original_stdout = sys.stdout.terminal
        sys.stdout = original_stdout


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='chikusei', choices=['cave', 'chikusei', 'pavia_university', 'xiongan_new_area'],
                       help='Dataset type to use: cave, chikusei, pavia_university or xiongan_new_area')
    parser.add_argument('--ratio', type=int, default=4, help='Upsample ratio')

    parser.add_argument('--H', type=int, default=512, help='Height of the high-resolution image (no longer used)')
    parser.add_argument('--W', type=int, default=512, help='Width of the high-resolution image (no longer used)')
    parser.add_argument('--channels', type=int, default=64, help='Feature channels')

    parser.add_argument('--use_ergas', type=bool, default=False, help='Use ERGAS loss for training or not')
    parser.add_argument('--ergas_hp', type=float, default=1e-4, help='Hyper-parameter for the ERGAS loss')
    parser.add_argument('--epoch', type=int, default=800, help='Epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Override DataLoader num_workers; default keeps the original auto policy')
    parser.add_argument('--lr', type=float, default=1.5e-4, help='Learning rate')
    parser.add_argument('--step', type=int, default=100, help='Step number')
    parser.add_argument('--decay', type=float, default=0.5, help='Learning rate decay')
    parser.add_argument('--ckpt', type=int, default=10, help='Checkpoint')
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--train_data_path', type=str,
                       default=None,
                       help='Path of the training dataset.')
    parser.add_argument('--val_data_path', type=str,
                       default=None,
                       help='Path of the validation dataset.')
    parser.add_argument('--test_h5_path', type=str, default=None,
                       help='Optional test H5 path used for experiment_record/test command generation.')
    parser.add_argument('--test_sam_cache_path', type=str, default=None,
                       help='Optional test SAM cache path used for experiment_record/test command generation.')
    parser.add_argument('--weight_dir', type=str, default='weights/', help='Dir of the weight.')
    parser.add_argument('--exp_code', type=str, default=None,
                       help='Short experiment code used as weight folder name, e.g. E1. If omitted, auto-increment E-code is used.')
    parser.add_argument('--resume', type=str, default=None, help='Path to latest checkpoint (default: None)')
    parser.add_argument('--val_freq', type=int, default=10, help='Validation frequency (epochs)')
    parser.add_argument('--epoch_log_interval', type=int, default=1, help='Epoch logging interval (1=every epoch, 5=every 5 epochs)')
    parser.add_argument('--mixed_precision', type=bool, default=True, help='Enable mixed precision training')

    parser.add_argument('--use_ase', action='store_true', help='use ASE (Attentive State-space Equation) module')
    parser.add_argument('--use_sam_ase', action='store_true', help='use SAM-guided ASE module (overrides original ASE)')
    parser.add_argument('--sam_checkpoint', type=str, default=None, help='path to SAM model checkpoint')
    parser.add_argument('--sam_prompt_dim', type=int, default=64, help='SAM prompt feature dimension')
    parser.add_argument('--use_learnable_prompts', action='store_true', help='use learnable prompt generator for SAM (improves prompt selection)')
    parser.add_argument('--num_learnable_prompts', type=int, default=16, help='number of learnable prompts for SAM')
    parser.add_argument('--use_soft_masks', action='store_true', help='use soft multi-region masks instead of binary masks (improves semantic granularity)')
    parser.add_argument('--num_soft_regions', type=int, default=8, help='number of soft mask regions (8-16 recommended)')
    parser.add_argument('--use_offline_sam_cache', action='store_true', help='use offline cached raw SAM outputs during train/val')
    parser.add_argument('--sam_cache_path', type=str, default=None, help='path to sidecar SAM cache h5 file')
    parser.add_argument('--sam_cache_strict', action='store_true', help='fail if SAM cache is missing or incompatible')
    parser.add_argument('--prefer_scene_sam_cache', action='store_true',
                       help='when offline SAM cache is enabled and no explicit cache path is provided, prefer *.whole_sam_cache.scene_region.h5 over the older patch/window cache variants')
    parser.add_argument('--prefer_multi_region_sam_cache', action='store_true',
                       help='when offline SAM cache is enabled and no explicit cache path is provided, prefer *.whole_sam_cache.multi_region.h5 over the default single-mask cache')
    parser.add_argument('--use_structure_guided_sam_ase', action='store_true',
                       help='enable structure-guided SAM-ASE step-1: semantic coarse grouping + wavelet texture fine ranking')
    parser.add_argument('--structure_texture_weight', type=float, default=0.25,
                       help='texture weight for structure-guided SAM-ASE sorting and gating (default: 0.25)')
    parser.add_argument('--route_reg_weight', type=float, default=0.002, help='weight for routing probability regularization')
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
    parser.add_argument('--use_sam_region_relation_loss', action='store_true',
                       help='use remapped whole-image SAM regions to regularize fusion feature relations during training')
    parser.add_argument('--sam_region_relation_weight', type=float, default=0.003,
                       help='weight for SAM region relation loss (default: 0.003)')
    parser.add_argument('--sam_region_relation_start_epoch', type=int, default=20,
                       help='start epoch for SAM region relation loss (default: 20)')
    parser.add_argument('--sam_region_relation_count', type=int, default=4,
                       help='top-k SAM regions used by relation loss (default: 4)')
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
    parser.add_argument('--use_sam_distillation', action='store_true',
                       help='use training-only SAM distillation on top of the current ASE mainline')
    parser.add_argument('--sam_distill_route_weight', type=float, default=0.005,
                       help='weight for SAM semantic route consistency loss (default: 0.005)')
    parser.add_argument('--sam_distill_boundary_recon_weight', type=float, default=0.01,
                       help='weight for SAM boundary-focused reconstruction loss (default: 0.01)')
    parser.add_argument('--sam_distill_start_epoch', type=int, default=20,
                       help='start epoch for SAM distillation losses (default: 20)')
    parser.add_argument('--use_hf_wavelet_loss', action='store_true',
                       help='enable training-only Haar high-frequency reconstruction loss on SR/GT')
    parser.add_argument('--hf_wavelet_loss_weight', type=float, default=0.01,
                       help='weight for HF wavelet reconstruction loss (default: 0.01)')
    parser.add_argument('--hf_wavelet_loss_start_epoch', type=int, default=5,
                       help='start epoch for HF wavelet reconstruction loss (default: 5)')
    parser.add_argument('--use_semantic_region_weighted_hf_wavelet_loss', action='store_true',
                       help='enable training-only semantic-region weighted Haar high-frequency loss')
    parser.add_argument('--semantic_region_hf_wavelet_loss_weight', type=float, default=0.003,
                       help='weight for semantic-region weighted HF wavelet loss (default: 0.003)')
    parser.add_argument('--semantic_region_hf_wavelet_loss_start_epoch', type=int, default=100,
                       help='start epoch for semantic-region weighted HF wavelet loss (default: 100)')
    parser.add_argument('--semantic_region_hf_wavelet_boundary_boost', type=float, default=0.5,
                       help='extra boost applied to semantic boundary regions in weighted HF wavelet loss (default: 0.5)')
    parser.add_argument('--use_boundary_selective_wavelet_loss', action='store_true',
                       help='enable boundary-selective semantic-frequency weighted Haar high-frequency loss')
    parser.add_argument('--boundary_selective_wavelet_loss_weight', type=float, default=0.003,
                       help='weight for boundary-selective wavelet loss (default: 0.003)')
    parser.add_argument('--boundary_selective_wavelet_loss_start_epoch', type=int, default=80,
                       help='start epoch for boundary-selective wavelet loss (default: 80)')
    parser.add_argument('--boundary_selective_wavelet_boundary_boost', type=float, default=0.75,
                       help='boundary emphasis in boundary-selective wavelet loss (default: 0.75)')
    parser.add_argument('--boundary_selective_wavelet_frequency_boost', type=float, default=0.5,
                       help='semantic-frequency prior emphasis in boundary-selective wavelet loss (default: 0.5)')

    parser.add_argument('--use_fass', action='store_true', help='use FASS (Frequency-Adaptive Sparse Scanning) module')
    parser.add_argument('--fass_compression_ratio', type=int, default=2, help='FASS high-frequency compression ratio (2=compress to 1/2)')
    parser.add_argument('--fass_threshold', type=float, default=0.5, help='[DEPRECATED] FASS gating network threshold (use Top-K instead)')
    parser.add_argument('--fass_sparsity_target', type=float, default=0.3, help='[DEPRECATED] FASS target sparsity (use fass_ll_sparsity and fass_hf_sparsity instead)')
    parser.add_argument('--fass_ll_sparsity', type=float, default=1.0, help='FASS LL branch keep ratio (1.0=dense, 0.25=keep 25%% tokens). Default: 1.0')
    parser.add_argument('--fass_hf_sparsity', type=float, default=0.20, help='FASS HF branch keep ratio (0.20=keep 20%% tokens, i.e., 80%% sparse). Default: 0.20 (HF moderately sparse)')
    parser.add_argument('--fass_d_state', type=int, default=16, help='FASS Mamba state dimension')
    parser.add_argument('--gating_loss_weight', type=float, default=1.0, help='Weight for gating network training loss (default: 1.0, increase if gating not learning)')
    parser.add_argument('--gating_reg_mode', type=str, default='legacy', choices=['legacy', 'budget'],
                       help='Gating regularization mode: legacy=push masks smaller, budget=match keep ratio targets')
    parser.add_argument('--gating_reg_weight', type=float, default=0.01,
                       help='Weight for gating regularization (default: 0.01)')
    parser.add_argument('--ll_target_keep', type=float, default=None,
                       help='Target LL keep ratio for budget gating regularization (default: use fass_ll_sparsity)')
    parser.add_argument('--hf_target_keep', type=float, default=None,
                       help='Target HF keep ratio for budget gating regularization (default: use fass_hf_sparsity)')
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

    parser.add_argument('--train_mode', type=str, default='auto', choices=['dense', 'sparse', 'auto'],
                       help='FASS training mode: dense=full training, sparse=sparse training, auto=auto switch (default: auto)')
    parser.add_argument('--dense_epochs', type=int, default=100,
                       help='Number of epochs for dense training before switching to sparse (default: 100)')
    parser.set_defaults(
        gating_use_semantic_mask=True,
        gating_use_prompt_strength=True,
        gating_use_local_contrast=True,
        sam_prior_use_boundary=True,
        sam_prior_use_confidence=True,
        use_ase=True,
        use_ase_fusion_residual=True,
        use_offline_sam_cache=True,
        prefer_multi_region_sam_cache=True,
        use_wavelet_priors=True,
        use_joint_spatial_spectral_wavelet_prior=True,
        use_sam_state_organizer_v1=True,
        use_boundary_selective_wavelet_loss=True,
    )
    args = parser.parse_args()


    if args.dataset.lower() == 'chikusei':

        if args.train_data_path is None:
            args.train_data_path = './data/chikusei'
        if args.val_data_path is None:
            args.val_data_path = './data/chikusei'


        if args.channels == 32:
            args.channels = 64
        print(f"[INFO] CHIKUSEI数据集调整: 特征通道数从32增加到64以提高特征表达能力")
        print(f"[INFO] CHIKUSEI数据集路径调整: train_data_path={args.train_data_path}, val_data_path={args.val_data_path}")
    elif args.dataset.lower() == 'pavia_university':

        if args.train_data_path is None:
            args.train_data_path = './data/pavia_university_x4'
        if args.val_data_path is None:
            args.val_data_path = './data/pavia_university_x4'
        if args.test_h5_path is None:
            args.test_h5_path = './data/pavia_university_x4/pavia_university_test.h5'

        if args.channels == 32:
            args.channels = 64
        print(f"[INFO] Pavia University数据集调整: 特征通道数从32增加到64以适配HSI-MSI融合主线")
        print(
            f"[INFO] Pavia University数据集路径调整: train_data_path={args.train_data_path}, "
            f"val_data_path={args.val_data_path}, test_h5_path={args.test_h5_path}"
        )
    elif args.dataset.lower() == 'xiongan_new_area':

        if args.train_data_path is None:
            args.train_data_path = './data/xiongan_new_area_x4'
        if args.val_data_path is None:
            args.val_data_path = './data/xiongan_new_area_x4'
        if args.test_h5_path is None:
            args.test_h5_path = './data/xiongan_new_area_x4/xiongan_new_area_test.h5'

        if args.channels == 32:
            args.channels = 64
        print(f"[INFO] Xiongan New Area数据集调整: 特征通道数从32增加到64以适配HSI-MSI融合主线")
        print(
            f"[INFO] Xiongan New Area数据集路径调整: train_data_path={args.train_data_path}, "
            f"val_data_path={args.val_data_path}, test_h5_path={args.test_h5_path}"
        )
    else:
        if args.train_data_path is None:
            args.train_data_path = './data/train/CAVE/train'
        if args.val_data_path is None:
            args.val_data_path = './data/train/CAVE/val'


    custom_weight_dir = create_custom_weight_dir(args)


    training_data_loader, validate_data_loader = prepare_training_data(args)


    train(args, training_data_loader, validate_data_loader, custom_weight_dir)
