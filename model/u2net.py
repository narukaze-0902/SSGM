import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
from model.fusion_mamba import FusionMamba
from mamba_ssm.modules.mamba_simple import Mamba
from utils.wavelet_utils import (
    build_joint_spatial_spectral_prior,
    build_region_frequency_statistics,
    build_wavelet_guidance_map,
    should_use_wavelet_priors,
)


class ResBlock2D(nn.Module):
    def __init__(self, dim, res_se_ratio):
        super().__init__()
        hidden_dim = int(res_se_ratio * dim)
        self.conv0 = nn.Conv2d(dim, hidden_dim, 3, 1, 1)
        self.conv1 = nn.Conv2d(hidden_dim, dim, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        rs1 = self.relu(self.conv0(x))
        rs1 = self.conv1(rs1)
        rs = torch.add(x, rs1)
        return rs


class PixelShuffle(nn.Module):
    def __init__(self, dim, scale):
        super().__init__()
        self.upsamle = nn.Sequential(
            nn.Conv2d(dim, dim*(scale**2), 3, 1, 1, bias=False),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        return self.upsamle(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale, upsample='default', combine='add'):
        super().__init__()
        if upsample == 'bilinear':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.LeakyReLU()
            )
        elif upsample == 'bicubic':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.LeakyReLU()
            )
        elif upsample == 'pixelshuffle':
            self.up = nn.Sequential(
                PixelShuffle(in_channels, scale),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.LeakyReLU()
            )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, scale, scale, 0),
                nn.LeakyReLU()
            )
        if combine == 'concat':
            self.conv = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels * 2, 3, 1, 1, groups=out_channels*2),
                nn.Conv2d(out_channels * 2, out_channels, 1, 1, 0),
                nn.LeakyReLU()
                )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
                nn.Conv2d(out_channels, out_channels, 1, 1, 0),
                nn.LeakyReLU()
                )
        self.combine = combine

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.combine == 'concat':
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1 + x2
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, scale, downsample='default'):
        super().__init__()
        if downsample == 'maxpooling':
            self.down = nn.Sequential(
                nn.MaxPool2d(scale),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
                nn.LeakyReLU()
            )
        else:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, scale, scale, 0),
                nn.LeakyReLU()
            )

    def forward(self, x):
        return self.down(x)


class Stage(nn.Module):
    def __init__(self, in_channels, out_channels, H=64, W=64, scale=2, sample_mode='down',
                 use_ase=False, num_ase_prompts=32, ase_rank=8, use_sam_ase=False, sam_checkpoint=None, sam_prompt_dim=64,
                 ase_prompt_mode='hard', ase_route_temperature=1.0, ase_prompt_soft_mix=0.5,
                 ase_scope='all', use_ase_fusion_residual=False, ase_fusion_res_scale=0.3,
                 use_learnable_ase_fusion_res_scale=False,
                 use_sam_semantic_prompt_bank=False, sam_semantic_prompt_bank_scale=0.1,
                 use_sam_region_prototype_bank=False, sam_region_prototype_bank_scale=0.1,
                 sam_region_prototype_count=8,
                 use_wavelet_guided_sam_prototype_scaling=False, wavelet_guided_sam_prototype_scale=0.1,
                 use_sam_region_prompt_mixture=False, sam_region_prompt_mixture_scale=0.05,
                 sam_region_prompt_mixture_count=8,
                 use_sam_guided_semantic_scanning=False, sam_semantic_scanning_count=6,
                 use_sam_feature_cluster_scanning=False, sam_feature_cluster_count=6,
                 sam_feature_cluster_iters=2, sam_feature_cluster_spatial_weight=0.05,
                 use_wavelet_augmented_ss1=False, wavelet_augmented_ss1_count=6,
                 wavelet_augmented_ss1_topk_ratio=0.25,
                 wavelet_augmented_ss1_strength=0.5,
                 wavelet_augmented_ss1_mode='stable_intra_region',
                 use_sam_boundary_aware_state_propagation=False, sam_boundary_aware_state_scale=0.2,
                 use_sam_state_reset_stronger=False, sam_state_reset_scale=0.35,
                 use_sam_state_organizer_v1=False, sam_state_organizer_count=6,
                 sam_state_organizer_boundary_scale=0.1, sam_state_organizer_reset_scale=0.15,
                 use_sam_region_prompt_subspace=False, sam_region_prompt_subspace_scale=0.05,
                 sam_region_prompt_subspace_count=6,
                 use_wavelet_guided_semantic_state_organization=False, wavelet_guided_semantic_state_count=6,
                 wavelet_guided_semantic_state_scale=0.05,
                 wavelet_guided_semantic_boundary_scale=0.1,
                 wavelet_guided_semantic_reset_scale=0.15,
                 use_joint_spatial_spectral_wavelet_prior=False,
                 joint_wavelet_spatial_weight=1.0,
                 joint_wavelet_spectral_weight=1.0,
                 use_dual_prototype_bank=False,
                 dual_prototype_semantic_scale=0.05,
                 dual_prototype_frequency_scale=0.05,
                 dual_prototype_count=6,
                 use_semantic_frequency_state_modulation=False,
                 semantic_frequency_state_count=6,
                 semantic_frequency_state_write_scale=0.08,
                 semantic_frequency_state_read_scale=0.08,
                 semantic_frequency_state_delta_scale=0.05,
                 use_wavelet_local_bias=False, wavelet_local_bias_scale=0.1,
                 use_wavelet_local_gate=False, wavelet_local_gate_scale=0.1,
                 use_sam_local_gate=False, sam_local_gate_scale=0.1,
                 use_learnable_prompts=False, num_learnable_prompts=16,
                 use_soft_masks=False, num_soft_regions=8, input_channels=4,
                 use_structure_guided_sam_ase=False, structure_texture_weight=0.25,
                 use_fass=False, fass_compression_ratio=2, fass_threshold=0.5,
                 fass_sparsity_target=0.3, fass_ll_sparsity=0.25, fass_hf_sparsity=0.08,
                 fass_d_state=16,
                 train_mode='auto', dense_epochs=100,
                 gating_loss_weight=None):
        super().__init__()

        self.fm = FusionMamba(in_channels, H, W, use_ase=use_ase, num_ase_prompts=num_ase_prompts, ase_rank=ase_rank,
                             ase_prompt_mode=ase_prompt_mode, ase_route_temperature=ase_route_temperature, ase_prompt_soft_mix=ase_prompt_soft_mix,
                             ase_scope=ase_scope, use_ase_fusion_residual=use_ase_fusion_residual, ase_fusion_res_scale=ase_fusion_res_scale,
                             use_learnable_ase_fusion_res_scale=use_learnable_ase_fusion_res_scale,
                             use_sam_semantic_prompt_bank=use_sam_semantic_prompt_bank, sam_semantic_prompt_bank_scale=sam_semantic_prompt_bank_scale,
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
                             use_dual_prototype_bank=use_dual_prototype_bank,
                             dual_prototype_semantic_scale=dual_prototype_semantic_scale,
                             dual_prototype_frequency_scale=dual_prototype_frequency_scale,
                             dual_prototype_count=dual_prototype_count,
                             use_semantic_frequency_state_modulation=use_semantic_frequency_state_modulation,
                             semantic_frequency_state_count=semantic_frequency_state_count,
                             semantic_frequency_state_write_scale=semantic_frequency_state_write_scale,
                             semantic_frequency_state_read_scale=semantic_frequency_state_read_scale,
                             semantic_frequency_state_delta_scale=semantic_frequency_state_delta_scale,
                             use_wavelet_local_bias=use_wavelet_local_bias, wavelet_local_bias_scale=wavelet_local_bias_scale,
                             use_wavelet_local_gate=use_wavelet_local_gate, wavelet_local_gate_scale=wavelet_local_gate_scale,
                             use_sam_local_gate=use_sam_local_gate, sam_local_gate_scale=sam_local_gate_scale,
                             use_sam_ase=use_sam_ase, sam_checkpoint=sam_checkpoint, sam_prompt_dim=sam_prompt_dim,
                             use_learnable_prompts=use_learnable_prompts, num_learnable_prompts=num_learnable_prompts,
                             use_soft_masks=use_soft_masks, num_soft_regions=num_soft_regions, input_channels=input_channels,
                             use_structure_guided_sam_ase=use_structure_guided_sam_ase,
                             structure_texture_weight=structure_texture_weight,
                             use_fass=use_fass, fass_compression_ratio=fass_compression_ratio, fass_threshold=fass_threshold,
                             fass_sparsity_target=fass_sparsity_target, fass_ll_sparsity=fass_ll_sparsity, fass_hf_sparsity=fass_hf_sparsity,
                             fass_d_state=fass_d_state,
                             train_mode=train_mode, dense_epochs=dense_epochs,
                             gating_loss_weight=gating_loss_weight)
        if sample_mode == 'down':
            self.sample = Down(in_channels, out_channels, scale)
        elif sample_mode == 'up':
            self.sample = Up(in_channels, out_channels, scale)

    def forward(self, pan, ms, pan_pre=None, ms_pre=None, sam_prompt=None, semantic_masks=None,
                original_msi=None, wavelet_guidance=None, current_epoch=0):

        if hasattr(self, 'current_sam_prior_bank'):
            self.fm.current_sam_prior_bank = self.current_sam_prior_bank
        if hasattr(self, 'current_sam_region_context'):
            self.fm.current_sam_region_context = self.current_sam_region_context
        pan, ms = self.fm(
            pan,
            ms,
            sam_prompt=sam_prompt,
            semantic_masks=semantic_masks,
            original_msi=original_msi,
            wavelet_guidance=wavelet_guidance,
            current_epoch=current_epoch,
        )
        if pan_pre is None:
            pan_skip = pan
            ms_skip = ms
            pan = self.sample(pan)
            ms = self.sample(ms)
            return pan, ms, pan_skip, ms_skip
        else:
            pan = self.sample(pan, pan_pre)
            ms = self.sample(ms, ms_pre)
            return pan, ms


class SpeAttention(nn.Module):
    def __init__(self, spe_channels, se_ratio=8, mode='mamba', channels=32):
        super().__init__()

        self.mode = mode
        self.pooling = nn.AdaptiveMaxPool2d((1, 1))
        if mode == 'mamba':
            self.block = nn.Sequential(
                nn.Linear(1, channels),
                nn.LayerNorm(channels),
                Mamba(channels, expand=1, d_state=8, bimamba_type='v2', if_devide_out=True, use_norm=True),
                nn.Linear(channels, 1)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(spe_channels, spe_channels // se_ratio, 1, 1, 0, bias=False),
                nn.LeakyReLU(),
                nn.Conv2d(spe_channels // se_ratio, spe_channels, 1, 1, 0, bias=False),
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        input = self.pooling(input)
        if self.mode == 'mamba':
            output = self.block(input.squeeze(-1)).unsqueeze(-1)
        else:
            output = self.block(input)
        return self.sigmoid(output)


class U2Net(nn.Module):
    def __init__(self, dim=32, lr_hsi_dim=128, hr_msi_dim=None, H=64, W=64, scale=4,
                 use_ase=False, use_sam_ase=False, sam_checkpoint=None, sam_prompt_dim=64,
                 ase_prompt_mode='hard', ase_route_temperature=1.0, ase_prompt_soft_mix=0.5,
                 ase_scope='all', ase_stage_scope='all_stages',
                 use_ase_fusion_residual=False, ase_fusion_res_scale=0.3, ase_stage_res_scales=None,
                 use_learnable_ase_fusion_res_scale=False,
                 use_sam_semantic_prompt_bank=False, sam_semantic_prompt_bank_scale=0.1,
                 use_sam_region_prototype_bank=False, sam_region_prototype_bank_scale=0.1,
                 sam_region_prototype_count=8,
                 use_wavelet_guided_sam_prototype_scaling=False, wavelet_guided_sam_prototype_scale=0.1,
                 use_sam_region_prompt_mixture=False, sam_region_prompt_mixture_scale=0.05,
                 sam_region_prompt_mixture_count=8,
                 use_sam_guided_semantic_scanning=False, sam_semantic_scanning_count=6,
                 use_sam_feature_cluster_scanning=False, sam_feature_cluster_count=6,
                 sam_feature_cluster_iters=2, sam_feature_cluster_spatial_weight=0.05,
                 use_wavelet_augmented_ss1=False, wavelet_augmented_ss1_count=6,
                 wavelet_augmented_ss1_topk_ratio=0.25,
                 wavelet_augmented_ss1_strength=0.5,
                 wavelet_augmented_ss1_mode='stable_intra_region',
                 use_sam_boundary_aware_state_propagation=False, sam_boundary_aware_state_scale=0.2,
                 use_sam_state_reset_stronger=False, sam_state_reset_scale=0.35,
                 use_sam_state_organizer_v1=False, sam_state_organizer_count=6,
                 sam_state_organizer_boundary_scale=0.1, sam_state_organizer_reset_scale=0.15,
                 use_sam_region_prompt_subspace=False, sam_region_prompt_subspace_scale=0.05,
                 sam_region_prompt_subspace_count=6,
                 use_wavelet_guided_semantic_state_organization=False, wavelet_guided_semantic_state_count=6,
                 wavelet_guided_semantic_state_scale=0.05,
                 wavelet_guided_semantic_boundary_scale=0.1,
                 wavelet_guided_semantic_reset_scale=0.15,
                 use_joint_spatial_spectral_wavelet_prior=False,
                 joint_wavelet_spatial_weight=1.0,
                 joint_wavelet_spectral_weight=1.0,
                 use_dual_prototype_bank=False,
                 dual_prototype_semantic_scale=0.05,
                 dual_prototype_frequency_scale=0.05,
                 dual_prototype_count=6,
                 use_semantic_frequency_state_modulation=False,
                 semantic_frequency_state_count=6,
                 semantic_frequency_state_write_scale=0.08,
                 semantic_frequency_state_read_scale=0.08,
                 semantic_frequency_state_delta_scale=0.05,
                 use_wavelet_local_bias=False, wavelet_local_bias_scale=0.1,
                 use_wavelet_local_gate=False, wavelet_local_gate_scale=0.1,
                 use_sam_local_gate=False, sam_local_gate_scale=0.1,
                  use_learnable_prompts=False, num_learnable_prompts=16,
                  use_soft_masks=False, num_soft_regions=8,
                  use_wavelet=False, use_wavelet_priors=False,
                  use_structure_guided_sam_ase=False, structure_texture_weight=0.25,
                 use_fass=False, fass_compression_ratio=2, fass_threshold=0.5,
                 fass_sparsity_target=0.3, fass_ll_sparsity=0.25, fass_hf_sparsity=0.08,
                 fass_d_state=16,
                 train_mode='auto', dense_epochs=100,
                 gating_loss_weight=None):
        super().__init__()

        self.use_ase = use_ase
        self.ase_prompt_mode = ase_prompt_mode
        self.ase_scope = ase_scope
        self.ase_stage_scope = ase_stage_scope
        self.use_ase_fusion_residual = use_ase_fusion_residual
        self.ase_fusion_res_scale = ase_fusion_res_scale
        self.ase_stage_res_scales = ase_stage_res_scales
        self.use_learnable_ase_fusion_res_scale = use_learnable_ase_fusion_res_scale
        self.use_wavelet_local_bias = use_wavelet_local_bias
        self.wavelet_local_bias_scale = wavelet_local_bias_scale
        self.use_wavelet_local_gate = use_wavelet_local_gate
        self.wavelet_local_gate_scale = wavelet_local_gate_scale
        self.use_sam_local_gate = use_sam_local_gate
        self.sam_local_gate_scale = sam_local_gate_scale
        self.ase_route_temperature = ase_route_temperature
        self.ase_prompt_soft_mix = ase_prompt_soft_mix
        self.use_sam_ase = use_sam_ase
        self.use_learnable_prompts = use_learnable_prompts
        self.num_learnable_prompts = num_learnable_prompts
        self.use_soft_masks = use_soft_masks
        self.num_soft_regions = num_soft_regions
        self.use_wavelet = use_wavelet
        self.use_wavelet_priors = should_use_wavelet_priors(
            use_wavelet_legacy=use_wavelet,
            use_wavelet_priors=use_wavelet_priors,
            use_joint_spatial_spectral_wavelet_prior=use_joint_spatial_spectral_wavelet_prior,
            use_structure_guided_sam_ase=use_structure_guided_sam_ase,
            use_wavelet_local_bias=use_wavelet_local_bias,
            use_wavelet_local_gate=use_wavelet_local_gate,
            use_wavelet_guided_sam_prototype_scaling=use_wavelet_guided_sam_prototype_scaling,
            use_wavelet_guided_semantic_state_organization=use_wavelet_guided_semantic_state_organization,
            use_dual_prototype_bank=use_dual_prototype_bank,
            use_semantic_frequency_state_modulation=use_semantic_frequency_state_modulation,
        )
        self.use_joint_spatial_spectral_wavelet_prior = use_joint_spatial_spectral_wavelet_prior
        self.joint_wavelet_spatial_weight = float(joint_wavelet_spatial_weight)
        self.joint_wavelet_spectral_weight = float(joint_wavelet_spectral_weight)
        self.use_structure_guided_sam_ase = use_structure_guided_sam_ase
        self.structure_texture_weight = structure_texture_weight
        self._wavelet_fusion_warning_emitted = False
        self.use_fass = use_fass
        self.fass_compression_ratio = fass_compression_ratio
        self.fass_threshold = fass_threshold
        self.fass_sparsity_target = fass_sparsity_target
        self.fass_ll_sparsity = fass_ll_sparsity
        self.fass_hf_sparsity = fass_hf_sparsity
        self.fass_d_state = fass_d_state
        self.scale = scale
        self.lr_hsi_dim = lr_hsi_dim
        self.hr_msi_dim = hr_msi_dim if hr_msi_dim is not None else 4


        self.train_mode = train_mode
        self.dense_epochs = dense_epochs
        self.current_epoch = 0
        self.gating_loss_weight = gating_loss_weight


        self.upsample = PixelShuffle(lr_hsi_dim, scale)


        if use_wavelet:
            self.wavelet_fusion = nn.Sequential(
                nn.Conv2d(lr_hsi_dim + 4*lr_hsi_dim, lr_hsi_dim, 3, 1, 1),
                nn.LeakyReLU()
            )


        self.raise_lsi_dim = nn.Sequential(
            nn.Conv2d(lr_hsi_dim, dim, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.raise_ms_dim = nn.Sequential(
            nn.Conv2d(hr_msi_dim, dim, 3, 1, 1),
            nn.LeakyReLU()
        )

        self.to_hrlsi = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(dim, lr_hsi_dim, 3, 1, 1)
        )


        dim0 = dim
        dim1 = int(dim0 * 2)
        dim2 = int(dim1 * 2)

        stage_ase_usage = self._resolve_ase_stage_usage(use_ase, ase_stage_scope)
        self.ase_stage_usage = stage_ase_usage
        stage_ase_res_scales = self._resolve_ase_stage_res_scales(ase_fusion_res_scale, ase_stage_res_scales)
        self.resolved_ase_stage_res_scales = stage_ase_res_scales
        self.use_sam_semantic_prompt_bank = use_sam_semantic_prompt_bank
        self.sam_semantic_prompt_bank_scale = sam_semantic_prompt_bank_scale
        self.use_sam_region_prototype_bank = use_sam_region_prototype_bank
        self.sam_region_prototype_bank_scale = sam_region_prototype_bank_scale
        self.sam_region_prototype_count = sam_region_prototype_count
        self.use_wavelet_guided_sam_prototype_scaling = use_wavelet_guided_sam_prototype_scaling
        self.wavelet_guided_sam_prototype_scale = wavelet_guided_sam_prototype_scale
        self.use_sam_region_prompt_mixture = use_sam_region_prompt_mixture
        self.sam_region_prompt_mixture_scale = sam_region_prompt_mixture_scale
        self.sam_region_prompt_mixture_count = sam_region_prompt_mixture_count
        self.use_sam_guided_semantic_scanning = use_sam_guided_semantic_scanning
        self.sam_semantic_scanning_count = sam_semantic_scanning_count
        self.use_sam_feature_cluster_scanning = use_sam_feature_cluster_scanning
        self.sam_feature_cluster_count = sam_feature_cluster_count
        self.sam_feature_cluster_iters = sam_feature_cluster_iters
        self.sam_feature_cluster_spatial_weight = sam_feature_cluster_spatial_weight
        self.use_wavelet_augmented_ss1 = use_wavelet_augmented_ss1
        self.wavelet_augmented_ss1_count = wavelet_augmented_ss1_count
        self.wavelet_augmented_ss1_topk_ratio = wavelet_augmented_ss1_topk_ratio
        self.wavelet_augmented_ss1_strength = wavelet_augmented_ss1_strength
        self.wavelet_augmented_ss1_mode = wavelet_augmented_ss1_mode
        self.use_sam_boundary_aware_state_propagation = use_sam_boundary_aware_state_propagation
        self.sam_boundary_aware_state_scale = sam_boundary_aware_state_scale
        self.use_sam_state_reset_stronger = use_sam_state_reset_stronger
        self.sam_state_reset_scale = sam_state_reset_scale
        self.use_sam_state_organizer_v1 = use_sam_state_organizer_v1
        self.sam_state_organizer_count = sam_state_organizer_count
        self.sam_state_organizer_boundary_scale = sam_state_organizer_boundary_scale
        self.sam_state_organizer_reset_scale = sam_state_organizer_reset_scale
        self.use_sam_region_prompt_subspace = use_sam_region_prompt_subspace
        self.sam_region_prompt_subspace_scale = sam_region_prompt_subspace_scale
        self.sam_region_prompt_subspace_count = sam_region_prompt_subspace_count
        self.use_wavelet_guided_semantic_state_organization = use_wavelet_guided_semantic_state_organization
        self.wavelet_guided_semantic_state_count = wavelet_guided_semantic_state_count
        self.wavelet_guided_semantic_state_scale = wavelet_guided_semantic_state_scale
        self.wavelet_guided_semantic_boundary_scale = wavelet_guided_semantic_boundary_scale
        self.wavelet_guided_semantic_reset_scale = wavelet_guided_semantic_reset_scale
        self.use_dual_prototype_bank = use_dual_prototype_bank
        self.dual_prototype_semantic_scale = dual_prototype_semantic_scale
        self.dual_prototype_frequency_scale = dual_prototype_frequency_scale
        self.dual_prototype_count = dual_prototype_count
        self.use_semantic_frequency_state_modulation = use_semantic_frequency_state_modulation
        self.semantic_frequency_state_count = semantic_frequency_state_count
        self.semantic_frequency_state_write_scale = semantic_frequency_state_write_scale
        self.semantic_frequency_state_read_scale = semantic_frequency_state_read_scale
        self.semantic_frequency_state_delta_scale = semantic_frequency_state_delta_scale

        if use_wavelet_local_bias:
            self.wavelet_local_bias_proj = nn.Sequential(
                nn.Conv2d(3 * lr_hsi_dim, dim0, 1, 1, 0),
                nn.LeakyReLU(),
                nn.Conv2d(dim0, dim0, 3, 1, 1)
            )
            nn.init.zeros_(self.wavelet_local_bias_proj[-1].weight)
            if self.wavelet_local_bias_proj[-1].bias is not None:
                nn.init.zeros_(self.wavelet_local_bias_proj[-1].bias)

        if use_wavelet_local_gate:
            self.wavelet_local_gate_proj = nn.Sequential(
                nn.Conv2d(3 * lr_hsi_dim, dim0, 1, 1, 0),
                nn.LeakyReLU(),
                nn.Conv2d(dim0, dim0, 3, 1, 1)
            )
            nn.init.zeros_(self.wavelet_local_gate_proj[-1].weight)
            if self.wavelet_local_gate_proj[-1].bias is not None:
                nn.init.zeros_(self.wavelet_local_gate_proj[-1].bias)

        if use_sam_local_gate:
            def build_sam_local_gate_proj(stage_dim):
                proj = nn.Sequential(
                    nn.Conv2d(2, stage_dim, 1, 1, 0),
                    nn.LeakyReLU(),
                    nn.Conv2d(stage_dim, stage_dim, 3, 1, 1)
                )
                nn.init.zeros_(proj[-1].weight)
                if proj[-1].bias is not None:
                    nn.init.zeros_(proj[-1].bias)
                return proj

            self.sam_local_gate_proj_stage0 = build_sam_local_gate_proj(dim0)
            self.sam_local_gate_proj_stage1 = build_sam_local_gate_proj(dim1)
            self.sam_local_gate_proj_stage2 = build_sam_local_gate_proj(dim2)
            self.sam_local_gate_proj_stage3 = build_sam_local_gate_proj(dim1)
            self.sam_local_gate_proj_stage4 = build_sam_local_gate_proj(dim0)


        self.stage0 = Stage(dim0, dim1, H, W, sample_mode='down', use_ase=stage_ase_usage['stage0'],
                           ase_prompt_mode=ase_prompt_mode, ase_route_temperature=ase_route_temperature, ase_prompt_soft_mix=ase_prompt_soft_mix,
                           ase_scope=ase_scope, use_ase_fusion_residual=use_ase_fusion_residual, ase_fusion_res_scale=stage_ase_res_scales['stage0'],
                           use_learnable_ase_fusion_res_scale=use_learnable_ase_fusion_res_scale,
                           use_sam_semantic_prompt_bank=use_sam_semantic_prompt_bank, sam_semantic_prompt_bank_scale=sam_semantic_prompt_bank_scale,
                           use_sam_region_prototype_bank=use_sam_region_prototype_bank, sam_region_prototype_bank_scale=sam_region_prototype_bank_scale,
                           sam_region_prototype_count=sam_region_prototype_count,
                           use_wavelet_guided_sam_prototype_scaling=use_wavelet_guided_sam_prototype_scaling,
                           wavelet_guided_sam_prototype_scale=wavelet_guided_sam_prototype_scale,
                           use_sam_region_prompt_mixture=use_sam_region_prompt_mixture, sam_region_prompt_mixture_scale=sam_region_prompt_mixture_scale,
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
                           use_dual_prototype_bank=use_dual_prototype_bank,
                           dual_prototype_semantic_scale=dual_prototype_semantic_scale,
                           dual_prototype_frequency_scale=dual_prototype_frequency_scale,
                           dual_prototype_count=dual_prototype_count,
                           use_semantic_frequency_state_modulation=use_semantic_frequency_state_modulation,
                           semantic_frequency_state_count=semantic_frequency_state_count,
                           semantic_frequency_state_write_scale=semantic_frequency_state_write_scale,
                           semantic_frequency_state_read_scale=semantic_frequency_state_read_scale,
                           semantic_frequency_state_delta_scale=semantic_frequency_state_delta_scale,
                           use_wavelet_local_bias=use_wavelet_local_bias, wavelet_local_bias_scale=wavelet_local_bias_scale,
                           use_wavelet_local_gate=use_wavelet_local_gate, wavelet_local_gate_scale=wavelet_local_gate_scale,
                           use_sam_local_gate=use_sam_local_gate, sam_local_gate_scale=sam_local_gate_scale,
                           use_sam_ase=use_sam_ase, sam_checkpoint=sam_checkpoint, sam_prompt_dim=sam_prompt_dim,
                           use_learnable_prompts=use_learnable_prompts, num_learnable_prompts=num_learnable_prompts,
                           use_soft_masks=use_soft_masks, num_soft_regions=num_soft_regions, input_channels=self.hr_msi_dim,
                           use_structure_guided_sam_ase=use_structure_guided_sam_ase,
                           structure_texture_weight=structure_texture_weight,
                           use_fass=use_fass, fass_compression_ratio=fass_compression_ratio, fass_threshold=fass_threshold,
                           fass_sparsity_target=fass_sparsity_target, fass_ll_sparsity=fass_ll_sparsity, fass_hf_sparsity=fass_hf_sparsity,
                           fass_d_state=fass_d_state,
                           train_mode=train_mode, dense_epochs=dense_epochs,
                           gating_loss_weight=gating_loss_weight)
        self.stage1 = Stage(dim1, dim2, H//2, W//2, sample_mode='down', use_ase=stage_ase_usage['stage1'],
                           ase_prompt_mode=ase_prompt_mode, ase_route_temperature=ase_route_temperature, ase_prompt_soft_mix=ase_prompt_soft_mix,
                           ase_scope=ase_scope, use_ase_fusion_residual=use_ase_fusion_residual, ase_fusion_res_scale=stage_ase_res_scales['stage1'],
                           use_learnable_ase_fusion_res_scale=use_learnable_ase_fusion_res_scale,
                           use_sam_semantic_prompt_bank=use_sam_semantic_prompt_bank, sam_semantic_prompt_bank_scale=sam_semantic_prompt_bank_scale,
                           use_sam_region_prototype_bank=use_sam_region_prototype_bank, sam_region_prototype_bank_scale=sam_region_prototype_bank_scale,
                           sam_region_prototype_count=sam_region_prototype_count,
                           use_wavelet_guided_sam_prototype_scaling=use_wavelet_guided_sam_prototype_scaling,
                           wavelet_guided_sam_prototype_scale=wavelet_guided_sam_prototype_scale,
                           use_sam_region_prompt_mixture=use_sam_region_prompt_mixture, sam_region_prompt_mixture_scale=sam_region_prompt_mixture_scale,
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
                           use_dual_prototype_bank=use_dual_prototype_bank,
                           dual_prototype_semantic_scale=dual_prototype_semantic_scale,
                           dual_prototype_frequency_scale=dual_prototype_frequency_scale,
                           dual_prototype_count=dual_prototype_count,
                           use_semantic_frequency_state_modulation=use_semantic_frequency_state_modulation,
                           semantic_frequency_state_count=semantic_frequency_state_count,
                           semantic_frequency_state_write_scale=semantic_frequency_state_write_scale,
                           semantic_frequency_state_read_scale=semantic_frequency_state_read_scale,
                           semantic_frequency_state_delta_scale=semantic_frequency_state_delta_scale,
                           use_wavelet_local_bias=use_wavelet_local_bias, wavelet_local_bias_scale=wavelet_local_bias_scale,
                           use_wavelet_local_gate=use_wavelet_local_gate, wavelet_local_gate_scale=wavelet_local_gate_scale,
                           use_sam_local_gate=use_sam_local_gate, sam_local_gate_scale=sam_local_gate_scale,
                           use_sam_ase=use_sam_ase, sam_checkpoint=sam_checkpoint, sam_prompt_dim=sam_prompt_dim,
                           use_learnable_prompts=use_learnable_prompts, num_learnable_prompts=num_learnable_prompts,
                           use_soft_masks=use_soft_masks, num_soft_regions=num_soft_regions, input_channels=self.hr_msi_dim,
                           use_structure_guided_sam_ase=use_structure_guided_sam_ase,
                           structure_texture_weight=structure_texture_weight,
                           use_fass=use_fass, fass_compression_ratio=fass_compression_ratio, fass_threshold=fass_threshold,
                           fass_sparsity_target=fass_sparsity_target, fass_ll_sparsity=fass_ll_sparsity, fass_hf_sparsity=fass_hf_sparsity,
                           fass_d_state=fass_d_state,
                           train_mode=train_mode, dense_epochs=dense_epochs,
                           gating_loss_weight=gating_loss_weight)
        self.stage2 = Stage(dim2, dim1, H//4, W//4, sample_mode='up', use_ase=stage_ase_usage['stage2'],
                           ase_prompt_mode=ase_prompt_mode, ase_route_temperature=ase_route_temperature, ase_prompt_soft_mix=ase_prompt_soft_mix,
                           ase_scope=ase_scope, use_ase_fusion_residual=use_ase_fusion_residual, ase_fusion_res_scale=stage_ase_res_scales['stage2'],
                           use_learnable_ase_fusion_res_scale=use_learnable_ase_fusion_res_scale,
                           use_sam_semantic_prompt_bank=use_sam_semantic_prompt_bank, sam_semantic_prompt_bank_scale=sam_semantic_prompt_bank_scale,
                           use_sam_region_prototype_bank=use_sam_region_prototype_bank, sam_region_prototype_bank_scale=sam_region_prototype_bank_scale,
                           sam_region_prototype_count=sam_region_prototype_count,
                           use_wavelet_guided_sam_prototype_scaling=use_wavelet_guided_sam_prototype_scaling,
                           wavelet_guided_sam_prototype_scale=wavelet_guided_sam_prototype_scale,
                           use_sam_region_prompt_mixture=use_sam_region_prompt_mixture, sam_region_prompt_mixture_scale=sam_region_prompt_mixture_scale,
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
                           use_dual_prototype_bank=use_dual_prototype_bank,
                           dual_prototype_semantic_scale=dual_prototype_semantic_scale,
                           dual_prototype_frequency_scale=dual_prototype_frequency_scale,
                           dual_prototype_count=dual_prototype_count,
                           use_semantic_frequency_state_modulation=use_semantic_frequency_state_modulation,
                           semantic_frequency_state_count=semantic_frequency_state_count,
                           semantic_frequency_state_write_scale=semantic_frequency_state_write_scale,
                           semantic_frequency_state_read_scale=semantic_frequency_state_read_scale,
                           semantic_frequency_state_delta_scale=semantic_frequency_state_delta_scale,
                           use_wavelet_local_bias=use_wavelet_local_bias, wavelet_local_bias_scale=wavelet_local_bias_scale,
                           use_wavelet_local_gate=use_wavelet_local_gate, wavelet_local_gate_scale=wavelet_local_gate_scale,
                           use_sam_local_gate=use_sam_local_gate, sam_local_gate_scale=sam_local_gate_scale,
                           use_sam_ase=use_sam_ase, sam_checkpoint=sam_checkpoint, sam_prompt_dim=sam_prompt_dim,
                           use_learnable_prompts=use_learnable_prompts, num_learnable_prompts=num_learnable_prompts,
                           use_soft_masks=use_soft_masks, num_soft_regions=num_soft_regions, input_channels=self.hr_msi_dim,
                           use_structure_guided_sam_ase=use_structure_guided_sam_ase,
                           structure_texture_weight=structure_texture_weight,
                           use_fass=use_fass, fass_compression_ratio=fass_compression_ratio, fass_threshold=fass_threshold,
                           fass_sparsity_target=fass_sparsity_target, fass_ll_sparsity=fass_ll_sparsity, fass_hf_sparsity=fass_hf_sparsity,
                           fass_d_state=fass_d_state,
                           train_mode=train_mode, dense_epochs=dense_epochs,
                           gating_loss_weight=gating_loss_weight)
        self.stage3 = Stage(dim1, dim0, H//2, W//2, sample_mode='up', use_ase=stage_ase_usage['stage3'],
                           ase_prompt_mode=ase_prompt_mode, ase_route_temperature=ase_route_temperature, ase_prompt_soft_mix=ase_prompt_soft_mix,
                           ase_scope=ase_scope, use_ase_fusion_residual=use_ase_fusion_residual, ase_fusion_res_scale=stage_ase_res_scales['stage3'],
                           use_learnable_ase_fusion_res_scale=use_learnable_ase_fusion_res_scale,
                           use_sam_semantic_prompt_bank=use_sam_semantic_prompt_bank, sam_semantic_prompt_bank_scale=sam_semantic_prompt_bank_scale,
                           use_sam_region_prototype_bank=use_sam_region_prototype_bank, sam_region_prototype_bank_scale=sam_region_prototype_bank_scale,
                           sam_region_prototype_count=sam_region_prototype_count,
                           use_wavelet_guided_sam_prototype_scaling=use_wavelet_guided_sam_prototype_scaling,
                           wavelet_guided_sam_prototype_scale=wavelet_guided_sam_prototype_scale,
                           use_sam_region_prompt_mixture=use_sam_region_prompt_mixture, sam_region_prompt_mixture_scale=sam_region_prompt_mixture_scale,
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
                           use_dual_prototype_bank=use_dual_prototype_bank,
                           dual_prototype_semantic_scale=dual_prototype_semantic_scale,
                           dual_prototype_frequency_scale=dual_prototype_frequency_scale,
                           dual_prototype_count=dual_prototype_count,
                           use_semantic_frequency_state_modulation=use_semantic_frequency_state_modulation,
                           semantic_frequency_state_count=semantic_frequency_state_count,
                           semantic_frequency_state_write_scale=semantic_frequency_state_write_scale,
                           semantic_frequency_state_read_scale=semantic_frequency_state_read_scale,
                           semantic_frequency_state_delta_scale=semantic_frequency_state_delta_scale,
                           use_wavelet_local_bias=use_wavelet_local_bias, wavelet_local_bias_scale=wavelet_local_bias_scale,
                           use_wavelet_local_gate=use_wavelet_local_gate, wavelet_local_gate_scale=wavelet_local_gate_scale,
                           use_sam_local_gate=use_sam_local_gate, sam_local_gate_scale=sam_local_gate_scale,
                           use_sam_ase=use_sam_ase, sam_checkpoint=sam_checkpoint, sam_prompt_dim=sam_prompt_dim,
                           use_learnable_prompts=use_learnable_prompts, num_learnable_prompts=num_learnable_prompts,
                           use_soft_masks=use_soft_masks, num_soft_regions=num_soft_regions, input_channels=self.hr_msi_dim,
                           use_structure_guided_sam_ase=use_structure_guided_sam_ase,
                           structure_texture_weight=structure_texture_weight,
                           use_fass=use_fass, fass_compression_ratio=fass_compression_ratio, fass_threshold=fass_threshold,
                           fass_sparsity_target=fass_sparsity_target, fass_ll_sparsity=fass_ll_sparsity, fass_hf_sparsity=fass_hf_sparsity,
                           fass_d_state=fass_d_state,
                           train_mode=train_mode, dense_epochs=dense_epochs,
                           gating_loss_weight=gating_loss_weight)
        self.stage4 = FusionMamba(dim0, H, W, final=True, use_ase=stage_ase_usage['stage4'], use_sam_ase=use_sam_ase,
                                 ase_prompt_mode=ase_prompt_mode, ase_route_temperature=ase_route_temperature, ase_prompt_soft_mix=ase_prompt_soft_mix,
                                 ase_scope=ase_scope, use_ase_fusion_residual=use_ase_fusion_residual, ase_fusion_res_scale=stage_ase_res_scales['stage4'],
                                 use_learnable_ase_fusion_res_scale=use_learnable_ase_fusion_res_scale,
                                 use_sam_semantic_prompt_bank=use_sam_semantic_prompt_bank, sam_semantic_prompt_bank_scale=sam_semantic_prompt_bank_scale,
                                 use_sam_region_prototype_bank=use_sam_region_prototype_bank, sam_region_prototype_bank_scale=sam_region_prototype_bank_scale,
                                 sam_region_prototype_count=sam_region_prototype_count,
                                 use_wavelet_guided_sam_prototype_scaling=use_wavelet_guided_sam_prototype_scaling,
                                 wavelet_guided_sam_prototype_scale=wavelet_guided_sam_prototype_scale,
                                 use_sam_region_prompt_mixture=use_sam_region_prompt_mixture, sam_region_prompt_mixture_scale=sam_region_prompt_mixture_scale,
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
                                 use_dual_prototype_bank=use_dual_prototype_bank,
                                 dual_prototype_semantic_scale=dual_prototype_semantic_scale,
                                 dual_prototype_frequency_scale=dual_prototype_frequency_scale,
                                 dual_prototype_count=dual_prototype_count,
                                 use_semantic_frequency_state_modulation=use_semantic_frequency_state_modulation,
                                 semantic_frequency_state_count=semantic_frequency_state_count,
                                 semantic_frequency_state_write_scale=semantic_frequency_state_write_scale,
                                 semantic_frequency_state_read_scale=semantic_frequency_state_read_scale,
                                 semantic_frequency_state_delta_scale=semantic_frequency_state_delta_scale,
                                 use_wavelet_local_bias=use_wavelet_local_bias, wavelet_local_bias_scale=wavelet_local_bias_scale,
                                 use_wavelet_local_gate=use_wavelet_local_gate, wavelet_local_gate_scale=wavelet_local_gate_scale,
                                 use_sam_local_gate=use_sam_local_gate, sam_local_gate_scale=sam_local_gate_scale,
                                 sam_checkpoint=sam_checkpoint, sam_prompt_dim=sam_prompt_dim,
                                 use_learnable_prompts=use_learnable_prompts, num_learnable_prompts=num_learnable_prompts,
                                 use_soft_masks=use_soft_masks, num_soft_regions=num_soft_regions, input_channels=self.hr_msi_dim,
                                 use_wavelet=use_wavelet,
                                 use_structure_guided_sam_ase=use_structure_guided_sam_ase,
                                 structure_texture_weight=structure_texture_weight,
                                 use_fass=use_fass, fass_compression_ratio=fass_compression_ratio, fass_threshold=fass_threshold,
                                 fass_sparsity_target=fass_sparsity_target, fass_ll_sparsity=fass_ll_sparsity, fass_hf_sparsity=fass_hf_sparsity,
                                 fass_d_state=fass_d_state,
                                 train_mode=train_mode, dense_epochs=dense_epochs,
                                 gating_loss_weight=gating_loss_weight)


        self.spe_attn = SpeAttention(lr_hsi_dim, 16, 'mamba', dim)


        if use_ase:

            self.route_fusion = nn.Sequential(
                nn.Conv2d(dim * 2, dim, 3, 1, 1),
                nn.LeakyReLU(),
                nn.Conv2d(dim, 4, 1, 1, 0),
                nn.Softmax(dim=1)
            )
        else:
            self.route_fusion = None


        if use_sam_ase or use_sam_local_gate or use_sam_semantic_prompt_bank:
            from model.sam_ase_mamba import SAMFeatureExtractor
            self.sam_extractor = SAMFeatureExtractor(
                sam_checkpoint_path=sam_checkpoint,
                feature_dim=256,
                output_dim=sam_prompt_dim,
                use_frozen_sam=True,
                use_adapter=True,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            print(f"[SAM-ASE] SAM特征提取器已创建，将在输入层对MSI图像提取语义信息")

    @staticmethod
    def _resolve_ase_stage_usage(use_ase, ase_stage_scope):
        stage_usage_map = {
            'all_stages': {'stage0': True, 'stage1': True, 'stage2': True, 'stage3': True, 'stage4': True},
            'deep34': {'stage0': False, 'stage1': False, 'stage2': False, 'stage3': True, 'stage4': True},
            'deep234': {'stage0': False, 'stage1': False, 'stage2': True, 'stage3': True, 'stage4': True},
            'stage4_only': {'stage0': False, 'stage1': False, 'stage2': False, 'stage3': False, 'stage4': True},
        }
        selected = stage_usage_map.get(ase_stage_scope, stage_usage_map['all_stages'])
        if not use_ase:
            return {k: False for k in selected}
        return selected.copy()

    @staticmethod
    def _resolve_ase_stage_res_scales(default_scale, ase_stage_res_scales):
        stage_names = ['stage0', 'stage1', 'stage2', 'stage3', 'stage4']
        if ase_stage_res_scales is None:
            values = [float(default_scale)] * len(stage_names)
        elif isinstance(ase_stage_res_scales, str):
            parts = [part.strip() for part in ase_stage_res_scales.split(',') if part.strip()]
            if len(parts) != len(stage_names):
                raise ValueError(
                    f"ase_stage_res_scales expects 5 comma-separated values, got {len(parts)}: {ase_stage_res_scales}"
                )
            values = [float(part) for part in parts]
        elif isinstance(ase_stage_res_scales, (list, tuple)):
            if len(ase_stage_res_scales) != len(stage_names):
                raise ValueError(
                    f"ase_stage_res_scales expects 5 values, got {len(ase_stage_res_scales)}"
                )
            values = [float(v) for v in ase_stage_res_scales]
        else:
            raise TypeError(
                f"Unsupported ase_stage_res_scales type: {type(ase_stage_res_scales).__name__}"
            )
        return {stage_name: value for stage_name, value in zip(stage_names, values)}

    def set_current_epoch(self, epoch):
        self.current_epoch = epoch

    def _build_wavelet_guidance_map(self, lr_hsi_details, target_hw, device, dtype):
        if not self.use_wavelet_priors or lr_hsi_details is None:
            return None
        return build_wavelet_guidance_map(
            lr_hsi_details,
            target_hw=target_hw,
            device=device,
            dtype=dtype,
        )

    def _build_wavelet_region_guidance_map(self, lr_hsi_details, target_hw, device, dtype):
        if not self.use_wavelet_priors or lr_hsi_details is None:
            return None
        return build_wavelet_guidance_map(
            lr_hsi_details,
            target_hw=target_hw,
            device=device,
            dtype=dtype,
        )

    def _build_joint_spatial_spectral_prior_maps(self, lr_hsi, lr_hsi_details, target_hw, device, dtype):
        if not self.use_joint_spatial_spectral_wavelet_prior:
            return None, None, None
        return build_joint_spatial_spectral_prior(
            lr_hsi,
            lr_hsi_details,
            target_hw=target_hw,
            device=device,
            dtype=dtype,
            spatial_weight=self.joint_wavelet_spatial_weight,
            spectral_weight=self.joint_wavelet_spectral_weight,
        )

    def _prepare_wavelet_fusion_inputs(self, lr_hsi, lr_hsi_approx, lr_hsi_details):
        if lr_hsi_approx is None or lr_hsi_details is None:
            return None, None

        approx_ok = (
            lr_hsi_approx.dim() == 4
            and lr_hsi_approx.shape[1] == self.lr_hsi_dim
        )
        details_ok = (
            lr_hsi_details.dim() == 5
            and lr_hsi_details.shape[1] == 3
            and lr_hsi_details.shape[2] == self.lr_hsi_dim
        )

        if not (approx_ok and details_ok):
            if not self._wavelet_fusion_warning_emitted:
                print(
                    f"[WAVELET] Skip wavelet_fusion: incompatible coeff shapes "
                    f"approx={tuple(lr_hsi_approx.shape)}, details={tuple(lr_hsi_details.shape)}. "
                    f"Wavelet coeffs will still be used for structure guidance when enabled."
                )
                self._wavelet_fusion_warning_emitted = True
            return None, None

        lr_hsi_approx_up = F.interpolate(
            lr_hsi_approx,
            size=(lr_hsi.shape[2], lr_hsi.shape[3]),
            mode='bilinear',
            align_corners=True,
        )
        lr_hsi_details_merged = lr_hsi_details.reshape(
            lr_hsi_details.shape[0],
            -1,
            lr_hsi_details.shape[3],
            lr_hsi_details.shape[4],
        )
        lr_hsi_details_merged = F.interpolate(
            lr_hsi_details_merged,
            size=(lr_hsi.shape[2], lr_hsi.shape[3]),
            mode='bilinear',
            align_corners=True,
        )
        return lr_hsi_approx_up, lr_hsi_details_merged

    def _build_wavelet_local_bias(self, lr_hsi_details, target_hw, device, dtype):
        if not self.use_wavelet_local_bias or lr_hsi_details is None:
            return None

        details_ok = (
            lr_hsi_details.dim() == 5
            and lr_hsi_details.shape[1] == 3
            and lr_hsi_details.shape[2] == self.lr_hsi_dim
        )
        if not details_ok:
            return None

        details = lr_hsi_details.to(device=device, dtype=dtype).abs()
        details = details.reshape(
            details.shape[0],
            -1,
            details.shape[3],
            details.shape[4],
        )

        if details.shape[-2:] != target_hw:
            details = F.interpolate(details, size=target_hw, mode='bilinear', align_corners=False)

        flat = details.flatten(1)
        min_val = flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        max_val = flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        details = (details - min_val) / (max_val - min_val + 1e-6)
        return self.wavelet_local_bias_proj(details)

    def _build_wavelet_local_gate(self, lr_hsi_details, target_hw, device, dtype):
        if not self.use_wavelet_local_gate or lr_hsi_details is None:
            return None

        details_ok = (
            lr_hsi_details.dim() == 5
            and lr_hsi_details.shape[1] == 3
            and lr_hsi_details.shape[2] == self.lr_hsi_dim
        )
        if not details_ok:
            return None

        details = lr_hsi_details.to(device=device, dtype=dtype).abs()
        details = details.reshape(
            details.shape[0],
            -1,
            details.shape[3],
            details.shape[4],
        )

        if details.shape[-2:] != target_hw:
            details = F.interpolate(details, size=target_hw, mode='bilinear', align_corners=False)

        projected_gate = self.wavelet_local_gate_proj(details)
        return 1.0 + self.wavelet_local_gate_scale * torch.tanh(projected_gate)

    def _build_sam_local_gate(self, target_hw, stage_key, device, dtype):
        if not self.use_sam_local_gate:
            return None

        prior_bank = getattr(self, 'current_sam_prior_bank', None)
        if prior_bank is None:
            return None

        region_map = prior_bank.get('region_map')
        prompt_strength_map = prior_bank.get('prompt_strength_map')
        if region_map is None or prompt_strength_map is None:
            return None

        if region_map.dim() == 3:
            region_map = region_map.unsqueeze(1)
        if prompt_strength_map.dim() == 3:
            prompt_strength_map = prompt_strength_map.unsqueeze(1)

        gate_input = torch.cat([region_map, prompt_strength_map], dim=1).to(device=device, dtype=torch.float32)
        if gate_input.shape[-2:] != target_hw:
            gate_input = F.interpolate(gate_input, size=target_hw, mode='bilinear', align_corners=False)

        proj = getattr(self, f'sam_local_gate_proj_{stage_key}', None)
        if proj is None:
            return None

        projected_gate = proj(gate_input)
        gate = 1.0 + self.sam_local_gate_scale * torch.tanh(projected_gate)
        return gate.to(dtype=dtype)

    def forward(self, hr_msi, lr_hsi, lr_hsi_approx=None, lr_hsi_details=None,
                cached_sam_features=None, cached_sam_masks=None):

        lrlsi = lr_hsi
        expected_batch = hr_msi.shape[0]

        def _align_cached_sam_tensor(tensor, name):
            if tensor is None:
                return None
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.dim() >= 4 and tensor.shape[0] != expected_batch:
                if tensor.shape[0] == 1:
                    expand_shape = [expected_batch] + [-1] * (tensor.dim() - 1)
                    tensor = tensor.expand(*expand_shape).contiguous()
                else:
                    raise ValueError(
                        f"{name} batch mismatch: got {tuple(tensor.shape)}, expected batch {expected_batch}"
                    )
            return tensor

        cached_sam_features = _align_cached_sam_tensor(cached_sam_features, "cached_sam_features")
        cached_sam_masks = _align_cached_sam_tensor(cached_sam_masks, "cached_sam_masks")


        if (self.use_sam_ase or self.use_sam_local_gate or self.use_sam_semantic_prompt_bank) and hasattr(self, 'sam_extractor'):


            sam_prompt, semantic_masks = self.sam_extractor(
                hr_msi,
                cached_raw_features=cached_sam_features,
                cached_raw_masks=cached_sam_masks,
            )


            self.current_sam_prompt = sam_prompt
            self.current_semantic_masks = semantic_masks
            self.current_sam_prior_bank = getattr(self.sam_extractor, '_last_prior_bank', None)
        else:
            self.current_sam_prompt = None
            self.current_semantic_masks = None
            self.current_sam_prior_bank = None
        self.current_wavelet_region_guidance = self._build_wavelet_region_guidance_map(
            lr_hsi_details,
            (hr_msi.shape[2], hr_msi.shape[3]),
            hr_msi.device,
            hr_msi.dtype,
        )
        (
            self.current_spatial_prior_map,
            self.current_spectral_prior_map,
            self.current_semantic_frequency_prior_map,
        ) = self._build_joint_spatial_spectral_prior_maps(
            lrlsi,
            lr_hsi_details,
            (hr_msi.shape[2], hr_msi.shape[3]),
            hr_msi.device,
            hr_msi.dtype,
        )
        self.current_region_frequency_stats = None
        if (
            cached_sam_masks is not None
            and (
                self.use_dual_prototype_bank
                or self.use_semantic_frequency_state_modulation
                or self.use_joint_spatial_spectral_wavelet_prior
            )
        ):
            self.current_region_frequency_stats = build_region_frequency_statistics(
                cached_sam_masks,
                spatial_prior_map=self.current_spatial_prior_map,
                spectral_prior_map=self.current_spectral_prior_map,
                joint_prior_map=self.current_semantic_frequency_prior_map,
                target_hw=(hr_msi.shape[2], hr_msi.shape[3]),
            )
        if (
            self.use_sam_region_prototype_bank
            or self.use_sam_region_prompt_mixture
            or self.use_sam_guided_semantic_scanning
            or self.use_sam_boundary_aware_state_propagation
            or self.use_sam_state_reset_stronger
            or self.use_sam_state_organizer_v1
            or self.use_sam_region_prompt_subspace
            or self.use_wavelet_guided_semantic_state_organization
            or self.use_joint_spatial_spectral_wavelet_prior
            or self.use_dual_prototype_bank
            or self.use_semantic_frequency_state_modulation
        ) and cached_sam_features is not None and cached_sam_masks is not None:
            self.current_sam_region_context = {
                'sam_features': cached_sam_features,
                'sam_masks': cached_sam_masks,
                'wavelet_guidance': self.current_wavelet_region_guidance,
                'wavelet_prior_map': self.current_spatial_prior_map,
                'spectral_prior_map': self.current_spectral_prior_map,
                'semantic_frequency_prior_map': self.current_semantic_frequency_prior_map,
                'region_frequency_stats': self.current_region_frequency_stats,
            }
        else:
            self.current_sam_region_context = None
        self.current_wavelet_guidance = self._build_wavelet_guidance_map(
            lr_hsi_details,
            (hr_msi.shape[2], hr_msi.shape[3]),
            hr_msi.device,
            hr_msi.dtype,
        )
        self.current_wavelet_local_bias = self._build_wavelet_local_bias(
            lr_hsi_details,
            (hr_msi.shape[2], hr_msi.shape[3]),
            hr_msi.device,
            hr_msi.dtype,
        )
        self.current_wavelet_local_gate = self._build_wavelet_local_gate(
            lr_hsi_details,
            (hr_msi.shape[2], hr_msi.shape[3]),
            hr_msi.device,
            hr_msi.dtype,
        )
        self.current_sam_local_gates = {
            'stage0': self._build_sam_local_gate((hr_msi.shape[2], hr_msi.shape[3]), 'stage0', hr_msi.device, hr_msi.dtype),
            'stage1': self._build_sam_local_gate((hr_msi.shape[2] // 2, hr_msi.shape[3] // 2), 'stage1', hr_msi.device, hr_msi.dtype),
            'stage2': self._build_sam_local_gate((hr_msi.shape[2] // 4, hr_msi.shape[3] // 4), 'stage2', hr_msi.device, hr_msi.dtype),
            'stage3': self._build_sam_local_gate((hr_msi.shape[2] // 2, hr_msi.shape[3] // 2), 'stage3', hr_msi.device, hr_msi.dtype),
            'stage4': self._build_sam_local_gate((hr_msi.shape[2], hr_msi.shape[3]), 'stage4', hr_msi.device, hr_msi.dtype),
        }
        for stage_module in [self.stage0.fm, self.stage1.fm, self.stage2.fm, self.stage3.fm, self.stage4]:
            stage_module.current_sam_prior_bank = self.current_sam_prior_bank
            stage_module.current_sam_region_context = self.current_sam_region_context
            stage_module.current_wavelet_guidance = self.current_wavelet_guidance
            stage_module.current_wavelet_local_bias = None
            stage_module.current_wavelet_local_gate = None
            stage_module.current_sam_local_gate = None
        self.stage4.current_wavelet_local_bias = self.current_wavelet_local_bias
        self.stage4.current_wavelet_local_gate = self.current_wavelet_local_gate
        self.stage0.fm.current_sam_local_gate = self.current_sam_local_gates['stage0']
        self.stage1.fm.current_sam_local_gate = self.current_sam_local_gates['stage1']
        self.stage2.fm.current_sam_local_gate = self.current_sam_local_gates['stage2']
        self.stage3.fm.current_sam_local_gate = self.current_sam_local_gates['stage3']
        self.stage4.current_sam_local_gate = self.current_sam_local_gates['stage4']


        lr_hsi = self.upsample(lr_hsi)

        if self.use_wavelet and lr_hsi_approx is not None and lr_hsi_details is not None:
            approx_ok = (
                lr_hsi_approx.dim() == 4
                and lr_hsi_approx.shape[1] == self.lr_hsi_dim
            )
            details_ok = (
                lr_hsi_details.dim() == 5
                and lr_hsi_details.shape[1] == 3
                and lr_hsi_details.shape[2] == self.lr_hsi_dim
            )
            if not (approx_ok and details_ok):
                if not self._wavelet_fusion_warning_emitted:
                    print(
                        f"[WAVELET] Skip wavelet_fusion: incompatible coeff shapes "
                        f"approx={tuple(lr_hsi_approx.shape)}, details={tuple(lr_hsi_details.shape)}. "
                        f"Wavelet coeffs will still be used for structure guidance when enabled."
                    )
                    self._wavelet_fusion_warning_emitted = True
                lr_hsi_approx = None
                lr_hsi_details = None


        if self.use_wavelet and lr_hsi_approx is not None and lr_hsi_details is not None:

            lr_hsi_approx_up, lr_hsi_details_merged = self._prepare_wavelet_fusion_inputs(
                lr_hsi,
                lr_hsi_approx,
                lr_hsi_details,
            )
            if lr_hsi_approx_up is not None and lr_hsi_details_merged is not None:
                lr_hsi = torch.cat([lr_hsi, lr_hsi_approx_up, lr_hsi_details_merged], dim=1)
                lr_hsi = self.wavelet_fusion(lr_hsi)


        skip = lr_hsi


        original_hr_msi = hr_msi


        hr_msi = self.raise_ms_dim(hr_msi)
        lr_hsi = self.raise_lsi_dim(lr_hsi)


        lr_hsi, hr_msi, lrlsi_skip0, hr_msi_skip0 = self.stage0(
            lr_hsi, hr_msi,
            sam_prompt=self.current_sam_prompt,
            semantic_masks=self.current_semantic_masks,
            original_msi=original_hr_msi,
            wavelet_guidance=self.current_wavelet_guidance,
            current_epoch=self.current_epoch
        )
        lr_hsi, hr_msi, lrlsi_skip1, hr_msi_skip1 = self.stage1(
            lr_hsi, hr_msi,
            sam_prompt=self.current_sam_prompt,
            semantic_masks=self.current_semantic_masks,
            original_msi=original_hr_msi,
            wavelet_guidance=self.current_wavelet_guidance,
            current_epoch=self.current_epoch
        )
        lr_hsi, hr_msi = self.stage2(
            lr_hsi, hr_msi, lrlsi_skip1, hr_msi_skip1,
            sam_prompt=self.current_sam_prompt,
            semantic_masks=self.current_semantic_masks,
            original_msi=original_hr_msi,
            wavelet_guidance=self.current_wavelet_guidance,
            current_epoch=self.current_epoch
        )
        lr_hsi, hr_msi = self.stage3(
            lr_hsi, hr_msi, lrlsi_skip0, hr_msi_skip0,
            sam_prompt=self.current_sam_prompt,
            semantic_masks=self.current_semantic_masks,
            original_msi=original_hr_msi,
            wavelet_guidance=self.current_wavelet_guidance,
            current_epoch=self.current_epoch
        )
        output = self.stage4(
            lr_hsi, hr_msi,
            sam_prompt=self.current_sam_prompt,
            semantic_masks=self.current_semantic_masks,
            original_msi=original_hr_msi,
            wavelet_guidance=self.current_wavelet_guidance,
            current_epoch=self.current_epoch
        )
        self.current_fusion_feature_map = output


        spe_attn = self.spe_attn(skip)


        output = self.to_hrlsi(output)
        output = output * spe_attn + skip


        routing_probs = None
        if self.use_ase:
            real_routing_probs = []
            for stage_module in [self.stage0.fm, self.stage1.fm, self.stage2.fm, self.stage3.fm, self.stage4]:
                stage_routes = getattr(stage_module, 'current_ase_routing_probs', None)
                if stage_routes:
                    real_routing_probs.append(stage_routes)

            if len(real_routing_probs) > 0:
                return output, real_routing_probs

            if self.route_fusion is not None:
                combined_features = torch.cat([lr_hsi, hr_msi], dim=1)
                routing_probs = self.route_fusion(combined_features)
                return output, routing_probs

        return output
