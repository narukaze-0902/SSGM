import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from model.fass_module import FASSModule


from model.sam_ase_mamba import SAMASEModule


class FASSSAMASEMambaBlock(nn.Module):
    def __init__(
        self,
        d_model,
        sam_prompt_dim=64,
        d_state=8,
        use_sam=True,
        sam_checkpoint=None,
        use_wavelet=False,
        use_learnable_prompts=False,
        num_learnable_prompts=16,
        use_soft_masks=False,
        num_soft_regions=8,
        input_channels=4,
        modal_type='single',

        use_fass=True,
        fass_compression_ratio=2,
        fass_threshold=0.5,
        fass_sparsity_target=0.3,
        fass_d_state=16,
        train_mode='auto',
        dense_epochs=100
    ):
        super().__init__()
        self.d_model = d_model
        self.input_channels = input_channels
        self.use_fass = use_fass
        self.use_wavelet = use_wavelet
        self.modal_type = modal_type


        if use_fass:
            self.fass_module = FASSModule(
                in_channels=input_channels,
                compression_ratio=fass_compression_ratio,
                threshold=fass_threshold,
                sparsity_target=fass_sparsity_target,
                d_state=fass_d_state,
                train_mode=train_mode,
                dense_epochs=dense_epochs
            )
            print(f"[FASS+SAM-ASE] FASS模块已启用")
            print(f"  - 压缩比: {fass_compression_ratio}")
            print(f"  - 阈值: {fass_threshold}")
            print(f"  - 目标稀疏度: {fass_sparsity_target}")


        self.sam_ase_module = SAMASEModule(
            dim=d_model,
            d_state=d_state,
            input_resolution=(64, 64),
            sam_prompt_dim=sam_prompt_dim,
            use_sam=use_sam,
            sam_checkpoint=sam_checkpoint,
            use_learnable_prompts=use_learnable_prompts,
            num_learnable_prompts=num_learnable_prompts,
            use_soft_masks=use_soft_masks,
            num_soft_regions=num_soft_regions,
            input_channels=input_channels
        )
        print(f"[FASS+SAM-ASE] SAM-ASE模块已启用")


        fusion_input_dim = d_model * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, d_model),
            nn.LayerNorm(d_model)
        )

        print(f"[FASS+SAM-ASE] 融合模块已启用")


        self.out_proj = nn.Linear(d_model, d_model)


        if use_wavelet and not use_fass:
            self.wavelet_conv = nn.Sequential(
                nn.Conv2d(d_model, d_model * 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(d_model * 2, d_model, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            print(f"[FASS+SAM-ASE] 小波变换层已启用")

    def forward(self, x, original_msi=None, sam_prompt=None, semantic_masks=None, current_epoch=0):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))


        if original_msi is None:
            original_msi = x.permute(0, 2, 1).reshape(B, C, H, W)
        else:

            _, _, msi_h, msi_w = original_msi.shape
            if msi_h != H or msi_w != W:

                original_msi = F.interpolate(original_msi, size=(H, W), mode='bilinear', align_corners=False)


        if self.use_fass:

            fass_output = self.fass_module(original_msi, epoch=current_epoch)


            fass_features = fass_output.reshape(B, -1, H * W).permute(0, 2, 1)


            if fass_features.shape[-1] != self.d_model:
                if not hasattr(self, 'fass_proj'):
                    self.fass_proj = nn.Linear(fass_features.shape[-1], self.d_model).to(x.device)
                fass_features = self.fass_proj(fass_features)


            fass_features = fass_features + x
        else:
            fass_features = x


        x_size = (H, W)
        sam_features = self.sam_ase_module(x, x_size, original_msi=original_msi)


        fused = torch.cat([fass_features, sam_features], dim=-1)
        fused = self.fusion(fused)


        if hasattr(self, 'wavelet_conv'):

            fused_spatial = fused.permute(0, 2, 1).reshape(B, C, H, W)


            wavelet_out = self.wavelet_conv(fused_spatial)


            wavelet_out = wavelet_out.reshape(B, C, H * W).permute(0, 2, 1)


            fused = fused + wavelet_out


        output = self.out_proj(fused)


        output = output + x

        return output


class FASSSAMASEMambaBlockV2(nn.Module):
    def __init__(
        self,
        d_model,
        sam_prompt_dim=64,
        d_state=8,
        use_sam=True,
        sam_checkpoint=None,
        use_learnable_prompts=False,
        num_learnable_prompts=16,
        use_soft_masks=False,
        num_soft_regions=8,
        input_channels=4,
        modal_type='single',

        use_fass=True,
        fass_compression_ratio=2,
        fass_threshold=0.5,
        fass_sparsity_target=0.3,
        fass_d_state=16,
        train_mode='auto',
        dense_epochs=100
    ):
        super().__init__()
        self.d_model = d_model
        self.input_channels = input_channels
        self.use_fass = use_fass
        self.modal_type = modal_type


        if use_fass:
            self.fass_module = FASSModule(
                in_channels=input_channels,
                compression_ratio=fass_compression_ratio,
                threshold=fass_threshold,
                sparsity_target=fass_sparsity_target,
                d_state=fass_d_state,
                train_mode=train_mode,
                dense_epochs=dense_epochs
            )
            print(f"[FASS+SAM-ASE V2] 深度融合模式：SAM使用FASS输出")


        self.sam_ase_module = SAMASEModule(
            dim=d_model,
            d_state=d_state,
            input_resolution=(64, 64),
            sam_prompt_dim=sam_prompt_dim,
            use_sam=use_sam,
            sam_checkpoint=sam_checkpoint,
            use_learnable_prompts=use_learnable_prompts,
            num_learnable_prompts=num_learnable_prompts,
            use_soft_masks=use_soft_masks,
            num_soft_regions=num_soft_regions,
            input_channels=input_channels
        )


        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, original_msi=None, sam_prompt=None, semantic_masks=None, current_epoch=0):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))


        if original_msi is None:
            original_msi = x.permute(0, 2, 1).reshape(B, C, H, W)
        else:

            _, _, msi_h, msi_w = original_msi.shape
            if msi_h != H or msi_w != W:

                original_msi = F.interpolate(original_msi, size=(H, W), mode='bilinear', align_corners=False)


        if self.use_fass:

            enhanced_msi = self.fass_module(original_msi, epoch=current_epoch)


            fass_features = enhanced_msi.reshape(B, -1, H * W).permute(0, 2, 1)


            if fass_features.shape[-1] != self.d_model:
                if not hasattr(self, 'fass_proj'):
                    self.fass_proj = nn.Linear(fass_features.shape[-1], self.d_model).to(x.device)
                fass_features = self.fass_proj(fass_features)


            fass_features = fass_features + x


            sam_input_msi = enhanced_msi
        else:
            fass_features = x
            sam_input_msi = original_msi


        x_size = (H, W)
        sam_features = self.sam_ase_module(x, x_size, original_msi=sam_input_msi)


        fused = torch.cat([fass_features, sam_features], dim=-1)
        fused = self.fusion(fused)


        output = self.out_proj(fused)
        output = output + x

        return output


if __name__ == "__main__":
    print("=" * 60)
    print("FASS+SAM-ASE 融合模块测试")
    print("=" * 60)


    B, L, C = 2, 4096, 32
    x = torch.randn(B, L, C)
    original_msi = torch.randn(B, 4, 64, 64)

    print(f"\n输入形状: {x.shape}")
    print(f"原始MSI形状: {original_msi.shape}")


    print("\n" + "=" * 60)
    print("测试 V1：串联架构")
    print("=" * 60)

    fass_sam_ase_v1 = FASSSAMASEMambaBlock(
        d_model=32,
        sam_prompt_dim=64,
        use_sam=False,
        use_fass=True,
        fass_compression_ratio=2,
        fass_threshold=0.5
    )

    output_v1 = fass_sam_ase_v1(x, original_msi)
    print(f"V1输出形状: {output_v1.shape}")
    assert output_v1.shape == x.shape, "V1输出形状不匹配！"
    print("✅ V1形状验证通过")


    print("\n" + "=" * 60)
    print("测试 V2：深度融合")
    print("=" * 60)

    fass_sam_ase_v2 = FASSSAMASEMambaBlockV2(
        d_model=32,
        sam_prompt_dim=64,
        use_sam=False,
        use_fass=True,
        fass_compression_ratio=2
    )

    output_v2 = fass_sam_ase_v2(x, original_msi)
    print(f"V2输出形状: {output_v2.shape}")
    assert output_v2.shape == x.shape, "V2输出形状不匹配！"
    print("✅ V2形状验证通过")


    params_v1 = sum(p.numel() for p in fass_sam_ase_v1.parameters())
    params_v2 = sum(p.numel() for p in fass_sam_ase_v2.parameters())

    print(f"\nV1参数量: {params_v1:,}")
    print(f"V2参数量: {params_v2:,}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
