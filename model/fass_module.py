import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleDWT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape


        Yl = F.avg_pool2d(x, kernel_size=2, stride=2)


        x_down = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)


        x_up = F.interpolate(x_down, scale_factor=2, mode='bilinear', align_corners=False)


        high_freq_full = x - x_up


        high_freq = F.avg_pool2d(high_freq_full, kernel_size=2, stride=2)


        Yh_lh = high_freq
        Yh_hl = high_freq
        Yh_hh = high_freq

        return Yl, [Yh_lh, Yh_hl, Yh_hh]


class SimpleIDWT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Yl, Yh_list):
        Yh_lh, Yh_hl, Yh_hh = Yh_list


        Yl_up = F.interpolate(Yl, scale_factor=2, mode='bilinear', align_corners=False)


        Yh_avg = (Yh_lh + Yh_hl + Yh_hh) / 3.0


        Yh_up = F.interpolate(Yh_avg, scale_factor=2, mode='bilinear', align_corners=False)


        x = Yl_up + Yh_up

        return x


class FASSModule(nn.Module):
    def __init__(
        self,
        in_channels=4,
        compression_ratio=2,
        threshold=0.5,
        sparsity_target=0.3,
        d_state=16,
        train_mode='auto',
        dense_epochs=100
    ):
        super().__init__()
        self.in_channels = in_channels
        self.compression_ratio = compression_ratio
        self.threshold = threshold
        self.sparsity_target = sparsity_target
        self.train_mode = train_mode
        self.dense_epochs = dense_epochs
        self.is_gating_frozen = False


        self.dwt = SimpleDWT()
        self.idwt = SimpleIDWT()


        self.ll_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)


        hf_dim_raw = in_channels * 3


        hf_dim_compressed = hf_dim_raw // compression_ratio
        self.hf_proj = nn.Conv2d(hf_dim_raw, hf_dim_compressed, kernel_size=1)
        self.hf_backproj = nn.Conv2d(hf_dim_compressed, hf_dim_raw, kernel_size=1)

        print(f"[FASS] 高频通道配置: {hf_dim_raw} → {hf_dim_compressed} → {hf_dim_raw} "
              f"(压缩比={compression_ratio})")


        self.gating_net = nn.Sequential(
            nn.Conv2d(hf_dim_compressed, hf_dim_compressed // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hf_dim_compressed // 2, hf_dim_compressed // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hf_dim_compressed // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )


        try:
            from mamba_ssm import Mamba
            self.mamba = Mamba(
                d_model=hf_dim_compressed,
                d_state=d_state,
                d_conv=4,
                expand=2
            )
            self.use_mamba = True
            print(f"[FASS] Mamba SSM enabled (d_model={hf_dim_compressed}, d_state={d_state})")
        except Exception as e:
            print(f"[WARNING] mamba_ssm not found, using fallback Linear layer: {e}")
            self.mamba = nn.Linear(hf_dim_compressed, hf_dim_compressed)
            self.use_mamba = False


        self._channel_config = {
            'in_channels': in_channels,
            'hf_dim_raw': hf_dim_raw,
            'hf_dim_compressed': hf_dim_compressed,
            'compression_ratio': compression_ratio
        }

    def forward(self, x, epoch=0):
        B, C, H, W = x.shape


        if self.train_mode == 'auto':
            current_mode = 'dense' if epoch < self.dense_epochs else 'sparse'
        else:
            current_mode = self.train_mode


        if current_mode == 'sparse' and not self.is_gating_frozen:
            self.freeze_gating()


        if current_mode == 'dense':
            return self.forward_dense(x)
        else:
            return self.forward_sparse(x)

    def freeze_gating(self):
        for param in self.gating_net.parameters():
            param.requires_grad = False
        self.is_gating_frozen = True
        print(f"[FASS] Gating network frozen (switching to sparse training)")

    def forward_dense(self, x):

        B, C, H, W = x.shape
        assert C == self.in_channels,\
            f"[FASS-Dense] 输入通道数不匹配: 期望{self.in_channels}, 实际{C}"


        Yl, Yh_list = self.dwt(x)


        Yh_cat = torch.cat(Yh_list, dim=1)
        Yh_cat_raw = Yh_cat.clone()


        expected_hf_dim = self.in_channels * 3
        assert Yh_cat.shape[1] == expected_hf_dim,\
            f"[FASS-Dense] 高频通道数不匹配: 期望{expected_hf_dim}, 实际{Yh_cat.shape[1]}"


        Yl_out = self.ll_conv(Yl)


        Yh_compressed = self.hf_proj(Yh_cat)


        expected_compressed = expected_hf_dim // self.compression_ratio
        assert Yh_compressed.shape[1] == expected_compressed,\
            f"[FASS-Dense] 压缩后通道数不匹配: 期望{expected_compressed}, 实际{Yh_compressed.shape[1]}"


        Yh_out = self.dense_mamba_scan(Yh_compressed)


        assert Yh_out.shape[1] == expected_compressed,\
            f"[FASS-Dense] Mamba输出通道数不匹配: 期望{expected_compressed}, 实际{Yh_out.shape[1]}"


        Yh_out = self.hf_backproj(Yh_out)


        assert Yh_out.shape[1] == expected_hf_dim,\
            f"[FASS-Dense] 升维后通道数不匹配: 期望{expected_hf_dim}, 实际{Yh_out.shape[1]}"


        Yh_out = Yh_out + Yh_cat_raw


        Yh_out_list = torch.chunk(Yh_out, 3, dim=1)


        for i, Yh_i in enumerate(Yh_out_list):
            assert Yh_i.shape[1] == self.in_channels,\
                f"[FASS-Dense] 拆分后通道数{i}不匹配: 期望{self.in_channels}, 实际{Yh_i.shape[1]}"


        out = self.idwt(Yl_out, Yh_out_list)


        assert out.shape[1] == self.in_channels,\
            f"[FASS-Dense] 输出通道数不匹配: 期望{self.in_channels}, 实际{out.shape[1]}"

        return out

    def forward_sparse(self, x):

        B, C, H, W = x.shape
        assert C == self.in_channels,\
            f"[FASS-Sparse] 输入通道数不匹配: 期望{self.in_channels}, 实际{C}"


        Yl, Yh_list = self.dwt(x)


        Yh_cat = torch.cat(Yh_list, dim=1)
        Yh_cat_raw = Yh_cat.clone()


        expected_hf_dim = self.in_channels * 3
        assert Yh_cat.shape[1] == expected_hf_dim,\
            f"[FASS-Sparse] 高频通道数不匹配: 期望{expected_hf_dim}, 实际{Yh_cat.shape[1]}"


        Yl_out = self.ll_conv(Yl)


        Yh_compressed = self.hf_proj(Yh_cat)


        expected_compressed = expected_hf_dim // self.compression_ratio
        assert Yh_compressed.shape[1] == expected_compressed,\
            f"[FASS-Sparse] 压缩后通道数不匹配: 期望{expected_compressed}, 实际{Yh_compressed.shape[1]}"


        mask_prob = self.gating_net(Yh_compressed)


        if self.training:

            mask = (mask_prob > self.threshold).float()
        else:

            mask = (mask_prob > self.threshold).float()


        if self.training:
            sparsity = mask.mean().item()
            if torch.rand(1).item() < 0.01:
                print(f"[FASS-Sparse] 当前稀疏度: {sparsity:.3f} (目标: {1-self.sparsity_target:.3f})")


        Yh_out = self.sparse_mamba_scan(Yh_compressed, mask)


        assert Yh_out.shape[1] == expected_compressed,\
            f"[FASS-Sparse] Mamba输出通道数不匹配: 期望{expected_compressed}, 实际{Yh_out.shape[1]}"


        Yh_out = self.hf_backproj(Yh_out)


        assert Yh_out.shape[1] == expected_hf_dim,\
            f"[FASS-Sparse] 升维后通道数不匹配: 期望{expected_hf_dim}, 实际{Yh_out.shape[1]}"


        Yh_out = Yh_out + Yh_cat_raw


        Yh_out_list = torch.chunk(Yh_out, 3, dim=1)


        for i, Yh_i in enumerate(Yh_out_list):
            assert Yh_i.shape[1] == self.in_channels,\
                f"[FASS-Sparse] 拆分后通道数{i}不匹配: 期望{self.in_channels}, 实际{Yh_i.shape[1]}"


        out = self.idwt(Yl_out, Yh_out_list)


        assert out.shape[1] == self.in_channels,\
            f"[FASS-Sparse] 输出通道数不匹配: 期望{self.in_channels}, 实际{out.shape[1]}"

        return out

    def dense_mamba_scan(self, features):
        B, C, H, W = features.shape


        features_seq = features.flatten(2).permute(0, 2, 1)


        if self.use_mamba:
            output_seq = self.mamba(features_seq)
        else:
            output_seq = self.mamba(features_seq)


        output = output_seq.permute(0, 2, 1).reshape(B, C, H, W)

        return output


    def sparse_mamba_scan(self, features, mask):
        B, C, H, W = features.shape
        output_list = []

        for b in range(B):

            feat_b = features[b:b+1]
            mask_b = mask[b:b+1]


            flat_feat = feat_b.reshape(C, -1).t()
            flat_mask = mask_b.reshape(-1)


            active_indices = (flat_mask > self.threshold).nonzero(as_tuple=True)[0]

            if len(active_indices) > 1:

                active_tokens = flat_feat[active_indices, :]


                if self.use_mamba:
                    active_tokens = active_tokens.unsqueeze(0)
                    processed_tokens = self.mamba(active_tokens)
                    processed_tokens = processed_tokens.squeeze(0)
                else:

                    processed_tokens = self.mamba(active_tokens)


                flat_output = torch.zeros_like(flat_feat)
                flat_output[active_indices, :] = processed_tokens
            else:

                flat_output = flat_feat


            out_b = flat_output.t().reshape(1, C, H, W)
            output_list.append(out_b)


        output = torch.cat(output_list, dim=0)

        return output

    def get_channel_config(self):
        return self._channel_config


if __name__ == "__main__":
    print("=" * 60)
    print("FASS模块测试")
    print("=" * 60)


    B, C, H, W = 2, 4, 64, 64
    x = torch.randn(B, C, H, W)


    fass = FASSModule(
        in_channels=4,
        compression_ratio=2,
        threshold=0.5,
        sparsity_target=0.3
    )


    print("\n通道配置:")
    config = fass.get_channel_config()
    for key, value in config.items():
        print(f"  {key}: {value}")


    print(f"\n输入形状: {x.shape}")
    output = fass(x)
    print(f"输出形状: {output.shape}")


    assert output.shape == x.shape, "输出形状不匹配！"
    print("\n✅ 形状验证通过")


    total_params = sum(p.numel() for p in fass.parameters())
    print(f"\n总参数量: {total_params:,}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
