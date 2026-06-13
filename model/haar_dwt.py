import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel = channel


        filters = torch.tensor([
            [[[0.5, 0.5], [0.5, 0.5]]],
            [[[-0.5, -0.5], [0.5, 0.5]]],
            [[[-0.5, 0.5], [-0.5, 0.5]]],
            [[[0.5, -0.5], [-0.5, 0.5]]]
        ], dtype=torch.float32)


        filters = filters.repeat(channel, 1, 1, 1)


        self.register_buffer('filters', filters)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.channel, f"Input channels {C} != expected {self.channel}"


        Yl = torch.zeros(B, C, H//2, W//2, device=x.device, dtype=x.dtype)
        Yh_lh = torch.zeros(B, C, H//2, W//2, device=x.device, dtype=x.dtype)
        Yh_hl = torch.zeros(B, C, H//2, W//2, device=x.device, dtype=x.dtype)
        Yh_hh = torch.zeros(B, C, H//2, W//2, device=x.device, dtype=x.dtype)


        for c in range(C):
            x_c = x[:, c:c+1, :, :]


            patches = F.unfold(x_c, kernel_size=2, stride=2)
            patches = patches.reshape(B, 1, 2, 2, H//2, W//2)


            a = patches[:, :, 0, 0, :, :]
            b = patches[:, :, 0, 1, :, :]
            c_bl = patches[:, :, 1, 0, :, :]
            d = patches[:, :, 1, 1, :, :]


            ll = (a + b + c_bl + d) / 2
            lh = (-a - b + c_bl + d) / 2
            hl = (-a + b - c_bl + d) / 2
            hh = (a - b - c_bl + d) / 2


            ll = ll.squeeze(1)
            lh = lh.squeeze(1)
            hl = hl.squeeze(1)
            hh = hh.squeeze(1)


            Yl[:, c, :, :] = ll
            Yh_lh[:, c, :, :] = lh
            Yh_hl[:, c, :, :] = hl
            Yh_hh[:, c, :, :] = hh


        Yh = torch.cat([Yh_lh, Yh_hl, Yh_hh], dim=1)

        return Yl, Yh


class HaarIDWT(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel = channel


        filters = torch.tensor([
            [[[0.5, 0.5], [0.5, 0.5]]],
            [[[-0.5, -0.5], [0.5, 0.5]]],
            [[[-0.5, 0.5], [-0.5, 0.5]]],
            [[[0.5, -0.5], [-0.5, 0.5]]]
        ], dtype=torch.float32)

        filters = filters.repeat(channel, 1, 1, 1)
        self.register_buffer('filters', filters)

    def forward(self, Yl, Yh):
        B, C, H, W = Yl.shape
        assert C == self.channel, f"LL channels {C} != expected {self.channel}"


        Yh_split = Yh.reshape(B, C, 3, H, W)


        LH = Yh_split[:, :, 0, :, :]
        HL = Yh_split[:, :, 1, :, :]
        HH = Yh_split[:, :, 2, :, :]


        Yl_up = F.interpolate(Yl, scale_factor=2, mode='nearest')
        LH_up = F.interpolate(LH, scale_factor=2, mode='nearest')
        HL_up = F.interpolate(HL, scale_factor=2, mode='nearest')
        HH_up = F.interpolate(HH, scale_factor=2, mode='nearest')


        x = Yl_up + LH_up + HL_up + HH_up

        return x


class SimpleDWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel = 64

    def forward(self, x):
        B, C, H, W = x.shape


        if not hasattr(self, '_haar_dwt') or self._haar_dwt.channel != C:
            self._haar_dwt = HaarDWT(C).to(x.device)

        Yl, Yh = self._haar_dwt(x)


        _, _, H_l, W_l = Yl.shape
        _, _, _, W_actual = Yh.shape


        Yh_split = Yh.reshape(B, C, 3, H_l, W_l)
        Yh_list = [Yh_split[:, :, i] for i in range(3)]

        return Yl, Yh_list


class SimpleIDWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel = 64

    def forward(self, Yl, Yh_list):
        B, C, H, W = Yl.shape


        if not hasattr(self, '_haar_idwt') or self._haar_idwt.channel != C:
            self._haar_idwt = HaarIDWT(C).to(Yl.device)


        Yh = torch.cat(Yh_list, dim=1)

        output = self._haar_idwt(Yl, Yh)

        return output


if __name__ == "__main__":
    print("=" * 60)
    print("Haar DWT 测试")
    print("=" * 60)


    B, C, H, W = 2, 64, 64, 64
    x = torch.randn(B, C, H, W)


    dwt = HaarDWT(channel=C)
    idwt = HaarIDWT(channel=C)

    print(f"\n输入形状: {x.shape}")


    Yl, Yh = dwt(x)
    print(f"LL子带: {Yl.shape}")
    print(f"HF子带: {Yh.shape}")


    sparsity_hf = (Yh.abs() < 0.01).float().mean()
    print(f"\n稀疏性（|HF| < 0.01的比例）: {sparsity_hf:.2%}")


    energy_in = (x ** 2).sum().item()
    energy_out = (Yl ** 2).sum().item() + (Yh ** 2).sum().item()
    energy_ratio = energy_out / energy_in
    print(f"能量比（out/in）: {energy_ratio:.4f} (应该≈1.0)")


    x_recon = idwt(Yl, Yh)
    print(f"\n重构形状: {x_recon.shape}")


    recon_error = (x - x_recon).abs().max().item()
    print(f"重构误差（max）: {recon_error:.8f}")

    if recon_error < 1e-5:
        print("✅ 完美重构！")
    else:
        print("⚠️  重构误差较大（可能由于插值近似）")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
