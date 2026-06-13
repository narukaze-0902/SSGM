import math
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def semantic_neighbor(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x


class SelectiveScan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank


        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj


        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs


        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device)
        A = A[None, :].repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = A_log[None, :, :].repeat(copies, 1, 1)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = D[None, :].repeat(copies, 1)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, prompt, state_write_gate=None, state_read_gate=None, state_delta_gate=None):
        B, L, C = x.shape
        K = 1

        xs = x.permute(0, 2, 1).view(B, 1, C, L).contiguous()

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)


        Cs = Cs.float().view(B, K, -1, L) + prompt
        if state_write_gate is not None:
            state_write_gate = state_write_gate.to(device=Bs.device, dtype=Bs.dtype).view(B, 1, 1, L)
            Bs = Bs * state_write_gate
        if state_read_gate is not None:
            state_read_gate = state_read_gate.to(device=Cs.device, dtype=Cs.dtype).view(B, 1, 1, L)
            Cs = Cs * state_read_gate
        if state_delta_gate is not None:
            state_delta_gate = state_delta_gate.to(device=dts.device, dtype=dts.dtype).view(B, 1, L)
            dts = dts * state_delta_gate

        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, prompt, state_write_gate=None, state_read_gate=None, state_delta_gate=None, **kwargs):
        b, l, c = prompt.shape
        prompt = prompt.permute(0, 2, 1).contiguous().view(b, 1, c, l)
        y = self.forward_core(
            x,
            prompt,
            state_write_gate=state_write_gate,
            state_read_gate=state_read_gate,
            state_delta_gate=state_delta_gate,
        )
        y = y.permute(0, 2, 1).contiguous()
        return y


class ASEModule(nn.Module):
    def __init__(self, dim, d_state, input_resolution, num_tokens=64, inner_rank=128, mlp_ratio=2.,
                 prompt_mode='hard', route_temperature=1.0, prompt_soft_mix=0.5):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank
        self.prompt_mode = prompt_mode
        self.route_temperature = route_temperature
        self.prompt_soft_mix = prompt_soft_mix


        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state


        self.selectiveScan = SelectiveScan(d_model=hidden, d_state=self.d_state, expand=1)


        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)


        self.in_proj = nn.Sequential(
            nn.Conv2d(self.dim, hidden, 1, 1, 0),
        )


        self.CPE = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
        )


        self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)
        self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)


        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim // 3),
            nn.GELU(),
            nn.Linear(self.dim // 3, self.num_tokens),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, x_size, token, prompt_bank_bias=None, route_logit_bias=None,
                token_prompt_residual=None, semantic_scan_indices=None,
                semantic_scan_reverse_indices=None, semantic_region_ids=None,
                semantic_boundary_scale=0.0, semantic_state_reset_scale=0.0,
                semantic_boundary_gate=None, semantic_state_reset_gate=None,
                semantic_state_write_gate=None, semantic_state_read_gate=None,
                semantic_state_delta_gate=None):
        B, n, C = x.shape
        H, W = x_size

        if semantic_scan_indices is not None:
            if semantic_scan_indices.dim() != 2 or semantic_scan_indices.shape != (B, n):
                raise ValueError(
                    f"semantic_scan_indices must be [B, L], got {tuple(semantic_scan_indices.shape)}"
                )
            x = semantic_neighbor(x, semantic_scan_indices)

        if token_prompt_residual is not None:
            if token_prompt_residual.dim() != 3 or token_prompt_residual.shape != (B, n, self.d_state):
                raise ValueError(
                    f"token_prompt_residual must be [B, L, d_state], got {tuple(token_prompt_residual.shape)}"
                )
            if semantic_scan_indices is not None:
                token_prompt_residual = semantic_neighbor(token_prompt_residual, semantic_scan_indices)

        if semantic_region_ids is not None:
            if semantic_region_ids.dim() != 2 or semantic_region_ids.shape != (B, n):
                raise ValueError(
                    f"semantic_region_ids must be [B, L], got {tuple(semantic_region_ids.shape)}"
                )
            if semantic_scan_indices is not None:
                semantic_region_ids = semantic_neighbor(semantic_region_ids, semantic_scan_indices)

        if semantic_boundary_gate is not None:
            if semantic_boundary_gate.dim() != 2 or semantic_boundary_gate.shape != (B, n):
                raise ValueError(
                    f"semantic_boundary_gate must be [B, L], got {tuple(semantic_boundary_gate.shape)}"
                )
            if semantic_scan_indices is not None:
                semantic_boundary_gate = semantic_neighbor(semantic_boundary_gate, semantic_scan_indices)

        if semantic_state_reset_gate is not None:
            if semantic_state_reset_gate.dim() != 2 or semantic_state_reset_gate.shape != (B, n):
                raise ValueError(
                    f"semantic_state_reset_gate must be [B, L], got {tuple(semantic_state_reset_gate.shape)}"
                )
            if semantic_scan_indices is not None:
                semantic_state_reset_gate = semantic_neighbor(semantic_state_reset_gate, semantic_scan_indices)
        if semantic_state_write_gate is not None:
            if semantic_state_write_gate.dim() != 2 or semantic_state_write_gate.shape != (B, n):
                raise ValueError(
                    f"semantic_state_write_gate must be [B, L], got {tuple(semantic_state_write_gate.shape)}"
                )
            if semantic_scan_indices is not None:
                semantic_state_write_gate = semantic_neighbor(semantic_state_write_gate, semantic_scan_indices)
        if semantic_state_read_gate is not None:
            if semantic_state_read_gate.dim() != 2 or semantic_state_read_gate.shape != (B, n):
                raise ValueError(
                    f"semantic_state_read_gate must be [B, L], got {tuple(semantic_state_read_gate.shape)}"
                )
            if semantic_scan_indices is not None:
                semantic_state_read_gate = semantic_neighbor(semantic_state_read_gate, semantic_scan_indices)
        if semantic_state_delta_gate is not None:
            if semantic_state_delta_gate.dim() != 2 or semantic_state_delta_gate.shape != (B, n):
                raise ValueError(
                    f"semantic_state_delta_gate must be [B, L], got {tuple(semantic_state_delta_gate.shape)}"
                )
            if semantic_scan_indices is not None:
                semantic_state_delta_gate = semantic_neighbor(semantic_state_delta_gate, semantic_scan_indices)


        full_embedding = self.embeddingB.weight @ token.weight


        route_log_prob = self.route(x)
        if route_logit_bias is not None:
            if route_logit_bias.dim() != 2 or route_logit_bias.shape[0] != B or route_logit_bias.shape[1] != self.num_tokens:
                raise ValueError(
                    f"route_logit_bias must be [B, num_tokens], got {tuple(route_logit_bias.shape)}"
                )
            route_log_prob = F.log_softmax(
                route_log_prob + route_logit_bias.unsqueeze(1).to(route_log_prob.dtype),
                dim=-1,
            )
        route_prob = route_log_prob.exp()
        soft_route_prob = F.softmax(route_log_prob / max(self.route_temperature, 1e-6), dim=-1)
        if self.training:
            hard_route_policy = F.gumbel_softmax(
                route_log_prob,
                tau=self.route_temperature,
                hard=True,
                dim=-1,
            )
        else:
            route_index = torch.argmax(route_log_prob, dim=-1)
            hard_route_policy = F.one_hot(route_index, num_classes=self.num_tokens).to(route_log_prob.dtype)


        if self.prompt_mode == 'hard':
            prompt_policy = hard_route_policy
        elif self.prompt_mode == 'soft':
            prompt_policy = soft_route_prob
        elif self.prompt_mode == 'hybrid':
            soft_mix = float(self.prompt_soft_mix)
            prompt_policy = soft_mix * soft_route_prob + (1.0 - soft_mix) * hard_route_policy
        else:
            raise ValueError(f"Unsupported ASE prompt_mode: {self.prompt_mode}")
        if prompt_bank_bias is not None:
            if prompt_bank_bias.dim() != 3 or prompt_bank_bias.shape[0] != B:
                raise ValueError(
                    f"prompt_bank_bias must be [B, num_tokens, d_state], got {tuple(prompt_bank_bias.shape)}"
                )
            if prompt_bank_bias.shape[1] != self.num_tokens or prompt_bank_bias.shape[2] != self.d_state:
                raise ValueError(
                    f"prompt_bank_bias shape mismatch: expected [B, {self.num_tokens}, {self.d_state}], got {tuple(prompt_bank_bias.shape)}"
                )
            refined_embedding = full_embedding.unsqueeze(0).to(prompt_bank_bias.dtype) + prompt_bank_bias
            prompt = torch.matmul(prompt_policy, refined_embedding).view(B, n, self.d_state)
        else:
            prompt = torch.matmul(prompt_policy, full_embedding).view(B, n, self.d_state)
        if token_prompt_residual is not None:
            prompt = prompt + token_prompt_residual.to(prompt.dtype)


        detached_index = torch.argmax(hard_route_policy.detach(), dim=-1, keepdim=False).view(B, n)

        x_sort_values, x_sort_indices = torch.sort(detached_index, dim=-1, stable=True)
        x_sort_indices_reverse = index_reverse(x_sort_indices)


        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.in_proj(x)
        x = x * torch.sigmoid(self.CPE(x))
        cc = x.shape[1]
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)


        semantic_x = semantic_neighbor(x, x_sort_indices)

        semantic_prompt = semantic_neighbor(prompt, x_sort_indices)
        if semantic_state_write_gate is not None:
            semantic_state_write_gate = semantic_neighbor(semantic_state_write_gate, x_sort_indices)
        if semantic_state_read_gate is not None:
            semantic_state_read_gate = semantic_neighbor(semantic_state_read_gate, x_sort_indices)
        if semantic_state_delta_gate is not None:
            semantic_state_delta_gate = semantic_neighbor(semantic_state_delta_gate, x_sort_indices)

        if semantic_region_ids is not None and (
            semantic_boundary_scale > 0
            or semantic_state_reset_scale > 0
            or semantic_boundary_gate is not None
            or semantic_state_reset_gate is not None
        ):
            semantic_region_ids = semantic_neighbor(semantic_region_ids, x_sort_indices)
            if semantic_boundary_gate is not None:
                semantic_boundary_gate = semantic_neighbor(semantic_boundary_gate, x_sort_indices)
            if semantic_state_reset_gate is not None:
                semantic_state_reset_gate = semantic_neighbor(semantic_state_reset_gate, x_sort_indices)
            boundary_transitions = torch.zeros_like(semantic_region_ids, dtype=semantic_x.dtype)
            prev_regions = semantic_region_ids[:, :-1]
            curr_regions = semantic_region_ids[:, 1:]
            cross_region = (curr_regions != prev_regions) & ((curr_regions > 0) | (prev_regions > 0))
            boundary_transitions[:, 1:] = cross_region.to(dtype=semantic_x.dtype)
            if semantic_boundary_scale > 0 or semantic_boundary_gate is not None:
                if semantic_boundary_gate is not None:
                    boundary_strength = semantic_boundary_gate.to(semantic_x.dtype).clamp(min=0.0, max=1.0)
                else:
                    boundary_strength = semantic_boundary_scale * torch.ones_like(boundary_transitions, dtype=semantic_x.dtype)
                boundary_gate = (1.0 - boundary_strength * boundary_transitions).clamp(min=0.0).unsqueeze(-1)
                semantic_x = semantic_x * boundary_gate
                semantic_prompt = semantic_prompt * boundary_gate
            if semantic_state_reset_scale > 0 or semantic_state_reset_gate is not None:
                if semantic_state_reset_gate is not None:
                    reset_keep = (1.0 - semantic_state_reset_gate.to(semantic_x.dtype).clamp(min=0.0, max=1.0)).unsqueeze(-1)
                else:
                    reset_keep = torch.full_like(boundary_transitions, max(1.0 - float(semantic_state_reset_scale), 0.0), dtype=semantic_x.dtype).unsqueeze(-1)
                reset_gate = torch.where(
                    boundary_transitions.unsqueeze(-1) > 0,
                    reset_keep,
                    torch.ones_like(reset_keep),
                )
                semantic_x = semantic_x * reset_gate
                semantic_prompt = semantic_prompt * reset_gate


        y = self.selectiveScan(
            semantic_x,
            semantic_prompt,
            state_write_gate=semantic_state_write_gate,
            state_read_gate=semantic_state_read_gate,
            state_delta_gate=semantic_state_delta_gate,
        )
        y = self.out_proj(self.out_norm(y))


        x = semantic_neighbor(y, x_sort_indices_reverse)

        if semantic_scan_reverse_indices is not None:
            if semantic_scan_reverse_indices.dim() != 2 or semantic_scan_reverse_indices.shape != (B, n):
                raise ValueError(
                    f"semantic_scan_reverse_indices must be [B, L], got {tuple(semantic_scan_reverse_indices.shape)}"
                )
            x = semantic_neighbor(x, semantic_scan_reverse_indices)
            route_prob = semantic_neighbor(route_prob, semantic_scan_reverse_indices)

        return x, route_prob


class ASEMambaBlock(nn.Module):
    def __init__(self, d_model, num_prompts=32, rank=8, modal_type='single', d_state=8, use_wavelet=False,
                 shared_embeddingA=None, prompt_mode='hard', route_temperature=1.0, prompt_soft_mix=0.5):
        super().__init__()
        self.d_model = d_model
        self.modal_type = modal_type
        self.use_wavelet = use_wavelet
        self.num_prompts = num_prompts
        self.prompt_d_state = d_state


        self.ase_module = AttentiveStateSpace(
            dim=d_model,
            d_state=d_state,
            input_resolution=(32, 32),
            num_tokens=num_prompts,
            inner_rank=rank,
            shared_embeddingA=shared_embeddingA,
            prompt_mode=prompt_mode,
            route_temperature=route_temperature,
            prompt_soft_mix=prompt_soft_mix,
        )


        self.out_proj = nn.Linear(d_model, d_model)


        if self.use_wavelet:
            self.wavelet_conv = nn.Sequential(
                nn.Conv2d(d_model, d_model // 2, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(d_model // 2, d_model, kernel_size=1)
            )

    def forward(self, x, extra_emb=None, semantic_prompt_bank_bias=None, semantic_route_logit_bias=None,
                semantic_token_prompt_residual=None, semantic_scan_indices=None,
                semantic_scan_reverse_indices=None, semantic_region_ids=None,
                semantic_boundary_scale=0.0, semantic_state_reset_scale=0.0,
                semantic_boundary_gate=None, semantic_state_reset_gate=None,
                semantic_state_write_gate=None, semantic_state_read_gate=None,
                semantic_state_delta_gate=None):
        B, L, C = x.shape


        H = W = int(round(math.sqrt(L)))


        if H * W != L:

            found = False
            for h in range(int(math.sqrt(L)), 0, -1):
                if L % h == 0:
                    H = h
                    W = L // h
                    found = True
                    break


            if not found:
                H = int(math.sqrt(L))
                W = L // H

                if H * W < L:
                    W += 1

        x_size = (H, W)


        output, routing_probs = self.ase_module(
            x,
            x_size,
            semantic_prompt_bank_bias=semantic_prompt_bank_bias,
            semantic_route_logit_bias=semantic_route_logit_bias,
            semantic_token_prompt_residual=semantic_token_prompt_residual,
            semantic_scan_indices=semantic_scan_indices,
            semantic_scan_reverse_indices=semantic_scan_reverse_indices,
            semantic_region_ids=semantic_region_ids,
            semantic_boundary_scale=semantic_boundary_scale,
            semantic_state_reset_scale=semantic_state_reset_scale,
            semantic_boundary_gate=semantic_boundary_gate,
            semantic_state_reset_gate=semantic_state_reset_gate,
            semantic_state_write_gate=semantic_state_write_gate,
            semantic_state_read_gate=semantic_state_read_gate,
            semantic_state_delta_gate=semantic_state_delta_gate,
        )


        if extra_emb is not None:

            output = output + extra_emb


        output = self.out_proj(output)


        if self.use_wavelet:

            output_2d = output.permute(0, 2, 1).reshape(B, C, H, W)


            def apply_2d_wavelet(img):
                B, C, H, W = img.shape
                reconstructed_channels = []

                for b in range(B):
                    batch_reconstructed = []
                    for c in range(C):

                        channel = img[b, c, :, :].cpu().detach().numpy()


                        coeffs = pywt.wavedec2(channel, 'sym3', level=2, mode='symmetric')


                        reconstructed = pywt.waverec2(coeffs, 'sym3', mode='symmetric')


                        reconstructed = reconstructed[:H, :W] if reconstructed.shape != (H, W) else reconstructed


                        reconstructed = torch.from_numpy(reconstructed).to(img.device)
                        batch_reconstructed.append(reconstructed)


                    batch_reconstructed = torch.stack(batch_reconstructed, dim=0)
                    reconstructed_channels.append(batch_reconstructed)


                reconstructed = torch.stack(reconstructed_channels, dim=0)
                return reconstructed


            wavelet_reconstructed = apply_2d_wavelet(output_2d)


            wavelet_out = self.wavelet_conv(wavelet_reconstructed)


            wavelet_out = wavelet_out.reshape(B, C, L).permute(0, 2, 1)


            output = output + wavelet_out

        return output, routing_probs


class AttentiveStateSpace(nn.Module):
    def __init__(self, dim, d_state, input_resolution, num_tokens=64, inner_rank=128, mlp_ratio=2.,
                 shared_embeddingA=None, prompt_mode='hard', route_temperature=1.0, prompt_soft_mix=0.5):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank


        self.ase = ASEModule(
            dim,
            d_state,
            input_resolution,
            num_tokens,
            inner_rank,
            mlp_ratio,
            prompt_mode=prompt_mode,
            route_temperature=route_temperature,
            prompt_soft_mix=prompt_soft_mix,
        )


        if shared_embeddingA is not None:
            self.embeddingA = shared_embeddingA
        else:
            self.embeddingA = nn.Embedding(self.inner_rank, d_state)
            self.embeddingA.weight.data.uniform_(-1 / self.inner_rank, 1 / self.inner_rank)


        self.norm = nn.LayerNorm(dim)

    def forward(self, x, x_size, semantic_prompt_bank_bias=None, semantic_route_logit_bias=None,
                semantic_token_prompt_residual=None, semantic_scan_indices=None,
                semantic_scan_reverse_indices=None, semantic_region_ids=None,
                semantic_boundary_scale=0.0, semantic_state_reset_scale=0.0,
                semantic_boundary_gate=None, semantic_state_reset_gate=None,
                semantic_state_write_gate=None, semantic_state_read_gate=None,
                semantic_state_delta_gate=None):

        x_norm = self.norm(x)


        x_ase, routing_probs = self.ase(
            x_norm,
            x_size,
            self.embeddingA,
            prompt_bank_bias=semantic_prompt_bank_bias,
            route_logit_bias=semantic_route_logit_bias,
            token_prompt_residual=semantic_token_prompt_residual,
            semantic_scan_indices=semantic_scan_indices,
            semantic_scan_reverse_indices=semantic_scan_reverse_indices,
            semantic_region_ids=semantic_region_ids,
            semantic_boundary_scale=semantic_boundary_scale,
            semantic_state_reset_scale=semantic_state_reset_scale,
            semantic_boundary_gate=semantic_boundary_gate,
            semantic_state_reset_gate=semantic_state_reset_gate,
            semantic_state_write_gate=semantic_state_write_gate,
            semantic_state_read_gate=semantic_state_read_gate,
            semantic_state_delta_gate=semantic_state_delta_gate,
        )


        return x + x_ase, routing_probs
