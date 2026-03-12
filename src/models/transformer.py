from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal time embedding."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B] or [B, 1], float in [0, 1] typically
        returns: [B, dim]
        """
        if t.dim() == 2 and t.shape[-1] == 1:
            t = t.squeeze(-1)
        assert t.dim() == 1, f"Expected t to have shape [B] or [B,1], got {t.shape}"

        device = t.device
        half_dim = self.dim // 2
        if half_dim == 0:
            return t[:, None]

        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


def find_multiple(n: int, k: int) -> int:
    return n if n % k == 0 else n + k - (n % k)


# -----------------------------------------------------------------------------
# Normalization / MLP
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


class FeedForward(nn.Module):
    """SwiGLU FFN."""
    def __init__(self, dim: int, multiple_of: int = 256, ffn_dim_multiplier: Optional[float] = None,
                 dropout_p: float = 0.0):
        super().__init__()
        hidden_dim = int(2 * (4 * dim) / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# -----------------------------------------------------------------------------
# Rotary embeddings (2D)
# -----------------------------------------------------------------------------

def precompute_freqs_cis_2d(grid_h: int, grid_w: int, head_dim: int, base: float = 10000.0) -> torch.Tensor:
    """
    Returns:
        freqs_cis: [grid_h * grid_w, head_dim // 2, 2]
    Notes:
        head_dim must be even. We split head_dim into half for y and half for x.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    half_dim = head_dim // 2
    assert half_dim % 2 == 0, "head_dim//2 must be even for 2D RoPE"

    device = torch.device("cpu")

    def _axis_freqs(n_pos: int, dim_axis: int) -> torch.Tensor:
        freqs = 1.0 / (base ** (torch.arange(0, dim_axis, 2, device=device).float() / dim_axis))
        t = torch.arange(n_pos, device=device)
        freqs = torch.outer(t, freqs)  # [n_pos, dim_axis//2]
        return freqs

    freqs_y = _axis_freqs(grid_h, half_dim)
    freqs_x = _axis_freqs(grid_w, half_dim)

    freqs_grid = torch.cat(
        [
            freqs_y[:, None, :].expand(-1, grid_w, -1),
            freqs_x[None, :, :].expand(grid_h, -1, -1),
        ],
        dim=-1,
    )  # [H, W, head_dim//2]

    cache = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1)
    cache = cache.view(grid_h * grid_w, head_dim // 2, 2)
    return cache


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    x: [B, N, n_head, head_dim]
    freqs_cis: [N, head_dim//2, 2]
    """
    B, N, H, Dh = x.shape
    x_ = x.float().reshape(B, N, H, Dh // 2, 2)
    freqs_cis = freqs_cis.view(1, N, 1, Dh // 2, 2)

    out = torch.stack(
        [
            x_[..., 0] * freqs_cis[..., 0] - x_[..., 1] * freqs_cis[..., 1],
            x_[..., 1] * freqs_cis[..., 0] + x_[..., 0] * freqs_cis[..., 1],
        ],
        dim=-1,
    )
    out = out.flatten(-2)
    return out.type_as(x)


# -----------------------------------------------------------------------------
# Attention / Transformer blocks
# -----------------------------------------------------------------------------

class BidirectionalAttention(nn.Module):
    def __init__(self, dim: int, n_head: int, attn_dropout_p: float = 0.0, resid_dropout_p: float = 0.0):
        super().__init__()
        assert dim % n_head == 0, "dim must be divisible by n_head"
        self.dim = dim
        self.n_head = n_head
        self.head_dim = dim // n_head

        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, dim]
        freqs_cis: [N, head_dim//2, 2]
        """
        B, N, _ = x.shape
        q, k, v = self.wqkv(x).chunk(3, dim=-1)

        q = q.view(B, N, self.n_head, self.head_dim)
        k = k.view(B, N, self.n_head, self.head_dim)
        v = v.view(B, N, self.n_head, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q = q.transpose(1, 2)  # [B, H, N, Dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Full bidirectional attention: no causal mask.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=False,
        )

        out = out.transpose(1, 2).contiguous().view(B, N, self.dim)
        out = self.resid_dropout(self.wo(out))
        return out


class TransformerBlock(nn.Module):
    """
    AdaLN-style block conditioned only on time.
    """
    def __init__(
        self,
        dim: int,
        n_head: int,
        norm_eps: float,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        attn_dropout_p: float,
        resid_dropout_p: float,
        ffn_dropout_p: float,
        drop_path_p: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

        self.attn = BidirectionalAttention(
            dim=dim,
            n_head=n_head,
            attn_dropout_p=attn_dropout_p,
            resid_dropout_p=resid_dropout_p,
        )
        self.ffn = FeedForward(
            dim=dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            dropout_p=ffn_dropout_p,
        )
        self.drop_path = DropPath(drop_path_p) if drop_path_p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, dim]
        cond: [B, 4 * dim]
        """
        shift_msa, scale_msa, shift_mlp, scale_mlp = cond.chunk(4, dim=-1)

        x_attn = self.attn_norm(x) * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        x = x + self.drop_path(self.attn(x_attn, freqs_cis))

        x_ffn = self.ffn_norm(x) * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        x = x + self.drop_path(self.ffn(x_ffn))
        return x


# -----------------------------------------------------------------------------
# CatFlow Transformer
# -----------------------------------------------------------------------------

@dataclass
class CatFlowTransformerConfig:
    # Data
    num_classes: int          # K
    seq_len: int              # D
    codebook_dim: int         # C_vae

    # Optional 2D layout for RoPE
    grid_h: int
    grid_w: int

    # Model
    dim: int = 512
    n_layer: int = 8
    n_head: int = 8
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    initializer_range: float = 0.02

    # Regularization
    input_dropout_p: float = 0.0
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.0
    ffn_dropout_p: float = 0.0
    drop_path_rate: float = 0.0

    # Time conditioning MLP width
    time_mlp_hidden_dim: Optional[int] = None


class CatFlowTransformer(nn.Module):
    """
    Bidirectional transformer encoder for CatFlow.

    Input:
        x_t: [B, D, K]  (simplex / one-hot / relaxed categorical states)
        t:   [B] or [B, 1]

    Output:
        logits: [B, D, K]

    Internally:
        x_t @ codebook -> [B, D, C_vae]
        then projected to model dim and processed by transformer blocks.
    """
    def __init__(self, config: CatFlowTransformerConfig, codebook: torch.Tensor):
        super().__init__()
        self.config = config

        assert config.grid_h * config.grid_w == config.seq_len, \
            f"Expected grid_h * grid_w == seq_len, got {config.grid_h} * {config.grid_w} != {config.seq_len}"

        if not isinstance(codebook, torch.Tensor):
            raise TypeError("codebook must be a torch.Tensor")
        if codebook.shape != (config.num_classes, config.codebook_dim):
            raise ValueError(
                f"Expected codebook shape {(config.num_classes, config.codebook_dim)}, got {tuple(codebook.shape)}"
            )

        # Store codebook as a buffer by default: the VQ-VAE codebook is usually fixed.
        # If you want to finetune it jointly, change this to nn.Parameter.
        self.register_buffer("codebook", codebook.clone())

        self.input_norm = RMSNorm(config.codebook_dim, eps=config.norm_eps)
        self.input_proj = nn.Linear(config.codebook_dim, config.dim, bias=False)
        self.input_dropout = nn.Dropout(config.input_dropout_p)

        self.time_emb = SinusoidalPosEmb(config.dim)
        time_hidden = config.time_mlp_hidden_dim or (4 * config.dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.dim, time_hidden),
            nn.SiLU(),
            nn.Linear(time_hidden, 4 * config.dim),
        )

        dpr = torch.linspace(0, config.drop_path_rate, config.n_layer).tolist()
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=config.dim,
                n_head=config.n_head,
                norm_eps=config.norm_eps,
                multiple_of=config.multiple_of,
                ffn_dim_multiplier=config.ffn_dim_multiplier,
                attn_dropout_p=config.attn_dropout_p,
                resid_dropout_p=config.resid_dropout_p,
                ffn_dropout_p=config.ffn_dropout_p,
                drop_path_p=dpr[i],
            )
            for i in range(config.n_layer)
        ])

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.num_classes, bias=False)

        head_dim = config.dim // config.n_head
        assert config.dim % config.n_head == 0, "dim must be divisible by n_head"
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        freqs_cis = precompute_freqs_cis_2d(
            grid_h=config.grid_h,
            grid_w=config.grid_w,
            head_dim=head_dim,
            base=config.rope_base,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def embed_with_codebook(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        x_t: [B, D, K]
        returns: [B, D, C_vae]
        """
        return x_t @ self.codebook

    def forward(self, t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """
        t:   [B] or [B, 1]
        x_t: [B, D, K]
        returns logits: [B, D, K]
        """
        if x_t.dim() != 3:
            raise ValueError(f"Expected x_t shape [B, D, K], got {tuple(x_t.shape)}")

        B, D, K = x_t.shape
        if D != self.config.seq_len:
            raise ValueError(f"Expected D={self.config.seq_len}, got D={D}")
        if K != self.config.num_classes:
            raise ValueError(f"Expected K={self.config.num_classes}, got K={K}")

        # [B, D, K] -> [B, D, C_vae]
        z = self.embed_with_codebook(x_t)

        # [B, D, C_vae] -> [B, D, dim]
        h = self.input_norm(z)
        h = self.input_proj(h)
        h = self.input_dropout(h)

        # time conditioning -> [B, 4 * dim]
        t_emb = self.time_emb(t)
        cond = self.time_mlp(t_emb)

        freqs_cis = self.freqs_cis[:D].to(device=h.device, dtype=h.dtype)

        for layer in self.layers:
            h = layer(h, freqs_cis=freqs_cis, cond=cond)

        h = self.norm(h)
        logits = self.output(h).float()  # [B, D, K]
        return logits