from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image

from src.models.transformer import CatFlowTransformer, CatFlowTransformerConfig
from src.models.vfm_wrapper import CatFlow, CodebookCatFlow
from src.models.vq_model import VQ_Cifar_L


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # weights_only=False is required because the checkpoint stores a dataclass config object.
    return torch.load(path, map_location=device, weights_only=False)


def _to_transformer_config(raw_cfg: Any) -> CatFlowTransformerConfig:
    if isinstance(raw_cfg, CatFlowTransformerConfig):
        return raw_cfg
    if is_dataclass(raw_cfg):
        return CatFlowTransformerConfig(**asdict(raw_cfg))
    if isinstance(raw_cfg, dict):
        return CatFlowTransformerConfig(**raw_cfg)
    raise TypeError(f"Unsupported transformer config type: {type(raw_cfg)}")


def load_vqvae_from_checkpoint(vq_checkpoint_path: Path, device: torch.device) -> nn.Module:
    ckpt = load_checkpoint(vq_checkpoint_path, device=device)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"Missing `model_state_dict` in VQ-VAE checkpoint: {vq_checkpoint_path}")
    model = VQ_Cifar_L().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_catflow_from_checkpoint(
    flow_checkpoint_path: Path,
    vq_model: nn.Module,
    device: torch.device,
    sigma_min: float = 1e-6,
) -> tuple[CatFlow, CatFlowTransformerConfig, tuple[int, int]]:
    ckpt = load_checkpoint(flow_checkpoint_path, device=device)
    if "model_config" not in ckpt or "model_state_dict" not in ckpt:
        raise KeyError(
            f"Missing `model_config` or `model_state_dict` in CatFlow checkpoint: {flow_checkpoint_path}"
        )

    model_config = ckpt["model_config"]
    raw_cfg = model_config.get("model_cfg")
    if raw_cfg is None:
        raise KeyError("Missing `model_cfg` in checkpoint['model_config'].")

    transformer_cfg = _to_transformer_config(raw_cfg)
    obs_dim = tuple(model_config.get("obs_dim", (transformer_cfg.seq_len, transformer_cfg.num_classes)))
    if len(obs_dim) != 2:
        raise ValueError(f"`obs_dim` must have 2 entries (D, K), got {obs_dim}")

    codebook = vq_model.quantize.embedding.weight.detach()
    expected_shape = (transformer_cfg.num_classes, transformer_cfg.codebook_dim)
    if tuple(codebook.shape) != expected_shape:
        raise ValueError(
            "Codebook shape mismatch between VQ-VAE and transformer config: "
            f"expected {expected_shape}, got {tuple(codebook.shape)}"
        )

    model = CatFlowTransformer(transformer_cfg, codebook).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    flow = CatFlow(
        model=model,
        obs_dim=obs_dim,
        sigma_min=sigma_min,
    ).to(device)
    if "flow_state_dict" in ckpt:
        # Includes the same model weights with `model.` prefix, plus wrapper-owned state if any.
        flow.load_state_dict(ckpt["flow_state_dict"], strict=False)
    flow.clamp_t = 0.005
    flow.eval()

    return flow, transformer_cfg, obs_dim

def load_catflow_checkpoint(
    flow_checkpoint_path: Path,
    device: torch.device,
) -> tuple[CatFlow, CatFlowTransformerConfig, tuple[int, int]]:
    ckpt = load_checkpoint(flow_checkpoint_path, device=device)
    if "model_config" not in ckpt or "model_state_dict" not in ckpt:
        raise KeyError(
            f"Missing `model_config` or `model_state_dict` in CatFlow checkpoint: {flow_checkpoint_path}"
        )

    model_config = ckpt["model_config"]
    raw_cfg = model_config.get("model_cfg")
    if raw_cfg is None:
        raise KeyError("Missing `model_cfg` in checkpoint['model_config'].")

    transformer_cfg = _to_transformer_config(raw_cfg)
    obs_dim = tuple(model_config.get("obs_dim", (transformer_cfg.seq_len, transformer_cfg.num_classes)))
    if len(obs_dim) != 2:
        raise ValueError(f"`obs_dim` must have 2 entries (D, K), got {obs_dim}")

    codebook = None

    model = CatFlowTransformer(transformer_cfg, codebook).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    flow = CatFlow(
        model=model,
        obs_dim=obs_dim,
    ).to(device)
    if "flow_state_dict" in ckpt:
        # Includes the same model weights with `model.` prefix, plus wrapper-owned state if any.
        flow.load_state_dict(ckpt["flow_state_dict"], strict=False)
    flow.clamp_t = 0.005
    flow.eval()

    return flow, transformer_cfg, obs_dim

def load_codebook_catflow_from_checkpoint(
    flow_checkpoint_path: Path,
    vq_model: nn.Module,
    device: torch.device,
    temperature: float = 0.8,
) -> tuple[CodebookCatFlow, CatFlowTransformerConfig, tuple[int, int]]:
    ckpt = load_checkpoint(flow_checkpoint_path, device=device)
    if "model_config" not in ckpt or "model_state_dict" not in ckpt:
        raise KeyError(
            f"Missing `model_config` or `model_state_dict` in CatFlow checkpoint: {flow_checkpoint_path}"
        )

    model_config = ckpt["model_config"]
    raw_cfg = model_config.get("model_cfg")
    if raw_cfg is None:
        raise KeyError("Missing `model_cfg` in checkpoint['model_config'].")

    transformer_cfg = _to_transformer_config(raw_cfg)
    obs_dim = tuple(model_config.get("obs_dim", (transformer_cfg.seq_len, transformer_cfg.num_classes)))
    if len(obs_dim) != 2:
        raise ValueError(f"`obs_dim` must have 2 entries (D, K), got {obs_dim}")

    codebook = vq_model.quantize.embedding.weight.detach()
    expected_shape = (transformer_cfg.num_classes, transformer_cfg.codebook_dim)
    if tuple(codebook.shape) != expected_shape:
        raise ValueError(
            "Codebook shape mismatch between VQ-VAE and transformer config: "
            f"expected {expected_shape}, got {tuple(codebook.shape)}"
        )

    model = CatFlowTransformer(transformer_cfg, codebook).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    flow = CodebookCatFlow(
        model=model,
        vq_model=vq_model,
        obs_dim=obs_dim,
        temperature=temperature
    ).to(device)
    flow.eval()
    print(f"Flow observed dim: {flow.obs_dim}")
    return flow, transformer_cfg, obs_dim


@torch.no_grad()
def sample_probability_sequences(
    flow: CatFlow,
    n_samples: int,
    method: str = "euler",
    n_steps: int = 10,
) -> torch.Tensor:
    probs = flow.sample(n_samples=n_samples, method=method, n_steps=n_steps)  # (B, S, K)
    return probs.detach()


def argmax_readout(prob_sequences: torch.Tensor) -> torch.Tensor:
    return torch.argmax(prob_sequences, dim=-1).to(dtype=torch.long)


@torch.no_grad()
def decode_indices_with_vqvae(
    vq_model: nn.Module,
    indices: torch.Tensor,
    codebook_dim: int,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    if indices.dim() != 2:
        raise ValueError(f"Expected indices with shape [B, D], got {tuple(indices.shape)}")
    b, d = indices.shape
    if d != grid_h * grid_w:
        raise ValueError(f"Expected D={grid_h * grid_w}, got D={d}")

    flat_codes = indices.reshape(-1)
    quant_shape = (b, codebook_dim, grid_h, grid_w)
    recon = vq_model.decode_code(flat_codes, shape=quant_shape, channel_first=True)
    return recon.detach()

@torch.no_grad()
def generate_in_codebook_space(
    flow: CatFlow,
    vq_model: nn.Module,
    n_samples: int,
    method: str = "euler",
) -> torch.Tensor:
    x_final = flow.sample(n_samples, method=method)  # (B, S, K)
    print(f"Sampled logits shape: {tuple(x_final.shape)}")

    codebook = vq_model.quantize.embedding.weight    # (K, D)
    print(f"Codebook shape: {tuple(codebook.shape)}")

    B, S, K = x_final.shape
    K_cb, D = codebook.shape
    assert K == K_cb, f"Logit dim {K} must match number of codebook entries {K_cb}"

    # 1) logits -> probabilities
    probs = torch.softmax(x_final, dim=-1)           # (B, S, K)
    print(f"Probabilities shape: {tuple(probs.shape)}")

    # 2) expected vector in codebook space
    mean_codes = torch.matmul(probs, codebook)       # (B, S, D)
    print(f"Mean code vectors shape: {tuple(mean_codes.shape)}")

    # 3) nearest codebook vector
    mean_codes_flat = mean_codes.reshape(-1, D)      # (B*S, D)
    print(f"Flattened mean codes shape: {tuple(mean_codes_flat.shape)}")

    d = (
        torch.sum(mean_codes_flat ** 2, dim=1, keepdim=True)
        + torch.sum(codebook ** 2, dim=1)
        - 2 * torch.matmul(mean_codes_flat, codebook.t())
    )                                                # (B*S, K)
    print(f"Pairwise distances shape: {tuple(d.shape)}")

    indices = torch.argmin(d, dim=1)                 # (B*S,)
    indices = indices.reshape(B, S)                  # (B, S)
    print(f"Indices shape: {tuple(indices.shape)}")
    print(f"Indices sample:\n{indices[:, :10]}")
    print(f"Indices from probabilities sample:\n{torch.argmax(probs, dim=-1)[:, :10]}")

    # 4) reshape sequence into latent grid
    obs_dim = flow.obs_dim
    block_size = obs_dim[0]
    h = int(block_size ** 0.5)
    w = h
    assert h * w == S, f"Sequence length {S} does not match grid size {h}x{w}"

    indices = indices.reshape(B, h, w)               # (B, H, W)
    print(f"Grid indices shape: {tuple(indices.shape)}")

    decoded_img = vq_model.decode_code(indices, shape=(B, -1, h, w))
    return decoded_img


def print_probability_sequences(
    prob_sequences: torch.Tensor,
    max_tokens: int = 8,
    max_classes: int = 16,
) -> None:
    print(f"[sample] Probability sequences shape: {tuple(prob_sequences.shape)}")
    if prob_sequences.numel() <= 4096:
        print(prob_sequences)
        return
    sample = prob_sequences[:, :max_tokens, :max_classes].detach().cpu()
    print(
        "[sample] Tensor is large; printing truncated view "
        f"[:, :{max_tokens}, :{max_classes}]"
    )
    print(sample)


def plot_first_positions_distributions(
    prob_sequences: torch.Tensor,
    n_positions: int = 3,
    show: bool = True,
) -> None:
    if prob_sequences.dim() != 3:
        raise ValueError(
            f"Expected probability sequences with shape [B, D, K], got {tuple(prob_sequences.shape)}"
        )
    if n_positions <= 0:
        raise ValueError("`n_positions` must be > 0")
    if not show:
        return

    import matplotlib.pyplot as plt

    probs = prob_sequences.detach().cpu().float()
    bsz, seq_len, num_classes = probs.shape
    n_plot = min(n_positions, seq_len)

    for sample_idx in range(bsz):
        fig, axes = plt.subplots(1, n_plot, figsize=(5 * n_plot, 3), squeeze=False)
        sample = probs[sample_idx]

        for pos in range(n_plot):
            p = sample[pos].clamp_min(0.0)
            denom = p.sum()
            if denom <= 0:
                p = torch.full_like(p, 1.0 / num_classes)
            else:
                p = p / denom

            ax = axes[0, pos]
            x = np.arange(num_classes)
            ax.bar(x, p.numpy(), width=1.0, color="#4C78A8", edgecolor="none")
            ax.set_title(f"Sample {sample_idx} - position {pos}")
            ax.set_xlabel("Category")
            ax.set_ylabel("Probability")
            ax.set_xlim(0, num_classes - 1)
            ax.set_ylim(0.0, float(p.max().item()) * 1.1 + 1e-8)

        fig.tight_layout()
        if show:
            plt.show()
        plt.close(fig)


def print_index_sequences(indices: torch.Tensor, max_tokens: int = 32) -> None:
    print(f"[readout] Index sequences shape: {tuple(indices.shape)}")
    if indices.numel() <= 2048:
        print(indices.detach().cpu())
        return
    sample = indices[:, :max_tokens].detach().cpu()
    print(f"[readout] Tensor is large; printing truncated view [:, :{max_tokens}]")
    print(sample)


def _to_display_range(images: torch.Tensor) -> torch.Tensor:
    return ((images + 1.0) * 0.5).clamp(0.0, 1.0)


def save_image_grid(
    images: torch.Tensor,
    output_path: Path,
    nrow: int,
    show: bool = True,
    title: str | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display_images = _to_display_range(images).detach().cpu()
    grid = make_grid(display_images, nrow=nrow, padding=2)
    save_image(grid, str(output_path))

    if show:
        import matplotlib.pyplot as plt

        grid_np = grid.permute(1, 2, 0).numpy()
        plt.figure(figsize=(8, 8))
        plt.imshow(grid_np)
        if title:
            plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()
    return output_path
