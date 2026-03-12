from __future__ import annotations

import argparse
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.fid import compute_fid
from src.evaluation.precision_recall import compute_precision_recall, precision_recall_knn_blockwise
from src.models.vfm_wrapper import CatFlow
from src.models.vq_model import VQ_Cifar_L
from src.models.transformer import CatFlowTransformer, CatFlowTransformerConfig


@dataclass
class TrainConfig:
    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"
    q_train_dataset_name: str = "cifar10_q_train.pt"
    q_test_dataset_name: str = "cifar10_q_test.pt"
    cifar10_dino_features: str = "data/processed/cifar10_dinov2.pt"
    checkpoints_dir: str = "checkpoints"
    vq_checkpoint_path: str = "checkpoints/vq_cifar_epoch_20.pt"

    batch_size: int = 128
    num_workers: int = 4
    epochs: int = 300
    lr: float = 1e-4
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    ckpt_every: int = 5000
    eval_every: int = 1000
    eval_num_samples: int = 5000
    eval_val_batches: int = 8
    eval_nll_samples: int = 256
    eval_nll_hutchinson: int = 4
    seed: int = 42

    num_channels: int = 128
    proj_channels: int = 64
    catflow_sigma_min: float = 1e-6
    tqdm_disable: bool = True

    wandb_project: str = "closedform-catflow"
    wandb_run_name: str | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_cifar10(batch_size: int, data_root: str, train: bool = True, num_workers: int = 4) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.CIFAR10(root=data_root, train=train, download=True, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _extract_indices_from_vq_info(info: tuple[Any, Any, torch.Tensor], batch_size: int, h: int, w: int) -> torch.Tensor:
    min_encoding_indices = info[2]
    if min_encoding_indices is None:
        raise RuntimeError("VQ encoder did not return codebook indices.")
    return min_encoding_indices.view(batch_size, h * w)


def maybe_prepare_quantized_dataset(train:bool, config: TrainConfig, device: torch.device, vq_model: nn.Module) -> dict[str, Any]:
    processed_dir = ROOT / config.data_processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    q_path = processed_dir / (config.q_train_dataset_name if train else config.q_test_dataset_name)

    if q_path.exists():
        print(f"[data] Using cached quantized dataset: {q_path}")
        return torch.load(q_path, map_location="cpu")

    print("[data] Quantized dataset not found. Loading CIFAR-10 and quantizing...")
    loader = load_cifar10(config.batch_size, str(ROOT / config.data_raw_dir), train=train, num_workers=config.num_workers)

    all_indices: list[torch.Tensor] = []
    latent_h: int | None = None
    latent_w: int | None = None

    vq_model.eval()

    pbar = tqdm(loader, desc="Quantizing CIFAR-10", unit="batch", disable=config.tqdm_disable)
    print_perc = 0.0
    n_batch = 0
    with torch.inference_mode():
        for images, _ in pbar:
            images = images.to(device)
            quant, _, info = vq_model.encode(images)
            b, _, h, w = quant.shape
            if latent_h is None:
                latent_h, latent_w = h, w
            batch_indices = _extract_indices_from_vq_info(info, b, h, w)
            all_indices.append(batch_indices.cpu().to(torch.int16))
            curr_perc = (n_batch + 1) / len(pbar) * 100
            if config.tqdm_disable and curr_perc >= print_perc:
                print(f"[quant] Progress: {print_perc:.1f}%")
                print_perc += 10.0
            n_batch += 1
    if latent_h is None or latent_w is None:
        raise RuntimeError("No CIFAR-10 data was processed during quantization.")

    indices = torch.cat(all_indices, dim=0).contiguous()
    payload = {
        "indices": indices,  # [N, D], int16
        "latent_h": latent_h,
        "latent_w": latent_w,
        "codebook_size": int(vq_model.config.codebook_size),
        "codebook_embed_dim": int(vq_model.config.codebook_embed_dim),
    }
    torch.save(payload, q_path)
    print(f"[data] Saved quantized CIFAR-10 to: {q_path}")
    return payload

@torch.no_grad()
def get_vq_shapes(vq_model: nn.Module, image_size=(32, 32), in_channels=3, device="cpu"):
    vq_model = vq_model.to(device).eval()
    H, W = image_size

    x = torch.zeros(1, in_channels, H, W, device=device)
    h = vq_model.encoder(x)          # [1, z_channels, H_lat, W_lat]
    h = vq_model.quant_conv(h)       # [1, Cvae, H_lat, W_lat]

    _, Cvae, H_lat, W_lat = h.shape
    D = H_lat * W_lat
    K = vq_model.quantize.n_e

    return {
        "K": K,
        "Cvae": Cvae,
        "H_lat": H_lat,
        "W_lat": W_lat,
        "D": D,
    }


def build_dinov2_extractor(device: torch.device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return model, mean, std


def extract_dinov2_features(
    images_minus1_1: torch.Tensor,
    model: nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int = 128,
) -> torch.Tensor:
    feats = []
    with torch.inference_mode():
        for start in range(0, images_minus1_1.shape[0], batch_size):
            end = min(start + batch_size, images_minus1_1.shape[0])
            batch = images_minus1_1[start:end]
            batch = (batch + 1.0) * 0.5
            batch = batch.clamp(0.0, 1.0)
            batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
            batch = (batch - mean) / std
            out = model(batch)
            if isinstance(out, (tuple, list)):
                out = out[0]
            feats.append(out.detach().cpu())
    return torch.cat(feats, dim=0)


def load_training_features(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing training DINOv2 features at {path}. "
            "Please create data/processed/cifar10_dinov2.pt first."
        )

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for key in ("features", "train_features", "feats", "x"):
            if key in obj and isinstance(obj[key], torch.Tensor):
                return obj[key]
    raise ValueError(f"Unsupported feature file format: {path}")


def warmup_lr_lambda(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup_steps)


def save_checkpoint(
    path: Path,
    step: int,
    epoch: int,
    model: nn.Module,
    flow: nn.Module,
    optim: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LambdaLR,
    model_config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "flow_state_dict": flow.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
        },
        path,
    )


def evaluate(
    flow: CatFlow,
    vq_model: nn.Module,
    train_dino_feat: torch.Tensor,
    test_loader: DataLoader,
    dinov2_model: nn.Module,
    dino_mean: torch.Tensor,
    dino_std: torch.Tensor,
    eval_num_samples: int,
    eval_val_batches: int,
    eval_nll_samples: int,
    eval_nll_hutchinson: int,
    codebook_size: int,
    latent_h: int,
    latent_w: int,
    embed_dim: int,
    device: torch.device,
) -> dict[str, float]:
    metrics: dict[str, float] = {}

    # 1) Image-space quality metrics from generated samples.
    with torch.inference_mode():
        batch_size = 256
        remaining = eval_num_samples
        gen_feat = torch.empty(eval_num_samples, train_dino_feat.shape[1], device=torch.device("cpu"))
        print(f"[eval] Generating {eval_num_samples} samples...")
        while remaining > 0:
            curr_batch = min(batch_size, remaining)
            samples = flow.sample(curr_batch).detach()  # [B, D, K]
            indices = torch.argmax(samples, dim=-1).to(dtype=torch.long)

            codes_flat = indices.reshape(-1)
            quant_shape = (curr_batch, embed_dim, latent_h, latent_w)
            recon = vq_model.decode_code(codes_flat, shape=quant_shape, channel_first=True).to(device)

            gen_feat_b = extract_dinov2_features(recon, dinov2_model, dino_mean, dino_std, batch_size=128)
            start = eval_num_samples - remaining
            gen_feat[start : start + curr_batch] = gen_feat_b.to("cpu")
            remaining -= curr_batch
        
        print("[eval] Computing FID...")
        fid = compute_fid(train_dino_feat, gen_feat)
        # precision, recall = precision_recall_knn_blockwise(train_dino_feat, gen_feat, k=5)
        metrics["eval/fid_dinov2"] = float(fid)
        # metrics["eval/precision"] = float(precision)
        # metrics["eval/recall"] = float(recall)

        # 2) Fast validation proxy in token space (same objective as training).
        print("[eval] Computing validation loss proxy...")
        val_losses: list[float] = []
        for batch_idx, (x1,) in enumerate(test_loader):
            if batch_idx >= eval_val_batches:
                break
            x1 = x1.to(device=device, dtype=torch.long)
            bs = x1.shape[0]
            x0 = flow.sample_prior(bs, *list(flow.obs_dim), device=device)
            t = torch.rand(bs, device=device)
            val_losses.append(float(flow.criterion(t, x0, x1).item()))
        if val_losses:
            metrics["eval/val_fm_ce"] = float(np.mean(val_losses))

    # 3) Small-subset validation NLL estimate via reverse ODE (expensive; keep small).
    eps = 1e-4
    processed = 0
    nll_chunks: list[torch.Tensor] = []
    print(f"[eval] Computing NLL estimate...")
    for (x1_idx,) in test_loader:
        if processed >= eval_nll_samples:
            break
        keep = min(eval_nll_samples - processed, x1_idx.shape[0])
        x1_idx = x1_idx[:keep].to(device=device, dtype=torch.long)
        x1 = F.one_hot(x1_idx, num_classes=codebook_size).to(dtype=torch.float32)
        x1 = x1 * (1.0 - eps) + (eps / codebook_size)

        logp = flow.logp(x1, n_samples=eval_nll_hutchinson).to(device=device)
        nll_chunks.append((-logp).detach())
        processed += keep

    if nll_chunks:
        nll = torch.cat(nll_chunks, dim=0).mean().item()
        bits_per_token = nll / (np.log(2.0) * flow.obs_dim[0])
        metrics["eval/val_nll"] = float(nll)
        metrics["eval/val_bits_per_token"] = float(bits_per_token)

    return metrics


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train CatFlow UNet on VQ-quantized CIFAR-10.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--ckpt_every", type=int, default=5000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--proj_channels", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--tqdm_enable", action="store_true", help="Enable tqdm progress bars")
    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.batch_size = args.batch_size
    cfg.epochs = args.epochs
    cfg.lr = args.lr
    cfg.warmup_steps = args.warmup_steps
    cfg.ckpt_every = args.ckpt_every
    cfg.eval_every = args.eval_every
    cfg.proj_channels = args.proj_channels
    cfg.num_workers = args.num_workers
    cfg.seed = args.seed
    cfg.wandb_run_name = args.wandb_run_name
    cfg.tqdm_disable = not args.tqdm_enable
    return cfg


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    device = pick_device()
    
    try:
        from torchcfm.models.unet.unet import UNetModelWrapper
    except Exception as exc:
        raise ImportError(
            "Could not import torchcfm UNetModelWrapper. "
            "Install torchcfm before running this script."
        ) from exc

    # 1) Load VQ model and (if needed) create quantized CIFAR-10 cache.
    vq_model = VQ_Cifar_L().to(device)
    vq_ckpt = torch.load(ROOT / config.vq_checkpoint_path, map_location=device)
    vq_model.load_state_dict(vq_ckpt["model_state_dict"])
    vq_model.eval()

    q_data = maybe_prepare_quantized_dataset(train=True, config=config, device=device, vq_model=vq_model)
    indices = q_data["indices"].to(torch.long)
    latent_h = int(q_data["latent_h"])
    latent_w = int(q_data["latent_w"])
    codebook_size = int(q_data["codebook_size"])
    codebook_embed_dim = int(q_data["codebook_embed_dim"])
    obs_dim = (latent_h * latent_w, codebook_size)

    train_loader = DataLoader(
        TensorDataset(indices),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_data = maybe_prepare_quantized_dataset(train=False, config=config, device=device, vq_model=vq_model)
    test_indices = test_data["indices"].to(torch.long)
    test_loader = DataLoader(
        TensorDataset(test_indices),
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # 2) Load bidirectional transformer

    codebook = vq_model.quantize.embedding.weight   # [K, Cvae]

    model_cfg = CatFlowTransformerConfig(
        num_classes=codebook_size,
        seq_len=obs_dim[0],
        codebook_dim=codebook_embed_dim,
        grid_h=latent_h,
        grid_w=latent_w,
        dim=512,
        n_layer=8,
        n_head=8,
        input_dropout_p=0.1,
        resid_dropout_p=0.1,
        ffn_dropout_p=0.1,
    )

    model = CatFlowTransformer(model_cfg, codebook)

    flow = CatFlow(
        model=model,
        obs_dim=obs_dim,
        sigma_min=config.catflow_sigma_min,
    ).to(device)

    # 3) Optim + scheduler.
    # `adapter` already owns `net_model` as a submodule; avoid duplicated params in Adam.
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda step: warmup_lr_lambda(step, config.warmup_steps),
    )

    # 4) Evaluation setup (DINOv2 features from train set + extractor).
    train_dino_feat = load_training_features(ROOT / config.cifar10_dino_features)
    dinov2_model, dino_mean, dino_std = build_dinov2_extractor(device)

    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=asdict(config) | {"obs_dim": obs_dim, "model_cfg": model_cfg, "train_cfg": asdict(config)},
    )

    step = 0
    print("[train] Starting training loop...")
    for epoch in range(config.epochs):
        flow.train()
        if config.tqdm_disable:
            print(f"[train] Epoch {epoch+1}/{config.epochs}")
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", unit="batch", disable=config.tqdm_disable)
        for (x1,) in pbar:
            x1 = x1.to(device=device, dtype=torch.long)
            bs = x1.shape[0]
            x0 = flow.sample_prior(bs, *list(obs_dim), device=device)
            t = torch.rand(bs, device=device)

            loss = flow.criterion(t, x0, x1)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip).item()
            # print(f"Step {step}: loss={loss.item():.4f}, grad_norm={total_norm:.4f}")
            optim.step()
            sched.step()
            pbar.set_postfix({"loss": loss.item(), "lr": sched.get_last_lr()[0], "grad_norm": total_norm})

            step += 1
            if step % 20 == 0:
                wandb.log(
                    {
                        "train/loss": float(loss.item()),
                        "train/lr": float(sched.get_last_lr()[0]),
                        "train/grad_norm": float(total_norm),
                    },
                    step=step,
                )

            if step % config.ckpt_every == 0:
                ckpt_path = ROOT / config.checkpoints_dir / f"unet_{step}.pt"
                save_checkpoint(
                    ckpt_path,
                    step,
                    epoch,
                    model,
                    flow,
                    optim,
                    sched,
                    model_config={"model_cfg": model_cfg, "obs_dim": obs_dim},
                )
                print(f"[ckpt] Saved {ckpt_path}")

            if step % config.eval_every == 0:
                flow.eval()
                metrics = evaluate(
                    flow=flow,
                    vq_model=vq_model,
                    train_dino_feat=train_dino_feat,
                    test_loader=test_loader,
                    dinov2_model=dinov2_model,
                    dino_mean=dino_mean,
                    dino_std=dino_std,
                    eval_num_samples=config.eval_num_samples,
                    eval_val_batches=config.eval_val_batches,
                    eval_nll_samples=config.eval_nll_samples,
                    eval_nll_hutchinson=config.eval_nll_hutchinson,
                    codebook_size=codebook_size,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    embed_dim=codebook_embed_dim,
                    device=device,
                )
                wandb.log(metrics, step=step)
                print(
                    "[eval] "
                    + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
                    + f" @ step={step}"
                )
                flow.train()

    # Final checkpoint
    final_step = step
    final_path = ROOT / config.checkpoints_dir / f"unet_{final_step}.pt"
    save_checkpoint(
        final_path,
        final_step,
        config.epochs - 1,
        model,
        flow,
        optim,
        sched,
        model_config={"model_cfg": model_cfg, "obs_dim": obs_dim},
    )
    print(f"[done] Training complete. Final checkpoint: {final_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
