from __future__ import annotations

import argparse
import gzip
import math
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.request import urlopen

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.vfm_wrapper import CatFlow
from src.models.transformer import CatFlowTransformer, CatFlowTransformerConfig


@dataclass
class TrainConfig:
    data_train_path: str = "data/processed/uniprot_train.pt"
    data_test_path: str = "data/processed/uniprot_test.pt"
    train_key: str | None = None
    test_key: str | None = None
    num_classes: int | None = None

    batch_size: int = 128
    num_workers: int = 4
    epochs: int = 300
    lr: float = 2e-4
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    ckpt_every: int = 2000
    eval_every: int = 500
    eval_val_batches: int = 16
    seed: int = 42

    catflow_sigma_min: float = 1e-6
    tqdm_disable: bool = True

    grid_h: int = 1
    grid_w: int | None = None
    model_dim: int = 512
    n_layer: int = 8
    n_head: int = 8
    input_dropout_p: float = 0.0
    resid_dropout_p: float = 0.0
    ffn_dropout_p: float = 0.0

    checkpoints_dir: str = "checkpoints/uniprot_catflow"
    wandb_project: str = "closedform-catflow-uniprot"
    wandb_run_name: str | None = None
    disable_wandb: bool = False

    uniprot_url: str = (
        "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/"
        "uniprot_sprot.fasta.gz"
    )
    uniprot_raw_path: str = "data/raw/uniprot_sprot.fasta.gz"
    uniprot_seq_len: int = 128
    uniprot_test_fraction: float = 0.1
    uniprot_min_seq_len: int = 80
    uniprot_max_sequences: int = 50000
    uniprot_force_reprocess: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
PAD_TOKEN = 0
UNK_TOKEN = len(AA_ALPHABET) + 1
AA_TO_INDEX = {aa: idx + 1 for idx, aa in enumerate(AA_ALPHABET)}


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[data] Downloading UniProt from {url}")
    with urlopen(url) as response, destination.open("wb") as out_file:
        shutil.copyfileobj(response, out_file)
    print(f"[data] Download complete: {destination}")


def _iter_fasta_sequences(path: Path) -> Iterator[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt"
    with opener(path, mode, encoding="utf-8", errors="ignore") as handle:
        seq_chunks: list[str] = []
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_chunks:
                    yield "".join(seq_chunks)
                    seq_chunks = []
                continue
            seq_chunks.append(line.upper())
        if seq_chunks:
            yield "".join(seq_chunks)


def _tokenize_sequence(seq: str, seq_len: int) -> list[int]:
    tokens = [AA_TO_INDEX.get(char, UNK_TOKEN) for char in seq[:seq_len]]
    if len(tokens) < seq_len:
        tokens.extend([PAD_TOKEN] * (seq_len - len(tokens)))
    return tokens


def _sequence_list_to_tensor(sequences: list[str], seq_len: int) -> torch.Tensor:
    out = torch.full((len(sequences), seq_len), PAD_TOKEN, dtype=torch.int16)
    for idx, seq in enumerate(sequences):
        out[idx] = torch.tensor(_tokenize_sequence(seq, seq_len), dtype=torch.int16)
    return out


def _infer_split_from_path(path: Path) -> str:
    stem = path.stem.lower()
    if "train" in stem:
        return "train"
    if "test" in stem or "val" in stem or "valid" in stem:
        return "test"
    raise ValueError(
        f"Could not infer split name from path {path}. Include 'train' or 'test' in filename."
    )


def process_uniprot(config: TrainConfig, requested_split: str) -> Path:
    if requested_split not in {"train", "test"}:
        raise ValueError(f"Unsupported split '{requested_split}'. Expected 'train' or 'test'.")
    if config.uniprot_seq_len <= 0:
        raise ValueError(f"uniprot_seq_len must be > 0, got {config.uniprot_seq_len}")
    if not (0.0 < config.uniprot_test_fraction < 1.0):
        raise ValueError(
            f"uniprot_test_fraction must be in (0, 1), got {config.uniprot_test_fraction}"
        )
    if config.uniprot_min_seq_len <= 0:
        raise ValueError(f"uniprot_min_seq_len must be > 0, got {config.uniprot_min_seq_len}")

    raw_path = ROOT / config.uniprot_raw_path
    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    canonical_train = processed_dir / "uniprot_train.pt"
    canonical_test = processed_dir / "uniprot_test.pt"

    if (
        not config.uniprot_force_reprocess
        and canonical_train.exists()
        and canonical_test.exists()
    ):
        print("[data] Found existing processed UniProt train/test splits.")
        return canonical_train if requested_split == "train" else canonical_test

    if not raw_path.exists():
        _download_file(config.uniprot_url, raw_path)
    else:
        print(f"[data] Using cached raw UniProt file: {raw_path}")

    sequences: list[str] = []
    max_sequences = config.uniprot_max_sequences if config.uniprot_max_sequences > 0 else None
    for seq in _iter_fasta_sequences(raw_path):
        if len(seq) < config.uniprot_min_seq_len:
            continue
        sequences.append(seq)
        if max_sequences is not None and len(sequences) >= max_sequences:
            break

    if len(sequences) < 2:
        raise RuntimeError("Need at least 2 sequences to build train/test splits.")

    rng = random.Random(config.seed)
    rng.shuffle(sequences)
    n_test = max(1, int(round(len(sequences) * config.uniprot_test_fraction)))
    n_test = min(n_test, len(sequences) - 1)

    test_sequences = sequences[:n_test]
    train_sequences = sequences[n_test:]

    print(
        f"[data] Tokenizing UniProt: total={len(sequences)}, "
        f"train={len(train_sequences)}, test={len(test_sequences)}, seq_len={config.uniprot_seq_len}"
    )
    train_tensor = _sequence_list_to_tensor(train_sequences, config.uniprot_seq_len)
    test_tensor = _sequence_list_to_tensor(test_sequences, config.uniprot_seq_len)

    torch.save(train_tensor, canonical_train)
    torch.save(test_tensor, canonical_test)
    # Compatibility with requested naming in user instructions.
    print(f"[data] Saved processed UniProt train split to: {canonical_train}")
    print(f"[data] Saved processed UniProt test split to: {canonical_test}")

    return canonical_train if requested_split == "train" else canonical_test


def _extract_split_tensor(obj: Any, split_key: str | None, source_path: Path) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, dict):
        if split_key is not None:
            if split_key not in obj:
                raise KeyError(f"Key '{split_key}' not found in {source_path}. Available keys: {list(obj.keys())}")
            value = obj[split_key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Key '{split_key}' in {source_path} is not a tensor.")
            return value

        for key in ("indices", "tokens", "data", "x"):
            value = obj.get(key)
            if isinstance(value, torch.Tensor):
                return value

        tensor_values = [value for value in obj.values() if isinstance(value, torch.Tensor)]
        if len(tensor_values) == 1:
            return tensor_values[0]

    raise ValueError(
        f"Could not extract a categorical tensor from {source_path}. "
        "Use --train_key/--test_key if the split is stored under a specific dictionary key."
    )


def load_categorical_split(path: Path, split_key: str | None, config: TrainConfig) -> torch.Tensor:
    if not path.exists():
        split = _infer_split_from_path(path)
        print(f"[data] Missing processed split at {path}. Triggering process_uniprot('{split}')...")
        generated_path = process_uniprot(config=config, requested_split=split)
        if path != generated_path and generated_path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(generated_path, path)
            print(f"[data] Copied generated split from {generated_path} to requested path {path}")
        if not path.exists():
            if generated_path.exists():
                path = generated_path
            else:
                raise FileNotFoundError(
                    f"UniProt processing completed but could not find generated split for {split}: "
                    f"{generated_path}"
                )

    payload = torch.load(path, map_location="cpu")
    x = _extract_split_tensor(payload, split_key=split_key, source_path=path)
    if x.dim() == 1:
        raise ValueError(f"Expected 2D categorical tensor [N, D], got shape {tuple(x.shape)} from {path}.")
    if x.dim() > 2:
        x = x.view(x.shape[0], -1)

    if torch.is_floating_point(x):
        if not torch.allclose(x, x.round()):
            raise ValueError(f"Found non-integer floating-point values in {path}.")
        x = x.round()

    x = x.to(torch.long).contiguous()
    if x.numel() == 0:
        raise ValueError(f"Loaded empty categorical dataset from {path}.")
    if torch.any(x < 0):
        min_val = int(x.min().item())
        raise ValueError(f"Expected non-negative category ids in {path}, found minimum {min_val}.")
    return x


def infer_num_classes(train_x: torch.Tensor, test_x: torch.Tensor | None, override: int | None) -> int:
    max_train = int(train_x.max().item())
    max_test = int(test_x.max().item()) if test_x is not None else -1
    inferred = max(max_train, max_test) + 1
    if override is None:
        return inferred
    if override < inferred:
        raise ValueError(f"--num_classes={override} is too small for the data (requires at least {inferred}).")
    return override


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


@torch.no_grad()
def evaluate_validation_ce(
    flow: CatFlow,
    val_loader: DataLoader,
    obs_dim: tuple[int, int],
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    losses: list[float] = []
    for batch_idx, (x1,) in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        x1 = x1.to(device=device, dtype=torch.long)
        batch_size = x1.shape[0]
        x0 = flow.sample_prior(batch_size, *list(obs_dim), device=device)
        t = torch.rand(batch_size, device=device)
        losses.append(float(flow.criterion(t, x0, x1).item()))
    return {"eval/val_fm_ce": float(sum(losses) / len(losses))} if losses else {}


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train CatFlow transformer on categorical UniProt data.")
    parser.add_argument("--data_train_path", type=str, default="data/processed/uniprot_train.pt")
    parser.add_argument("--data_test_path", type=str, default="data/processed/uniprot_test.pt")
    parser.add_argument("--train_key", type=str, default=None)
    parser.add_argument("--test_key", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--ckpt_every", type=int, default=3000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_val_batches", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid_h", type=int, default=1)
    parser.add_argument("--grid_w", type=int, default=None)
    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--input_dropout_p", type=float, default=0.0)
    parser.add_argument("--resid_dropout_p", type=float, default=0.0)
    parser.add_argument("--ffn_dropout_p", type=float, default=0.0)
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints/uniprot_catflow")
    parser.add_argument("--wandb_project", type=str, default="closedform-catflow-uniprot")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument(
        "--uniprot_url",
        type=str,
        default=(
            "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/"
            "uniprot_sprot.fasta.gz"
        ),
    )
    parser.add_argument("--uniprot_raw_path", type=str, default="data/raw/uniprot_sprot.fasta.gz")
    parser.add_argument("--uniprot_seq_len", type=int, default=128)
    parser.add_argument("--uniprot_test_fraction", type=float, default=0.1)
    parser.add_argument("--uniprot_min_seq_len", type=int, default=80)
    parser.add_argument("--uniprot_max_sequences", type=int, default=50000)
    parser.add_argument("--uniprot_force_reprocess", action="store_true")
    parser.add_argument("--tqdm_enable", action="store_true", help="Enable tqdm progress bars")
    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.data_train_path = args.data_train_path
    cfg.data_test_path = args.data_test_path
    cfg.train_key = args.train_key
    cfg.test_key = args.test_key
    cfg.num_classes = args.num_classes
    cfg.batch_size = args.batch_size
    cfg.epochs = args.epochs
    cfg.lr = args.lr
    cfg.warmup_steps = args.warmup_steps
    cfg.ckpt_every = args.ckpt_every
    cfg.eval_every = args.eval_every
    cfg.eval_val_batches = args.eval_val_batches
    cfg.num_workers = args.num_workers
    cfg.seed = args.seed
    cfg.grid_h = args.grid_h
    cfg.grid_w = args.grid_w
    cfg.model_dim = args.model_dim
    cfg.n_layer = args.n_layer
    cfg.n_head = args.n_head
    cfg.input_dropout_p = args.input_dropout_p
    cfg.resid_dropout_p = args.resid_dropout_p
    cfg.ffn_dropout_p = args.ffn_dropout_p
    cfg.checkpoints_dir = args.checkpoints_dir
    cfg.wandb_project = args.wandb_project
    cfg.wandb_run_name = args.wandb_run_name
    cfg.disable_wandb = args.disable_wandb
    cfg.uniprot_url = args.uniprot_url
    cfg.uniprot_raw_path = args.uniprot_raw_path
    cfg.uniprot_seq_len = args.uniprot_seq_len
    cfg.uniprot_test_fraction = args.uniprot_test_fraction
    cfg.uniprot_min_seq_len = args.uniprot_min_seq_len
    cfg.uniprot_max_sequences = args.uniprot_max_sequences
    cfg.uniprot_force_reprocess = args.uniprot_force_reprocess
    cfg.tqdm_disable = not args.tqdm_enable
    return cfg


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    device = pick_device()

    train_indices = load_categorical_split(
        ROOT / config.data_train_path,
        split_key=config.train_key,
        config=config,
    )
    test_indices = load_categorical_split(
        ROOT / config.data_test_path,
        split_key=config.test_key,
        config=config,
    )

    seq_len = int(train_indices.shape[1])
    if int(test_indices.shape[1]) != seq_len:
        raise ValueError(
            f"Train/test sequence lengths differ: train={seq_len}, test={int(test_indices.shape[1])}."
        )
    if config.grid_h <= 0:
        raise ValueError(f"grid_h must be >= 1, got {config.grid_h}")
    if config.grid_w is None:
        if seq_len % config.grid_h != 0:
            raise ValueError(
                f"seq_len={seq_len} is not divisible by grid_h={config.grid_h}. "
                "Set --grid_h/--grid_w manually so grid_h * grid_w == seq_len."
            )
        grid_w = seq_len // config.grid_h
    else:
        grid_w = config.grid_w
        if config.grid_h * grid_w != seq_len:
            raise ValueError(
                f"Expected grid_h * grid_w == seq_len, got {config.grid_h} * {grid_w} != {seq_len}."
            )

    num_classes = infer_num_classes(train_indices, test_indices, override=config.num_classes)
    obs_dim = (seq_len, num_classes)
    codebook = None

    train_loader = DataLoader(
        TensorDataset(train_indices),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        TensorDataset(test_indices),
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model_cfg = CatFlowTransformerConfig(
        num_classes=num_classes,
        seq_len=seq_len,
        codebook_dim=None,
        grid_h=config.grid_h,
        grid_w=grid_w,
        dim=config.model_dim,
        n_layer=config.n_layer,
        n_head=config.n_head,
        input_dropout_p=config.input_dropout_p,
        resid_dropout_p=config.resid_dropout_p,
        ffn_dropout_p=config.ffn_dropout_p,
    )
    model = CatFlowTransformer(model_cfg, codebook=codebook)
    flow = CatFlow(
        model=model,
        obs_dim=obs_dim,
        sigma_min=config.catflow_sigma_min,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda step: warmup_lr_lambda(step, config.warmup_steps),
    )

    if not config.disable_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=asdict(config)
            | {
                "obs_dim": obs_dim,
                "model_cfg": asdict(model_cfg),
                "train_samples": int(train_indices.shape[0]),
                "val_samples": int(test_indices.shape[0]),
            },
        )

    print(
        "[data] "
        f"train={tuple(train_indices.shape)}, "
        f"val={tuple(test_indices.shape)}, "
        f"seq_len={seq_len}, num_classes={num_classes}, obs_dim={obs_dim}, device={device}"
    )
    print("[train] Starting training loop...")

    step = 0
    for epoch in range(config.epochs):
        flow.train()
        if config.tqdm_disable:
            print(f"[train] Epoch {epoch + 1}/{config.epochs}")
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.epochs}",
            unit="batch",
            disable=config.tqdm_disable,
        )
        for (x1,) in pbar:
            x1 = x1.to(device=device, dtype=torch.long)
            batch_size = x1.shape[0]
            x0 = flow.sample_prior(batch_size, *list(obs_dim), device=device)
            t = torch.rand(batch_size, device=device)

            loss = flow.criterion(t, x0, x1)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip).item()
            optim.step()
            sched.step()

            if not config.tqdm_disable:
                pbar.set_postfix(
                    {"loss": float(loss.item()), "lr": float(sched.get_last_lr()[0]), "grad_norm": total_norm}
                )

            step += 1
            if step % 20 == 0:
                logs = {
                    "train/loss": float(loss.item()),
                    "train/lr": float(sched.get_last_lr()[0]),
                    "train/grad_norm": float(total_norm),
                }
                if not config.disable_wandb:
                    wandb.log(logs, step=step)
                else:
                    print(f"[step {step}] " + ", ".join(f"{k}={v:.6f}" for k, v in logs.items()))

            if step % config.ckpt_every == 0:
                ckpt_path = ROOT / config.checkpoints_dir / f"step_{step}.pt"
                save_checkpoint(
                    ckpt_path,
                    step,
                    epoch,
                    model,
                    flow,
                    optim,
                    sched,
                    model_config={"model_cfg": asdict(model_cfg), "obs_dim": obs_dim},
                )
                print(f"[ckpt] Saved {ckpt_path}")

            if step % config.eval_every == 0:
                flow.eval()
                metrics = evaluate_validation_ce(
                    flow=flow,
                    val_loader=test_loader,
                    obs_dim=obs_dim,
                    device=device,
                    max_batches=config.eval_val_batches,
                )
                if metrics:
                    if not config.disable_wandb:
                        wandb.log(metrics, step=step)
                    print("[eval] " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()]) + f" @ step={step}")
                flow.train()

    final_path = ROOT / config.checkpoints_dir / f"step_{step}.pt"
    save_checkpoint(
        final_path,
        step,
        config.epochs - 1,
        model,
        flow,
        optim,
        sched,
        model_config={"model_cfg": asdict(model_cfg), "obs_dim": obs_dim},
    )
    print(f"[done] Training complete. Final checkpoint: {final_path}")
    if not config.disable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
