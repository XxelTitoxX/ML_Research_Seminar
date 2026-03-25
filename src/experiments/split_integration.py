import os
import sys
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Dataset
from pathlib import Path
from tqdm import tqdm

import torch.nn.functional as F
from typing import Optional
import json

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
CHECKPOINTS = ROOT / "checkpoints"
sys.path.insert(0, str(ROOT))

from src.experiments.closedform import u_star
from src.experiments.utils import load_catflow_checkpoint

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
def load_uniprot(data_root: str, batch_size: int, train: bool = True, labels: bool = False, shuffle: bool = True) -> DataLoader:
    split = "train" if train else "test"
    proportion = "9k" if train else "1k"
    data_path = Path(data_root) / f"uniprot_{split}_seq32_{proportion}.pt"
    indices_tensor = torch.load(data_path, map_location="cpu")
    if labels:
        dataset = TensorDataset(indices_tensor, torch.zeros(len(indices_tensor), dtype=torch.long))  # dummy labels
    else:
        dataset = TensorDataset(indices_tensor)
    num_classes = 22
    def collate_fn(batch):
        indices_batch = torch.stack([item[0] for item in batch]).to(dtype=torch.long)  # shape: (batch_size, seq_len)
        one_hot_batch = F.one_hot(indices_batch, num_classes=num_classes).float()  # shape: (batch_size, seq_len, num_classes)
        if len(batch[0]) == 2:  # If labels are present, also collate them
            labels_batch = torch.tensor([item[1] for item in batch], dtype=torch.long)  # shape: (batch_size,)
            return one_hot_batch, labels_batch
        return one_hot_batch
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def load_uniprot_dataset(data_root: str, n_samples: Optional[int], train: bool = True, shuffle: bool = True) -> DataLoader:
    split = "train" if train else "test"
    proportion = "9k" if train else "1k"
    data_path = Path(data_root) / f"uniprot_{split}_seq32_{proportion}.pt"
    indices_tensor = torch.load(data_path, map_location="cpu")
    if n_samples is not None:
        indices_tensor = indices_tensor[:n_samples]  # Take only the first n_samples
    # Shuffle the dataset if requested
    if shuffle:
        perm = torch.randperm(indices_tensor.size(0))
        indices_tensor = indices_tensor[perm]
    return F.one_hot(indices_tensor.to(dtype=torch.long), num_classes=22).float()  # shape: (n_samples, seq_len, num_classes)

def load_uniprot_indices(data_root: str, n_samples: Optional[int], train: bool = True, shuffle: bool = True) -> DataLoader:
    split = "train" if train else "test"
    proportion = "9k" if train else "1k"
    data_path = Path(data_root) / f"uniprot_{split}_seq32_{proportion}.pt"
    indices_tensor = torch.load(data_path, map_location="cpu")
    if n_samples is not None:
        indices_tensor = indices_tensor[:n_samples]  # Take only the first n_samples
    if shuffle:
        perm = torch.randperm(indices_tensor.size(0))
        indices_tensor = indices_tensor[perm]
    return indices_tensor.to(dtype=torch.long)  # shape: (n_samples, seq_len)

def accuracy(generated_indices: torch.Tensor, train_sequence_lookup: set[bytes]) -> tuple[int, float]:
    if generated_indices.dim() != 2:
        raise ValueError(f"Expected generated indices shape [N, D], got {tuple(generated_indices.shape)}")
    if generated_indices.shape[0] == 0:
        return 0, 0.0

    generated_cpu = generated_indices.to(dtype=torch.int16, device=torch.device("cpu")).contiguous()
    match_count = 0
    for row in generated_cpu:
        if row.numpy().tobytes() in train_sequence_lookup:
            match_count += 1

    return match_count, match_count / generated_cpu.shape[0]

def build_sequence_lookup(sequences: torch.Tensor) -> set[bytes]:
    seq_cpu = sequences.to(dtype=torch.int16, device=torch.device("cpu")).contiguous()
    return {row.numpy().tobytes() for row in seq_cpu}
    


# Global plotting style for paper figures
mpl.rcParams.update({
    "figure.figsize": (8, 5), 
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 2.0,
    "axes.grid": False,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def make_plot():
    results_path = ROOT / "src" / "experiments" / "split_integration_results.json"
    with open(results_path, "r") as f:
        results = json.load(f)
    taus = sorted(float(k) for k in results.keys())
    accuracies = [results[str(tau)] for tau in taus]
    fig, ax = plt.subplots()

    ax.plot(taus, accuracies)   # no marker
    ax.set_xlabel("Switch time $\\tau$")
    ax.set_ylabel("Exact match accuracy")

    # subtle horizontal grid only
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.set_ylim(0, 1.1)

    fig.savefig("acc_kld_gs.png")
    plt.close(fig)

def main():
    torch.backends.cudnn.benchmark = True
    device = pick_device()
    device = torch.device("cpu") 

    flow, model_cfg, obs_dim = load_catflow_checkpoint(
        flow_checkpoint_path=CHECKPOINTS / "cfm_uni_9k_seq32_21000.pt",
        device=device,
    )
    flow.eval()
    flow.loss = "cfm"  # "kld" or "mse", override loss just for error computation
    flow.p0 = "uniform"  # "uniform" or "gaussian_scaled", override prior to make sure we have same config as checkpoint
    print(f"Loaded CatFlow checkpoint : loss={flow.loss}, prior={flow.p0}, obs_dim={obs_dim}")

    
    
    n_steps = 200
    clamp_t = 0.01
    tau = 0.1
    n_xt_samples = 24
    batch_size_xt = 8
    batch_size_x1prime = 128
    ts = torch.linspace(0, 1.0 - clamp_t, n_steps, device=device)

    # Create dataloader of sequences of one-hot encoded codebook indices
    # DO NOT USE SHUFFLE for x1' dataloader !
    dataloader_x1prime = load_uniprot(DATA_PROCESSED, batch_size=batch_size_x1prime, train=True, labels=False, shuffle=False)

    x0s = flow.sample_prior((n_xt_samples, obs_dim[0], obs_dim[1]), device=device)  # shape: (n_xt_samples, seq_len, codebook_size)
    xts = x0s.clone()


    # x1s = load_uniprot_dataset(DATA_PROCESSED, n_samples=n_xt_samples, train=True, shuffle=True).to(device)  # shape: (n_xt_samples, seq_len, codebook_size)
    with torch.inference_mode():
        for i, t in enumerate(ts):
            print(f"Processing time step {i+1}/{len(ts)}: t={t.item():.4f}")
            xt_processed = 0

            pbar = tqdm(range(n_xt_samples//batch_size_xt), desc=f"t={t.item():.4f}")
            for b_idx in pbar:
                if xt_processed >= n_xt_samples:
                    break
                remaining = n_xt_samples - xt_processed
                batch_size = min(batch_size_xt, remaining)
                xt_b = xts[xt_processed:xt_processed+batch_size]
                t_b = t * torch.ones(batch_size, 1, device=device)
                dt = ts[i+1] - t if i < len(ts) - 1 else 1.0 - t

                # Compute closed form velocity
                if t.item() < tau:
                    u_star_b = u_star(t_b, xt_b, dataloader_x1prime, flow.prior_logp0) # shape: (batch_size, seq_len, codebook_dim)
                    xt_star_b = xt_b + u_star_b * dt  # Euler step to get next xt_star
                    assert not torch.isnan(xt_star_b).any(), f"NaN values found in xt_star_b at t={t.item():.4f}"
                    new_xt_b = xt_star_b

                # Compute model velocity
                else:
                    with torch.no_grad():
                        model_velocity_b = flow.velocity(t_b.reshape(-1), xt_b) # shape: (batch_size, seq_len, codebook_dim)
                    xt_model_b = (xt_b + model_velocity_b * dt)  # Euler step to get next xt_model
                    new_xt_b = xt_model_b
                if flow.p0 == "uniform":
                    min_xt_b = new_xt_b.min(dim=-1).values
                    contains_neg_values = min_xt_b < 0.0
                    new_xt_b = new_xt_b - min_xt_b.unsqueeze(-1) * contains_neg_values.unsqueeze(-1).float()  # Shift to make all values non-negative
                    den = new_xt_b.sum(dim=-1, keepdim=True)
                    zero_mask = den <= 1e-8
                    new_xt_b = new_xt_b / den.clamp(min=1e-8)  # Normalize to get probabilities, avoid division by zero
                    new_xt_b[zero_mask.expand_as(new_xt_b)] = 1.0 / new_xt_b.shape[-1]
                    if t.item() < tau:
                        pass
                assert not torch.isnan(new_xt_b).any(), f"NaN values found in new_xt_b at t={t.item():.4f}"
                xts[xt_processed:xt_processed+batch_size] = new_xt_b

                xt_processed += batch_size

    train_sequence_lookup = build_sequence_lookup(load_uniprot_indices(DATA_PROCESSED, n_samples=None, train=True, shuffle=False))  # Build lookup set from x1' sequences
    generated_indices = torch.argmax(xts, dim=-1).to(dtype=torch.long)
    _, match_acc = accuracy(generated_indices, train_sequence_lookup)

    # Test accuracy
    some_training_sequences = load_uniprot_dataset(DATA_PROCESSED, n_samples=100, train=True, shuffle=True)
    some_training_indices = torch.argmax(some_training_sequences, dim=-1).to(dtype=torch.long)
    training_match_count, training_acc = accuracy(some_training_indices, train_sequence_lookup)
    print(f"Sanity check - Accuracy on some training sequences: {training_acc:.4f} ({training_match_count}/100)")

    # Add (tau, match_count) point to json results file for later plotting
    results_path = ROOT / "src" / "experiments" / "split_integration_results.json"
    if results_path.exists():
        with open(results_path, "r") as f:
            results = json.load(f)
    else:
        results = {}
    results[str(tau)] = match_acc
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Match accuracy at tau={tau}: {match_acc:.4f}")
        



if __name__ == "__main__":
    main()
    #make_plot()