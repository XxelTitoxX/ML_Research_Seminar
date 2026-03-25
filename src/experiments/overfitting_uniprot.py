import os
import sys
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
from pathlib import Path
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
CHECKPOINTS = ROOT / "checkpoints"
sys.path.insert(0, str(ROOT))

from src.experiments.closedform import u_star
from src.experiments.utils import load_catflow_checkpoint
from src.models.transformer import CatFlowTransformer, CatFlowTransformerConfig

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

def load_uniprot_indices(data_root: str, train: bool = True) -> torch.Tensor:
    split = "train" if train else "test"
    proportion = "9k" if train else "1k"
    data_path = Path(data_root) / f"uniprot_{split}_seq32_{proportion}.pt"
    indices_tensor = torch.load(data_path, map_location="cpu")
    return indices_tensor

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

@torch.no_grad()
def evaluate_generation_accuracy(
    flow,
    train_sequence_lookup: set[bytes],
    num_samples: int,
    sample_batch_size: int,
    method: str = "euler",
    n_steps: int = 10,
) -> dict[str, float]:
    if num_samples <= 0:
        return {}
    if sample_batch_size <= 0:
        raise ValueError(f"sample_batch_size must be > 0, got {sample_batch_size}")

    total_matches = 0
    generated = 0
    pbar = tqdm(total=num_samples, desc="Evaluating generation accuracy")
    while generated < num_samples:
        curr_batch = min(sample_batch_size, num_samples - generated)
        samples = flow.sample(n_samples=curr_batch, method=method, n_steps=n_steps).detach()
        generated_indices = torch.argmax(samples, dim=-1).to(dtype=torch.long)
        match_count, _ = accuracy(generated_indices, train_sequence_lookup)
        total_matches += match_count
        generated += curr_batch
        pbar.set_postfix({"batch_match_count": match_count, "total_matches": total_matches, "generated": generated})
        pbar.update(curr_batch)
    pbar.close()
    acc = total_matches / num_samples
    return {
        "eval/train_exact_match_acc": float(acc),
    }

def accuracy_plot():
    # bar plot of accuracy for different modes (e.g. mse_uniform, mse_gauss_scaled, kld_uniform, kld_gauss_scaled)
    # unit is per-thousand (i.e. 1.9 accuracy = 1.9 per-thousand)
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    accuracies = {
        "Cat+Uni": 0.0,
        "Cat+Gauss": 1.9,
        "CFM+Uni": 0.0,
        "CFM+Gauss": 0.2,
        }
    plt.figure(figsize=(8, 6))
    plt.bar(accuracies.keys(), accuracies.values(), color=[default_colors[1], default_colors[0], default_colors[2], default_colors[3]])
    plt.ylabel("Exact Match Accuracy ‰ (per-thousand)")
    plt.grid(axis="y", alpha=0.25)
    plt.savefig("uniprot_accuracy_comparison.png")
    plt.close()

def main():
    torch.backends.cudnn.benchmark = True
    device = pick_device()

    flow, model_cfg, obs_dim = load_catflow_checkpoint(
        flow_checkpoint_path=CHECKPOINTS / "cfm_uni_9k_seq32_21000.pt",
        device=device,
    )
    flow.eval()
    flow.loss = "mse"  # override loss just for error computation
    flow.p0 = "uniform"  # override prior just for error computation
    print(f"Loaded CatFlow checkpoint : loss={flow.loss}, prior={flow.p0}, obs_dim={obs_dim}")

    cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    
    num_samples = 10000
    sample_batch_size = 128

    # Create dataloader of sequences of one-hot encoded codebook indices
    # DO NOT USE SHUFFLE for x1' dataloader !
    uniprot_indices = load_uniprot_indices(data_root=DATA_PROCESSED, train=True)
    

    train_sequence_lookup = build_sequence_lookup(uniprot_indices)

    accuracy = evaluate_generation_accuracy(
        flow=flow,
        train_sequence_lookup=train_sequence_lookup,
        num_samples=num_samples,
        sample_batch_size=sample_batch_size,
    )

    print(f"Training set exact match accuracy: {accuracy['eval/train_exact_match_acc']:.4f}")

        



if __name__ == "__main__":
    # main()
    accuracy_plot()