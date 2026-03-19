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
    data_path = Path(data_root) / f"uniprot_{split}{proportion}.pt"
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

def make_plot(ts, error_t, out_path, logy=False):
    fig, ax = plt.subplots()

    ax.plot(ts.cpu(), error_t.cpu())   # no marker
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Average $\\ell_2$ error" + (" (log scale)" if logy else ""))

    if logy:
        ax.set_yscale("log")

    # subtle horizontal grid only
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")

    fig.savefig(out_path)
    plt.close(fig)

def main():
    torch.backends.cudnn.benchmark = True
    device = pick_device()

    flow, model_cfg, obs_dim = load_catflow_checkpoint(
        flow_checkpoint_path=CHECKPOINTS / "kld_9k_uniform_21000.pt",
        device=device,
    )
    flow.eval()
    flow.loss = "kld"  # override loss just for error computation
    flow.p0 = "uniform"  # override prior just for error computation
    print(f"Loaded CatFlow checkpoint : loss={flow.loss}, prior={flow.p0}, obs_dim={obs_dim}")

    cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    
    n_steps = 20
    clamp_t = 0.01
    n_xt_samples = 128
    batch_size_xt = 8
    batch_size_x1prime = 128
    ts = torch.linspace(0, 1.0 - clamp_t, n_steps, device=device)

    # Create dataloader of sequences of one-hot encoded codebook indices
    # DO NOT USE SHUFFLE for x1' dataloader !
    dataloader_xt = load_uniprot(DATA_PROCESSED, batch_size=batch_size_xt, train=True, labels=True, shuffle=True)
    dataloader_x1prime = load_uniprot(DATA_PROCESSED, batch_size=batch_size_x1prime, train=True, labels=False, shuffle=False)

    error_t = torch.zeros_like(ts)
    with torch.inference_mode():
        for i, t in enumerate(ts):
            print(f"Processing time step {i+1}/{len(ts)}: t={t.item():.4f}")
            xt_processed = 0
            pbar = tqdm(dataloader_xt, total=n_xt_samples // batch_size_xt, desc="Processing xt batches")
            for x1_b, labels in pbar:
                if xt_processed >= n_xt_samples:
                    break
                xt_b_size = min(batch_size_xt, n_xt_samples - xt_processed)
                x1_b = x1_b[:xt_b_size].to(device)
                labels = labels[:xt_b_size].to(device)
                x0_b = flow.sample_prior(*x1_b.shape, device=device)
                xt_b = x0_b * (1 - t) + x1_b * t
                t_b = t * torch.ones(xt_b_size, 1, device=device)

                # Compute closed form velocity
                u_star_b = u_star(t_b, xt_b, dataloader_x1prime, flow.prior_logp0) # shape: (batch_size, seq_len, codebook_dim)

                # Compute model velocity
                with torch.no_grad():
                    model_velocity_b = flow.velocity(t_b.reshape(-1), xt_b) # shape: (batch_size, seq_len, codebook_dim)
                
                # Compute error
                error_b = torch.norm(u_star_b - model_velocity_b, dim=-1).sum().item()
                pbar.set_postfix({"error": error_b})
                error_t[i] += error_b
                xt_processed += xt_b_size
            error_t[i] /= n_xt_samples
            print(f"Average L2 error at t={t.item():.4f}: {error_t[i].item():.4f}")
    
    make_plot(ts, error_t, "velocity_error.png", logy=False)
    make_plot(ts, error_t, "velocity_error_log.png", logy=True)

        



if __name__ == "__main__":
    main()