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

def load_uniprot_dataset(data_root: str, n_samples: int, train: bool = True, shuffle: bool = True) -> DataLoader:
    split = "train" if train else "test"
    proportion = "9k" if train else "1k"
    data_path = Path(data_root) / f"uniprot_{split}_seq32_{proportion}.pt"
    indices_tensor = torch.load(data_path, map_location="cpu")
    indices_tensor = indices_tensor[:n_samples]  # Take only the first n_samples
    return F.one_hot(indices_tensor, num_classes=22).float()  # shape: (n_samples, seq_len, num_classes)
    


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

def make_plot_cossim(ts, error_t, out_path):
    fig, ax = plt.subplots()

    ax.plot(ts.cpu(), error_t.cpu())   # no marker
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Average Cosine Similarity")

    # subtle horizontal grid only
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")

    fig.savefig(out_path)
    plt.close(fig)

def main():
    torch.backends.cudnn.benchmark = True
    device = pick_device()
    device = torch.device("cpu") 

    flow, model_cfg, obs_dim = load_catflow_checkpoint(
        flow_checkpoint_path=CHECKPOINTS / "cfm_gauss_scaled_9k_seq32_21000.pt",
        device=device,
    )
    flow.eval()
    flow.loss = "mse"  # "kld" or "mse", override loss just for error computation
    flow.p0 = "gaussian_scaled"  # "uniform" or "gaussian_scaled", override prior to make sure we have same config as checkpoint
    print(f"Loaded CatFlow checkpoint : loss={flow.loss}, prior={flow.p0}, obs_dim={obs_dim}")

    
    
    n_steps = 100
    clamp_t = 0.01
    n_xt_samples = 128
    batch_size_xt = 8
    batch_size_x1prime = 128
    ts = torch.linspace(0, 1.0 - clamp_t, n_steps, device=device)

    # Create dataloader of sequences of one-hot encoded codebook indices
    # DO NOT USE SHUFFLE for x1' dataloader !
    dataloader_x1prime = load_uniprot(DATA_PROCESSED, batch_size=batch_size_x1prime, train=True, labels=False, shuffle=False)

    x0s = flow.sample_prior((n_xt_samples, obs_dim[0], obs_dim[1]), device=device)  # shape: (n_xt_samples, seq_len, codebook_size)
    xts_star = x0s.clone()  # shape: (n_xt_samples, seq_len, codebook_size)
    xts_model = x0s.clone()  # shape: (n_xt_samples, seq_len, codebook_size)

    entropies_star = []
    entropies_model = []
    # x1s = load_uniprot_dataset(DATA_PROCESSED, n_samples=n_xt_samples, train=True, shuffle=True).to(device)  # shape: (n_xt_samples, seq_len, codebook_size)
    with torch.inference_mode():
        for i, t in enumerate(ts):
            print(f"Processing time step {i+1}/{len(ts)}: t={t.item():.4f}")
            xt_processed = 0
            entropy_star_sum = 0.0
            entropy_model_sum = 0.0
            pbar = tqdm(range(n_xt_samples//batch_size_xt), desc=f"t={t.item():.4f}")
            for b_idx in pbar:
                if xt_processed >= n_xt_samples:
                    break
                remaining = n_xt_samples - xt_processed
                batch_size = min(batch_size_xt, remaining)
                xt_star_b = xts_star[xt_processed:xt_processed+batch_size]
                xt_model_b = xts_model[xt_processed:xt_processed+batch_size]
                t_b = t * torch.ones(batch_size, 1, device=device)
                dt = ts[i+1] - t if i < len(ts) - 1 else 1.0 - t

                # Compute closed form velocity
                u_star_b = u_star(t_b, xt_star_b, dataloader_x1prime, flow.prior_logp0) # shape: (batch_size, seq_len, codebook_dim)
                xt_star_b = xt_star_b + u_star_b * dt  # Euler step to get next xt_star
                assert not torch.isnan(xt_star_b).any(), f"NaN values found in xt_star_b at t={t.item():.4f}"
                xts_star[xt_processed:xt_processed+batch_size] = xt_star_b


                # Compute model velocity
                with torch.no_grad():
                    model_velocity_b = flow.velocity(t_b.reshape(-1), xt_model_b) # shape: (batch_size, seq_len, codebook_dim)
                xt_model_b = (xt_model_b + model_velocity_b * dt)  # Euler step to get next xt_model
                xts_model[xt_processed:xt_processed+batch_size] = xt_model_b

                # Compute entropies
                min_xt_star_b = xt_star_b.min(dim=-1).values
                contains_neg_values = min_xt_star_b < 0.0
                xt_star_b = xt_star_b - min_xt_star_b.unsqueeze(-1) * contains_neg_values.unsqueeze(-1).float()  # Shift to make all values non-negative
                den = xt_star_b.sum(dim=-1, keepdim=True)
                zero_mask = den <= 1e-8
                xt_star_b = xt_star_b / den.clamp(min=1e-8)  # Normalize to get probabilities, avoid division by zero
                xt_star_b[zero_mask.expand_as(xt_star_b)] = 1.0 / xt_star_b.shape[-1]
                entropy_star_b = torch.special.entr(xt_star_b).sum()  # Compute entropy
                entropy_star_sum += entropy_star_b.item()

                min_xt_model_b = xt_model_b.min(dim=-1).values
                contains_neg_values_model = min_xt_model_b < 0.0
                xt_model_b = xt_model_b - min_xt_model_b.unsqueeze(-1) * contains_neg_values_model.unsqueeze(-1).float()  # Shift to make all values non-negative
                den_model = xt_model_b.sum(dim=-1, keepdim=True)
                zero_mask_model = den_model <= 1e-8
                xt_model_b = xt_model_b / den_model.clamp(min=1e-8)  # Normalize to get probabilities, avoid division by zero
                xt_model_b[zero_mask_model.expand_as(xt_model_b)] = 1.0 / xt_model_b.shape[-1]  # If zero vector, set to uniform distribution
                entropy_model_b = torch.special.entr(xt_model_b).sum()  # Compute entropy
                entropy_model_sum += entropy_model_b.item()

                pbar.set_postfix({"ent_star": entropy_star_b.item(), "ent_model": entropy_model_b.item()})

                xt_processed += batch_size
            avg_entropy_star = entropy_star_sum / n_xt_samples
            avg_entropy_model = entropy_model_sum / n_xt_samples
            entropies_star.append(avg_entropy_star)
            entropies_model.append(avg_entropy_model)
            print(f"Average entropy at t={t.item():.4f}: star={avg_entropy_star:.4f}, model={avg_entropy_model:.4f}")
            
    entropies_star = torch.tensor(entropies_star)
    entropies_model = torch.tensor(entropies_model)
    max_entropy = torch.log(torch.tensor(22.0))*obs_dim[0]  # Max entropy for uniform distribution over 22 categories (sum over seq_len positions)

    fig, ax = plt.subplots()

    ax.plot(ts.cpu(), entropies_star.cpu())   # no marker
    ax.plot(ts.cpu(), entropies_model.cpu())   # no marker
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Average Entropy")
    ax.legend(["Closed-form Velocity", "Model Velocity"])
    # Show horizontal line for max entropy
    ax.axhline(max_entropy.item(), color="gray", linestyle="--", label="Max Entropy (Uniform Dist)")
    ax.legend(loc="upper right")

    # subtle horizontal grid only
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")

    fig.savefig("entropy_comparison.png")
    plt.close(fig)
    

        



if __name__ == "__main__":
    main()