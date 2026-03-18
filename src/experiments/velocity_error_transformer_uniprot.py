import os
import sys
import torch
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
    
def load_uniprot(data_root: str, batch_size: int, train: bool = True, shuffle: bool = True) -> DataLoader:
    split = "train" if train else "test"
    data_path = Path(data_root) / f"uniprot_{split}.pt"
    data = torch.load(data_path, map_location="cpu")
    indices = data["indices"]  # shape: (N, seq_len)
    seq_len = indices.shape[1]
    num_classes = 22
    dataset = TensorDataset(indices)
    def collate_fn(batch):
        indices_batch = torch.stack([item[0] for item in batch]).to(dtype=torch.long)  # shape: (batch_size, seq_len)
        one_hot_batch = F.one_hot(indices_batch, num_classes=num_classes).float()  # shape: (batch_size, seq_len, num_classes)
        return one_hot_batch
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def main():
    torch.backends.cudnn.benchmark = True
    device = pick_device()

    flow, model_cfg, obs_dim = load_catflow_checkpoint(
        flow_checkpoint_path=CHECKPOINTS / "step_25000.pt",
        device=device,
    )
    flow.eval()

    cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    
    n_steps = 20
    clamp_t = 0.01
    n_xt_samples = 128
    batch_size_xt = 8
    batch_size_x1prime = 128
    ts = torch.linspace(0, 1.0 - clamp_t, n_steps, device=device)

    # Create dataloader of sequences of one-hot encoded codebook indices
    # DO NOT USE SHUFFLE for x1' dataloader !
    dataloader_xt = load_uniprot(DATA_PROCESSED, batch_size=batch_size_xt, train=True, shuffle=True)
    dataloader_x1prime = load_uniprot(DATA_PROCESSED, batch_size=batch_size_x1prime, train=True, shuffle=False)

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
    
    plt.figure(figsize=(8, 5))
    plt.plot(ts.cpu(), error_t.cpu(), marker='o')
    plt.title("Velocity Error vs Time")
    plt.xlabel("Time (t)")
    plt.ylabel("Average L2 Error")
    plt.grid()
    plt.savefig("velocity_error.png")

    plt.figure(figsize=(8, 5))
    plt.plot(ts.cpu(), error_t.cpu(), marker='o')
    plt.title("Velocity Error vs Time")
    plt.xlabel("Time (t)")
    plt.ylabel("Average L2 Error")
    plt.ylim(0, error_t[:len(ts)//2].max().item() * 1.1)  # zoom in on first half of time steps
    plt.grid()
    plt.savefig("velocity_error_zoomed.png")

        



if __name__ == "__main__":
    main()