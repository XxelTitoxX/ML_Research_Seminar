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
sys.path.insert(0, str(ROOT))

from src.models.vq_model import VQ_Cifar_L
from src.experiments.closedform import u_star, u_star_with_labels
from src.experiments.utils import load_catflow_from_checkpoint

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
def load_cifar10(data_root: str, train: bool = True) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.CIFAR10(root=data_root, train=train, download=True, transform=transform)
    return dataset

def load_q_cifar10() -> TensorDataset:
    q_data = torch.load("data/processed/cifar10_q_train_l.pt", map_location="cpu")
    q_indices = q_data["indices"]  # shape: (N, seq_len)
    return TensorDataset(q_indices)

def load_q_cifar10_with_labels() -> TensorDataset:
    q_data = torch.load("data/processed/cifar10_q_train_l.pt", map_location="cpu")
    q_indices = q_data["indices"]  # shape: (N, seq_len)
    q_labels = q_data["labels"]  # shape: (N,)
    return TensorDataset(q_indices, q_labels)

def get_vec_dataloader(batch_size, codebook, shuffle: bool = True) -> DataLoader:
    dataset = load_q_cifar10()
    def collate_fn(batch):
        # batch is a list of tuples (q_indices,)
        q_indices_batch = torch.stack([item[0] for item in batch]).to(dtype=torch.long)  # shape: (batch_size, seq_len)
        codebook_vectors = codebook[q_indices_batch]  # shape: (batch_size, seq_len, codebook_dim)
        return codebook_vectors
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def get_vec_dataloader_with_labels(batch_size, codebook, shuffle: bool = True) -> DataLoader:
    dataset = load_q_cifar10_with_labels()
    def collate_fn(batch):
        # batch is a list of tuples (q_indices, q_labels)
        q_indices_batch = torch.stack([item[0] for item in batch]).to(dtype=torch.long)  # shape: (batch_size, seq_len)
        q_labels_batch = torch.stack([item[1] for item in batch])  # shape: (batch_size,)
        codebook_vectors = codebook[q_indices_batch]  # shape: (batch_size, seq_len, codebook_dim)
        return codebook_vectors, q_labels_batch
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def get_q_loader(batch_size, codebook_size, shuffle: bool = True) -> DataLoader:
    dataset = load_q_cifar10()
    def collate_fn(batch):
        # one-hot encode the q_indices
        q_indices_batch = torch.stack([item[0] for item in batch]).to(dtype=torch.long)  # shape: (batch_size, seq_len)
        one_hot = F.one_hot(q_indices_batch.view(-1), num_classes=codebook_size).view(*q_indices_batch.shape, -1)  # shape: (batch_size, seq_len, codebook_size)
        return one_hot
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def get_q_loader_with_labels(batch_size, codebook_size, shuffle: bool = True) -> DataLoader:
    dataset = load_q_cifar10_with_labels()
    def collate_fn(batch):
        # batch is a list of tuples (q_indices, q_labels)
        q_indices_batch = torch.stack([item[0] for item in batch]).to(dtype=torch.long)  # shape: (batch_size, seq_len)
        q_labels_batch = torch.stack([item[1] for item in batch])  # shape: (batch_size,)
        one_hot = F.one_hot(q_indices_batch.view(-1), num_classes=codebook_size).view(*q_indices_batch.shape, -1)  # shape: (batch_size, seq_len, codebook_size)
        return one_hot, q_labels_batch
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def maybe_prepare_quantized_dataset(train:bool, device: torch.device, vq_model):
    q_path = Path("data/processed/cifar10_q_{}_l.pt".format("train" if train else "test"))

    if q_path.exists():
        print(f"[data] Using cached quantized dataset: {q_path}")
        return torch.load(q_path, map_location="cpu")

    print("[data] Quantized dataset not found. Loading CIFAR-10 and quantizing...")
    dataset = load_cifar10("data/raw", train=train)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, drop_last=False)

    all_indices: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    latent_h: int | None = None
    latent_w: int | None = None

    vq_model.eval()

    pbar = tqdm(loader, desc="Quantizing CIFAR-10", unit="batch")
    print_perc = 0.0
    n_batch = 0
    with torch.inference_mode():
        for images, labels in pbar:
            images = images.to(device)
            quant, _, info = vq_model.encode(images)
            b, _, h, w = quant.shape
            if latent_h is None:
                latent_h, latent_w = h, w
            batch_indices = info[2].view(b, h * w)  # shape: (batch_size, seq_len)
            all_indices.append(batch_indices.cpu().to(torch.int16))
            all_labels.append(labels.cpu())
            n_batch += 1
    if latent_h is None or latent_w is None:
        raise RuntimeError("No CIFAR-10 data was processed during quantization.")

    indices = torch.cat(all_indices, dim=0).contiguous()
    payload = {
        "indices": indices,  # [N, D], int16
        "labels": torch.cat(all_labels, dim=0),  # [N], int64
        "latent_h": latent_h,
        "latent_w": latent_w,
        "codebook_size": int(vq_model.config.codebook_size),
        "codebook_embed_dim": int(vq_model.config.codebook_embed_dim),
    }
    torch.save(payload, q_path)
    print(f"[data] Saved quantized CIFAR-10 to: {q_path}")
    return payload

def main():
    torch.backends.cudnn.benchmark = True
    device = pick_device()

    vq_model = VQ_Cifar_L().to(device)
    vq_model.load_state_dict(torch.load("checkpoints/vq_cifar_epoch_20.pt", map_location=device)["model_state_dict"])
    vq_model.eval()

    maybe_prepare_quantized_dataset(train=True, device=device, vq_model=vq_model)

    flow, model_cfg, obs_dim = load_catflow_from_checkpoint(
        flow_checkpoint_path=Path("checkpoints/step_25000.pt"),
        vq_model=vq_model,
        device=device,
        sigma_min=1e-6,
    )
    flow.eval()

    cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    codebook_size = vq_model.config.codebook_size
    
    n_steps = 20
    clamp_t = 0.01
    n_xt_samples = 128
    batch_size_xt = 8
    batch_size_x1prime = 128
    ts = torch.linspace(0, 1.0 - clamp_t, n_steps, device=device)

    # Create dataloader of sequences of one-hot encoded codebook indices
    # DO NOT USE SHUFFLE for x1' dataloader !
    dataloader_xt = get_q_loader_with_labels(batch_size=batch_size_xt, codebook_size=codebook_size, shuffle=True)
    dataloader_x1prime = get_q_loader(batch_size=batch_size_x1prime, codebook_size=codebook_size, shuffle=False)

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
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(ts.cpu(), error_t.cpu(), marker='o')
    plt.title("Velocity Error vs Time")
    plt.xlabel("Time (t)")
    plt.ylabel("Average L2 Error")
    plt.ylim(0, error_t[:len(ts)//2].max().item() * 1.1)  # zoom in on first half of time steps
    plt.grid()
    plt.savefig("velocity_error.png")
    plt.show()

        



if __name__ == "__main__":
    main()