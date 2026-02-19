import math
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from models.catflow.mlp import MLP


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_fake_categorical_data(
    n_samples: int,
    d: int,
    k: int,
) -> torch.Tensor:
    """
    Create a simple multi-modal categorical dataset:
    all dims are class 0 or class K-1 except one feature.
    """
    if k < 2:
        raise ValueError("k must be >= 2.")
    data = torch.zeros((n_samples, d), dtype=torch.long)
    for i in range(n_samples):
        base_class = 0 if (i % 2 == 0) else (k - 1)
        special_dim = i % d
        data[i].fill_(base_class)
        data[i, special_dim] = (k - 1) if base_class == 0 else 0
    return data


def to_one_hot(c: torch.Tensor, k: int) -> torch.Tensor:
    # c: [B, D] -> one hot: [B, D, K]
    return F.one_hot(c, num_classes=k).float()


def catflow_loss(
    model: torch.nn.Module,
    c1: torch.Tensor,
    k: int,
    device: torch.device,
) -> torch.Tensor:
    # c1: [B, D] int
    b, d = c1.shape
    x1 = to_one_hot(c1, k).view(b, d * k).to(device)  # [B, M]
    x0 = torch.randn_like(x1)
    t = torch.rand((b, 1), device=device)
    x_t = t * x1 + (1.0 - t) * x0
    model_in = torch.cat([x_t, t], dim=1)
    logits = model(model_in).view(b, d, k)
    loss = F.cross_entropy(logits.view(b * d, k), c1.view(b * d).to(device))
    return loss


@torch.no_grad()
def sample_catflow(
    model: torch.nn.Module,
    n_samples: int,
    d: int,
    k: int,
    n_steps: int = 50,
    eps: float = 1e-3,
    device: torch.device | None = None,
) -> torch.Tensor:
    device = device or torch.device("cpu")
    m = d * k
    x = torch.randn((n_samples, m), device=device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((n_samples, 1), i * dt, device=device)
        model_in = torch.cat([x, t], dim=1)
        logits = model(model_in).view(n_samples, d, k)
        mu = F.softmax(logits, dim=-1).view(n_samples, m)
        v = (mu - x) / (1.0 - t + eps)
        x = x + dt * v
    x = x.view(n_samples, d, k)
    c_hat = torch.argmax(x, dim=-1)
    return c_hat.cpu()


def main() -> None:
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data + model params
    n_samples = 2000
    d = 4
    k = 10
    hidden_dim = 128
    num_layers = 3
    batch_size = 256
    n_steps = 4
    lr = 1e-3

    # Fake data
    data = make_fake_categorical_data(n_samples, d, k)

    # Model: input is x_t concat t, output is logits for all categories
    input_dim = d * k + 1
    output_dim = d * k
    model = MLP(input_dim, hidden_dim, output_dim, num_layers=num_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for step in range(1, n_steps + 1):
        idx = torch.randint(0, n_samples, (batch_size,))
        batch = data[idx].to(device)
        loss = catflow_loss(model, batch, k, device)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"step {step:04d} | loss {loss.item():.4f}")

    # Generation
    model.eval()
    n_gen_samples = 1000
    gen = sample_catflow(model, n_gen_samples, d, k, n_steps=60, device=device)

    # Plot first two dims (integer classes)
    train_xy = data[:, :2].numpy()
    gen_xy = gen[:, :2].numpy()

    plt.figure(figsize=(6, 5))
    plt.scatter(
        train_xy[:, 0],
        train_xy[:, 1],
        s=18,
        alpha=0.6,
        label="train",
        marker="o",
    )
    plt.scatter(
        gen_xy[:, 0],
        gen_xy[:, 1],
        s=18,
        alpha=0.6,
        label="generated",
        marker="x",
    )
    plt.xlabel("dim 1 (class)")
    plt.ylabel("dim 2 (class)")
    plt.xticks(range(k))
    plt.yticks(range(k))
    plt.legend()
    plt.tight_layout()

    os.makedirs("runs", exist_ok=True)
    out_path = os.path.join("runs", "catflow_example.png")
    plt.savefig(out_path, dpi=150)
    print(f"saved plot to {out_path}")


if __name__ == "__main__":
    main()
