"""Precision/Recall in feature space using k-NN radii.

Inputs:
  train_feat: (N, D) features from training set
  gen_feat: (M, D) features from generated samples
"""

from __future__ import annotations

import numpy as np
import torch

from .metrics_common import to_torch


def _knn_radius(x: torch.Tensor, k: int, batch_size: int = 1024) -> torch.Tensor:
    n = x.shape[0]
    if k >= n:
        raise ValueError(f"k={k} must be smaller than number of samples {n}")
    radii = torch.empty(n, dtype=x.dtype)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        d = torch.cdist(x[start:end], x)
        # Exclude self distances
        rows = torch.arange(start, end)
        d[torch.arange(end - start), rows] = float("inf")
        knn = torch.topk(d, k, largest=False).values[:, -1]
        radii[start:end] = knn
    return radii


def compute_precision_recall(
    train_feat,
    gen_feat,
    k: int = 3,
    subset_size: int = 5000,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    train_feat = to_torch(train_feat)
    gen_feat = to_torch(gen_feat)

    n_train = train_feat.shape[0]
    n_gen = gen_feat.shape[0]
    n_sub = min(subset_size, n_train)
    idx = rng.choice(n_train, size=n_sub, replace=False)
    train_sub = train_feat[idx]

    k = min(k, n_sub - 1, n_gen - 1)
    if k <= 0:
        raise ValueError("Not enough samples to compute precision/recall.")

    radius_train = _knn_radius(train_sub, k)
    radius_gen = _knn_radius(gen_feat, k)

    # Precision: fraction of gen within train manifold
    prec_hits = 0
    for start in range(0, n_gen, 512):
        end = min(start + 512, n_gen)
        d = torch.cdist(gen_feat[start:end], train_sub)
        hit = (d <= radius_train[None, :]).any(dim=1)
        prec_hits += hit.sum().item()
    precision = prec_hits / n_gen

    # Recall: fraction of train within gen manifold
    rec_hits = 0
    for start in range(0, n_sub, 512):
        end = min(start + 512, n_sub)
        d = torch.cdist(train_sub[start:end], gen_feat)
        hit = (d <= radius_gen[None, :]).any(dim=1)
        rec_hits += hit.sum().item()
    recall = rec_hits / n_sub

    return float(precision), float(recall)
