"""Memorization/generalization metric: mean NN1/NN2 distance ratio."""

from __future__ import annotations

import numpy as np
import torch

from .metrics_common import to_torch


def compute_generalization_metric(
    train_feat,
    gen_feat,
    subset_size: int = 5000,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    train_feat = to_torch(train_feat)
    gen_feat = to_torch(gen_feat)

    n_train = train_feat.shape[0]
    n_gen = gen_feat.shape[0]
    n_sub = min(subset_size, n_train)
    idx = rng.choice(n_train, size=n_sub, replace=False)
    train_sub = train_feat[idx]

    ratios = []
    for start in range(0, n_gen, 512):
        end = min(start + 512, n_gen)
        d = torch.cdist(gen_feat[start:end], train_sub)
        nn2 = torch.topk(d, 2, largest=False).values
        ratio = (nn2[:, 0] / (nn2[:, 1] + 1e-12)).cpu().numpy()
        ratios.append(ratio)
    return float(np.mean(np.concatenate(ratios)))
