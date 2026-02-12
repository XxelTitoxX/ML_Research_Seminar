"""Frechet Inception Distance computed from features.

Inputs:
  train_feat: (N, D) features from training set
  gen_feat: (M, D) features from generated samples
"""

from __future__ import annotations

import torch

from .metrics_common import sqrtm_product, stats_from_features, to_torch


def compute_fid(train_feat, gen_feat) -> float:
    mu1, cov1 = stats_from_features(train_feat)
    mu2, cov2 = stats_from_features(gen_feat)

    diff = (mu1 - mu2)
    cov_sqrt = sqrtm_product(cov1, cov2)

    fid = diff.dot(diff) + torch.trace(cov1 + cov2 - 2.0 * cov_sqrt)
    return float(fid)
