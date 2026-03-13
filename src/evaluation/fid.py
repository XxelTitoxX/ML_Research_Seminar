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

def test_fid():
    train_dino_feat = torch.load("data/processed/cifar10_dinov2.pt", map_location="cpu")["features"]
    n_samples = 5000
    reference_subset = train_dino_feat[:5000]
    ref_v_ref_fid = compute_fid(reference_subset, reference_subset)
    print(f"FID of reference subset vs itself: {ref_v_ref_fid:.4f}")
    random_subset = train_dino_feat[torch.randperm(train_dino_feat.shape[0])[:n_samples]]
    ref_v_random_fid = compute_fid(reference_subset, random_subset)
    print(f"FID of reference subset vs random subset: {ref_v_random_fid:.4f}")

if __name__ == "__main__":
    test_fid()
