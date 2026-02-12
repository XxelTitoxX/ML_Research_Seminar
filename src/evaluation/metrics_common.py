"""Common helpers for evaluation metrics (no external FID libraries)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def to_torch(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return torch.from_numpy(x)


def _cov(x: torch.Tensor) -> torch.Tensor:
    # x: (N, D)
    x = x - x.mean(dim=0, keepdim=True)
    n = x.shape[0]
    return (x.T @ x) / (n - 1)


def _sqrtm_psd(mat: torch.Tensor) -> torch.Tensor:
    """Matrix square root for symmetric PSD matrices using eigen-decomposition."""
    # Ensure symmetry
    mat = (mat + mat.T) * 0.5
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = torch.clamp(eigvals, min=0.0)
    sqrt_eigvals = torch.sqrt(eigvals)
    return (eigvecs * sqrt_eigvals) @ eigvecs.T


def sqrtm_product(cov1: torch.Tensor, cov2: torch.Tensor) -> torch.Tensor:
    """Compute sqrtm(cov1 @ cov2) in a numerically stable way.

    Tries scipy.linalg.sqrtm if available; otherwise falls back to eigen-based PSD sqrt.
    """
    try:
        import scipy.linalg  # type: ignore

        mat = (cov1 @ cov2).cpu().numpy()
        sqrtm = scipy.linalg.sqrtm(mat)
        # Numerical errors can introduce small imaginary parts
        if np.iscomplexobj(sqrtm):
            sqrtm = sqrtm.real
        return torch.from_numpy(sqrtm)
    except Exception:
        # Fallback: symmetric product approximation
        prod = cov1 @ cov2
        return _sqrtm_psd(prod)


def stats_from_features(features: torch.Tensor | np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    feat = to_torch(features).float()
    mean = feat.mean(dim=0)
    cov = _cov(feat)
    return mean, cov
