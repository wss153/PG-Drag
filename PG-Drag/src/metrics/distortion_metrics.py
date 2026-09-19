"""Distortion evaluation metrics (SD-P99, etc.)."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from src.loss.symmetric_dirichlet import (
    area_weighted_mean,
    per_face_symmetric_dirichlet_energy,
    precompute_rest_triangle_areas,
)


@torch.no_grad()
def compute_sd_p99(
    poisson,
    curr_v: Tensor,
    rest_areas: Tensor,
    eps: float = 1e-3,
) -> Dict[str, float]:
    """
    99th percentile of per-face Symmetric Dirichlet energy (evaluation only).
    """
    j_cur = poisson.compute_per_triangle_jacobian(curr_v.to(torch.float64))
    e_f, sigma1, sigma2 = per_face_symmetric_dirichlet_energy(j_cur, eps=eps)
    e_f_np = e_f.detach().cpu().float().numpy()
    p99 = float(torch.quantile(e_f.float(), 0.99).item())
    sigma_min = torch.minimum(sigma1, sigma2)
    return {
        "sd_p99": p99,
        "sd_mean": float(area_weighted_mean(e_f, rest_areas).item()),
        "sigma_min_global": float(sigma_min.min().item()),
        "sigma_min_p99": float(torch.quantile(sigma_min.float(), 0.99).item()),
        "e_f_max": float(e_f.max().item()),
    }


def precompute_rest_areas_from_mesh(v0: Tensor, faces: Tensor) -> Tensor:
    return precompute_rest_triangle_areas(v0, faces)
