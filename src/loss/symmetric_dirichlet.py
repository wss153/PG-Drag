"""
Symmetric Dirichlet regularization for Stage 2 (SSP alternative).

References:
  - Smith & Schaefer, Bijective Parameterization with Free Boundaries, TOG 2015
  - Rabinovich et al., SLIM, TOG 2017
  - libigl MappingEnergyType::SYMMETRIC_DIRICHLET
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor


def precompute_rest_triangle_areas(
    v0: Tensor,
    faces: Tensor,
) -> Tensor:
    """Area of each rest triangle (detached, used as fixed weights)."""
    v0 = v0.detach()
    e1 = v0[faces[:, 1]] - v0[faces[:, 0]]
    e2 = v0[faces[:, 2]] - v0[faces[:, 0]]
    return 0.5 * torch.linalg.norm(torch.cross(e1, e2, dim=-1), dim=-1)


def per_face_symmetric_dirichlet_energy(
    jacobians: Tensor,
    eps: float = 1e-3,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Per-face Symmetric Dirichlet energy from 3x3 surface Jacobians.

    Args:
        jacobians: [F, 3, 3] from reconstructed mesh vertices.
        eps: clamp for singular values.

    Returns:
        E_f: [F] per-face energy (identity => 0).
        sigma1, sigma2: [F] top two singular values.
    """
    s = torch.linalg.svdvals(jacobians)
    sigma1 = torch.clamp(s[:, 0], min=eps)
    sigma2 = torch.clamp(s[:, 1], min=eps)
    e_f = (
        0.5 * (
            sigma1.pow(2)
            + sigma2.pow(2)
            + sigma1.pow(-2)
            + sigma2.pow(-2)
        )
        - 2.0
    )
    return e_f, sigma1, sigma2


def area_weighted_mean(values: Tensor, areas: Tensor) -> Tensor:
    """sum(A * v) / sum(A)."""
    denom = areas.sum().clamp_min(1e-12)
    return (areas * values).sum() / denom


class SymmetricDirichletRegularizer:
    """Stage-2 SD loss using reconstructed mesh Jacobians."""

    def __init__(
        self,
        poisson,
        rest_areas: Tensor,
        eps: float = 1e-3,
    ) -> None:
        self.poisson = poisson
        self.rest_areas = rest_areas.detach()
        self.eps = eps

    def jacobians_from_vertices(self, curr_v: Tensor) -> Tensor:
        """APAP-consistent per-triangle Jacobian from reconstructed vertices."""
        v_query = curr_v.to(torch.float64)
        return self.poisson.compute_per_triangle_jacobian(v_query)

    def forward(
        self,
        curr_v: Tensor,
        return_per_face: bool = False,
    ) -> Tuple[Tensor, Dict[str, float]]:
        j_cur = self.jacobians_from_vertices(curr_v)
        e_f, sigma1, sigma2 = per_face_symmetric_dirichlet_energy(j_cur, eps=self.eps)
        l_sd = area_weighted_mean(e_f, self.rest_areas)

        sigma_min = torch.minimum(sigma1, sigma2)
        stats = {
            "l_sd": float(l_sd.detach().item()),
            "e_f_mean": float(e_f.detach().mean().item()),
            "sigma_min": float(sigma_min.min().detach().item()),
            "sigma_min_mean": float(sigma_min.mean().detach().item()),
        }
        if return_per_face:
            stats["e_f"] = e_f.detach()
            stats["sigma1"] = sigma1.detach()
            stats["sigma2"] = sigma2.detach()
        return l_sd, stats

    def __call__(self, curr_v: Tensor, return_per_face: bool = False):
        return self.forward(curr_v, return_per_face=return_per_face)


def grad_norm_on_poisson_j(
    poisson,
    loss: Tensor,
) -> float:
    """||grad_J loss|| for diagnostic logging."""
    if not poisson.J.requires_grad:
        return 0.0
    if poisson.J.grad is not None:
        poisson.J.grad.zero_()
    loss.backward(retain_graph=True)
    if poisson.J.grad is None:
        return 0.0
    return float(poisson.J.grad.norm().item())
