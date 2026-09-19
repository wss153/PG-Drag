"""
ROI (Region of Interest) mask utilities for SDS gradient masking around handles.

This module provides functions to compute vertex masks that reduce SDS influence
near handle vertices, preventing local artifacts while maintaining global shape guidance.
"""

from collections import deque
from typing import List, Optional, Union

import numpy as np
import torch


def add_adjacency_edges(
    neighbors: List[List[int]],
    src: Union[List[int], np.ndarray, torch.Tensor],
    dst: Union[List[int], np.ndarray, torch.Tensor],
) -> int:
    """Add undirected edges (virtual seams, etc.) to an adjacency list."""
    if isinstance(src, torch.Tensor):
        src_list = src.detach().cpu().tolist()
    else:
        src_list = np.asarray(src).reshape(-1).tolist()
    if isinstance(dst, torch.Tensor):
        dst_list = dst.detach().cpu().tolist()
    else:
        dst_list = np.asarray(dst).reshape(-1).tolist()

    n_added = 0
    n = len(neighbors)
    for a, b in zip(src_list, dst_list):
        a, b = int(a), int(b)
        if a == b or not (0 <= a < n and 0 <= b < n):
            continue
        if b not in neighbors[a]:
            neighbors[a].append(b)
            n_added += 1
        if a not in neighbors[b]:
            neighbors[b].append(a)
    return n_added


def build_vertex_adjacency(num_verts: int, faces: torch.Tensor) -> List[List[int]]:
    """
    Build vertex adjacency list from mesh faces.
    
    Args:
        num_verts: Number of vertices
        faces: (F, 3) long tensor, face indices
    
    Returns:
        neighbors: List of lists, neighbors[i] contains indices of vertices adjacent to vertex i
    """
    neighbors = [[] for _ in range(num_verts)]
    f = faces.cpu().numpy()
    
    for i, j, k in f:
        neighbors[i].append(j)
        neighbors[i].append(k)
        neighbors[j].append(i)
        neighbors[j].append(k)
        neighbors[k].append(i)
        neighbors[k].append(j)
    
    # Remove duplicates
    for idx in range(num_verts):
        if neighbors[idx]:
            neighbors[idx] = list(set(neighbors[idx]))
    
    return neighbors


def bfs_handle_hops(
    num_verts: int,
    neighbors: List[List[int]],
    handle_idx: Union[List[int], torch.Tensor],
    max_hops: int,
) -> torch.Tensor:
    """
    Compute hop distance from handle vertices using BFS.
    
    Args:
        num_verts: Number of vertices
        neighbors: Adjacency list from build_vertex_adjacency
        handle_idx: Handle vertex indices (list or tensor)
        max_hops: Maximum BFS distance to compute
    
    Returns:
        dist: (num_verts,) float tensor, hop distance from nearest handle.
              Vertices beyond max_hops are set to +inf
    """
    import math
    
    dist = [float('inf')] * num_verts
    q = deque()
    
    # Convert handle_idx to list of ints
    if isinstance(handle_idx, torch.Tensor):
        handle_list = handle_idx.cpu().tolist()
    else:
        handle_list = handle_idx
    
    # Initialize handle vertices
    for h in handle_list:
        h = int(h)
        if 0 <= h < num_verts:
            dist[h] = 0
            q.append(h)
    
    # BFS
    while q:
        v = q.popleft()
        if dist[v] >= max_hops:
            continue
        
        for nb in neighbors[v]:
            if dist[nb] == float('inf'):
                dist[nb] = dist[v] + 1
                if dist[nb] < max_hops:
                    q.append(nb)
    
    # Convert to tensor
    dist_t = torch.tensor(dist, dtype=torch.float32)
    return dist_t


def build_sds_vertex_mask(
    dist: torch.Tensor,
    core_hops: int,
    band_hops: int,
    min_weight: float = 0.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build soft SDS mask from hop distances (DEPRECATED - use ratio-based version).
    
    Args:
        dist: (V,) float tensor, hop distance from handles
        core_hops: Core region threshold (dist <= core_hops → min_weight)
        band_hops: Transition band end (dist >= band_hops → weight=1.0)
        min_weight: Minimum weight in core region (0.0 = fully masked)
        device: Target device for output tensor
    
    Returns:
        mask: (V,) float tensor, SDS weights in [min_weight, 1.0]
              - Core region (dist <= core_hops): min_weight
              - Transition band (core_hops < dist < band_hops): linear interpolation
              - Far region (dist >= band_hops): 1.0
    """
    if device is None:
        device = dist.device
    
    # Ensure dist is on the target device
    dist = dist.to(device)
    
    mask = torch.ones_like(dist, dtype=torch.float32, device=device)
    
    # Core region: fully masked (or min_weight)
    core = dist <= float(core_hops)
    mask[core] = float(min_weight)
    
    # Transition band: linear interpolation from min_weight to 1.0
    band = (dist > float(core_hops)) & (dist < float(band_hops))
    if band.any():
        d_band = dist[band]
        alpha = (d_band - float(core_hops)) / float(band_hops - core_hops)
        # Linear interpolation: min_weight -> 1.0
        mask[band] = min_weight + alpha * (1.0 - min_weight)
    
    # Far region (dist >= band_hops): mask=1.0 (already set)
    
    return mask


def build_sds_vertex_mask_from_ratio(
    hop_dist: torch.Tensor,
    core_ratio: float = 0.2,   # Target: Core ≈ 20%
    band_ratio: float = 0.5,   # Target: Core+Band ≈ 50%
) -> tuple[torch.Tensor, int, int]:
    """
    Build SDS mask using percentile-based thresholds (automatic ROI sizing).
    
    Instead of fixed hop thresholds, this computes Core/Band boundaries
    based on the distribution of hop distances, ensuring consistent coverage
    across different mesh topologies.
    
    Args:
        hop_dist: (N,) float tensor, hop distance from handles
                  +inf / NaN = unreachable (no mesh or virtual-seam path)
        core_ratio: Target ratio of vertices in Core region (fully masked)
        band_ratio: Target ratio of vertices in Core+Band regions
    
    Returns:
        vertex_weight: (N,) float tensor in [0,1], SDS weights
        d_core: int, actual Core hop threshold
        d_band: int, actual Band end hop threshold
    """
    device = hop_dist.device
    
    # Percentiles only on reachable verts. Unreachable stay weight=0 so SDS
    # cannot pin disconnected components against the handle pull.
    valid = torch.isfinite(hop_dist) & (hop_dist >= 0)
    d = hop_dist[valid].float()  # (Nv,)
    
    if d.numel() == 0:
        full_w = torch.zeros_like(hop_dist, dtype=torch.float32, device=device)
        return full_w, 0, 0
    
    # 1) Compute thresholds using quantiles
    core_ratio = float(core_ratio)
    band_ratio = float(band_ratio)
    # Safety clamps
    core_ratio = max(0.05, min(core_ratio, 0.45))
    band_ratio = max(core_ratio + 0.05, min(band_ratio, 0.95))
    
    d_core_q = torch.quantile(d, core_ratio)   # e.g., 20th percentile
    d_band_q = torch.quantile(d, band_ratio)   # e.g., 50th percentile
    
    d_core = int(d_core_q.item())
    d_band = int(d_band_q.item())
    if d_band <= d_core:
        d_band = d_core + 1
    
    # 2) Compute weights for valid vertices
    w_valid = torch.ones_like(d, dtype=torch.float32)  # Far default=1.0
    
    # Core: fully masked → 0
    core_mask = d <= d_core
    w_valid[core_mask] = 0.0
    
    # Band: 0 → 1 linear transition
    band_mask = (d > d_core) & (d <= d_band)
    if band_mask.any():
        w_valid[band_mask] = (d[band_mask] - d_core) / float(d_band - d_core)
    
    # 3) Fill back to full (N,) tensor. Unreachable default = 0 (masked).
    full_w = torch.zeros_like(hop_dist, dtype=torch.float32, device=device)
    full_w[valid] = w_valid
    
    return full_w, d_core, d_band

