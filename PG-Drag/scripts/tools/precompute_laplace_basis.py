#!/usr/bin/env python3
"""
Pre-compute Laplace-Beltrami low-frequency bases for APAP 3D meshes.

Usage (run from repo root):
    conda activate YAPAP
    PYTHONPATH=. python scripts/tools/precompute_laplace_basis.py \
        --data-root data/apap_3d/processed \
        --out-root data/apap_3d_laplace \
        --k 32
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

# Add parent directory to path to import load_obj
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.geometry_utils import load_obj


def compute_laplace_basis(v: np.ndarray, f: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the first k non-constant Laplace-Beltrami eigenmodes.
    
    Args:
        v: (N, 3) vertex positions (float64)
        f: (F, 3) face indices (int32/int64)
        k: number of non-constant eigenmodes to keep
    
    Returns:
        Phi: (N, k) eigenvectors (low-frequency basis)
        mass_diag: (N,) diagonal of the mass matrix
    """
    # Build cotan Laplacian manually (since igl might not be available)
    # Simple uniform Laplacian as fallback
    N = v.shape[0]
    F = f.shape[0]
    
    # Build adjacency using faces
    from collections import defaultdict
    edges = defaultdict(int)
    for tri in f:
        for i in range(3):
            v0, v1 = sorted([tri[i], tri[(i+1)%3]])
            edges[(v0, v1)] += 1
    
    # Build Laplacian matrix (uniform for simplicity)
    L = sp.lil_matrix((N, N))
    for (v0, v1), count in edges.items():
        L[v0, v1] = -1.0
        L[v1, v0] = -1.0
        L[v0, v0] += 1.0
        L[v1, v1] += 1.0
    
    L = sp.csr_matrix(L)
    
    # Build mass matrix (lumped, uniform)
    M = sp.diags(np.ones(N))
    
    # Debug: verify matrices are sparse
    if False:  # Enable for debugging
        print(f"    L: type={type(L)}, shape={L.shape}, nnz={L.nnz}")
        print(f"    M: type={type(M)}, shape={M.shape}, nnz={M.nnz}")
    
    n_eigs = min(k + 1, N - 2)  # include constant mode, but not more than available
    
    # Solve generalized eigenproblem: -L phi = lambda M phi
    # Use shift-invert mode to find smallest eigenvalues (most stable & fastest)
    try:
        # Correct shift-invert mode: A=-L, sigma=1e-6, which='LM', tol=1e-3
        evals, evecs = sla.eigsh(
            A=-L,           # Note: -L (negative Laplacian)
            M=M,            # Mass matrix
            k=n_eigs,       # Number of eigenvalues
            sigma=1e-6,     # Small shift to avoid singularity
            which='LM',     # Largest magnitude (with shift-invert = smallest eigenvalues)
            tol=1e-3,       # Loose tolerance (sufficient for low-freq basis)
            maxiter=None,   # Use default
        )
        # Note: eigenvalues will be negative, take abs
        evals = -evals
    except Exception as e:
        print(f"    Warning: eigsh with shift-invert failed: {e}")
        try:
            # Fallback 1: Direct 'SA' (slower but more robust)
            print(f"    Trying fallback: which='SA' (slower)")
            evals, evecs = sla.eigsh(L, k=n_eigs, M=M, which='SA', tol=1e-3)
        except Exception as e2:
            print(f"    Warning: eigsh with M failed: {e2}")
            # Fallback 2: Simple Laplacian without mass matrix
            print(f"    Last resort: simple Laplacian (no mass matrix)")
            evals, evecs = sla.eigsh(L, k=n_eigs, which='SA', tol=1e-3)
    
    # Sort eigenpairs by eigenvalue
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]
    
    # Drop the first (constant) mode, keep next k modes
    actual_k = min(k, n_eigs - 1)
    Phi = evecs[:, 1 : 1 + actual_k].astype(np.float32)
    mass_diag = np.ones(N, dtype=np.float32)
    
    return Phi, mass_diag


def main():
    parser = argparse.ArgumentParser(
        description="Precompute Laplace-Beltrami basis for APAP 3D meshes"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="APAP 3D data root directory (e.g., data/apap_3d/processed)"
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="data/apap_3d_laplace",
        help="Output directory for npz files"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=32,
        help="Number of low-frequency modes to compute"
    )
    args = parser.parse_args()
    
    os.makedirs(args.out_root, exist_ok=True)
    
    # Check if data_root itself contains mesh.obj (single object mode)
    single_mesh = os.path.join(args.data_root, "mesh.obj")
    if os.path.exists(single_mesh):
        # Single object mode
        obj_name = os.path.basename(os.path.abspath(args.data_root))
        print(f"[laplace] processing {obj_name} (single object mode)")
        
        try:
            # Use same load_obj as deform_meshes.py to ensure vertex count matches
            v, f, _, _, _, _, _, _ = load_obj(Path(single_mesh))
            v = v.astype(np.float64)
            f = f.astype(np.int32)
            
            Phi, mass_diag = compute_laplace_basis(v, f, k=args.k)
            
            out_path = os.path.join(args.out_root, f"{obj_name}.npz")
            np.savez(out_path, phi=Phi, mass=mass_diag)
            print(f"  -> saved {out_path} (phi: {Phi.shape}, mass: {mass_diag.shape})")
        except Exception as e:
            print(f"  -> ERROR: {e}")
    else:
        # Multi-object mode: iterate through subdirectories
        for obj_name in sorted(os.listdir(args.data_root)):
            obj_dir = os.path.join(args.data_root, obj_name)
            if not os.path.isdir(obj_dir):
                continue
            
            mesh_path = os.path.join(obj_dir, "mesh.obj")
            if not os.path.exists(mesh_path):
                print(f"[skip] no mesh.obj for {obj_name}")
                continue
            
            print(f"[laplace] processing {obj_name}")
            
            try:
                # Use same load_obj as deform_meshes.py to ensure vertex count matches
                v, f, _, _, _, _, _, _ = load_obj(Path(mesh_path))
                v = v.astype(np.float64)
                f = f.astype(np.int32)
                
                Phi, mass_diag = compute_laplace_basis(v, f, k=args.k)
                
                out_path = os.path.join(args.out_root, f"{obj_name}.npz")
                np.savez(out_path, phi=Phi, mass=mass_diag)
                print(f"  -> saved {out_path} (phi: {Phi.shape}, mass: {mass_diag.shape})")
            except Exception as e:
                print(f"  -> ERROR: {e}")
                continue

if __name__ == "__main__":
    main()
