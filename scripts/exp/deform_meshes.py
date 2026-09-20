"""
deform_meshes.py

A script for mesh deformation experiments.
"""

from dataclasses import dataclass, field, fields
import json
from PIL import Image
from pathlib import Path
from typing import Any, List, Optional, Union

import imageio
from jaxtyping import jaxtyped
import numpy as np
import torch
from typeguard import typechecked
import tyro
import wandb

from configs.deform_meshes.pipeline import (
    StableDiffusion2DPipeConfig,
    StableDiffusion3DPipeConfig,
)
from src.geometry.poisson_system import PoissonSystem
from src.geometry.virtual_seams import (
    assert_original_geometry,
    build_virtual_seams,
    write_debug_visualization,
    write_seam_gap_curve,
)
from src.guidance.stable_diffusion import StableDiffusionGuidance
from src.renderer.camera import compute_lookat_mat, compute_proj_mat
from src.renderer.nvdiffrast import render, render_depth
from src.utils.geometry_utils import load_obj, save_obj
from src.utils.math_utils import (
    normalize_vs,
    quat_to_mat_torch,
)
from src.utils.random_utils import seed_everything
from src.utils.vis_utils import (
    render_mesh_360,
    render_mesh_with_markers_360,
)
from src.utils.smooth_arap_loss import SmoothARAP
from src.prior.sparse_shape_prior import (
    TemplateCache,
    SparseShapePrior,
    sparse_shape_hyper_from_config,
)
from src.utils.roi_mask import (
    add_adjacency_edges,
    build_vertex_adjacency,
    bfs_handle_hops,
    build_sds_vertex_mask,
    build_sds_vertex_mask_from_ratio,
)
from src.renderer.nvdiffrast import render_vertex_attribute
from src.loss.symmetric_dirichlet import (
    SymmetricDirichletRegularizer,
    grad_norm_on_poisson_j,
    precompute_rest_triangle_areas,
)
from src.metrics.distortion_metrics import compute_sd_p99


def compute_handle_error(
    verts: torch.Tensor,
    handle_idx: torch.Tensor,
    handle_target: torch.Tensor
) -> torch.Tensor:
    """
    Compute average L2 error between handle vertices and their target positions.
    
    Args:
        verts: (V, 3) current vertices
        handle_idx: (H,) long, handle vertex indices
        handle_target: (H, 3) handle target positions
    
    Returns:
        Scalar tensor, average L2 handle error
    """
    v_h = verts[handle_idx]  # (H, 3)
    err = torch.mean(torch.sum((v_h - handle_target) ** 2, dim=-1))
    return err


@dataclass
class Args:

    mesh_file: Path
    """A mesh file to deform"""
    handle_file: Path
    """A file holding handle indices and target positions"""
    anchor_file: Path
    """A file holding anchor indices and target positions"""
    out_dir: Path
    """Output directory"""
    wandb_grp: str = "deform_meshes"
    """W&B group name"""
    lora_dir: Optional[Path] = None
    """Path to LoRA checkpoint"""
    cam_radius: float = 1.5
    """Distance of SDS cameras from the origin (default 1.5 matches stock cam_locs)"""
    vis_radius: Optional[float] = None
    """Camera radius for 360 visualization. Defaults to cam_radius"""
    vis_img_size: int = 512
    """Resolution of 360 visualization / video frames"""
    vis_bg_color: Optional[tuple[float, float, float]] = None
    """Fixed RGB background for 360 renders and SDS (overrides use_random_bg when set)"""

    pipe_cfg: Union[
        StableDiffusion2DPipeConfig,
        StableDiffusion3DPipeConfig,
    ] = StableDiffusion3DPipeConfig()
    """Config for the experiment pipeline"""


@jaxtyped(typechecker=typechecked)
def main(args: Args) -> None:

    # reproducibility
    seed_everything(args.pipe_cfg.seed)

    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {str(args.out_dir.resolve())}")

    locs = args.pipe_cfg.cam_locs.float()
    norms = torch.linalg.norm(locs, dim=-1, keepdim=True).clamp(min=1e-8)
    args.pipe_cfg.cam_locs = locs / norms * float(args.cam_radius)
    vis_radius = (
        float(args.vis_radius)
        if args.vis_radius is not None
        else float(args.cam_radius)
    )
    vis_h = vis_w = int(args.vis_img_size)
    print(
        f"SDS cam_radius={args.cam_radius} "
        f"cam_locs={args.pipe_cfg.cam_locs.tolist()}"
    )
    print(f"vis radius={vis_radius} size={vis_h}x{vis_w}")

    vis_bg = torch.ones(3, device=args.pipe_cfg.device)
    if args.vis_bg_color is not None:
        vis_bg = torch.tensor(args.vis_bg_color, dtype=torch.float32, device=args.pipe_cfg.device)
        print(f"vis/sds background RGB={args.vis_bg_color}")

    # Save experiment config
    # TODO: recursively save guidance config
    with open(args.out_dir / "config.txt", mode="w") as file:
        for item in fields(args):
            name = item.name
            value = getattr(args, name)
            file.write(f"{name}: {value}\n")

    # Load mesh & metadata
    assert args.mesh_file.exists(), f"Not found: {str(args.mesh_file)}"
    (
        v, f, vcs, uvs, vns, tex_inds, vn_inds, tex
    ) = load_obj(args.mesh_file)
    v = torch.from_numpy(v).to(args.pipe_cfg.device)
    f = torch.from_numpy(f).to(args.pipe_cfg.device)
    uvs = torch.from_numpy(uvs).to(args.pipe_cfg.device)
    tex_inds = torch.from_numpy(tex_inds).to(args.pipe_cfg.device)
    tex = torch.from_numpy(tex).to(args.pipe_cfg.device)
    with open(args.mesh_file.parent / "metadata.json") as file:
        mesh_metadata = json.load(file)

    rest_v_np = v.detach().cpu().numpy()
    rest_f_np = f.detach().cpu().numpy()
    uvs_np = None if uvs is None else uvs.detach().cpu().numpy()
    tex_inds_np = None if tex_inds is None else tex_inds.detach().cpu().numpy()
    assert_original_geometry(
        rest_v_np,
        rest_f_np,
        uvs_np,
        tex_inds_np,
        args.mesh_file.parent / "original_geometry.json",
    )

    # Load handles
    handle_inds, handle_pos = [], []
    with open(args.handle_file, "r") as file:
        for l in file.readlines():
            ind, x, y, z = l.split()
            handle_inds.append(int(ind))
            handle_pos.append([float(x), float(y), float(z)])
    handle_inds = torch.tensor(handle_inds).to(args.pipe_cfg.device)
    handle_pos = torch.tensor(handle_pos, dtype=torch.float32).to(args.pipe_cfg.device)

    # Load anchors
    anchor_inds, anchor_pos = [], []
    with open(args.anchor_file, "r") as file:
        for l in file.readlines():
            ind, x, y, z = l.split()
            anchor_inds.append(int(ind))
            anchor_pos.append([float(x), float(y), float(z)])
    anchor_inds = torch.tensor(anchor_inds).to(args.pipe_cfg.device)
    anchor_pos = torch.tensor(anchor_pos, dtype=torch.float32).to(args.pipe_cfg.device)

    # Initialize W&B
    handle_id = str(args.handle_file.parent.stem)
    anchor_id = str(args.anchor_file.parent.stem)
    wandb_name = (
        f"mesh-{mesh_metadata['object_name']}_handle-{handle_id}_anchor-{anchor_id}"
    )
    wandb.init(
        project="apap",
        group=args.wandb_grp,
        name=wandb_name,
        save_code=True,
    )

    # Initialize video writer
    vid_dir = args.out_dir / "video"
    vid_dir.mkdir(parents=True, exist_ok=True)
    vid_writer = imageio.get_writer(
        vid_dir / "optim.mp4",
        format="FFMPEG",
        mode="I",
        fps=24,
        macro_block_size=1,
    )

    # Initialize guidance
    guidance = StableDiffusionGuidance(
        device=args.pipe_cfg.device,
        lora_path=args.lora_dir,
        lora_scale=args.pipe_cfg.lora_scale,
        grad_clamp_val=args.pipe_cfg.clamp_val,
    )
    print(f"Loaded guidance: {type(guidance)}")

    # Determine guidance prompt
    prompt = f"a photo of {mesh_metadata['object_name']}"
    if not args.lora_dir is None:
        ckpt_name = args.lora_dir.stem
        with open(args.lora_dir / "metadata.json") as file:
            lora_metadata = json.load(file)
        prompt = (
            f"a photo of {lora_metadata['special_token']} {lora_metadata['object_name']}"
        )
    print(f"Guidance Prompt: {str(prompt)}")

    train_v = args.pipe_cfg.train_V
    if train_v and args.pipe_cfg.train_J:
        raise ValueError("train_V and train_J are mutually exclusive")
    if train_v:
        args.pipe_cfg.train_J = False

    coupling = None
    poisson_anchor_inds = anchor_inds
    poisson_constraint_pos = anchor_pos

    hard_handle = args.pipe_cfg.hard_handle
    if hard_handle:
        overlap = set(handle_inds.detach().cpu().tolist()) & set(anchor_inds.detach().cpu().tolist())
        if overlap:
            raise ValueError(f"hard_handle: handle/anchor index overlap: {overlap}")
        poisson_anchor_inds = torch.cat([poisson_anchor_inds, handle_inds], dim=0)
        poisson_constraint_pos = torch.cat(
            [poisson_constraint_pos, handle_pos.to(dtype=poisson_constraint_pos.dtype)],
            dim=0,
        )
        args.pipe_cfg.enable_freeze = False
        print(
            f"[HardHandle] Poisson constraints: {len(anchor_inds)} anchor(s) + "
            f"{len(handle_inds)} handle(s) = {len(poisson_anchor_inds)} total"
        )
        print("[HardHandle] Soft handle loss disabled; Handle Freeze disabled")

    if args.pipe_cfg.virtual_seams:
        object_name = str(mesh_metadata.get("object_name", ""))
        building_tau = 0.06 if object_name.lower() == "building" else None
        is_building = object_name.lower() == "building"
        coupling = build_virtual_seams(
            rest_v_np,
            rest_f_np,
            handle_inds.detach().cpu().numpy(),
            anchor_inds.detach().cpu().numpy(),
            tau=args.pipe_cfg.seam_tau,
            tau_frac=args.pipe_cfg.seam_tau_frac,
            gap_delta_frac=args.pipe_cfg.seam_gap_delta_frac,
            building_tau=building_tau,
            per_pair_delta=is_building,
            dense_n_min=args.pipe_cfg.seam_dense_min,
            dense_n_max=args.pipe_cfg.seam_dense_max,
            focus_largest_major=is_building and args.pipe_cfg.seam_focus_largest_major,
            lambda_group=args.pipe_cfg.lambda_group,
        )
        write_debug_visualization(
            args.out_dir / "virtual_seams",
            rest_v_np,
            rest_f_np,
            coupling,
            handle_inds.detach().cpu().numpy(),
            anchor_inds.detach().cpu().numpy(),
        )
        if len(coupling.aux_pin_inds) > 0:
            aux_inds = torch.tensor(
                coupling.aux_pin_inds, dtype=torch.long, device=args.pipe_cfg.device
            )
            aux_pos = torch.tensor(
                coupling.aux_pin_pos, dtype=torch.float32, device=args.pipe_cfg.device
            )
            poisson_anchor_inds = torch.cat([anchor_inds, aux_inds], dim=0)
            poisson_constraint_pos = torch.cat([anchor_pos, aux_pos], dim=0)

    rel_i = rel_j = rel_off = rel_w = None
    if coupling is not None and coupling.n_pairs > 0:
        rel_i = torch.tensor(
            coupling.pair_i, dtype=torch.long, device=args.pipe_cfg.device
        )
        rel_j = torch.tensor(
            coupling.pair_j, dtype=torch.long, device=args.pipe_cfg.device
        )
        rel_off = torch.tensor(
            coupling.offset0, dtype=torch.float64, device=args.pipe_cfg.device
        )
        rel_w = torch.tensor(
            coupling.poisson_pair_w, dtype=torch.float64, device=args.pipe_cfg.device
        )

    # Initialize Poisson System
    poisson = PoissonSystem(
        v, f,
        args.pipe_cfg.device,
        train_J=args.pipe_cfg.train_J,
        anchor_inds=poisson_anchor_inds,
        rel_pair_i=rel_i,
        rel_pair_j=rel_j,
        rel_offsets=rel_off,
        rel_lambda=args.pipe_cfg.poisson_rel_lambda if rel_i is not None else 0.0,
        rel_pair_w=rel_w,
    )
    print(f"Initialized Poisson system (cholesky_ok={getattr(poisson, 'cholesky_ok', None)})")

    v_field: Optional[torch.Tensor] = None
    if train_v:
        v_field = v.detach().clone().requires_grad_(True)
        with torch.no_grad():
            v_field.data[poisson_anchor_inds] = poisson_constraint_pos
        print("[INFO] Optimizing vertex positions directly (train_V=True, no Jacobian field)")
    
    def get_current_mesh() -> tuple[torch.Tensor, torch.Tensor]:
        if train_v:
            assert v_field is not None
            curr_v = v_field.clone()
            curr_v[poisson_anchor_inds, ...] = poisson_constraint_pos
            return curr_v, f
        return poisson.get_current_mesh(
            poisson_constraint_pos,
            trans_mats=quat_to_mat_torch(normalize_vs(quat_field)),
        )
    
    # Initialize Smooth-ARAP geometry regularization (Route 2: surface smoothness)
    smooth_arap = SmoothARAP(
        verts0=v,  # Original mesh vertices V0
        faces=f,
        device=args.pipe_cfg.device,
        lambda_arap=args.pipe_cfg.smooth_arap_lambda_arap,
        lambda_smooth=args.pipe_cfg.smooth_arap_lambda_smooth,
    )
    print(f"[Smooth-ARAP] Initialized geometry regularization")
    print(f"  - N_verts: {v.shape[0]}, N_faces: {f.shape[0]}")
    print(f"  - lambda_arap: {args.pipe_cfg.smooth_arap_lambda_arap}")
    print(f"  - lambda_smooth: {args.pipe_cfg.smooth_arap_lambda_smooth}")
    print(f"  - Stage 1 weight: {args.pipe_cfg.lambda_smooth_arap_stage1}")
    print(f"  - Stage 2 weight: {args.pipe_cfg.lambda_smooth_arap_stage2}")

    # Symmetric Dirichlet vs SSP (mutually exclusive in Stage 2)
    sd_reg = None
    rest_areas = precompute_rest_triangle_areas(v, f).to(args.pipe_cfg.device)
    if args.pipe_cfg.sd_reg.enabled or args.pipe_cfg.sd_reg.stage1:
        if args.pipe_cfg.ssp.enabled and args.pipe_cfg.sd_reg.enabled:
            print("[SD-Reg] Stage2 enabled: disabling SSP (mutually exclusive)")
            args.pipe_cfg.ssp.enabled = False
        sd_reg = SymmetricDirichletRegularizer(
            poisson, rest_areas, eps=args.pipe_cfg.sd_reg.eps
        )
        print(f"\n[SD-Reg] Symmetric Dirichlet regularization")
        print(f"  - Stage 1: {args.pipe_cfg.sd_reg.stage1} (weight={args.pipe_cfg.sd_reg.stage1_weight})")
        print(f"  - Stage 2: {args.pipe_cfg.sd_reg.enabled} (weight={args.pipe_cfg.sd_reg.weight})")
        print(f"  - eps: {args.pipe_cfg.sd_reg.eps}")

    # Initialize SSP (Sparse Shape Prior) for Stage 2 geometry regularization
    sparse_shape_prior = None
    if args.pipe_cfg.ssp.enabled:
        ssp_hyper = sparse_shape_hyper_from_config(args.pipe_cfg.ssp)
        ssp_cache = TemplateCache(v.detach(), f.detach())
        sparse_shape_prior = SparseShapePrior(ssp_cache, ssp_hyper)
        print(f"\n[SSP] Initialized Sparse Shape Prior (Stage 2 only)")
        print(f"  - compute_every: {args.pipe_cfg.ssp.compute_every}")
        print(f"  - alpha/beta: {ssp_hyper.alpha}/{ssp_hyper.beta}")
        print(f"  - M_surface/K_volume: {ssp_hyper.M_surface}/{ssp_hyper.K_volume}")
        print(f"  - schedule: warmup={ssp_hyper.warmup}, hold={ssp_hyper.hold}, decay={ssp_hyper.decay}")
        print(f"  - lambda peak/tail: {ssp_hyper.peak}/{ssp_hyper.tail}")
    
    # Initialize SDS ROI masking (gradient attenuation around handles)
    sds_vertex_mask = None
    roi_d_core, roi_d_band = 0, 0
    if args.pipe_cfg.sds_roi.enabled:
        print(f"\n[SDS ROI] Initializing gradient masking around handles")
        print(f"  - Mode: {args.pipe_cfg.sds_roi.mode}")
        print(f"  - Apply to: Stage {args.pipe_cfg.sds_roi.stage} (1=Stage1, 2=Stage2, 0=both)")
        
        # Build adjacency and compute hop distances.
        # Virtual-seam pairs are extra edges so BFS can leave the handle component.
        neighbors = build_vertex_adjacency(v.shape[0], f)
        if coupling is not None and coupling.n_pairs > 0:
            n_extra = add_adjacency_edges(
                neighbors, coupling.pair_i, coupling.pair_j
            )
            print(
                f"  - Virtual-seam edges in ROI graph: {n_extra} "
                f"(from {coupling.n_pairs} pairs)"
            )
        
        if args.pipe_cfg.sds_roi.mode == "percentile":
            # Percentile-based mode (auto-sizing)
            print(f"  - Target: Core={args.pipe_cfg.sds_roi.core_ratio*100:.0f}%, Core+Band={args.pipe_cfg.sds_roi.band_ratio*100:.0f}%")
            
            max_hops = 200  # Large enough for BFS to reach all vertices
            dist = bfs_handle_hops(v.shape[0], neighbors, handle_inds, max_hops)
            
            # Build mask using percentile thresholds
            sds_vertex_mask, roi_d_core, roi_d_band = build_sds_vertex_mask_from_ratio(
                hop_dist=dist,
                core_ratio=args.pipe_cfg.sds_roi.core_ratio,
                band_ratio=args.pipe_cfg.sds_roi.band_ratio,
            )
            
            print(f"  - Computed thresholds: d_core={roi_d_core} hops, d_band={roi_d_band} hops")
            
        else:
            # Fixed hops mode (backward compatibility)
            print(f"  - Core region: {args.pipe_cfg.sds_roi.core_hops} hops (weight=0)")
            print(f"  - Transition band: {args.pipe_cfg.sds_roi.core_hops} -> {args.pipe_cfg.sds_roi.band_hops} hops")
            
            max_hops = args.pipe_cfg.sds_roi.band_hops + 10
            dist = bfs_handle_hops(v.shape[0], neighbors, handle_inds, max_hops)
            
            sds_vertex_mask = build_sds_vertex_mask(
                dist=dist,
                core_hops=args.pipe_cfg.sds_roi.core_hops,
                band_hops=args.pipe_cfg.sds_roi.band_hops,
                min_weight=args.pipe_cfg.sds_roi.min_weight,
                device=args.pipe_cfg.device,
            )
            roi_d_core = args.pipe_cfg.sds_roi.core_hops
            roi_d_band = args.pipe_cfg.sds_roi.band_hops
        
        # Print actual coverage statistics
        if args.pipe_cfg.sds_roi.verbose:
            valid = torch.isfinite(dist) & (dist >= 0)
            core_mask = valid & (dist <= roi_d_core)
            band_mask = valid & (dist > roi_d_core) & (dist <= roi_d_band)
            far_mask = valid & (dist > roi_d_band)
            unreach_mask = ~valid
            
            core_ratio_actual = core_mask.float().mean().item()
            band_ratio_actual = band_mask.float().mean().item()
            coreband_ratio_actual = (core_mask | band_mask).float().mean().item()
            far_ratio_actual = far_mask.float().mean().item()
            unreach_ratio_actual = unreach_mask.float().mean().item()
            
            print(f"  - Actual coverage:")
            print(f"    · Core (masked):        {core_ratio_actual*100:5.1f}%")
            print(f"    · Band (transition):    {band_ratio_actual*100:5.1f}%")
            print(f"    · Core+Band total:      {coreband_ratio_actual*100:5.1f}%")
            print(f"    · Far (full SDS):       {far_ratio_actual*100:5.1f}%")
            print(f"    · Unreachable (masked): {unreach_ratio_actual*100:5.1f}%")
    
    # Initialize low-frequency subspace basis (Route 1: anti-spike) - DEPRECATED, using Smooth-ARAP instead
    lf_phi, lf_mass_col, v_rest = None, None, None
    if args.pipe_cfg.lowfreq.enabled:
        import os
        object_name = mesh_metadata['object_name']
        lf_dir = args.pipe_cfg.lowfreq.basis_root
        lf_path = os.path.join(lf_dir, f"{object_name}.npz")
        
        if not os.path.exists(lf_path):
            raise FileNotFoundError(
                f"[LowFreq] Basis file not found: {lf_path}\n"
                f"Please run: PYTHONPATH=. python scripts/tools/precompute_laplace_basis.py "
                f"--data-root data/apap_3d/processed --out-root {lf_dir} --k {args.pipe_cfg.lowfreq.k}"
            )
        
        data = np.load(lf_path)
        phi = data["phi"]  # (N, K0)
        mass = data["mass"]  # (N,)
        
        # Truncate to configured k
        k = min(args.pipe_cfg.lowfreq.k, phi.shape[1])
        phi = phi[:, :k]
        
        # Convert to torch tensors
        lf_phi = torch.from_numpy(phi).to(device=args.pipe_cfg.device, dtype=v.dtype)  # (N, K)
        lf_mass_col = torch.from_numpy(mass).to(device=args.pipe_cfg.device, dtype=v.dtype)[:, None]  # (N, 1)
        v_rest = v.clone().detach()  # (N, 3) rest pose
        
        print(f"[LowFreq] Loaded basis for '{object_name}': K={k}, weight={args.pipe_cfg.lowfreq.weight}")
        print(f"[LowFreq] Stage 1 will be regularized to low-frequency subspace (anti-spike)")

    # Initialize auxiliary learnable parameters
    quat_field = torch.zeros(
        (f.shape[0], 4),
        dtype=torch.float64,
        device=args.pipe_cfg.device,
    )
    quat_field[:, 0] = 1.0  # identity rotation
    if args.pipe_cfg.train_quat:
        quat_field.requires_grad_(True)
    # TODO: Add scaling field if necessary

    # =================================================================================
    # Optimization loop begins - Unified two-stage optimization with handle_freeze + lowfreq
    
    # Initialize optimizer with all trainable variables
    optim_vars = []
    if train_v:
        assert v_field is not None and v_field.requires_grad
        optim_vars.append(v_field)
    elif args.pipe_cfg.train_J:
        assert poisson.J.requires_grad, "Jacobian field must be trainable"
        optim_vars.append(poisson.J)
    if args.pipe_cfg.train_quat:
        quat_field.requires_grad_(True)
        optim_vars.append(quat_field)
    assert len(optim_vars) > 0, "No trainable variables found"

    optim = torch.optim.Adam(optim_vars, lr=args.pipe_cfg.lr_stage_1)
    
    # State variables for two-stage switching
    handle_frozen = False
    stage2_step = 0
    global_step = 0
    handle_v_stage1 = None  # Will store Stage 1 handle positions for soft constraint
    actual_stage2_steps = args.pipe_cfg.stage2_steps  # Will be dynamically adjusted
    sd_grad_diag_done = False
    train_curves: List[dict] = []
    
    stage1_end_step: Optional[int] = None
    stage1_handle_error: Optional[float] = None
    
    mode_label = "Hard Handle + Poisson" if hard_handle else "Soft Handle + Handle Freeze"
    print(f"[INFO] Starting unified two-stage optimization ({mode_label})")
    print(f"[INFO] Stage 1: Keypoint + Geometry + SDS (lambda_sds={args.pipe_cfg.lambda_sds_stage1}) + Smooth-ARAP")
    if hard_handle:
        print("[INFO] Hard handle: Poisson positional constraint (no soft l_kp, no Handle Freeze)")
        print(f"[INFO] Stage 1: fixed {args.pipe_cfg.max_stage1_steps} steps (no handle_err early stop)")
    else:
        print(f"[INFO] Stage 2: Geometry + SDS + Handle Freeze (soft constraint, weight={args.pipe_cfg.freeze_weight})")
        print(f"[INFO] Stage 1: max {args.pipe_cfg.max_stage1_steps} steps, early stop when handle_err < {args.pipe_cfg.eps_handle:.3e}")
    reg_name = "SD"
    if sd_reg is not None:
        parts = []
        if args.pipe_cfg.sd_reg.stage1:
            parts.append("S1-SD")
        if args.pipe_cfg.sd_reg.enabled:
            parts.append("S2-SD")
        reg_name = "+".join(parts) if parts else "SD"
    elif sparse_shape_prior is not None:
        reg_name = "SSP"
    else:
        reg_name = "none"
    freeze_suffix = "" if hard_handle else " + handle freeze"
    print(f"[INFO] Stage 2: fixed {args.pipe_cfg.stage2_steps} steps (SDS + {reg_name}{freeze_suffix})")

    # Render initial mesh (before optimization) — same layout as optim video frames
    with torch.no_grad():
        init_v, init_f = get_current_mesh()
        init_imgs = render_mesh_360(
            init_v, init_f, uvs, tex_inds, tex, n_step=4,
            radius=vis_radius, img_height=vis_h, img_width=vis_w,
            bg_color=vis_bg,
        )
        init_imgs_cat = np.concatenate(init_imgs, axis=1)
        Image.fromarray(init_imgs_cat).save(args.out_dir / "initial_mesh.png")
        vid_writer.append_data(init_imgs_cat)
    print(f"[INFO] Saved initial_mesh.png and appended to optim.mp4")
    
    # Main optimization loop
    # Safety upper limit: Stage 1 cap + fixed Stage 2 + buffer
    max_steps = args.pipe_cfg.max_stage1_steps + args.pipe_cfg.stage2_steps + 100
    for global_step in range(max_steps):
        
        # 1. Get current mesh
        curr_v, curr_f = get_current_mesh()
        
        # 2. Compute handle error
        handle_err = compute_handle_error(curr_v, handle_inds, handle_pos)
        
        # 3. Check if we should trigger handle_freeze (transition to stage 2)
        should_trigger = False
        trigger_reason = ""
        
        if not handle_frozen:
            if (
                not hard_handle
                and handle_err < args.pipe_cfg.eps_handle
            ):
                should_trigger = True
                trigger_reason = f"handle_err < eps_handle ({handle_err.item():.3e} < {args.pipe_cfg.eps_handle:.3e})"
            elif global_step >= args.pipe_cfg.max_stage1_steps:
                should_trigger = True
                trigger_reason = f"max_stage1_steps reached ({global_step} >= {args.pipe_cfg.max_stage1_steps})"
        
        if should_trigger:
            handle_frozen = True
            stage2_step = 0
            
            actual_stage2_steps = args.pipe_cfg.stage2_steps
            stage1_actual_steps = global_step
            stage1_end_step = global_step
            stage1_handle_error = float(handle_err.item())

            with torch.no_grad():
                s1_imgs = render_mesh_360(
                    curr_v, curr_f, uvs, tex_inds, tex, n_step=4,
                    radius=vis_radius, img_height=vis_h, img_width=vis_w,
                    bg_color=vis_bg,
                )
                Image.fromarray(np.concatenate(s1_imgs, axis=1)).save(
                    args.out_dir / "stage1_end_mesh.png"
                )
                stage1_mesh_dir = args.out_dir / "mesh"
                stage1_mesh_dir.mkdir(parents=True, exist_ok=True)
                save_obj(
                    stage1_mesh_dir / "stage1_end.obj",
                    curr_v.detach().cpu().numpy(),
                    curr_f.detach().cpu().numpy(),
                    uvs=uvs.detach().cpu().numpy(),
                    tex_inds=tex_inds.detach().cpu().numpy(),
                    tex=tex.detach().cpu().numpy(),
                )
            print(f"[INFO] Saved stage1_end_mesh.png and mesh/stage1_end.obj")
            
            # Save Stage 1 handle positions for soft constraint
            if args.pipe_cfg.enable_freeze:
                with torch.no_grad():
                    handle_v_stage1 = curr_v[handle_inds, ...].detach().clone()
                print(f"[INFO] Handle freeze triggered at global_step={global_step}")
                print(f"[INFO] Reason: {trigger_reason}")
                print(f"[INFO] Current handle_err={handle_err.item():.3e}")
                print(f"[INFO] Stage 1 completed in {stage1_actual_steps} steps")
                print(f"[INFO] Stage 1 handle positions saved for {len(handle_inds)} handles")
                print(f"[INFO] Stage 2: Soft constraint freeze (weight={args.pipe_cfg.freeze_weight})")
                print(f"[INFO] Stage 2 will run for {actual_stage2_steps} fixed steps")
            else:
                print(f"[INFO] Stage 2 started at global_step={global_step} (freeze disabled)")
                print(f"[INFO] Reason: {trigger_reason}")
                print(f"[INFO] Stage 2 will run for {actual_stage2_steps} steps")
            
            # Switch to stage 2 learning rate
            for param_group in optim.param_groups:
                param_group['lr'] = args.pipe_cfg.lr_stage_2
        
        # 4. Set loss weights based on current stage
        if hard_handle:
            lambda_kp = 0.0
            if not handle_frozen:
                lambda_sds = args.pipe_cfg.lambda_sds_stage1
                lambda_geom = args.pipe_cfg.lambda_geom_stage1
            else:
                lambda_sds = args.pipe_cfg.lambda_sds_stage2
                lambda_geom = args.pipe_cfg.lambda_geom_stage2
        elif not handle_frozen:
            # Stage 1: Keypoint + Geometry + SDS + LowFreq
            lambda_kp = args.pipe_cfg.lambda_kp_stage1
            lambda_sds = args.pipe_cfg.lambda_sds_stage1
            lambda_geom = args.pipe_cfg.lambda_geom_stage1
        else:
            # Stage 2: Geometry + SDS + Soft freeze constraint
            lambda_kp = args.pipe_cfg.lambda_kp_stage2
            lambda_sds = args.pipe_cfg.lambda_sds_stage2
            lambda_geom = args.pipe_cfg.lambda_geom_stage2
        
        # 5. Compute camera parameters (needed for SDS)
        if args.pipe_cfg.cam_schedule == "random":
            view_idx = int(np.random.randint(len(args.pipe_cfg.cam_locs)))
        elif args.pipe_cfg.cam_schedule == "sequential":
            view_idx = int(global_step % len(args.pipe_cfg.cam_locs))
        else:
            raise ValueError(f"Unknown camera schedule: {args.pipe_cfg.cam_schedule}")
        cam_loc = args.pipe_cfg.cam_locs[view_idx]
        cam2world = compute_lookat_mat(
            cam_loc.type(torch.float32).to(args.pipe_cfg.device),
            args.pipe_cfg.origin.type(torch.float32).to(args.pipe_cfg.device),
        )
        proj_mat = compute_proj_mat(
            args.pipe_cfg.aspect_ratio,
            args.pipe_cfg.fov,
            args.pipe_cfg.near,
            args.pipe_cfg.far,
            device=args.pipe_cfg.device,
        )
        
        # 6. Render RGB (needed for SDS)
        bg_color = vis_bg.clone()
        if args.vis_bg_color is None and args.pipe_cfg.use_random_bg:
            bg_color = torch.rand(3, device=args.pipe_cfg.device)
        img, img_grad = render(
            curr_v, curr_f,
            cam2world,
            proj_mat,
            args.pipe_cfg.img_height,
            args.pipe_cfg.img_width,
            uvs=uvs, tex_inds=tex_inds, tex=tex, ss_scale=4.0, bg_color=bg_color,
        )
        
        # 7. Compute individual losses
        # Keypoint loss
        l_kp = torch.sum((curr_v[handle_inds, ...] - handle_pos) ** 2) / len(handle_inds)
        
        # Geometry loss: SSP in Stage 2 (Sparse Shape Prior)
        l_geom = torch.tensor(0.0, device=args.pipe_cfg.device)
        ssp_l_surf = 0.0
        ssp_l_io = 0.0
        ssp_lambda_t = 0.0
        if (
            handle_frozen
            and sparse_shape_prior is not None
            and stage2_step % args.pipe_cfg.ssp.compute_every == 0
        ):
            ssp_out = sparse_shape_prior.compute(curr_v, curr_f, stage2_step)
            l_geom = ssp_out["lambda_t"] * ssp_out["L_shape"]
            ssp_l_surf = float(ssp_out["L_surf"].item())
            ssp_l_io = float(ssp_out["L_io"].item())
            ssp_lambda_t = float(ssp_out["lambda_t"].item())

        # Symmetric Dirichlet regularization (Stage 2 only, SSP alternative)
        l_sd = torch.tensor(0.0, device=args.pipe_cfg.device)
        sd_stats = {"l_sd": 0.0, "sigma_min": 0.0, "sigma_min_mean": 0.0}
        use_sd_s1 = (not handle_frozen) and sd_reg is not None and args.pipe_cfg.sd_reg.stage1
        use_sd_s2 = handle_frozen and sd_reg is not None and args.pipe_cfg.sd_reg.enabled
        if use_sd_s1 or use_sd_s2:
            l_sd, sd_stats = sd_reg(curr_v)
        
        # SDS loss (guidance) with optional ROI masking
        l_sds = torch.tensor(0.0, device=args.pipe_cfg.device)
        apply_roi = False
        if lambda_sds > 0.0:
            # Check if ROI masking should be applied in this stage
            apply_roi = (
                sds_vertex_mask is not None and
                args.pipe_cfg.sds_roi.enabled and
                (
                    args.pipe_cfg.sds_roi.stage == 0 or  # both stages
                    (args.pipe_cfg.sds_roi.stage == 1 and not handle_frozen) or  # stage 1 only
                    (args.pipe_cfg.sds_roi.stage == 2 and handle_frozen)  # stage 2 only
                )
            )
            
            if apply_roi:
                # Render vertex mask to per-pixel weight map
                sds_weight_map = render_vertex_attribute(
                    v=curr_v,
                    f=curr_f,
                    vertex_attr=sds_vertex_mask.unsqueeze(-1),  # (N, 1)
                    cam2world=cam2world,
                    proj_mat=proj_mat,
                    img_height=args.pipe_cfg.img_height,
                    img_width=args.pipe_cfg.img_width,
                    bg_value=1.0,  # far regions have weight=1
                    device=args.pipe_cfg.device,
                    ss_scale=4.0,
                )
                sds_weight_map = sds_weight_map.permute(2, 0, 1).unsqueeze(0)  # (1, 1, H, W)
            else:
                sds_weight_map = None
            
            l_sds = guidance(
                prompt,
                image=img[None].permute(0, 3, 1, 2),
                cfg_scale=args.pipe_cfg.cfg_scale,
                weight_map=sds_weight_map,
            )
        
        # Smooth-ARAP geometry regularization (both stages)
        # Weight is 0 in the current APAP 3D config; skip the call so a NaN
        # cotan weight on kitbash meshes cannot poison 0 * l_sarap.
        if not handle_frozen:
            w_sarap = args.pipe_cfg.lambda_smooth_arap_stage1
        else:
            w_sarap = args.pipe_cfg.lambda_smooth_arap_stage2
        if w_sarap > 0.0:
            l_sarap = smooth_arap(curr_v)
        else:
            l_sarap = torch.tensor(0.0, device=args.pipe_cfg.device)

        if coupling is not None:
            if not handle_frozen:
                coupling.set_stage_weights(
                    1,
                    disp_major=args.pipe_cfg.lambda_disp_major_stage1,
                    gap_major=args.pipe_cfg.lambda_gap_major_stage1,
                    disp_minor=args.pipe_cfg.lambda_disp_minor_stage1,
                    gap_minor=args.pipe_cfg.lambda_gap_minor_stage1,
                    normal_major=args.pipe_cfg.lambda_normal_major_stage1,
                    group=args.pipe_cfg.lambda_group,
                    gap_max=args.pipe_cfg.lambda_gap_max,
                )
            else:
                coupling.set_stage_weights(
                    2,
                    disp_major=args.pipe_cfg.lambda_disp_major_stage2,
                    gap_major=args.pipe_cfg.lambda_gap_major_stage2,
                    disp_minor=args.pipe_cfg.lambda_disp_minor_stage2,
                    gap_minor=args.pipe_cfg.lambda_gap_minor_stage2,
                    normal_major=args.pipe_cfg.lambda_normal_major_stage2,
                    group=args.pipe_cfg.lambda_group,
                    gap_max=args.pipe_cfg.lambda_gap_max,
                )
        l_disp = torch.tensor(0.0, device=args.pipe_cfg.device)
        l_gap = torch.tensor(0.0, device=args.pipe_cfg.device)
        l_group = torch.tensor(0.0, device=args.pipe_cfg.device)
        l_normal = torch.tensor(0.0, device=args.pipe_cfg.device)
        seam_stats = {
            "mean_gap": 0.0,
            "max_gap": 0.0,
            "mean_gap0": 0.0,
            "violation_ratio": 0.0,
            "e_group": 0.0,
            "e_normal": 0.0,
            "focus_mean_gap": 0.0,
            "focus_max_gap": 0.0,
            "largest_sep_seam": -1,
            "largest_sep_gap": 0.0,
            "seam_rows": [],
            "w_disp_focus": 0.0,
            "w_gap_focus": 0.0,
        }
        if coupling is not None and coupling.n_pairs > 0:
            l_disp, l_gap, l_group, l_normal, seam_stats = coupling.losses(curr_v)
        
        # Low-frequency projection prior (Stage 1 only) - DEPRECATED, using Smooth-ARAP instead
        l_lowfreq = torch.tensor(0.0, device=args.pipe_cfg.device)
        if (not handle_frozen) and args.pipe_cfg.lowfreq.enabled and lf_phi is not None:
            # Project current vertices to low-frequency subspace
            disp = curr_v - v_rest  # (N, 3) displacement field
            Mdisp = disp * lf_mass_col  # (N, 3) mass-weighted displacement
            c = lf_phi.transpose(0, 1) @ Mdisp  # (K, 3) low-frequency coefficients
            disp_low = lf_phi @ c  # (N, 3) low-frequency reconstruction
            # Penalize high-frequency residual: ||disp - disp_low||^2
            l_lowfreq = torch.mean(torch.sum((disp - disp_low) ** 2, dim=-1))
        
        # Quaternion regularization
        l_quat_reg = torch.tensor(0.0, device=args.pipe_cfg.device)
        if args.pipe_cfg.w_quat_reg > 0.0:
            quat_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=args.pipe_cfg.device)
            l_quat_reg = torch.sum((quat_field - quat_id[None]) ** 2)
        
        # Handle Freeze loss (Stage 2 soft constraint, CAPAP-style)
        l_handle_freeze = torch.tensor(0.0, device=args.pipe_cfg.device)
        if handle_frozen and args.pipe_cfg.enable_freeze and handle_v_stage1 is not None:
            # Soft constraint: penalize deviation from Stage 1 handle positions
            curr_handle_v = curr_v[handle_inds, ...]
            l_handle_freeze = torch.sum((curr_handle_v - handle_v_stage1) ** 2) / len(handle_inds)
            # Apply high weight (default 1000) to enforce strong constraint
            l_handle_freeze = args.pipe_cfg.freeze_weight * l_handle_freeze
        
        # Stage 2 gradient diagnostic (first effective iteration only)
        grad_j_base = 0.0
        grad_j_sd = 0.0
        lambda_sd_suggested = 0.0
        if (
            handle_frozen
            and stage2_step == 0
            and sd_reg is not None
            and not sd_grad_diag_done
            and not train_v
        ):
            sd_grad_diag_done = True
            optim.zero_grad()
            l_base_diag = (
                lambda_kp * l_kp
                + lambda_sds * l_sds
                + w_sarap * l_sarap
                + l_disp + l_gap + l_group + l_normal
                + l_handle_freeze
            )
            grad_j_base = grad_norm_on_poisson_j(poisson, l_base_diag)
            optim.zero_grad()
            grad_j_sd = grad_norm_on_poisson_j(poisson, l_sd)
            optim.zero_grad()
            lambda_sd_suggested = 0.1 * grad_j_base / (grad_j_sd + 1e-12)
            print(
                f"[SD-Reg] Gradient diagnostic (Stage 2 step 0): "
                f"||grad_J L_base||={grad_j_base:.6e} "
                f"||grad_J L_SD||={grad_j_sd:.6e} "
                f"lambda_10={lambda_sd_suggested:.6e}"
            )

        # 8. Assemble total loss
        lambda_sd = 0.0
        if use_sd_s1:
            lambda_sd = args.pipe_cfg.sd_reg.stage1_weight
        elif use_sd_s2:
            lambda_sd = args.pipe_cfg.sd_reg.weight
        l_total = (
            lambda_kp * l_kp +
            lambda_geom * l_geom +
            lambda_sds * l_sds +
            args.pipe_cfg.w_quat_reg * l_quat_reg +
            args.pipe_cfg.lowfreq.weight * l_lowfreq +  # DEPRECATED
            w_sarap * l_sarap +  # Smooth-ARAP (Route 2)
            l_disp +
            l_gap +
            l_group +
            l_normal +
            l_handle_freeze +  # Soft freeze constraint
            lambda_sd * l_sd
        )
        
        # 9. Optimization step - SPLIT BACKWARD for hard SDS masking
        optim.zero_grad()
        
        # 🎯 HARD SDS MASKING: Two-phase backward
        # Phase 1: SDS backward with vertex-level gradient masking (only in Stage 1 with ROI)
        if lambda_sds > 0.0 and apply_roi and sds_vertex_mask is not None:
            # Ensure vertex mask is on the same device as curr_v
            vertex_mask_cuda = sds_vertex_mask.to(curr_v.device)
            
            # Register hook to mask SDS gradients on Core vertices
            def sds_vertex_mask_hook(grad: torch.Tensor) -> torch.Tensor:
                """
                Hard masking: multiply SDS gradients by vertex_weight.
                Core vertices (weight=0) get ZERO SDS gradient.
                """
                return grad * vertex_mask_cuda.unsqueeze(-1)  # (N, 3) * (N, 1)
            
            # Attach hook to curr_v
            hook_handle = curr_v.register_hook(sds_vertex_mask_hook)
            
            # Backward SDS only (retain graph for second phase)
            (lambda_sds * l_sds).backward(retain_graph=True)
            
            # Remove hook before geometry backward
            hook_handle.remove()
            
            # Phase 2: Geometry/Keypoint backward (no masking)
            l_geom_total = (
                lambda_kp * l_kp +
                lambda_geom * l_geom +
                args.pipe_cfg.w_quat_reg * l_quat_reg +
                args.pipe_cfg.lowfreq.weight * l_lowfreq +
                w_sarap * l_sarap +
                l_disp +
                l_gap +
                l_group +
                l_normal +
                l_handle_freeze +
                lambda_sd * l_sd
            )
            l_geom_total.backward()
        else:
            # Standard single backward (no ROI masking or no SDS)
            l_total.backward()
        
        optim.step()
        if train_v:
            with torch.no_grad():
                v_field.data[poisson_anchor_inds, ...] = poisson_constraint_pos
        
        # 10. Track stage 2 progress
        if handle_frozen:
            stage2_step += 1
            if stage2_step >= actual_stage2_steps:
                total_steps = global_step + 1
                print(f"[INFO] Stage 2 completed after {stage2_step} steps")
                print(f"[INFO] Total optimization steps: {total_steps} (Stage1≤{args.pipe_cfg.max_stage1_steps} + Stage2={actual_stage2_steps})")
                break
        
        # 11. Logging
        log_dict = {
            "train/loss_total": l_total.item(),
            "train/loss_keypoint": l_kp.item(),
            "train/loss_geometry": l_geom.item(),
            "train/loss_sds": l_sds.item(),
            "train/loss_quat_reg": l_quat_reg.item(),
            "train/loss_lowfreq": l_lowfreq.item(),  # DEPRECATED
            "train/loss_smooth_arap": l_sarap.item(),  # Smooth-ARAP (Route 2)
            "train/handle_error": handle_err.item(),
            "train/stage": 2 if handle_frozen else 1,
            "train/stage2_step": stage2_step if handle_frozen else 0,
            "train/stage2_total_steps": actual_stage2_steps if handle_frozen else 0,
            "train/lambda_kp": lambda_kp,
            "train/lambda_sds": lambda_sds,
            "train/lambda_smooth_arap": w_sarap,
            "train/loss_disp": l_disp.item() if torch.is_tensor(l_disp) else float(l_disp),
            "train/loss_gap": l_gap.item() if torch.is_tensor(l_gap) else float(l_gap),
            "train/loss_group": l_group.item() if torch.is_tensor(l_group) else float(l_group),
            "train/loss_normal": l_normal.item() if torch.is_tensor(l_normal) else float(l_normal),
            "train/lambda_disp_focus": seam_stats.get("w_disp_focus", 0.0),
            "train/lambda_gap_focus": seam_stats.get("w_gap_focus", 0.0),
            "train/seam_mean_gap": seam_stats["mean_gap"],
            "train/seam_max_gap": seam_stats["max_gap"],
            "train/seam_gap0": seam_stats["mean_gap0"],
            "train/seam_violation_ratio": seam_stats["violation_ratio"],
            "train/seam_focus_mean_gap": seam_stats.get("focus_mean_gap", 0.0),
            "train/seam_focus_max_gap": seam_stats.get("focus_max_gap", 0.0),
            "train/sds_roi_active": 1 if apply_roi else 0,
            "train/ssp_l_surf": ssp_l_surf,
            "train/ssp_l_io": ssp_l_io,
            "train/ssp_lambda_t": ssp_lambda_t,
            "train/loss_sd": float(l_sd.item()) if torch.is_tensor(l_sd) else 0.0,
            "train/lambda_sd": lambda_sd,
            "train/sd_sigma_min": sd_stats.get("sigma_min", 0.0),
            "train/sd_sigma_min_mean": sd_stats.get("sigma_min_mean", 0.0),
        }
        
        if handle_frozen:
            train_curves.append({
                "global_step": int(global_step),
                "stage2_step": int(stage2_step),
                "handle_error": float(handle_err.item()),
                "loss_sds": float(l_sds.item()) if torch.is_tensor(l_sds) else 0.0,
                "loss_sd": float(l_sd.item()) if torch.is_tensor(l_sd) else 0.0,
                "sigma_min": sd_stats.get("sigma_min", 0.0),
                "lambda_sd": float(lambda_sd),
            })
            if stage2_step == 0 and sd_grad_diag_done:
                log_dict["train/grad_j_base"] = grad_j_base
                log_dict["train/grad_j_sd"] = grad_j_sd
                log_dict["train/lambda_sd_suggested"] = lambda_sd_suggested
        
        # Add handle freeze loss to logs
        if handle_frozen and args.pipe_cfg.enable_freeze:
            log_dict["train/loss_handle_freeze"] = l_handle_freeze.item()
        
        wandb.log(log_dict)
        
        # 12. Visualization + building seam debug
        if coupling is not None and coupling.n_pairs > 0 and (global_step + 1) % 20 == 0:
            notes = []
            if global_step + 1 >= 20:
                notes = coupling.adapt_gap_weights(seam_stats.get("seam_rows", []))
            print(coupling.format_debug_log(global_step + 1, seam_stats, notes))
            focus_row = None
            for rec in seam_stats.get("seam_rows", []):
                if rec["seam"] == coupling.focus_seam or (
                    coupling.focus_seam < 0 and rec.get("is_major")
                ):
                    focus_row = rec
                    break
            coupling.gap_history.append(
                {
                    "step": int(global_step + 1),
                    "mean_gap": seam_stats["mean_gap"],
                    "max_gap": seam_stats["max_gap"],
                    "focus_mean_gap": seam_stats.get("focus_mean_gap", 0.0),
                    "focus_max_gap": seam_stats.get("focus_max_gap", 0.0),
                    "focus_gap0": (
                        float(focus_row["original_mean_gap"]) if focus_row else seam_stats["mean_gap0"]
                    ),
                    "violation_ratio": seam_stats["violation_ratio"],
                    "largest_sep_seam": seam_stats.get("largest_sep_seam", -1),
                    "largest_sep_gap": seam_stats.get("largest_sep_gap", 0.0),
                }
            )
            write_seam_gap_curve(args.out_dir / "virtual_seams", coupling.gap_history)
            write_debug_visualization(
                args.out_dir / "virtual_seams",
                rest_v_np,
                rest_f_np,
                coupling,
                handle_inds.detach().cpu().numpy(),
                anchor_inds.detach().cpu().numpy(),
                curr_v=curr_v.detach().cpu().numpy(),
                filename_png=f"virtual_seams_step_{global_step+1:04d}.png",
                write_obj=False,
            )

        if (global_step + 1) % args.pipe_cfg.vis_every == 0:
            print(
                f"[VirtualSeam] step={global_step} mean_gap={seam_stats['mean_gap']:.4f} "
                f"max_gap={seam_stats['max_gap']:.4f} gap0={seam_stats['mean_gap0']:.4f} "
                f"viol={seam_stats['violation_ratio']:.3f} "
                f"E_disp={float(l_disp):.4e} E_gap={float(l_gap):.4e} "
                f"E_group={float(l_group):.4e} E_normal={float(l_normal):.4e}"
            )
            optim_img_dir = args.out_dir / "optim_imgs"
            optim_img_dir.mkdir(parents=True, exist_ok=True)

            vis_imgs = render_mesh_360(
                curr_v, curr_f, uvs, tex_inds, tex, n_step=4,
                radius=vis_radius, img_height=vis_h, img_width=vis_w,
                bg_color=vis_bg,
            )
            vis_imgs = np.concatenate(vis_imgs, axis=1) 

            # log video
            vid_writer.append_data(vis_imgs)

            # log image
            vis_imgs = Image.fromarray(vis_imgs)
            vis_imgs.save(optim_img_dir / f"{global_step:04d}.png")
            wandb.log({"eval/image": wandb.Image(vis_imgs)})

    # End of unified two-stage optimization
    print(f"[INFO] Optimization completed at global_step={global_step}")
    if coupling is not None and coupling.gap_history:
        write_seam_gap_curve(args.out_dir / "virtual_seams", coupling.gap_history)

    # Clean up
    vid_writer.close()

    # Optimization loop ends
    # =================================================================================
        
    # Save results
    mesh_dir = args.out_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = mesh_dir / "deformed.obj"
    save_obj(
        mesh_path,
        curr_v.detach().cpu().numpy(),
        curr_f.detach().cpu().numpy(),
        uvs=uvs.detach().cpu().numpy(),
        tex_inds=tex_inds.detach().cpu().numpy(),
        tex=tex.detach().cpu().numpy(),
    )
    print(f"Saved mesh at: {str(mesh_path)}")

    final_imgs = render_mesh_360(
        curr_v, curr_f, uvs, tex_inds, tex, n_step=4,
        radius=vis_radius, img_height=vis_h, img_width=vis_w,
        bg_color=vis_bg,
    )
    final_imgs = np.concatenate(final_imgs, axis=1)
    final_imgs_ = Image.fromarray(final_imgs)
    final_imgs_.save(args.out_dir / "deformed_mesh.png")

    final_marker_imgs = render_mesh_with_markers_360(
        curr_v,
        curr_f,
        torch.cat([handle_inds, anchor_inds], dim=0),
        torch.cat([handle_pos, anchor_pos], dim=0),
        n_step=4,
        radius=vis_radius,
        img_height=vis_h,
        img_width=vis_w,
    )
    final_marker_imgs = np.concatenate(final_marker_imgs, axis=1)
    final_marker_imgs_ = Image.fromarray(final_marker_imgs)
    final_marker_imgs_.save(args.out_dir / "deformed_mesh_markers.png")

    final_imgs = np.concatenate([final_imgs, final_marker_imgs], axis=0)
    final_imgs_ = Image.fromarray(final_imgs)
    final_imgs_.save(args.out_dir / "deformed_mesh_summary.png")

    # Save training curves and evaluation metrics
    metrics_dir = args.out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "train_curves.json", "w") as f:
        json.dump(train_curves, f, indent=2)

    eval_metrics = {
        "prompt": prompt,
        "seed": args.pipe_cfg.seed,
        "lambda_sds_stage1": args.pipe_cfg.lambda_sds_stage1,
        "stage2_steps": args.pipe_cfg.stage2_steps,
        "ssp_enabled": args.pipe_cfg.ssp.enabled,
        "sd_reg_enabled": args.pipe_cfg.sd_reg.enabled,
        "train_V": args.pipe_cfg.train_V,
        "train_J": args.pipe_cfg.train_J,
        "sd_reg_stage1": args.pipe_cfg.sd_reg.stage1,
        "lambda_sd": args.pipe_cfg.sd_reg.weight,
        "lambda_sd_stage1": args.pipe_cfg.sd_reg.stage1_weight,
        "sd_eps": args.pipe_cfg.sd_reg.eps,
        "hard_handle": args.pipe_cfg.hard_handle,
        "enable_freeze": args.pipe_cfg.enable_freeze,
        "stage1_end_step": stage1_end_step,
        "stage1_handle_error": stage1_handle_error,
        "final_handle_error": float(compute_handle_error(curr_v, handle_inds, handle_pos).item()),
    }
    if rest_areas is not None:
        eval_metrics.update(
            compute_sd_p99(
                poisson, curr_v, rest_areas,
                eps=args.pipe_cfg.sd_reg.eps if args.pipe_cfg.sd_reg.enabled else 1e-3,
            )
        )
    with open(metrics_dir / "eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    wandb.log(
        {
            "eval/video": wandb.Video(str(vid_dir / "optim.mp4")),
            "eval/final_rendering": wandb.Image(final_imgs_),
        }
    )

    print(f"Done! Logs can be found at: {str(args.out_dir.resolve())}")


if __name__ == "__main__":
    main(
        tyro.cli(Args)
    )
