"""
sd_configs.py

Config objects for experiments using Stable Diffusion guidance.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Union

from jaxtyping import Shaped
import torch


@dataclass
class SdsRoiConfig:
    """ROI masking configuration for SDS gradient attenuation around handles"""
    enabled: bool = True
    """Whether to enable SDS ROI masking"""
    stage: int = 1
    """Which stage to apply masking (1=Stage1, 2=Stage2, 0=both)"""
    mode: str = "percentile"
    """Masking mode: 'percentile' (auto-sizing) or 'fixed' (manual hops)"""
    core_ratio: float = 0.2
    """[Percentile mode] Target ratio of vertices in Core region (fully masked)"""
    band_ratio: float = 0.5
    """[Percentile mode] Target ratio of vertices in Core+Band regions (far=80%)"""
    core_hops: int = 38
    """[Fixed mode] Core region: vertices within this hop distance are fully masked"""
    band_hops: int = 70
    """[Fixed mode] Transition band end: linear interpolation from core_hops to band_hops"""
    min_weight: float = 0.0
    """Minimum SDS weight in core region (0.0 = fully masked)"""
    verbose: bool = True
    """Print ROI coverage statistics"""


@dataclass
class LowFreqConfig:
    """Configuration for low-frequency subspace regularization (Route 1: anti-spike)
    DEPRECATED: Using Smooth-ARAP (Route 2) instead
    """
    enabled: bool = False  # DISABLED, using Smooth-ARAP instead
    """Whether to enable low-frequency projection prior in Stage 1"""
    basis_root: str = "data/apap_3d_laplace"
    """Root directory containing precomputed Laplace basis npz files"""
    k: int = 32
    """Number of low-frequency modes to use (must be <= precomputed k)"""
    weight: float = 0.0  # Set to 0 since we're using Smooth-ARAP
    """Weight for low-frequency projection loss (higher = stronger smoothing, recommend 1.0~3.0)"""


@dataclass
class SdRegConfig:
    """Symmetric Dirichlet regularization (Stage 2 SSP alternative)."""
    enabled: bool = False
    """Enable SD reg in Stage 2 (mutually exclusive with SSP)."""
    stage1: bool = False
    """Also apply SD reg in Stage 1 (e.g. with lambda_sds_stage1=0)."""
    weight: float = 0.1
    """lambda_sd multiplier for L_SD in Stage 2."""
    stage1_weight: float = 0.1
    """lambda_sd multiplier for L_SD in Stage 1."""
    eps: float = 1e-3
    """Minimum singular value clamp."""


@dataclass
class SspConfig:
    """SSP (Sparse Shape Prior) configuration for Stage 2 - CAPAP compatible settings"""
    enabled: bool = True
    """Whether to enable SSP in Stage 2"""
    # Compute frequency
    compute_every: int = 6
    """Compute SSP loss every N steps (for efficiency)"""
    # Loss weights (L_shape = alpha * L_surf + beta * L_io)
    alpha: float = 1.0
    """Weight for L_surf (surface distance loss)"""
    beta: float = 0.3
    """Weight for L_io (inside-outside consistency loss)"""
    # Sampling parameters
    M_surface: int = 4096
    """Number of surface sample points"""
    K_volume: int = 4096
    """Number of volume sample points"""
    resample_every: int = 10
    """Resample points every N steps"""
    # Robustness parameters
    delta: float = 0.02
    """Huber threshold for L_surf"""
    tau: float = 0.05
    """tanh temperature for soft sign in L_io"""
    margin: float = 0.02
    """Hinge margin for L_io"""
    # Cosine schedule (lambda_t varies over Stage 2)
    warmup: int = 300
    """Warmup steps: lambda linearly increases from 0 to peak"""
    hold: int = 800
    """Hold steps: lambda stays at peak"""
    decay: int = 600
    """Decay steps: lambda cosine decays from peak to tail"""
    peak: float = 0.10
    """Peak lambda value"""
    tail: float = 0.02
    """Tail lambda value (after decay)"""


@dataclass
class StableDiffusion3DPipeConfig:

    # Guidance parameters
    guidance_type: Literal["sd"] = "sd"
    """Type of guidance to use"""
    lora_scale: float = 1.0
    """LoRA scale used for the experiment"""
    cfg_scale: float = 100.0
    """CFG scale used for the experiment"""
    clamp_val: float = 2.0
    """Clamping value for the guidance"""

    # Poisson parameters
    train_J: bool = True
    """Whether to train source Jacobian field"""
    train_V: bool = False
    """Optimize vertex positions directly (mutually exclusive with train_J)"""
    train_quat: bool = False
    """Whether to train quaternion field"""

    # Renderer parameters
    cam_schedule: Literal["random", "sequential"] = "random"
    """Camera scheduler"""
    cam_locs: Shaped[torch.Tensor, "* 3"] = torch.tensor(
        [
            [0.0, 0.0, 1.5],
            [1.5, 0.0, 0.0],
            [-1.5, 0.0, 0.0],
            [0.0, 0.0, -1.5],
        ]
    )
    """Camera locations"""
    origin: Shaped[torch.Tensor, "3"] = torch.tensor([0.0, 0.0, 0.0])
    """Origin of the scene"""
    img_height: int = 512
    """Height of the rendered image"""
    img_width: int = 512
    """Width of the rendered image"""
    aspect_ratio: float = 1.0
    """Aspect ratio of the rendered image"""
    fov: float = 53.14
    """Field of view of the rendered image"""
    near: float = 1e-1
    """Near plane of the rendered image"""
    far: float = 1e10
    """Far plane of the rendered image"""
    use_random_bg: bool = True
    """Whether to use random background"""

    # Loss configs
    w_guidance: float = 1.0
    """Weight for guidance loss"""
    w_kp: float = 1.0
    """Weight for keypoint matching loss"""
    w_quat_reg: float = 0.0
    """Weight for quaternion regularization loss"""
    
    # Low-frequency subspace regularization (Route 1: anti-spike) - DEPRECATED
    # Using Smooth-ARAP instead
    lowfreq: LowFreqConfig = field(default_factory=LowFreqConfig)
    """Low-frequency projection prior configuration (only active in Stage 1)"""
    
    # Smooth-ARAP geometry regularization (Route 2: surface smoothness)
    smooth_arap_lambda_arap: float = 1.0
    """Internal weight for ARAP term in Smooth-ARAP module"""
    smooth_arap_lambda_smooth: float = 1.0
    """Internal weight for Smooth term in Smooth-ARAP module"""
    lambda_smooth_arap_stage1: float = 0.0
    """Weight for Smooth-ARAP loss in Stage 1 (DISABLED)"""
    lambda_smooth_arap_stage2: float = 0.0
    """Weight for Smooth-ARAP loss in Stage 2 (DISABLED - let SDS dominate)"""
    
    # SDS ROI masking (gradient attenuation around handles)
    sds_roi: SdsRoiConfig = field(default_factory=SdsRoiConfig)
    """SDS ROI masking configuration"""
    
    # SSP (Sparse Shape Prior) - Stage 2 only, CAPAP compatible
    ssp: SspConfig = field(default_factory=SspConfig)
    """SSP configuration for Stage 2 geometry prior"""

    # Symmetric Dirichlet regularization - Stage 2 only (SSP alternative)
    sd_reg: SdRegConfig = field(default_factory=SdRegConfig)
    """Symmetric Dirichlet regularization for Stage 2"""
    
    # Handle freeze parameters (CAPAP-style soft constraint)
    eps_handle: float = 1e-3
    """Handle error threshold to trigger handle_freeze (L2 average error)"""
    max_stage1_steps: int = 400
    """Maximum steps for stage 1. If handle_err < eps_handle not reached, force transition to stage 2"""
    min_total_steps: int = 1200
    """Minimum total steps (Stage 1 + Stage 2). Stage 2 will be extended if needed."""
    freeze_weight: float = 1000.0
    """Weight for handle freeze constraint in stage 2 (soft constraint penalty)"""
    enable_freeze: bool = True
    """Whether to enable handle freeze in stage 2"""
    hard_handle: bool = False
    """Enforce handle targets via Poisson positional constraints (same as anchors).
    Disables soft handle loss and Handle Freeze."""
    
    lambda_kp_stage1: float = 1.0
    """Weight for keypoint loss in stage 1 (before handle freeze)"""
    lambda_sds_stage1: float = 0.1
    """Weight for SDS loss in stage 1 (before handle freeze) - Light guidance with ROI masking"""
    lambda_geom_stage1: float = 1.0
    """Weight for geometry loss in stage 1 (before handle freeze)"""
    
    lambda_kp_stage2: float = 0.0
    """Weight for keypoint loss in stage 2 (after handle freeze)"""
    lambda_sds_stage2: float = 1.0
    """Weight for SDS loss in stage 2 (after handle freeze) - Increase for semantic guidance"""
    lambda_geom_stage2: float = 1.0
    """Weight for geometry loss in stage 2 (after handle freeze)"""
    stage2_steps: int = 600
    """Number of fixed steps to run in stage 2 (after handle freeze)"""

    # Solver-only virtual seam coupling (disconnected CAD / kitbash meshes)
    virtual_seams: bool = True
    """Enable virtual component coupling without changing rest mesh topology"""
    seam_tau: Optional[float] = None
    """Absolute adjacency threshold. None = seam_tau_frac * bbox diagonal"""
    seam_tau_frac: float = 0.02
    """Fallback tau as a fraction of bbox diagonal (1%~3%)"""
    seam_gap_delta_frac: float = 0.0075
    """Allowed extra opening as a fraction of bbox diagonal (~0.5%~1%)"""
    lambda_disp_stage1: float = 50.0
    """Fallback / minor displacement-consistency weight in stage 1"""
    lambda_gap_stage1: float = 150.0
    """Fallback / minor anti-separation hinge weight in stage 1"""
    lambda_disp_stage2: float = 25.0
    """Fallback / minor displacement-consistency weight in stage 2"""
    lambda_gap_stage2: float = 80.0
    """Fallback / minor anti-separation hinge weight in stage 2"""
    lambda_disp_major_stage1: float = 150.0
    """Major structural seam displacement weight in stage 1"""
    lambda_gap_major_stage1: float = 400.0
    """Major structural seam anti-separation weight in stage 1"""
    lambda_disp_major_stage2: float = 100.0
    """Major structural seam displacement weight in stage 2"""
    lambda_gap_major_stage2: float = 300.0
    """Major structural seam anti-separation weight in stage 2"""
    lambda_disp_minor_stage1: float = 40.0
    """Minor / decorative seam displacement weight in stage 1"""
    lambda_gap_minor_stage1: float = 100.0
    """Minor / decorative seam anti-separation weight in stage 1"""
    lambda_disp_minor_stage2: float = 20.0
    """Minor / decorative seam displacement weight in stage 2"""
    lambda_gap_minor_stage2: float = 60.0
    """Minor / decorative seam anti-separation weight in stage 2"""
    lambda_normal_major_stage1: float = 150.0
    """Major seam normal anti-opening weight in stage 1"""
    lambda_normal_major_stage2: float = 100.0
    """Major seam normal anti-opening weight in stage 2"""
    lambda_group: float = 80.0
    """Structural-group relative centroid motion weight"""
    lambda_gap_max: float = 1000.0
    """Cap for adaptive per-seam lambda_gap"""
    seam_dense_min: int = 30
    """Minimum virtual pairs on a dense/major seam"""
    seam_dense_max: int = 50
    """Maximum virtual pairs on a dense/major seam"""
    seam_focus_largest_major: bool = False
    """If True, only one longest seam is major. False = all long/handle seams."""
    poisson_rel_lambda: float = 1.0
    """Relative seam weight inside the Poisson linear system (numerical coupling)"""

    # Experiment configs
    lr_stage_1: float = 1e-3
    """Optimizer learning rate for stage 1"""
    lr_stage_2: float = 1e-3
    """Optimizer learning rate for stage 2"""
    n_iter: int = 1300
    """Number of optimization iterations"""
    n_kp_iter: int = 300
    """Number of keypoint optimization only iterations"""
    vis_every: int = 10
    """Visualize every n iterations"""
    device = torch.device("cuda")
    """Device to use for optimization"""
    seed: int = 2024
    """Random seed"""

@dataclass
class StableDiffusion2DPipeConfig:

    # Guidance parameters
    guidance_type: Literal["sd"] = "sd"
    """Type of guidance to use"""
    lora_dir: Optional[Path] = None
    """Path to LoRA checkpoint"""
    lora_scale: float = 1.0
    """LoRA scale used for the experiment"""
    cfg_scale: float = 100.0
    """CFG scale used for the experiment"""
    clamp_val: float = 2.0
    """Clamping value for the guidance"""

    # Poisson parameters
    train_J: bool = True
    """Whether to train source Jacobian field"""
    train_V: bool = False
    """Optimize vertex positions directly (mutually exclusive with train_J)"""
    train_quat: bool = False
    """Whether to train quaternion field"""

    # Renderer parameters
    cam_schedule: Literal["random", "sequential"] = "random"
    """Camera scheduler"""
    cam_locs: Shaped[torch.Tensor, "* 3"] = torch.tensor(
        [
            [0.0, 0.0, 1.0],
        ]
    )
    """Camera locations"""
    origin: Shaped[torch.Tensor, "3"] = torch.tensor([0.0, 0.0, 0.0])
    """Origin of the scene"""
    img_height: int = 512
    """Height of the rendered image"""
    img_width: int = 512
    """Width of the rendered image"""
    aspect_ratio: float = 1.0
    """Aspect ratio of the rendered image"""
    fov: float = 53.14
    """Field of view of the rendered image"""
    near: float = 1e-1
    """Near plane of the rendered image"""
    far: float = 1e10
    """Far plane of the rendered image"""
    use_random_bg: bool = True
    """Whether to use random background"""

    # Loss configs
    w_guidance: float = 1.0
    """Weight for guidance loss"""
    w_kp: float = 1.0
    """Weight for keypoint matching loss"""
    w_quat_reg: float = 0.0
    """Weight for quaternion regularization loss"""

    # Experiment configs
    lr_stage_1: float = 1e-3
    """Optimizer learning rate for stage 1"""
    lr_stage_2: float = 1e-3
    """Optimizer learning rate for stage 2"""
    n_iter: int = 1300
    """Number of optimization iterations"""
    n_kp_iter: int = 300
    """Number of keypoint optimization only iterations"""
    vis_every: int = 10
    """Visualize every n iterations"""
    device = torch.device("cuda")
    """Device to use for optimization"""
    seed: int = 2024
    """Random seed"""
