# src/prior/sparse_shape_prior.py
import os, math, torch
from dataclasses import dataclass
from typing import Dict, Optional

try:
    import igl  # pip install libigl
except Exception:
    igl = None

# ------------------------- Hyper & Schedule ------------------------- #
@dataclass
class SparseShapeHyper:
    # loss weights & sampling
    alpha: float = 1.0      # L_surf
    beta: float = 0.3       # L_io (reduced from 1.0)
    gamma: float = 0.0      # L_norm (disabled - not used in Stage-1)
    M_surface: int = 4096   # increased from 2000
    K_volume: int = 4096    # increased from 4000
    resample_every: int = 10

    # robust & sign params
    delta: float = 0.02     # Huber threshold (increased from 0.01)
    tau: float = 0.05       # tanh temperature for soft sign (increased from 0.02)
    margin: float = 0.02    # hinge margin (decreased from 0.7)
    near_frac: float = 0.01 # L_norm only when dist < near_frac * diag (not used)

    # stage-2 scheduling for lambda_shape_t
    warmup: int = 300
    hold: int = 800
    decay: int = 600
    peak: float = 0.10
    tail: float = 0.02

    @staticmethod
    def from_env():
        h = SparseShapeHyper()
        h.alpha = float(os.getenv("APAP_SSP_ALPHA", h.alpha))  # L_surf权重
        h.beta = float(os.getenv("APAP_SSP_BETA", h.beta))     # L_io权重
        h.gamma = 0.0  # Always 0, ignore env var - L_norm disabled
        h.M_surface = int(os.getenv("APAP_SSP_M", h.M_surface))
        h.K_volume  = int(os.getenv("APAP_SSP_K", h.K_volume))
        h.resample_every = int(os.getenv("APAP_SSP_R", h.resample_every))
        h.delta = float(os.getenv("APAP_SSP_DELTA", h.delta))
        h.tau   = float(os.getenv("APAP_SSP_TAU", h.tau))
        h.margin= float(os.getenv("APAP_SSP_MARGIN", h.margin))
        h.warmup= int(os.getenv("APAP_SSP_WARMUP", h.warmup))
        h.hold  = int(os.getenv("APAP_SSP_HOLD", h.hold))
        h.decay = int(os.getenv("APAP_SSP_DECAY", h.decay))
        h.peak  = float(os.getenv("APAP_SSP_LPEAK", h.peak))
        h.tail  = float(os.getenv("APAP_SSP_LTAIL", h.tail))
        return h


def sparse_shape_hyper_from_config(cfg) -> SparseShapeHyper:
    """Build SparseShapeHyper from pipeline SspConfig, with env var overrides."""
    h = SparseShapeHyper(
        alpha=cfg.alpha,
        beta=cfg.beta,
        gamma=0.0,
        M_surface=cfg.M_surface,
        K_volume=cfg.K_volume,
        resample_every=cfg.resample_every,
        delta=cfg.delta,
        tau=cfg.tau,
        margin=cfg.margin,
        warmup=cfg.warmup,
        hold=cfg.hold,
        decay=cfg.decay,
        peak=cfg.peak,
        tail=cfg.tail,
    )
    h.alpha = float(os.getenv("APAP_SSP_ALPHA", h.alpha))
    h.beta = float(os.getenv("APAP_SSP_BETA", h.beta))
    h.M_surface = int(os.getenv("APAP_SSP_M", h.M_surface))
    h.K_volume = int(os.getenv("APAP_SSP_K", h.K_volume))
    h.resample_every = int(os.getenv("APAP_SSP_R", h.resample_every))
    h.delta = float(os.getenv("APAP_SSP_DELTA", h.delta))
    h.tau = float(os.getenv("APAP_SSP_TAU", h.tau))
    h.margin = float(os.getenv("APAP_SSP_MARGIN", h.margin))
    h.warmup = int(os.getenv("APAP_SSP_WARMUP", h.warmup))
    h.hold = int(os.getenv("APAP_SSP_HOLD", h.hold))
    h.decay = int(os.getenv("APAP_SSP_DECAY", h.decay))
    h.peak = float(os.getenv("APAP_SSP_LPEAK", h.peak))
    h.tail = float(os.getenv("APAP_SSP_LTAIL", h.tail))
    return h


def _huber(x, delta):
    ax = x.abs()
    return torch.where(ax < delta, 0.5*(x**2)/delta, ax - 0.5*delta)

def _cosine_lambda(t, h: SparseShapeHyper):
    if t < h.warmup:                # 0 -> peak
        return h.peak * t / max(1, h.warmup)
    t -= h.warmup
    if t < h.hold:                  # hold
        return h.peak
    t = min(h.decay, max(0, t - h.hold))
    if h.decay <= 0:
        return h.peak
    cosw = 0.5 * (1 + math.cos(math.pi * t / h.decay))
    return h.tail + (h.peak - h.tail) * cosw

# --------------------------- Template Cache ------------------------- #
class TemplateCache:
    def __init__(self, V0: torch.Tensor, F0: torch.Tensor):
        """V0: (N0,3) float32; F0: (M0,3) long"""
        self.V0 = V0.contiguous()
        self.F0 = F0.contiguous()
        self.device = V0.device
        with torch.no_grad():
            self.aabb_min = self.V0.min(0).values
            self.aabb_max = self.V0.max(0).values
            # precompute vertex normals for template (no grad needed)
            v0 = self.V0[self.F0[:,0]]; v1 = self.V0[self.F0[:,1]]; v2 = self.V0[self.F0[:,2]]
            fn = torch.nn.functional.normalize(torch.cross(v1-v0, v2-v0), dim=-1)
            self.vn0 = torch.zeros_like(self.V0)
            self.vn0.index_add_(0, self.F0[:,0], fn)
            self.vn0.index_add_(0, self.F0[:,1], fn)
            self.vn0.index_add_(0, self.F0[:,2], fn)
            self.vn0 = torch.nn.functional.normalize(self.vn0, dim=-1)

    @torch.no_grad()
    def sample_surface_points(self, M: int) -> torch.Tensor:
        V, F = self.V0, self.F0
        v0, v1, v2 = V[F[:,0]], V[F[:,1]], V[F[:,2]]
        area = 0.5 * torch.linalg.norm(torch.cross(v1 - v0, v2 - v0), dim=-1)
        prob = (area / (area.sum() + 1e-9)).clamp_min(1e-12)
        K = min(8192, F.shape[0])
        tri = F[torch.multinomial(prob, num_samples=K, replacement=True)]
        r1 = torch.sqrt(torch.rand(K,1, device=V.device))
        r2 = torch.rand(K,1, device=V.device)
        P = (1-r1)*V[tri[:,0]] + r1*(1-r2)*V[tri[:,1]] + r1*r2*V[tri[:,2]]
        # light FPS
        sel = [torch.randint(0, K, (1,), device=V.device)]
        d2 = torch.full((K,), float("inf"), device=V.device)
        for _ in range(min(M-1, 2048)):
            last = P[sel[-1]]
            d2 = torch.minimum(d2, ((P-last)**2).sum(-1))
            sel.append(torch.argmax(d2).view(1))
        return P[torch.unique(torch.cat(sel))][:M]

    @torch.no_grad()
    def sample_volume_points(self, K: int) -> torch.Tensor:
        u = torch.rand(K,3, device=self.device)
        return self.aabb_min + u * (self.aabb_max - self.aabb_min)

    @torch.no_grad()
    def label_inside_outside(self, Q: torch.Tensor) -> torch.Tensor:
        if igl is None:
            return torch.ones(Q.shape[0], device=Q.device)
        V0_np = self.V0.detach().cpu().numpy()
        F0_np = self.F0.detach().cpu().numpy().astype("int32")
        Q_np  = Q.detach().cpu().numpy()
        try:
            sdf, _, _, _ = igl.signed_distance(Q_np, V0_np, F0_np, return_normals=False)
            sig = torch.from_numpy((sdf <= 0).astype("float32")*2-1).to(Q.device)
        except Exception:
            wn = igl.winding_number(V0_np, F0_np, Q_np)
            sig = torch.from_numpy((wn >= 0.5).astype("float32")*2-1).to(Q.device)
        return sig.clamp(-1, 1)

# ------------------------------ Prior Loss --------------------------- #
class SparseShapePrior:
    """All ops wrt current mesh are differentiable."""
    def __init__(self, cache: TemplateCache, h: SparseShapeHyper):
        self.cache = cache
        self.h = h
        self._state: Dict[str, torch.Tensor] = {}

    def _ensure_samples(self, step: int):
        if ("P" not in self._state) or (step % self.h.resample_every == 0):
            P = self.cache.sample_surface_points(self.h.M_surface)  # (M,3)
            Q = self.cache.sample_volume_points(self.h.K_volume)    # (K,3)
            sigma = self.cache.label_inside_outside(Q)              # {-1,+1}
            self._state["P"], self._state["Q"], self._state["sigma"] = P, Q, sigma

    @staticmethod
    def _nearest_vertex(P: torch.Tensor, V: torch.Tensor):
        # returns indices of nearest vertices (piecewise diff)
        d2 = torch.cdist(P, V)           # (N,M)
        idx = d2.argmin(dim=1)           # (N,)
        return idx, torch.gather(V, 0, idx[:,None].repeat(1,3))

    @staticmethod
    def _vertex_normals(V: torch.Tensor, F: torch.Tensor):
        v0 = V[F[:,0]]; v1 = V[F[:,1]]; v2 = V[F[:,2]]
        fn = torch.nn.functional.normalize(torch.cross(v1-v0, v2-v0), dim=-1)
        vn = torch.zeros_like(V)
        vn.index_add_(0, F[:,0], fn)
        vn.index_add_(0, F[:,1], fn)
        vn.index_add_(0, F[:,2], fn)
        return torch.nn.functional.normalize(vn, dim=-1)

    def compute(self, V_cur: torch.Tensor, F: torch.Tensor, step: int, 
                per_vertex_weight: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute SSP loss with optional per-vertex weighting.
        
        Args:
            V_cur: Current mesh vertices (V, 3)
            F: Face indices (F, 3)
            step: Current optimization step
            per_vertex_weight: Optional per-vertex weight (V,) for ROI-aware SSP.
                              If provided, samples will be weighted based on their location.
        
        Returns:
            Dictionary with loss components
        """
        self._ensure_samples(step)
        P, Q, sigma = self._state["P"], self._state["Q"], self._state["sigma"]

        # current mesh vertex normals (diff)
        vn_cur = self._vertex_normals(V_cur, F)

        # ---- Compute sample weights from per_vertex_weight if provided ---- #
        w_surf = None
        w_vol = None
        if per_vertex_weight is not None:
            # For surface samples P: use nearest vertex weight
            idxP_for_weight, _ = self._nearest_vertex(P, V_cur)
            w_surf = per_vertex_weight[idxP_for_weight]
            # Normalize to prevent overall scaling
            w_surf = w_surf / (w_surf.mean() + 1e-8)
            
            # For volume samples Q: use nearest vertex weight
            idxQ_for_weight, _ = self._nearest_vertex(Q, V_cur)
            w_vol = per_vertex_weight[idxQ_for_weight]
            w_vol = w_vol / (w_vol.mean() + 1e-8)

        # ---- L_surf: Huber(dist to current mesh via nearest vertex) ---- #
        idxP, VnP = self._nearest_vertex(P, V_cur)
        distP = (P - VnP).norm(dim=1)
        surf_err = _huber(distP, self.h.delta)
        if w_surf is not None:
            L_surf = (w_surf * surf_err).mean()
        else:
            L_surf = surf_err.mean()

        # ---- L_io: one-sided hinge on soft sign (diff) ---- #
        idxQ, VnQ = self._nearest_vertex(Q, V_cur)
        nQ = vn_cur[idxQ]
        s_cur = ((Q - VnQ) * nQ).sum(-1)             # local signed distance approx
        sigma_cur = torch.tanh(-s_cur / self.h.tau)  # (-1,1)
        io_err = torch.clamp(self.h.margin - sigma * sigma_cur, min=0)
        if w_vol is not None:
            L_io = (w_vol * io_err).mean()
        else:
            L_io = io_err.mean()

        # ---- L_norm: disabled (set to 0, not used in Stage-1) ---- #
        # We keep the interface for compatibility but don't compute it
        L_norm = torch.zeros((), device=V_cur.device)

        # Only use L_surf + L_io (L_norm disabled via gamma=0)
        L_shape = self.h.alpha * L_surf + self.h.beta * L_io
        lam_t = torch.tensor(_cosine_lambda(step, self.h), device=V_cur.device)
        return {"L_shape": L_shape, "L_surf": L_surf.detach(),
                "L_io": L_io.detach(), "L_norm": L_norm.detach(),
                "lambda_t": lam_t}

