"""Solver-only virtual seam coupling for disconnected CAD / kitbash meshes.

Does not change mesh vertices, faces, UVs, or indices. Component adjacency and
vertex pairs are used as Poisson relative constraints and as deformation losses.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.spatial import cKDTree


def geometry_fingerprint(
    v: np.ndarray,
    f: np.ndarray,
    uvs: Optional[np.ndarray] = None,
    tex_inds: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    def _h(arr: Optional[np.ndarray]) -> str:
        if arr is None:
            return "none"
        a = np.ascontiguousarray(arr)
        return hashlib.sha1(a.tobytes()).hexdigest()

    return {
        "n_vertices": int(v.shape[0]),
        "n_faces": int(f.shape[0]),
        "n_vt": 0 if uvs is None else int(uvs.shape[0]),
        "v_hash": _h(np.asarray(v, dtype=np.float32)),
        "f_hash": _h(np.asarray(f, dtype=np.int32)),
        "uv_hash": _h(None if uvs is None else np.asarray(uvs, dtype=np.float32)),
        "uv_inds_hash": _h(
            None if tex_inds is None else np.asarray(tex_inds, dtype=np.int32)
        ),
    }


def assert_original_geometry(
    v: np.ndarray,
    f: np.ndarray,
    uvs: Optional[np.ndarray],
    tex_inds: Optional[np.ndarray],
    meta_path: Path,
    atol: float = 1e-8,
) -> None:
    """Fail if the loaded mesh drifted from the restored original OBJ."""
    if not meta_path.exists():
        print(f"[VirtualSeam] no fingerprint at {meta_path}, skip assert")
        return
    expected = json.loads(meta_path.read_text(encoding="utf-8"))
    got = geometry_fingerprint(v, f, uvs, tex_inds)
    for key in ("n_vertices", "n_faces", "n_vt", "f_hash", "uv_hash", "uv_inds_hash"):
        if expected.get(key) != got.get(key):
            raise AssertionError(
                f"Original mesh topology/UV changed: {key} "
                f"expected {expected.get(key)} got {got.get(key)}"
            )
    if expected.get("v_hash") != got.get("v_hash"):
        raise AssertionError(
            "Original vertex coordinates changed "
            f"(hash mismatch, nV={got['n_vertices']}). "
            "Virtual-seam coupling must not weld or move rest vertices."
        )


def connected_components(nv: int, faces: np.ndarray) -> list[np.ndarray]:
    adj: list[list[int]] = [[] for _ in range(nv)]
    for a, b, c in faces.astype(np.int64):
        for u, w in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            adj[u].append(w)
            adj[w].append(u)
    seen = np.zeros(nv, dtype=bool)
    comps: list[np.ndarray] = []
    for i in range(nv):
        if seen[i]:
            continue
        q = deque([i])
        seen[i] = True
        cur = [i]
        while q:
            u = q.popleft()
            for w in adj[u]:
                if not seen[w]:
                    seen[w] = True
                    q.append(w)
                    cur.append(w)
        comps.append(np.asarray(cur, dtype=np.int64))
    comps.sort(key=len, reverse=True)
    return comps


def bbox_diagonal(v: np.ndarray) -> float:
    lo = v.min(0)
    hi = v.max(0)
    return float(np.linalg.norm(hi - lo))


def nearest_component_distance(
    va: np.ndarray, vb: np.ndarray
) -> tuple[float, int, int]:
    if len(va) <= len(vb):
        tree = cKDTree(vb)
        d, nn = tree.query(va, k=1, workers=-1)
        k = int(np.argmin(d))
        return float(d[k]), k, int(nn[k])
    tree = cKDTree(va)
    d, nn = tree.query(vb, k=1, workers=-1)
    k = int(np.argmin(d))
    return float(d[k]), int(nn[k]), k


def _fps_subset(points: np.ndarray, n_keep: int, start: int = 0) -> list[int]:
    n = len(points)
    if n_keep >= n:
        return list(range(n))
    if n_keep <= 0:
        return []
    chosen = [int(start)]
    dist = np.full(n, np.inf, dtype=np.float64)
    for _ in range(n_keep - 1):
        last = points[chosen[-1]]
        dist = np.minimum(dist, np.linalg.norm(points - last, axis=1))
        dist[np.asarray(chosen, dtype=np.int64)] = -1.0
        chosen.append(int(np.argmax(dist)))
    return chosen


def _collect_unique_pairs(
    verts: np.ndarray,
    ia: np.ndarray,
    ib: np.ndarray,
    band: float,
) -> list[tuple[float, int, int]]:
    """One-to-one mutual NN, then greedy unique NN fill. No vertex reused."""
    if len(ia) == 0 or len(ib) == 0:
        return []
    va = verts[ia]
    vb = verts[ib]
    d_ab, nn_ab = cKDTree(vb).query(va, k=1, workers=-1)
    d_ba, nn_ba = cKDTree(va).query(vb, k=1, workers=-1)
    d_ab = np.asarray(d_ab, dtype=np.float64).reshape(-1)
    nn_ab = np.asarray(nn_ab, dtype=np.int64).reshape(-1)
    nn_ba = np.asarray(nn_ba, dtype=np.int64).reshape(-1)

    used_a: set[int] = set()
    used_b: set[int] = set()
    mutual: list[tuple[float, int, int]] = []
    for si in range(len(ia)):
        d = float(d_ab[si])
        if not np.isfinite(d) or d > band:
            continue
        tj = int(nn_ab[si])
        if int(nn_ba[tj]) != si:
            continue
        a = int(ia[si])
        b = int(ib[tj])
        if a in used_a or b in used_b:
            continue
        used_a.add(a)
        used_b.add(b)
        mutual.append((d, a, b))

    cands: list[tuple[float, int, int]] = []
    for si in range(len(ia)):
        d = float(d_ab[si])
        if not np.isfinite(d) or d > band:
            continue
        cands.append((d, int(ia[si]), int(ib[int(nn_ab[si])])))
    cands.sort(key=lambda x: x[0])
    for d, a, b in cands:
        if a in used_a or b in used_b:
            continue
        used_a.add(a)
        used_b.add(b)
        mutual.append((d, a, b))
    return mutual


def _spread_keep_indices(mids: np.ndarray, n_keep: int, thirds: bool) -> list[int]:
    n = len(mids)
    n_keep = min(int(n_keep), n)
    if n_keep <= 0:
        return []
    if (not thirds) or n_keep < 9 or n < 9:
        return _fps_subset(mids, n_keep, start=0)

    xz = mids[:, [0, 2]]
    xz = xz - xz.mean(0, keepdims=True)
    cov = xz.T @ xz
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]
    t = xz @ axis
    tmin = float(t.min())
    tmax = float(t.max())
    span = max(tmax - tmin, 1e-8)
    cuts = [tmin - 1e-9, tmin + span / 3.0, tmin + 2.0 * span / 3.0, tmax + 1e-9]
    n_each = [n_keep // 3, n_keep // 3, n_keep - 2 * (n_keep // 3)]
    chosen: list[int] = []
    used: set[int] = set()
    for b in range(3):
        idx = [k for k in range(n) if cuts[b] <= float(t[k]) < cuts[b + 1]]
        if not idx:
            # empty bin: take the extreme along this third of the axis
            lo, hi = cuts[b], cuts[b + 1]
            idx = [int(np.argmin(np.abs(t - 0.5 * (lo + hi))))]
        n_b = min(max(1, n_each[b]), len(idx))
        sub = mids[np.asarray(idx)]
        local = _fps_subset(sub, n_b, start=0)
        for li in local:
            gi = int(idx[li])
            if gi not in used:
                used.add(gi)
                chosen.append(gi)
    if len(chosen) < n_keep:
        rest_pts = mids
        extra = _fps_subset(rest_pts, n_keep, start=chosen[0] if chosen else 0)
        for gi in extra:
            if gi not in used:
                used.add(int(gi))
                chosen.append(int(gi))
            if len(chosen) >= n_keep:
                break
    return chosen[:n_keep]


def sample_small_side_all(
    verts: np.ndarray,
    small: np.ndarray,
    large: np.ndarray,
    band: float,
) -> list[tuple[float, int, int]]:
    """Pair every small-side vertex to its NN on the large side (handle/roof)."""
    if len(small) == 0 or len(large) == 0:
        return []
    d, nn = cKDTree(verts[large]).query(verts[small], k=1, workers=-1)
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    nn = np.asarray(nn, dtype=np.int64).reshape(-1)
    out: list[tuple[float, int, int]] = []
    for si in range(len(small)):
        dist = float(d[si])
        if not np.isfinite(dist) or dist > band:
            continue
        out.append((dist, int(small[si]), int(large[int(nn[si])])))
    return out


def sample_interface_pairs(
    verts: np.ndarray,
    ia: np.ndarray,
    ib: np.ndarray,
    tau: float,
    n_pairs: int,
    spread_thirds: bool = False,
) -> list[tuple[float, int, int]]:
    """Mutual nearest neighbors on the near-interface, then farthest-point spread."""
    if len(ia) == 0 or len(ib) == 0 or n_pairs <= 0:
        return []
    band = max(float(tau), 1e-8)
    mutual = _collect_unique_pairs(verts, ia, ib, band)
    if not mutual:
        return []
    n_keep = min(int(n_pairs), len(mutual))
    mids = np.stack([(verts[a] + verts[b]) * 0.5 for _, a, b in mutual], axis=0)
    keep = _spread_keep_indices(mids, n_keep, thirds=spread_thirds)
    out = [mutual[k] for k in keep]
    out.sort(key=lambda x: x[0])
    return out


def pair_budget(n_a: int, n_b: int) -> int:
    n_if = min(n_a, n_b)
    if n_if >= 80:
        return 16
    if n_if >= 30:
        return 10
    return int(min(8, max(3, n_if // 2)))


def dense_pair_budget(n_available: int, n_min: int = 30, n_max: int = 50) -> int:
    if n_available <= 0:
        return 0
    if n_available < n_min:
        return int(n_available)
    return int(min(n_max, n_available))


class _DSU:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(x) for x in obj]
    return obj


def _interface_metrics(
    verts: np.ndarray,
    ia: np.ndarray,
    ib: np.ndarray,
    sampled: list[tuple[float, int, int]],
) -> dict[str, float]:
    ca = verts[ia].mean(0)
    cb = verts[ib].mean(0)
    if sampled:
        mids = np.stack([(verts[a] + verts[b]) * 0.5 for _, a, b in sampled])
        span = mids.max(0) - mids.min(0)
    else:
        span = np.zeros(3, dtype=np.float64)
    hspan = float(max(span[0], span[2]))
    vspan = float(span[1])
    return {
        "hspan": hspan,
        "vspan": vspan,
        "dy": float(abs(ca[1] - cb[1])),
        "cy_a": float(ca[1]),
        "cy_b": float(cb[1]),
        "nx": float(ca[0] - cb[0]),
        "ny": float(ca[1] - cb[1]),
        "nz": float(ca[2] - cb[2]),
    }


def _is_major_horizontal(
    n_a: int,
    n_b: int,
    hspan: float,
    vspan: float,
    dy: float,
    bbox_h: float,
    n_pool: int,
    touches_handle: bool = False,
) -> bool:
    """Long, flat, stacked interface, or any handle-adjacent roof/floor seam."""
    if hspan < 1e-6:
        return False
    # Roof cornice / handle piece, or a handle-adjacent fracture (gear split).
    if touches_handle and max(hspan, vspan) >= 0.08 * max(bbox_h, 1e-6):
        return True
    # Two large pieces with a long interface, any orientation.
    if (
        min(n_a, n_b) >= 80
        and n_pool >= 20
        and max(hspan, vspan) >= 0.25 * max(bbox_h, 1e-6)
    ):
        return True
    if max(n_a, n_b) < 80:
        return False
    if min(n_a, n_b) < 12:
        return False
    if n_pool < 8:
        return False
    if hspan < 0.25 * max(bbox_h, 1e-6):
        return False
    if vspan > 0.25 * hspan:
        return False
    if dy < 0.05:
        return False
    return True


def cluster_structural_groups(
    verts: np.ndarray,
    comps: list[np.ndarray],
    min_large: int = 20,
    y_gap: float = 0.08,
    handle_comps: Optional[list[int]] = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """1D agglomerative clustering of large components by centroid Y."""
    n = len(comps)
    centroids = np.stack(
        [verts[c].mean(0) if len(c) else np.zeros(3) for c in comps], axis=0
    )
    sizes = np.asarray([len(c) for c in comps], dtype=np.int64)
    large = [i for i in range(n) if int(sizes[i]) >= min_large]
    if not large:
        large = [int(np.argmax(sizes))]

    order = sorted(large, key=lambda i: float(centroids[i, 1]))
    clusters: list[list[int]] = []
    cur = [order[0]]
    for ci in order[1:]:
        if float(centroids[ci, 1]) - float(centroids[cur[-1], 1]) > y_gap:
            clusters.append(cur)
            cur = [ci]
        else:
            cur.append(ci)
    clusters.append(cur)

    comp_group = np.full(n, -1, dtype=np.int64)
    for gi, members in enumerate(clusters):
        for ci in members:
            comp_group[ci] = gi

    large_centroids = centroids[np.asarray(large)]
    for ci in range(n):
        if comp_group[ci] >= 0:
            continue
        d = np.linalg.norm(large_centroids - centroids[ci], axis=1)
        nearest = int(large[int(np.argmin(d))])
        comp_group[ci] = int(comp_group[nearest])

    # Keep a small handle component as its own group so E_group can
    # stop the roof cornice from drifting away inside a huge facade group.
    if handle_comps:
        for ci in handle_comps:
            if 0 <= int(ci) < n and int(sizes[int(ci)]) < 80:
                new_id = int(comp_group.max()) + 1 if n else 0
                comp_group[int(ci)] = new_id

    n_g = int(comp_group.max()) + 1 if n else 0
    groups: list[dict[str, Any]] = []
    y_means = []
    for gi in range(n_g):
        members = [i for i in range(n) if int(comp_group[i]) == gi]
        gverts = (
            np.concatenate([comps[i] for i in members])
            if members
            else np.zeros((0,), dtype=np.int64)
        )
        cy = float(verts[gverts, 1].mean()) if len(gverts) else 0.0
        y_means.append(cy)
        groups.append(
            {
                "id": gi,
                "members": members,
                "n_verts": int(len(gverts)),
                "centroid_y": cy,
                "name": "",
            }
        )

    # name by height rank: roof / upper / middle / lower / base
    rank = list(np.argsort(y_means)[::-1])
    labels_by_count = {
        1: ["body"],
        2: ["upper", "lower"],
        3: ["upper", "middle", "lower"],
        4: ["roof", "upper", "middle", "lower"],
        5: ["roof", "upper", "middle", "lower", "base"],
    }
    labels = labels_by_count.get(
        n_g, ["roof", "upper", "middle", "lower", "base"] + [f"g{k}" for k in range(5, n_g)]
    )
    labels = labels[:n_g]
    for k, gi in enumerate(rank):
        groups[gi]["name"] = labels[k] if k < len(labels) else f"g{gi}"
        groups[gi]["height_rank"] = k

    return comp_group, groups


@dataclass
class VirtualSeamCoupling:
    tau: float
    bbox_diag: float
    gap_delta: float
    comps: list[np.ndarray]
    vert_comp: np.ndarray
    adjacency: list[dict[str, Any]]
    pair_i: np.ndarray
    pair_j: np.ndarray
    offset0: np.ndarray
    dist0: np.ndarray
    pair_delta: np.ndarray
    pair_seam: np.ndarray
    pair_is_major: np.ndarray
    seam_is_major: np.ndarray
    seam_comp_a: np.ndarray
    seam_comp_b: np.ndarray
    seam_normal: np.ndarray
    rest_v: np.ndarray
    group_id: np.ndarray
    group_verts: list[np.ndarray]
    group_meta: list[dict[str, Any]]
    group_adj: list[tuple[int, int]]
    aux_pin_inds: np.ndarray
    aux_pin_pos: np.ndarray
    n_groups: int
    n_struct_groups: int
    poisson_pair_w: np.ndarray
    logs: dict[str, Any] = field(default_factory=dict)
    lambda_disp_major: float = 150.0
    lambda_gap_major: float = 400.0
    lambda_disp_minor: float = 40.0
    lambda_gap_minor: float = 100.0
    lambda_normal_major: float = 150.0
    lambda_group: float = 80.0
    lambda_gap_max: float = 1000.0
    gap_scale: np.ndarray = field(default_factory=lambda: np.ones((0,), dtype=np.float64))
    gap_history: list[dict[str, Any]] = field(default_factory=list)
    focus_seam: int = -1

    @property
    def n_pairs(self) -> int:
        return int(self.pair_i.shape[0])

    @property
    def n_seams(self) -> int:
        return int(self.seam_is_major.shape[0])

    def set_stage_weights(
        self,
        stage: int,
        *,
        disp_major: float,
        gap_major: float,
        disp_minor: float,
        gap_minor: float,
        normal_major: float,
        group: float,
        gap_max: float,
    ) -> None:
        self.lambda_disp_major = float(disp_major)
        self.lambda_gap_major = float(gap_major)
        self.lambda_disp_minor = float(disp_minor)
        self.lambda_gap_minor = float(gap_minor)
        self.lambda_normal_major = float(normal_major)
        self.lambda_group = float(group)
        self.lambda_gap_max = float(gap_max)

    def _seam_lambdas(self) -> tuple[np.ndarray, np.ndarray]:
        n_s = self.n_seams
        w_disp = np.where(
            self.seam_is_major, self.lambda_disp_major, self.lambda_disp_minor
        ).astype(np.float64)
        w_gap = np.where(
            self.seam_is_major, self.lambda_gap_major, self.lambda_gap_minor
        ).astype(np.float64)
        if len(self.gap_scale) != n_s:
            self.gap_scale = np.ones(n_s, dtype=np.float64)
        w_gap = np.minimum(w_gap * self.gap_scale, self.lambda_gap_max)
        return w_disp, w_gap

    def _per_seam_gap_stats(
        self, gap: np.ndarray, extra: np.ndarray
    ) -> list[dict[str, Any]]:
        rows = []
        for s in range(self.n_seams):
            m = self.pair_seam == s
            if not np.any(m):
                continue
            g = gap[m]
            e = extra[m]
            d0 = self.dist0[m]
            delta = self.pair_delta[m]
            viol = e > 0
            rec = {
                "seam": int(s),
                "is_major": bool(self.seam_is_major[s]),
                "comp_a": int(self.seam_comp_a[s]),
                "comp_b": int(self.seam_comp_b[s]),
                "num_pairs": int(m.sum()),
                "original_mean_gap": float(d0.mean()),
                "current_mean_gap": float(g.mean()),
                "current_max_gap": float(g.max()),
                "extra_mean": float((g - d0).mean()),
                "extra_max": float((g - d0).max()),
                "violation_ratio": float(viol.mean()),
                "delta_mean": float(delta.mean()),
                "d0_mean_plus_2delta": float(d0.mean() + 2.0 * delta.mean()),
            }
            rows.append(rec)
        return rows

    def adapt_gap_weights(self, seam_rows: list[dict[str, Any]]) -> list[str]:
        """Per-seam lambda_gap bump. Does not raise weights on stable seams."""
        notes = []
        if len(self.gap_scale) != self.n_seams:
            self.gap_scale = np.ones(self.n_seams, dtype=np.float64)
        w_disp, w_gap = self._seam_lambdas()
        for rec in seam_rows:
            s = int(rec["seam"])
            if not rec["is_major"]:
                continue
            hot = rec["violation_ratio"] > 0.2 or rec.get("extra_max", 0.0) > max(
                2.0 * rec.get("delta_mean", 0.01), 0.015
            )
            if not hot:
                continue
            if float(w_gap[s]) >= self.lambda_gap_max - 1e-6:
                notes.append(f"seam {s} lambda_gap already at cap {self.lambda_gap_max}")
                continue
            self.gap_scale[s] *= 1.5
            _, w_gap2 = self._seam_lambdas()
            notes.append(
                f"seam {s} C{rec['comp_a']}-C{rec['comp_b']} "
                f"lambda_gap -> {float(w_gap2[s]):.1f} "
                f"(viol={rec['violation_ratio']:.3f} max={rec['current_max_gap']:.4f})"
            )
        return notes

    def losses(self, curr_v):
        """Weighted SUM E_disp / E_gap / E_normal / E_group. curr_v: (V, 3)."""
        import torch

        z = curr_v.new_zeros(())
        empty_stats = {
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
        }
        if self.n_pairs == 0:
            return z, z, z, z, empty_stats

        i = torch.as_tensor(self.pair_i, device=curr_v.device, dtype=torch.long)
        j = torch.as_tensor(self.pair_j, device=curr_v.device, dtype=torch.long)
        off0 = torch.as_tensor(self.offset0, device=curr_v.device, dtype=curr_v.dtype)
        d0 = torch.as_tensor(self.dist0, device=curr_v.device, dtype=curr_v.dtype)
        delta = torch.as_tensor(
            self.pair_delta, device=curr_v.device, dtype=curr_v.dtype
        )
        seam_ids = torch.as_tensor(
            self.pair_seam, device=curr_v.device, dtype=torch.long
        )
        w_disp_s, w_gap_s = self._seam_lambdas()
        w_disp_p = torch.as_tensor(
            w_disp_s, device=curr_v.device, dtype=curr_v.dtype
        )[seam_ids]
        w_gap_p = torch.as_tensor(
            w_gap_s, device=curr_v.device, dtype=curr_v.dtype
        )[seam_ids]
        is_maj = torch.as_tensor(
            self.pair_is_major, device=curr_v.device, dtype=torch.bool
        )

        vi = curr_v[i]
        vj = curr_v[j]
        rel = (vi - vj) - off0
        disp2 = torch.sum(rel * rel, dim=-1)
        e_disp = torch.sum(w_disp_p * disp2)

        gap = torch.linalg.norm(vi - vj, dim=-1)
        extra = torch.clamp(gap - d0 - delta, min=0.0)
        e_gap = torch.sum(w_gap_p * extra * extra)

        e_normal = z
        if bool(is_maj.any()):
            nrm = torch.as_tensor(
                self.seam_normal, device=curr_v.device, dtype=curr_v.dtype
            )
            n_p = nrm[seam_ids]
            n_dot = torch.sum(rel * n_p, dim=-1)
            e_normal = self.lambda_normal_major * torch.sum(
                n_dot[is_maj] * n_dot[is_maj]
            )

        e_group = z
        if self.group_adj and self.lambda_group > 0:
            acc = z
            for ga, gb in self.group_adj:
                va_idx = torch.as_tensor(
                    self.group_verts[ga], device=curr_v.device, dtype=torch.long
                )
                vb_idx = torch.as_tensor(
                    self.group_verts[gb], device=curr_v.device, dtype=torch.long
                )
                if va_idx.numel() == 0 or vb_idx.numel() == 0:
                    continue
                c_a = curr_v[va_idx].mean(0)
                c_b = curr_v[vb_idx].mean(0)
                c_a0 = torch.as_tensor(
                    self.rest_v[self.group_verts[ga]].mean(0),
                    device=curr_v.device,
                    dtype=curr_v.dtype,
                )
                c_b0 = torch.as_tensor(
                    self.rest_v[self.group_verts[gb]].mean(0),
                    device=curr_v.device,
                    dtype=curr_v.dtype,
                )
                dlt = (c_a - c_b) - (c_a0 - c_b0)
                acc = acc + torch.sum(dlt * dlt)
            e_group = self.lambda_group * acc

        gap_np = gap.detach().cpu().numpy()
        extra_np = extra.detach().cpu().numpy()
        seam_rows = self._per_seam_gap_stats(gap_np, extra_np)
        viol = float((extra > 0).float().mean().detach())
        largest = max(seam_rows, key=lambda r: r.get("extra_max", 0.0)) if seam_rows else None
        focus = None
        if self.focus_seam >= 0:
            focus = next((r for r in seam_rows if r["seam"] == self.focus_seam), None)
        elif seam_rows:
            majors = [r for r in seam_rows if r["is_major"]]
            focus = max(majors or seam_rows, key=lambda r: r["num_pairs"])

        stats = {
            "mean_gap": float(gap.mean().detach()),
            "max_gap": float(gap.max().detach()),
            "mean_gap0": float(d0.mean().detach()),
            "violation_ratio": viol,
            "e_group": float(e_group.detach()) if torch.is_tensor(e_group) else 0.0,
            "e_normal": float(e_normal.detach()) if torch.is_tensor(e_normal) else 0.0,
            "focus_mean_gap": float(focus["current_mean_gap"]) if focus else 0.0,
            "focus_max_gap": float(focus["current_max_gap"]) if focus else 0.0,
            "largest_sep_seam": int(largest["seam"]) if largest else -1,
            "largest_sep_gap": float(largest.get("extra_max", 0.0)) if largest else 0.0,
            "largest_sep_comp_a": int(largest["comp_a"]) if largest else -1,
            "largest_sep_comp_b": int(largest["comp_b"]) if largest else -1,
            "seam_rows": seam_rows,
            "w_disp_focus": float(
                w_disp_s[focus["seam"]] if focus is not None else 0.0
            ),
            "w_gap_focus": float(w_gap_s[focus["seam"]] if focus is not None else 0.0),
        }
        return e_disp, e_gap, e_group, e_normal, stats

    def format_debug_log(self, step: int, stats: dict[str, Any], notes: list[str]) -> str:
        w_disp, w_gap = self._seam_lambdas()
        lines = ["[building seam debug]", f"step={step}"]
        majors = [r for r in stats.get("seam_rows", []) if r["is_major"]]
        minors = [r for r in stats.get("seam_rows", []) if not r["is_major"]]
        lines.append(f"major seam count: {len(majors)}")
        lines.append(f"minor seam count: {len(minors)}")
        show = list(majors)
        hot_minors = sorted(minors, key=lambda r: r.get("extra_max", 0.0), reverse=True)[:3]
        show.extend(hot_minors)
        for rec in show:
            s = rec["seam"]
            tag = "MAJOR" if rec["is_major"] else "minor"
            ga = int(self.group_id[rec["comp_a"]]) if len(self.group_id) else -1
            gb = int(self.group_id[rec["comp_b"]]) if len(self.group_id) else -1
            na = self.group_meta[ga]["name"] if 0 <= ga < len(self.group_meta) else "?"
            nb = self.group_meta[gb]["name"] if 0 <= gb < len(self.group_meta) else "?"
            lines.append(
                f"  [{tag}] component/group pair: C{rec['comp_a']}-C{rec['comp_b']} "
                f"({na}/{nb})  num_pairs={rec['num_pairs']}"
            )
            lines.append(
                f"    original mean gap={rec['original_mean_gap']:.4f}  "
                f"current mean gap={rec['current_mean_gap']:.4f}  "
                f"current max gap={rec['current_max_gap']:.4f}  "
                f"extra_max={rec.get('extra_max', 0.0):.4f}"
            )
            lines.append(
                f"    violation ratio={rec['violation_ratio']:.3f}  "
                f"lambda_disp={float(w_disp[s]):.1f}  lambda_gap={float(w_gap[s]):.1f}"
            )
        lines.append(
            f"largest separation seam: {stats.get('largest_sep_seam')}  "
            f"group A: C{stats.get('largest_sep_comp_a')}  "
            f"group B: C{stats.get('largest_sep_comp_b')}  "
            f"max gap: {stats.get('largest_sep_gap', 0.0):.4f}"
        )
        for n in notes:
            lines.append(f"  adapt: {n}")
        return "\n".join(lines)


def build_virtual_seams(
    verts: np.ndarray,
    faces: np.ndarray,
    handle_inds: np.ndarray,
    anchor_inds: np.ndarray,
    tau: Optional[float] = None,
    tau_frac: float = 0.02,
    gap_delta_frac: float = 0.0075,
    building_tau: Optional[float] = None,
    per_pair_delta: bool = False,
    dense_n_min: int = 30,
    dense_n_max: int = 50,
    focus_largest_major: bool = False,
    lambda_group: float = 80.0,
) -> VirtualSeamCoupling:
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    handle_inds = np.asarray(handle_inds, dtype=np.int64).reshape(-1)
    anchor_inds = np.asarray(anchor_inds, dtype=np.int64).reshape(-1)

    comps = connected_components(len(verts), faces)
    n_comp = len(comps)
    vert_comp = np.full(len(verts), -1, dtype=np.int64)
    for ci, c in enumerate(comps):
        vert_comp[c] = ci

    diag = bbox_diagonal(verts)
    bbox_h = float(max(verts[:, 0].ptp(), verts[:, 2].ptp()))
    if tau is None:
        tau = max(float(tau_frac) * diag, 1e-4)
        if building_tau is not None:
            tau = max(tau, float(building_tau))
    gap_delta_global = float(gap_delta_frac) * diag

    handle_comps = sorted({int(vert_comp[i]) for i in handle_inds if 0 <= i < len(verts)})
    anchor_comps = sorted({int(vert_comp[i]) for i in anchor_inds if 0 <= i < len(verts)})

    sizes = [int(len(c)) for c in comps]
    face_counts = []
    for cset in comps:
        mask = np.isin(faces, cset).all(axis=1)
        face_counts.append(int(mask.sum()))

    min_dists: list[tuple[float, int, int]] = []
    cand_seams: list[dict[str, Any]] = []

    if n_comp > 1:
        for i in range(n_comp):
            for j in range(i + 1, n_comp):
                d, _, _ = nearest_component_distance(verts[comps[i]], verts[comps[j]])
                min_dists.append((d, i, j))
                if d > tau:
                    continue
                # Do not inflate the band: 3*dmin was pairing far decoration
                # verts and made rest gaps look like 0.15 openings.
                if building_tau is not None:
                    band = float(tau)
                else:
                    band = max(float(tau), 3.0 * float(d))
                pool = _collect_unique_pairs(verts, comps[i], comps[j], band)
                metrics = _interface_metrics(verts, comps[i], comps[j], pool)
                touches_handle = i in handle_comps or j in handle_comps
                is_major = _is_major_horizontal(
                    len(comps[i]),
                    len(comps[j]),
                    metrics["hspan"],
                    metrics["vspan"],
                    metrics["dy"],
                    bbox_h,
                    len(pool),
                    touches_handle=touches_handle,
                )
                cand_seams.append(
                    {
                        "comp_a": i,
                        "comp_b": j,
                        "min_dist": float(d),
                        "band": float(band),
                        "pool": pool,
                        "n_pool": len(pool),
                        "is_major": bool(is_major),
                        "touches_handle": bool(touches_handle),
                        **metrics,
                    }
                )

    major_cands = [s for s in cand_seams if s["is_major"]]
    focus_key = None
    if major_cands:
        focus = max(
            major_cands,
            key=lambda s: float(s["hspan"]) * float(min(sizes[s["comp_a"]], sizes[s["comp_b"]])),
        )
        focus_key = (focus["comp_a"], focus["comp_b"])
        if focus_largest_major:
            for s in cand_seams:
                s["is_major"] = (s["comp_a"], s["comp_b"]) == focus_key

    pair_i_list: list[int] = []
    pair_j_list: list[int] = []
    pair_seam_list: list[int] = []
    pair_major_list: list[bool] = []
    adjacency: list[dict[str, Any]] = []
    seam_is_major: list[bool] = []
    seam_comp_a: list[int] = []
    seam_comp_b: list[int] = []
    seam_normal: list[np.ndarray] = []

    for s in cand_seams:
        is_major = bool(s["is_major"])
        if is_major:
            if s.get("touches_handle"):
                ia, ib = comps[s["comp_a"]], comps[s["comp_b"]]
                if len(ia) <= len(ib):
                    sampled = sample_small_side_all(verts, ia, ib, s["band"])
                else:
                    sampled = sample_small_side_all(verts, ib, ia, s["band"])
                    sampled = [(d, b, a) for d, a, b in sampled]
                if len(sampled) > dense_n_max:
                    mids = np.stack(
                        [(verts[a] + verts[b]) * 0.5 for _, a, b in sampled]
                    )
                    keep = _spread_keep_indices(mids, dense_n_max, thirds=True)
                    sampled = [sampled[k] for k in keep]
            else:
                n_want = dense_pair_budget(s["n_pool"], dense_n_min, dense_n_max)
                sampled = sample_interface_pairs(
                    verts,
                    comps[s["comp_a"]],
                    comps[s["comp_b"]],
                    s["band"],
                    n_want,
                    spread_thirds=True,
                )
        else:
            n_want = pair_budget(len(comps[s["comp_a"]]), len(comps[s["comp_b"]]))
            sampled = sample_interface_pairs(
                verts,
                comps[s["comp_a"]],
                comps[s["comp_b"]],
                s["band"],
                n_want,
                spread_thirds=False,
            )
        if not sampled:
            continue
        sid = len(adjacency)
        nrm = np.asarray([s["nx"], s["ny"], s["nz"]], dtype=np.float64)
        nlen = float(np.linalg.norm(nrm))
        nrm = nrm / nlen if nlen > 1e-8 else np.array([0.0, 1.0, 0.0])
        adjacency.append(
            {
                "seam": sid,
                "comp_a": s["comp_a"],
                "comp_b": s["comp_b"],
                "min_dist": s["min_dist"],
                "n_pairs": len(sampled),
                "n_pool": s["n_pool"],
                "is_major": is_major,
                "touches_handle": bool(s.get("touches_handle")),
                "hspan": s["hspan"],
                "vspan": s["vspan"],
                "dy": s["dy"],
                "mean_pair_dist": float(np.mean([p[0] for p in sampled])),
            }
        )
        seam_is_major.append(is_major)
        seam_comp_a.append(int(s["comp_a"]))
        seam_comp_b.append(int(s["comp_b"]))
        seam_normal.append(nrm)
        for dist, va, vb in sampled:
            pair_i_list.append(int(va))
            pair_j_list.append(int(vb))
            pair_seam_list.append(sid)
            pair_major_list.append(is_major)

    if pair_i_list:
        pair_i = np.asarray(pair_i_list, dtype=np.int64)
        pair_j = np.asarray(pair_j_list, dtype=np.int64)
        offset0 = verts[pair_i] - verts[pair_j]
        dist0 = np.linalg.norm(offset0, axis=1)
        pair_seam = np.asarray(pair_seam_list, dtype=np.int64)
        pair_is_major = np.asarray(pair_major_list, dtype=bool)
    else:
        pair_i = np.zeros((0,), dtype=np.int64)
        pair_j = np.zeros((0,), dtype=np.int64)
        offset0 = np.zeros((0, 3), dtype=np.float64)
        dist0 = np.zeros((0,), dtype=np.float64)
        pair_seam = np.zeros((0,), dtype=np.int64)
        pair_is_major = np.zeros((0,), dtype=bool)

    handle_pair = np.zeros(len(pair_i), dtype=bool)
    if len(pair_i):
        handle_pair = np.array(
            [
                int(vert_comp[int(a)]) in handle_comps
                or int(vert_comp[int(b)]) in handle_comps
                for a, b in zip(pair_i.tolist(), pair_j.tolist())
            ],
            dtype=bool,
        )

    if per_pair_delta and len(dist0):
        pair_delta = np.maximum(0.3 * dist0, 0.005 * diag)
        pair_delta = np.minimum(pair_delta, 0.01 * diag)
        if handle_pair.any():
            hd = np.minimum(np.maximum(0.15 * dist0, 0.003 * diag), 0.006 * diag)
            pair_delta = np.where(handle_pair, hd, pair_delta)
    else:
        pair_delta = np.full_like(dist0, gap_delta_global)

    seam_is_major_arr = np.asarray(seam_is_major, dtype=bool)
    seam_comp_a_arr = np.asarray(seam_comp_a, dtype=np.int64)
    seam_comp_b_arr = np.asarray(seam_comp_b, dtype=np.int64)
    seam_normal_arr = (
        np.stack(seam_normal, axis=0)
        if seam_normal
        else np.zeros((0, 3), dtype=np.float64)
    )

    focus_seam = -1
    handle_majors = [s for s in cand_seams if s.get("is_major") and s.get("touches_handle")]
    if handle_majors:
        focus = max(handle_majors, key=lambda s: float(s["hspan"]))
        focus_key = (focus["comp_a"], focus["comp_b"])
    if focus_key is not None:
        for rec in adjacency:
            if (rec["comp_a"], rec["comp_b"]) == focus_key:
                focus_seam = int(rec["seam"])
                break

    poisson_pair_w = np.where(handle_pair, 6.0, np.where(pair_is_major, 3.0, 1.0)).astype(
        np.float64
    )

    dsu = _DSU(n_comp)
    for rec in adjacency:
        dsu.union(rec["comp_a"], rec["comp_b"])
    groups: dict[int, list[int]] = {}
    for ci in range(n_comp):
        groups.setdefault(dsu.find(ci), []).append(ci)
    n_groups = len(groups)

    anchor_group_roots = {dsu.find(c) for c in anchor_comps} if len(anchor_inds) else set()
    aux_pins: list[int] = []
    for root, members in groups.items():
        has_anchor = root in anchor_group_roots or any(m in anchor_comps for m in members)
        if has_anchor:
            continue
        group_verts = np.concatenate([comps[m] for m in members])
        group_handles = [int(h) for h in handle_inds if int(vert_comp[h]) in members]
        if group_handles:
            hpos = verts[np.asarray(group_handles, dtype=np.int64)]
            dmin = np.min(
                np.linalg.norm(verts[group_verts, None, :] - hpos[None, :, :], axis=2),
                axis=1,
            )
            aux_pins.append(int(group_verts[int(np.argmax(dmin))]))
        else:
            centroid = verts[group_verts].mean(0)
            d = np.linalg.norm(verts[group_verts] - centroid, axis=1)
            aux_pins.append(int(group_verts[int(np.argmin(d))]))

    if len(anchor_inds) == 0 and not aux_pins:
        aux_pins.append(int(comps[0][0]))

    aux_pin_inds = np.asarray(aux_pins, dtype=np.int64)
    aux_pin_pos = verts[aux_pin_inds] if len(aux_pin_inds) else np.zeros((0, 3))

    comp_group, group_meta = cluster_structural_groups(
        verts, comps, handle_comps=handle_comps
    )
    n_struct = len(group_meta)
    group_verts_list: list[np.ndarray] = []
    for gi in range(n_struct):
        members = group_meta[gi]["members"]
        gverts = (
            np.concatenate([comps[i] for i in members])
            if members
            else np.zeros((0,), dtype=np.int64)
        )
        group_verts_list.append(gverts)

    struct_adj: set[tuple[int, int]] = set()
    for rec in adjacency:
        ga = int(comp_group[rec["comp_a"]])
        gb = int(comp_group[rec["comp_b"]])
        if ga == gb:
            continue
        struct_adj.add((min(ga, gb), max(ga, gb)))
    group_adj = sorted(struct_adj)

    dist_arr = (
        np.asarray([d for d, _, _ in min_dists], dtype=np.float64)
        if min_dists
        else np.zeros((0,))
    )
    logs = {
        "n_components": n_comp,
        "component_sizes": sizes,
        "component_n_faces": face_counts,
        "handle_components": handle_comps,
        "anchor_components": anchor_comps,
        "tau": float(tau),
        "bbox_diag": diag,
        "gap_delta": gap_delta_global,
        "per_pair_delta": bool(per_pair_delta),
        "focus_largest_major": bool(focus_largest_major),
        "focus_seam": int(focus_seam),
        "focus_comps": list(focus_key) if focus_key else None,
        "min_dist_count": int(len(min_dists)),
        "min_dist_p10": float(np.percentile(dist_arr, 10)) if len(dist_arr) else None,
        "min_dist_p50": float(np.percentile(dist_arr, 50)) if len(dist_arr) else None,
        "min_dist_p90": float(np.percentile(dist_arr, 90)) if len(dist_arr) else None,
        "n_adjacent_pairs": len(adjacency),
        "adjacency": adjacency,
        "n_virtual_pairs": int(len(pair_i)),
        "n_major_pairs": int(pair_is_major.sum()) if len(pair_is_major) else 0,
        "mean_pair_gap0": float(dist0.mean()) if len(dist0) else 0.0,
        "n_virtual_groups": n_groups,
        "n_struct_groups": n_struct,
        "struct_groups": group_meta,
        "struct_group_adj": group_adj,
        "aux_pin_indices": aux_pin_inds.tolist(),
        "lambda_group": float(lambda_group),
    }

    print("[VirtualSeam] ========================================")
    print(f"[VirtualSeam] components={n_comp} sizes={sizes[:12]}{'...' if n_comp>12 else ''}")
    print(f"[VirtualSeam] handle_comps={handle_comps} anchor_comps={anchor_comps}")
    print(
        f"[VirtualSeam] bbox_diag={diag:.4f} tau={tau:.4f} "
        f"gap_delta_global={gap_delta_global:.4f} per_pair_delta={per_pair_delta}"
    )
    print(
        f"[VirtualSeam] adjacent={len(adjacency)} pairs={len(pair_i)} "
        f"major_pairs={int(pair_is_major.sum()) if len(pair_is_major) else 0} "
        f"focus_seam={focus_seam} focus_comps={focus_key}"
    )
    print(
        f"[VirtualSeam] virtual_groups={n_groups} struct_groups={n_struct} "
        f"names={[g['name'] for g in group_meta]} aux_pins={aux_pin_inds.tolist()}"
    )
    for rec in adjacency:
        if not rec["is_major"]:
            continue
        print(
            f"[VirtualSeam]   [MAJOR] C{rec['comp_a']}-C{rec['comp_b']} "
            f"dmin={rec['min_dist']:.4f} pairs={rec['n_pairs']}/{rec['n_pool']} "
            f"hspan={rec['hspan']:.3f} vspan={rec['vspan']:.3f} dy={rec['dy']:.3f}"
        )
    print("[VirtualSeam] ========================================")

    return VirtualSeamCoupling(
        tau=float(tau),
        bbox_diag=diag,
        gap_delta=gap_delta_global,
        comps=comps,
        vert_comp=vert_comp,
        adjacency=adjacency,
        pair_i=pair_i,
        pair_j=pair_j,
        offset0=offset0,
        dist0=dist0,
        pair_delta=pair_delta,
        pair_seam=pair_seam,
        pair_is_major=pair_is_major,
        seam_is_major=seam_is_major_arr,
        seam_comp_a=seam_comp_a_arr,
        seam_comp_b=seam_comp_b_arr,
        seam_normal=seam_normal_arr,
        rest_v=verts.copy(),
        group_id=comp_group,
        group_verts=group_verts_list,
        group_meta=group_meta,
        group_adj=group_adj,
        aux_pin_inds=aux_pin_inds,
        aux_pin_pos=aux_pin_pos,
        n_groups=n_groups,
        n_struct_groups=n_struct,
        poisson_pair_w=poisson_pair_w,
        logs=logs,
        lambda_group=float(lambda_group),
        gap_scale=np.ones(len(adjacency), dtype=np.float64),
        focus_seam=int(focus_seam),
    )


def write_debug_visualization(
    out_dir: Path,
    verts: np.ndarray,
    faces: np.ndarray,
    coupling: VirtualSeamCoupling,
    handle_inds: np.ndarray,
    anchor_inds: np.ndarray,
    curr_v: Optional[np.ndarray] = None,
    filename_png: str = "virtual_seams_debug.png",
    write_obj: bool = True,
) -> None:
    """Colored components + seam lines. Not written into the experiment mesh."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    colors = rng.rand(max(len(coupling.comps), 1), 3)
    colors = 0.35 + 0.65 * colors
    draw_v = np.asarray(verts if curr_v is None else curr_v, dtype=np.float64)

    viol_mask = np.zeros(coupling.n_pairs, dtype=bool)
    if coupling.n_pairs > 0 and curr_v is not None:
        gi = np.linalg.norm(draw_v[coupling.pair_i] - draw_v[coupling.pair_j], axis=1)
        viol_mask = gi > (coupling.dist0 + coupling.pair_delta)

    if write_obj:
        obj_path = out_dir / "virtual_seams_debug.obj"
        with obj_path.open("w", encoding="utf-8") as fh:
            fh.write("# virtual seam debug (do not use as experiment mesh)\n")
            for i, p in enumerate(verts):
                r, g, b = colors[int(coupling.vert_comp[i])]
                fh.write(f"v {p[0]} {p[1]} {p[2]} {r:.4f} {g:.4f} {b:.4f}\n")
            for a, b, c in faces.astype(np.int64):
                fh.write(f"f {a+1} {b+1} {c+1}\n")
            for a, b in zip(coupling.pair_i.tolist(), coupling.pair_j.tolist()):
                fh.write(f"l {a+1} {b+1}\n")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(14, 6))
        views = [(22, -60), (22, 30), (90, 0), (0, 0)]
        titles = ["view0", "view1", "top", "front"]
        for k, ((elev, azim), title) in enumerate(zip(views, titles)):
            ax = fig.add_subplot(1, 4, k + 1, projection="3d")
            ax.view_init(elev=elev, azim=azim)
            for ci, comp in enumerate(coupling.comps):
                p = draw_v[comp]
                ax.scatter(
                    p[:, 0], p[:, 2], p[:, 1], s=1, c=[colors[ci]], depthshade=False
                )
            for pi, (a, b) in enumerate(zip(coupling.pair_i, coupling.pair_j)):
                pa, pb = draw_v[int(a)], draw_v[int(b)]
                if viol_mask[pi] if len(viol_mask) else False:
                    col, lw, alpha = "red", 1.4, 0.95
                elif coupling.pair_is_major[pi]:
                    col, lw, alpha = "limegreen", 1.1, 0.9
                else:
                    col, lw, alpha = "0.55", 0.5, 0.55
                ax.plot(
                    [pa[0], pb[0]],
                    [pa[2], pb[2]],
                    [pa[1], pb[1]],
                    color=col,
                    lw=lw,
                    alpha=alpha,
                )
            if len(coupling.aux_pin_inds):
                p = draw_v[coupling.aux_pin_inds]
                ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=42, c="blue", marker="^")
            if len(handle_inds):
                p = draw_v[np.asarray(handle_inds, dtype=np.int64)]
                ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=36, c="red", marker="o")
            if len(anchor_inds):
                p = draw_v[np.asarray(anchor_inds, dtype=np.int64)]
                ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=36, c="black", marker="s")
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            try:
                ax.set_box_aspect(
                    (
                        float(np.ptp(draw_v[:, 0]) + 1e-6),
                        float(np.ptp(draw_v[:, 2]) + 1e-6),
                        float(np.ptp(draw_v[:, 1]) + 1e-6),
                    )
                )
            except Exception:
                pass
        n_maj = int(coupling.pair_is_major.sum()) if coupling.n_pairs else 0
        fig.suptitle(
            f"virtual seams  comps={len(coupling.comps)}  "
            f"pairs={coupling.n_pairs} major={n_maj}  "
            f"aux_pins={len(coupling.aux_pin_inds)}  "
            f"green=major gray=minor red=violation",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(out_dir / filename_png, dpi=140)
        plt.close(fig)
    except Exception as exc:
        print(f"[VirtualSeam] debug plot failed: {exc}")

    (out_dir / "virtual_seams_log.json").write_text(
        json.dumps(_jsonify(coupling.logs), indent=2), encoding="utf-8"
    )
    print(f"[VirtualSeam] wrote debug to {out_dir}")


def write_seam_gap_curve(out_dir: Path, history: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "seam_gap_history.json").write_text(
        json.dumps(_jsonify(history), indent=2), encoding="utf-8"
    )
    if not history:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [h["step"] for h in history]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(steps, [h["focus_mean_gap"] for h in history], label="focus mean gap")
        ax.plot(steps, [h["focus_max_gap"] for h in history], label="focus max gap")
        ax.plot(steps, [h["mean_gap"] for h in history], "--", label="all-pair mean gap")
        ax.plot(steps, [h["max_gap"] for h in history], ":", label="all-pair max gap")
        if "focus_gap0" in history[0]:
            ax.axhline(
                history[0]["focus_gap0"],
                color="0.4",
                ls="-.",
                label=f"focus d0={history[0]['focus_gap0']:.4f}",
            )
        ax.set_xlabel("optimization step")
        ax.set_ylabel("seam gap")
        ax.set_title("Building major seam gap")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "seam_gap_curve.png", dpi=140)
        plt.close(fig)
    except Exception as exc:
        print(f"[VirtualSeam] gap curve failed: {exc}")
