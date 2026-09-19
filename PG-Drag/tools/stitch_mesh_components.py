#!/usr/bin/env python3
"""Stitch disconnected mesh components without touching UVs.

CAD / kitbash meshes often have many connected components with small gaps.
A global distance weld would also collapse nearby vertices on the same
surface. This script welds up to K nearest vertex pairs between every pair
of components whose gap is below --threshold (greedy matching). That makes
a seam instead of a one-vertex hinge, while vt / vn / texture stay unchanged.

Examples:
    python tools/stitch_mesh_components.py \
        --mesh data/apap_3d/processed/building/mesh.obj \
        --threshold 0.02 --pairs 6 \
        --handle data/apap_3d/processed/building/keypoints/000/user_single_keypoints.txt \
        --anchor data/apap_3d/processed/building/keypoints/000/constraint_single_keypoints.txt
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class FaceCorner:
    v: int
    t: int | None
    n: int | None
    raw: str


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int, prefer_a: bool | None = None) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if prefer_a is True:
            keep, drop = ra, rb
        elif prefer_a is False:
            keep, drop = rb, ra
        elif self.size[ra] >= self.size[rb]:
            keep, drop = ra, rb
        else:
            keep, drop = rb, ra
        self.parent[drop] = keep
        self.size[keep] += self.size[drop]
        return keep

    def n_sets(self) -> int:
        return sum(self.parent[i] == i for i in range(len(self.parent)))


def parse_obj(path: Path) -> dict:
    header: list[str] = []
    v_lines: list[str] = []
    verts: list[list[float]] = []
    vt_lines: list[str] = []
    vn_lines: list[str] = []
    other_before_faces: list[str] = []
    faces: list[list[FaceCorner]] = []
    face_prefix: list[str] = []  # usemtl / s / o / g immediately before faces, kept in order
    saw_vertex = False
    saw_face = False

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                saw_vertex = True
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                v_lines.append(line.rstrip("\n"))
            elif line.startswith("vt "):
                vt_lines.append(line.rstrip("\n"))
            elif line.startswith("vn "):
                vn_lines.append(line.rstrip("\n"))
            elif line.startswith("f "):
                saw_face = True
                corners: list[FaceCorner] = []
                for tok in line.split()[1:]:
                    bits = tok.split("/")
                    vi = int(bits[0]) - 1
                    ti = int(bits[1]) - 1 if len(bits) > 1 and bits[1] else None
                    ni = int(bits[2]) - 1 if len(bits) > 2 and bits[2] else None
                    corners.append(FaceCorner(vi, ti, ni, tok))
                faces.append(corners)
            elif not saw_vertex:
                header.append(line.rstrip("\n"))
            elif not saw_face:
                other_before_faces.append(line.rstrip("\n"))
            else:
                face_prefix.append(line.rstrip("\n"))

    return {
        "header": header,
        "v_lines": v_lines,
        "verts": np.asarray(verts, dtype=np.float64),
        "vt_lines": vt_lines,
        "vn_lines": vn_lines,
        "other_before_faces": other_before_faces,
        "faces": faces,
        "trailing": face_prefix,
    }


def connected_components(nv: int, faces: list[list[FaceCorner]]) -> list[np.ndarray]:
    adj: list[list[int]] = [[] for _ in range(nv)]
    for face in faces:
        vs = [c.v for c in face]
        for i, a in enumerate(vs):
            b = vs[(i + 1) % len(vs)]
            adj[a].append(b)
            adj[b].append(a)
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


def nearest_pair(va: np.ndarray, ia: np.ndarray, vb: np.ndarray, ib: np.ndarray) -> tuple[float, int, int]:
    if len(ia) <= len(ib):
        tree = cKDTree(vb)
        d, nn = tree.query(va, k=1, workers=-1)
        k = int(np.argmin(d))
        return float(d[k]), int(ia[k]), int(ib[int(nn[k])])
    tree = cKDTree(va)
    d, nn = tree.query(vb, k=1, workers=-1)
    k = int(np.argmin(d))
    return float(d[k]), int(ia[int(nn[k])]), int(ib[k])


def greedy_pairs(
    va: np.ndarray,
    ia: np.ndarray,
    vb: np.ndarray,
    ib: np.ndarray,
    threshold: float,
    n_pairs: int,
) -> list[tuple[float, int, int]]:
    """Greedy unique matching of up to n_pairs vertex pairs with dist <= threshold."""
    if len(ia) == 0 or len(ib) == 0 or n_pairs <= 0:
        return []
    if len(ia) <= len(ib):
        src_p, src_i, dst_p, dst_i = va, ia, vb, ib
        swap = False
    else:
        src_p, src_i, dst_p, dst_i = vb, ib, va, ia
        swap = True
    k_query = int(min(max(n_pairs, 1), len(dst_i)))
    tree = cKDTree(dst_p)
    dists, nn = tree.query(src_p, k=k_query, workers=-1)
    dists = np.asarray(dists, dtype=np.float64).reshape(len(src_i), k_query)
    nn = np.asarray(nn, dtype=np.int64).reshape(len(src_i), k_query)
    cands: list[tuple[float, int, int]] = []
    for si in range(len(src_i)):
        for qi in range(dists.shape[1]):
            d = float(dists[si, qi])
            if not np.isfinite(d) or d > threshold:
                continue
            a = int(src_i[si])
            b = int(dst_i[int(nn[si, qi])])
            if swap:
                a, b = b, a
            cands.append((d, a, b))
    cands.sort(key=lambda x: x[0])
    used_a: set[int] = set()
    used_b: set[int] = set()
    out: list[tuple[float, int, int]] = []
    for d, a, b in cands:
        if a in used_a or b in used_b:
            continue
        used_a.add(a)
        used_b.add(b)
        out.append((d, a, b))
        if len(out) >= n_pairs:
            break
    return out


def vertex_adjacency(nv: int, faces: list[list[FaceCorner]]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(nv)]
    for face in faces:
        vs = [c.v for c in face]
        for i, a in enumerate(vs):
            b = vs[(i + 1) % len(vs)]
            adj[a].append(b)
            adj[b].append(a)
    return adj


def first_attrib_for_vertex(nv: int, faces: list[list[FaceCorner]]) -> tuple[list[int], list[int]]:
    vt = [0] * nv
    vn = [0] * nv
    for face in faces:
        for c in face:
            if c.t is not None:
                vt[c.v] = c.t
            if c.n is not None:
                vn[c.v] = c.n
    return vt, vn


def make_corner(v: int, t: int, n: int) -> FaceCorner:
    return FaceCorner(v, t, n, f"{v+1}/{t+1}/{n+1}")


def nearby_component_matches(
    verts: np.ndarray,
    comps: list[np.ndarray],
    threshold: float,
    pairs_per: int,
) -> list[tuple[int, int, list[tuple[float, int, int]]]]:
    points = [verts[c] for c in comps]
    jobs: list[tuple[int, int, list[tuple[float, int, int]]]] = []
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            dist, _, _ = nearest_pair(points[i], comps[i], points[j], comps[j])
            if dist > threshold:
                continue
            k_here = min(pairs_per, len(comps[i]), len(comps[j]))
            matches = greedy_pairs(
                points[i], comps[i], points[j], comps[j], threshold, k_here
            )
            if matches:
                jobs.append((i, j, matches))
    return jobs


def boundary_edges(faces: list[list[FaceCorner]]) -> list[tuple[int, int]]:
    cnt: Counter[tuple[int, int]] = Counter()
    for face in faces:
        vs = [c.v for c in face]
        n = len(vs)
        for i in range(n):
            a, b = vs[i], vs[(i + 1) % n]
            e = (min(a, b), max(a, b))
            cnt[e] += 1
    return [e for e, k in cnt.items() if k == 1]


def zip_boundary_faces(
    verts: np.ndarray,
    faces: list[list[FaceCorner]],
    threshold: float,
    axis: int = 1,
) -> tuple[list[list[FaceCorner]], list[dict]]:
    """Fill remaining open rims by pairing nearby boundary edges into quads.

    Vertex welding only shares isolated points, so a floor slab can still show
    a horizontal crack where the facing loops never gained faces. This zips
    unmatched boundary edges whose midpoints are within --threshold.

    If axis >= 0, only pair edges whose gap is mainly along that axis (1=Y).
    That closes stacked floor seams without filling sideways window holes.
    """
    vt_of, vn_of = first_attrib_for_vertex(len(verts), faces)
    edges = boundary_edges(faces)
    extra: list[list[FaceCorner]] = []
    stitches: list[dict] = []
    if len(edges) < 2:
        return extra, stitches

    def tri(a: int, b: int, c: int) -> bool:
        if len({a, b, c}) < 3:
            return False
        pa, pb, pc = verts[a], verts[b], verts[c]
        cr = np.cross(pb - pa, pc - pa)
        area = 0.5 * float(np.linalg.norm(cr))
        if not np.isfinite(area) or area < 1e-10:
            return False
        eab = float(np.linalg.norm(pb - pa))
        ebc = float(np.linalg.norm(pc - pb))
        eca = float(np.linalg.norm(pa - pc))
        emin = min(eab, ebc, eca)
        emax = max(eab, ebc, eca)
        if emin < 1e-12 or emax / emin > 80.0:
            return False
        extra.append(
            [
                make_corner(a, vt_of[a], vn_of[a]),
                make_corner(b, vt_of[b], vn_of[b]),
                make_corner(c, vt_of[c], vn_of[c]),
            ]
        )
        return True

    mids = np.asarray([(verts[a] + verts[b]) * 0.5 for a, b in edges], dtype=np.float64)
    k_query = int(min(12, len(edges)))
    tree = cKDTree(mids)
    dists, nn = tree.query(mids, k=k_query, workers=-1)
    dists = np.asarray(dists, dtype=np.float64).reshape(len(edges), k_query)
    nn = np.asarray(nn, dtype=np.int64).reshape(len(edges), k_query)

    cands: list[tuple[float, int, int, int, float]] = []
    for i, (a, b) in enumerate(edges):
        for qi in range(k_query):
            j = int(nn[i, qi])
            if j <= i:
                continue
            dmid = float(dists[i, qi])
            if not np.isfinite(dmid) or dmid > threshold:
                continue
            if axis >= 0:
                delta = mids[i] - mids[j]
                if abs(float(delta[axis])) < 0.35 * max(dmid, 1e-8):
                    continue
            c, d = edges[j]
            if len({a, b, c, d}) < 4:
                continue
            d_ac_bd = max(
                float(np.linalg.norm(verts[a] - verts[c])),
                float(np.linalg.norm(verts[b] - verts[d])),
            )
            d_ad_bc = max(
                float(np.linalg.norm(verts[a] - verts[d])),
                float(np.linalg.norm(verts[b] - verts[c])),
            )
            if d_ac_bd <= d_ad_bc:
                pairing, dmax = 0, d_ac_bd
            else:
                pairing, dmax = 1, d_ad_bc
            if dmax > threshold * 1.8:
                continue
            cands.append((dmid, i, j, pairing, dmax))
    cands.sort(key=lambda x: x[0])

    used: set[int] = set()
    for dmid, i, j, pairing, dmax in cands:
        if i in used or j in used:
            continue
        a, b = edges[i]
        c, d = edges[j]
        n0 = len(extra)
        if pairing == 0:
            tri(a, b, d)
            tri(a, d, c)
        else:
            tri(a, b, c)
            tri(a, c, d)
        if len(extra) <= n0:
            continue
        used.add(i)
        used.add(j)
        stitches.append(
            {
                "dist": dmid,
                "comp_a": -1,
                "comp_b": -1,
                "vert_a": a,
                "vert_b": c,
                "edge_a": [a, b],
                "edge_b": [c, d],
                "type": "zip",
                "dmax": dmax,
            }
        )
    return extra, stitches


def bridge_faces(
    verts: np.ndarray,
    faces: list[list[FaceCorner]],
    comps: list[np.ndarray],
    threshold: float,
    pairs_per: int,
) -> tuple[list[list[FaceCorner]], list[dict]]:
    adj = vertex_adjacency(len(verts), faces)
    vt_of, vn_of = first_attrib_for_vertex(len(verts), faces)
    extra: list[list[FaceCorner]] = []
    stitches: list[dict] = []

    def tri(a: int, b: int, c: int) -> None:
        if len({a, b, c}) < 3:
            return
        pa, pb, pc = verts[a], verts[b], verts[c]
        cr = np.cross(pb - pa, pc - pa)
        area = 0.5 * float(np.linalg.norm(cr))
        if not np.isfinite(area) or area < 1e-10:
            return
        eab = float(np.linalg.norm(pb - pa))
        ebc = float(np.linalg.norm(pc - pb))
        eca = float(np.linalg.norm(pa - pc))
        emin = min(eab, ebc, eca)
        emax = max(eab, ebc, eca)
        if emin < 1e-12 or emax / emin > 80.0:
            return
        extra.append(
            [
                make_corner(a, vt_of[a], vn_of[a]),
                make_corner(b, vt_of[b], vn_of[b]),
                make_corner(c, vt_of[c], vn_of[c]),
            ]
        )

    for i, j, matches in nearby_component_matches(verts, comps, threshold, pairs_per):
        for dist, va, vb in matches:
            stitches.append(
                {
                    "dist": dist,
                    "comp_a": i,
                    "comp_b": j,
                    "vert_a": va,
                    "vert_b": vb,
                    "type": "bridge",
                }
            )
        if len(matches) >= 2:
            mids = np.array(
                [(verts[va] + verts[vb]) * 0.5 for _, va, vb in matches],
                dtype=np.float64,
            )
            axis = int(np.argmax(mids.max(0) - mids.min(0)))
            order = sorted(range(len(matches)), key=lambda t: float(mids[t, axis]))
            seq = [matches[t] for t in order]
            for t in range(len(seq) - 1):
                _, va, vb = seq[t]
                _, va2, vb2 = seq[t + 1]
                tri(va, vb, va2)
                tri(vb, vb2, va2)
        else:
            _, va, vb = matches[0]
            na = next((w for w in adj[va] if w != vb), None)
            nb = next((w for w in adj[vb] if w != va), None)
            if na is not None:
                tri(va, vb, na)
            if nb is not None:
                tri(vb, va, nb)
    return extra, stitches


def stitch_components(
    verts: np.ndarray,
    faces: list[list[FaceCorner]],
    threshold: float,
    pairs_per: int = 6,
) -> tuple[DSU, list[dict], int]:
    comps = connected_components(len(verts), faces)
    n_comp0 = len(comps)
    if n_comp0 <= 1:
        return DSU(len(verts)), [], n_comp0

    points = [verts[c] for c in comps]
    pair_jobs: list[tuple[float, int, int]] = []
    for i in range(n_comp0):
        for j in range(i + 1, n_comp0):
            dist, _, _ = nearest_pair(points[i], comps[i], points[j], comps[j])
            if dist <= threshold:
                pair_jobs.append((dist, i, j))
    pair_jobs.sort(key=lambda x: x[0])

    vert_dsu = DSU(len(verts))
    stitches: list[dict] = []
    for _, i, j in pair_jobs:
        keep_va = len(comps[i]) >= len(comps[j])
        k_here = min(pairs_per, len(comps[i]), len(comps[j]))
        matches = greedy_pairs(
            points[i], comps[i], points[j], comps[j], threshold, k_here
        )
        for dist, va, vb in matches:
            if vert_dsu.find(va) == vert_dsu.find(vb):
                continue
            vert_dsu.union(va, vb, prefer_a=keep_va)
            stitches.append(
                {
                    "dist": dist,
                    "comp_a": i,
                    "comp_b": j,
                    "vert_a": va,
                    "vert_b": vb,
                    "kept": int(va if keep_va else vb),
                    "dropped": int(vb if keep_va else va),
                }
            )
    return vert_dsu, stitches, n_comp0


def compact_and_remap(
    obj: dict,
    vert_dsu: DSU,
) -> tuple[list[str], list[list[FaceCorner]], np.ndarray, dict]:
    nv = len(obj["verts"])
    old_to_new = np.full(nv, -1, dtype=np.int64)
    new_v_lines: list[str] = []
    repr_to_new: dict[int, int] = {}
    for i in range(nv):
        r = vert_dsu.find(i)
        if r not in repr_to_new:
            repr_to_new[r] = len(new_v_lines)
            new_v_lines.append(obj["v_lines"][r])
        old_to_new[i] = repr_to_new[r]

    new_faces: list[list[FaceCorner]] = []
    n_degen = 0
    for face in obj["faces"]:
        vs = [int(old_to_new[c.v]) for c in face]
        if len(set(vs)) < 3:
            n_degen += 1
            continue
        new_faces.append(
            [
                FaceCorner(vs[k], face[k].t, face[k].n, face[k].raw)
                for k in range(len(face))
            ]
        )
    meta = {
        "n_vertices_in": nv,
        "n_vertices_out": len(new_v_lines),
        "n_faces_in": len(obj["faces"]),
        "n_faces_out": len(new_faces),
        "n_degenerate_dropped": n_degen,
    }
    return new_v_lines, new_faces, old_to_new, meta


def format_corner(c: FaceCorner) -> str:
    v = c.v + 1
    if c.t is None and c.n is None:
        return str(v)
    t = "" if c.t is None else str(c.t + 1)
    if c.n is None:
        return f"{v}/{t}"
    n = str(c.n + 1)
    return f"{v}/{t}/{n}"


def write_obj(
    path: Path,
    obj: dict,
    new_v_lines: list[str],
    new_faces: list[list[FaceCorner]],
) -> None:
    lines: list[str] = []
    lines.extend(obj["header"])
    lines.extend(new_v_lines)
    lines.extend(obj["vt_lines"])
    lines.extend(obj["vn_lines"])
    lines.extend(obj["other_before_faces"])
    for face in new_faces:
        lines.append("f " + " ".join(format_corner(c) for c in face))
    lines.extend(obj["trailing"])
    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


def load_keypoints(path: Path) -> list[tuple[int, float, float, float]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            ind, x, y, z = line.split()
            rows.append((int(ind), float(x), float(y), float(z)))
    return rows


def write_keypoints(path: Path, rows: list[tuple[int, float, float, float]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for ind, x, y, z in rows:
            handle.write(f"{ind} {x} {y} {z}\n")


def remap_keypoints(
    rows: list[tuple[int, float, float, float]],
    old_to_new: np.ndarray,
    new_verts: np.ndarray,
    snap_xyz_to_mesh: bool,
) -> list[tuple[int, float, float, float]]:
    out: list[tuple[int, float, float, float]] = []
    seen: set[int] = set()
    for ind, x, y, z in rows:
        if ind < 0 or ind >= len(old_to_new):
            raise IndexError(f"keypoint vertex {ind} out of range 0..{len(old_to_new)-1}")
        new_id = int(old_to_new[ind])
        if new_id in seen:
            continue
        seen.add(new_id)
        if snap_xyz_to_mesh:
            x, y, z = (float(c) for c in new_verts[new_id])
        out.append((new_id, x, y, z))
    return out


def backup_if_needed(path: Path, suffix: str) -> Path:
    bak = path.with_name(path.stem + suffix + path.suffix)
    if not bak.exists():
        shutil.copy2(path, bak)
    return bak


def count_components(nv: int, faces: list[list[FaceCorner]]) -> int:
    return len(connected_components(nv, faces))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument(
        "--pairs",
        type=int,
        default=6,
        help="Max unique vertex pairs to weld/bridge between each nearby component pair",
    )
    parser.add_argument(
        "--mode",
        choices=["merge", "bridge", "zip"],
        default="bridge",
        help="merge: weld xyz. bridge: add triangles between components. "
        "zip: fill remaining open boundary rims on the current mesh",
    )
    parser.add_argument("--handle", type=Path, default=None)
    parser.add_argument("--anchor", type=Path, default=None)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Read mesh.obj as-is instead of mesh_unwelded.obj (used by zip)",
    )
    args = parser.parse_args()

    mesh_path: Path = args.mesh
    bak_mesh = mesh_path.with_name(mesh_path.stem + "_unwelded" + mesh_path.suffix)
    use_current = args.in_place or args.mode == "zip"
    source_mesh = mesh_path if use_current or not bak_mesh.exists() else bak_mesh
    obj = parse_obj(source_mesh)
    verts = obj["verts"]
    comps0 = connected_components(len(verts), obj["faces"])
    n_comp0 = len(comps0)

    if args.mode == "zip":
        extra, stitches = zip_boundary_faces(verts, obj["faces"], args.threshold)
        new_v_lines = obj["v_lines"]
        new_faces = list(obj["faces"]) + extra
        old_to_new = np.arange(len(verts), dtype=np.int64)
        meta = {
            "n_vertices_in": len(verts),
            "n_vertices_out": len(verts),
            "n_faces_in": len(obj["faces"]),
            "n_faces_out": len(new_faces),
            "n_zip_faces": len(extra),
            "n_boundary_edges_before": len(boundary_edges(obj["faces"])),
            "n_boundary_edges_after": len(boundary_edges(new_faces)),
            "n_degenerate_dropped": 0,
        }
        new_verts = verts
    elif args.mode == "bridge":
        extra, stitches = bridge_faces(
            verts, obj["faces"], comps0, args.threshold, args.pairs
        )
        new_v_lines = obj["v_lines"]
        new_faces = list(obj["faces"]) + extra
        old_to_new = np.arange(len(verts), dtype=np.int64)
        meta = {
            "n_vertices_in": len(verts),
            "n_vertices_out": len(verts),
            "n_faces_in": len(obj["faces"]),
            "n_faces_out": len(new_faces),
            "n_bridge_faces": len(extra),
            "n_degenerate_dropped": 0,
        }
        new_verts = verts
    else:
        vert_dsu, stitches, n_comp0 = stitch_components(
            verts, obj["faces"], args.threshold, pairs_per=args.pairs
        )
        new_v_lines, new_faces, old_to_new, meta = compact_and_remap(obj, vert_dsu)
        new_verts = np.asarray(
            [[float(x) for x in line.split()[1:4]] for line in new_v_lines],
            dtype=np.float64,
        )
    n_comp1 = count_components(len(new_v_lines), new_faces)

    if not bak_mesh.exists():
        shutil.copy2(mesh_path, bak_mesh)
    write_obj(mesh_path, obj, new_v_lines, new_faces)

    pair_key = [
        (min(s["comp_a"], s["comp_b"]), max(s["comp_a"], s["comp_b"])) for s in stitches
    ]
    n_comp_pairs = len(set(pair_key))
    pair_mult = Counter(Counter(pair_key).values())

    report = {
        "mesh": str(mesh_path),
        "backup": str(bak_mesh),
        "mode": args.mode,
        "threshold": args.threshold,
        "pairs_per": args.pairs,
        "n_components_before": n_comp0,
        "n_components_after": n_comp1,
        "n_stitches": len(stitches),
        "n_component_pairs": n_comp_pairs,
        "stitches_per_pair_hist": {str(k): int(v) for k, v in sorted(pair_mult.items())},
        "max_stitch_dist": max((s["dist"] for s in stitches), default=0.0),
        "n_vt_unchanged": len(obj["vt_lines"]),
        "n_vn_unchanged": len(obj["vn_lines"]),
        **meta,
        "stitches": stitches,
        "old_to_new": old_to_new.tolist(),
    }

    if args.handle is not None:
        bak = backup_if_needed(args.handle, "_unwelded")
        rows = load_keypoints(bak)
        remapped = remap_keypoints(rows, old_to_new, new_verts, snap_xyz_to_mesh=False)
        write_keypoints(args.handle, remapped)
        report["handle_backup"] = str(bak)
        report["handle_before"] = rows
        report["handle_after"] = remapped

    if args.anchor is not None:
        bak = backup_if_needed(args.anchor, "_unwelded")
        rows = load_keypoints(bak)
        remapped = remap_keypoints(rows, old_to_new, new_verts, snap_xyz_to_mesh=True)
        write_keypoints(args.anchor, remapped)
        report["anchor_backup"] = str(bak)
        report["anchor_before"] = rows
        report["anchor_after"] = remapped

    # zip is a post-pass on an already-stitched mesh; do not clobber merge maps
    if args.mode == "zip":
        report_path = mesh_path.with_name("stitch_zip_report.json")
        slim = {k: v for k, v in report.items() if k != "old_to_new"}
        report_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    else:
        report_path = mesh_path.with_name("stitch_report.json")
        slim = {k: v for k, v in report.items() if k != "old_to_new"}
        report_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")
        map_path = mesh_path.with_name("stitch_old_to_new.json")
        map_path.write_text(json.dumps(report["old_to_new"]), encoding="utf-8")

    print(f"wrote {mesh_path}")
    print(f"backup {bak_mesh}")
    extra_note = ""
    if args.mode == "zip":
        extra_note = (
            f" | boundary {meta.get('n_boundary_edges_before')} -> "
            f"{meta.get('n_boundary_edges_after')} | zip_faces={meta.get('n_zip_faces')}"
        )
    print(
        f"components {n_comp0} -> {n_comp1} | "
        f"V {meta['n_vertices_in']} -> {meta['n_vertices_out']} | "
        f"F {meta['n_faces_in']} -> {meta['n_faces_out']} | "
        f"stitches {len(stitches)} across {n_comp_pairs} comp-pairs "
        f"hist={dict(sorted(pair_mult.items()))} "
        f"max_dist={report['max_stitch_dist']:.6f}"
        f"{extra_note}"
    )
    print(f"vt kept {len(obj['vt_lines'])}  vn kept {len(obj['vn_lines'])}")
    if args.handle:
        print(f"handle {report['handle_before']} -> {report['handle_after']}")
    if args.anchor:
        print(f"anchor {report['anchor_before']} -> {report['anchor_after']}")


if __name__ == "__main__":
    main()
