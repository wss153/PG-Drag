#!/usr/bin/env python3
"""Restore processed meshes from raw OBJs: normalize xyz only, keep topology/UV."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.geometry.virtual_seams import geometry_fingerprint
from src.utils.geometry_utils import load_obj
from tools.stitch_mesh_components import parse_obj, write_obj


def normalize_obj(obj: dict) -> tuple[np.ndarray, np.ndarray, float]:
    v = np.asarray(obj["verts"], dtype=np.float64)
    vmin = v.min(0)
    vmax = v.max(0)
    center = (vmin + vmax) * 0.5
    scale = float((vmax - vmin).max())
    v2 = (v - center) / scale
    new_lines = []
    for i, line in enumerate(obj["v_lines"]):
        parts = line.split()
        parts[1] = f"{v2[i, 0]:.10g}"
        parts[2] = f"{v2[i, 1]:.10g}"
        parts[3] = f"{v2[i, 2]:.10g}"
        new_lines.append(" ".join(parts))
    obj["verts"] = v2
    obj["v_lines"] = new_lines
    return v2, center, scale


def write_fingerprint(out_dir: Path, mesh_path: Path) -> dict:
    v, f, _vc, uvs, _vns, tex_inds, _vn_inds, _tex = load_obj(mesh_path)
    meta = geometry_fingerprint(v, f, uvs, tex_inds)
    (out_dir / "original_geometry.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta


def write_kp(path: Path, ids: np.ndarray, xyz: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.column_stack([ids.astype(np.int64), xyz])
    np.savetxt(path, rows, fmt="%d %.10g %.10g %.10g")


def restore_from_raw(
    raw_obj: Path,
    out_dir: Path,
    handle_ids: list[int],
    handle_offset: np.ndarray,
    anchor_ids: list[int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = out_dir / "mesh.obj"
    obj = parse_obj(raw_obj)
    verts, center, scale = normalize_obj(obj)
    write_obj(mesh_path, obj, obj["v_lines"], obj["faces"])
    unwelded = out_dir / "mesh_unwelded.obj"
    write_obj(unwelded, obj, obj["v_lines"], obj["faces"])
    fp = write_fingerprint(out_dir, mesh_path)
    hids = np.asarray(handle_ids, dtype=np.int64)
    aids = np.asarray(anchor_ids, dtype=np.int64)
    h_rest = verts[hids]
    h_tgt = h_rest + np.asarray(handle_offset, dtype=np.float64).reshape(1, 3)
    a_rest = verts[aids]
    kp_dir = out_dir / "keypoints" / "000"
    write_kp(kp_dir / "user_single_keypoints.txt", hids, h_tgt)
    write_kp(kp_dir / "user_single_keypoints_unwelded.txt", hids, h_tgt)
    write_kp(kp_dir / "constraint_single_keypoints.txt", aids, a_rest)
    write_kp(kp_dir / "constraint_single_keypoints_unwelded.txt", aids, a_rest)
    print(
        f"restored {out_dir.name}: V={fp['n_vertices']} F={fp['n_faces']} "
        f"vt={fp['n_vt']} scale={scale:.6g} center={center}"
    )
    print(f"  handle ids {hids.tolist()} rest_y={np.round(h_rest[:,1],4).tolist()} "
          f"tgt_y={np.round(h_tgt[:,1],4).tolist()}")
    print(f"  anchor ids {aids.tolist()} rest_y={np.round(a_rest[:,1],4).tolist()}")


def restore_copy_unwelded(
    out_dir: Path,
    handle_ids: list[int],
    handle_offset: np.ndarray,
    anchor_ids: list[int],
) -> None:
    src = out_dir / "mesh_unwelded.obj"
    dst = out_dir / "mesh.obj"
    dst.write_bytes(src.read_bytes())
    fp = write_fingerprint(out_dir, dst)
    from src.utils.geometry_utils import load_obj as _load
    verts, *_ = _load(dst)
    verts = verts.astype(np.float64)
    hids = np.asarray(handle_ids, dtype=np.int64)
    aids = np.asarray(anchor_ids, dtype=np.int64)
    h_rest = verts[hids]
    h_tgt = h_rest + np.asarray(handle_offset, dtype=np.float64).reshape(1, 3)
    a_rest = verts[aids]
    kp_dir = out_dir / "keypoints" / "000"
    write_kp(kp_dir / "user_single_keypoints.txt", hids, h_tgt)
    write_kp(kp_dir / "user_single_keypoints_unwelded.txt", hids, h_tgt)
    write_kp(kp_dir / "constraint_single_keypoints.txt", aids, a_rest)
    write_kp(kp_dir / "constraint_single_keypoints_unwelded.txt", aids, a_rest)
    print(f"restored {out_dir.name} from unwelded: V={fp['n_vertices']} F={fp['n_faces']}")
    print(f"  handle ids {hids.tolist()} rest={np.round(h_rest,4).tolist()} "
          f"tgt={np.round(h_tgt,4).tolist()}")


def main() -> None:
    root = Path("data/apap_3d/processed")
    restore_from_raw(
        Path("/home/dell/桌面/wss/building.obj"),
        root / "building",
        handle_ids=[4245, 4242, 4240, 4244],
        handle_offset=np.array([0.0, 0.12, 0.0]),
        anchor_ids=[987, 1478, 891, 1358],
    )
    restore_from_raw(
        Path("/home/dell/桌面/wss/gear.obj"),
        root / "gear",
        handle_ids=[1080],
        handle_offset=np.array([0.0, 0.15, 0.0]),
        anchor_ids=[626],
    )
    restore_copy_unwelded(
        root / "saw",
        handle_ids=[367],
        handle_offset=np.array([0.0, 0.2, 0.0]),
        anchor_ids=[948, 833],
    )


if __name__ == "__main__":
    main()
