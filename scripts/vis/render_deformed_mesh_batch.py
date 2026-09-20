#!/usr/bin/env python3
"""Render four-view textured previews (deformed_mesh.png) from deformed OBJ files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.geometry_utils import load_obj
from src.utils.vis_utils import render_mesh_360


def _load_mesh_for_render(mesh_path: Path, device: torch.device):
    expected_tex = mesh_path.parent / f"{mesh_path.stem}_texture.png"
    alt_tex = mesh_path.with_name(mesh_path.name.replace("_deformed.obj", "_texture.png"))
    linked = False
    if not expected_tex.is_file() and alt_tex.is_file():
        expected_tex.symlink_to(alt_tex.resolve())
        linked = True
    try:
        v, f, vc, uvs, vns, tex_inds, vn_inds, tex = load_obj(mesh_path)
    finally:
        if linked and expected_tex.is_symlink():
            expected_tex.unlink()

    v_t = torch.from_numpy(v).to(device)
    f_t = torch.from_numpy(f).to(device)
    uvs_t = torch.from_numpy(uvs).to(device) if uvs is not None else None
    tex_inds_t = torch.from_numpy(tex_inds).to(device) if tex_inds is not None else None
    tex_t = torch.from_numpy(tex).to(device) if tex is not None else None
    return v_t, f_t, uvs_t, tex_inds_t, tex_t


def output_png_path(out_dir: Path, mesh_path: Path) -> Path:
    if mesh_path.name == "deformed.obj" and mesh_path.parent.name == "mesh":
        return out_dir / "deformed_mesh.png"
    if mesh_path.name.endswith("_deformed.obj"):
        return out_dir / mesh_path.name.replace("_deformed.obj", "_deformed_mesh.png")
    return out_dir / f"{mesh_path.stem}_render.png"


def discover_cases(root: Path) -> list[tuple[Path, Path]]:
    root = root.resolve()
    cases: list[tuple[Path, Path]] = []

    if root.is_file() and root.suffix == ".obj":
        return [(root.parent, root)]

    if (root / "mesh" / "deformed.obj").is_file():
        cases.append((root, root / "mesh" / "deformed.obj"))
        return cases

    for case_dir in sorted(root.glob("mesh-*")):
        mesh_path = case_dir / "mesh" / "deformed.obj"
        if mesh_path.is_file():
            cases.append((case_dir, mesh_path))

    if not cases:
        for obj in sorted(root.glob("*_deformed.obj")):
            cases.append((obj.parent, obj))

    return cases


def render_case(
    out_dir: Path,
    mesh_path: Path,
    device: torch.device,
    *,
    force: bool,
    n_step: int,
    radius: float,
    img_size: int,
) -> str:
    out_path = output_png_path(out_dir, mesh_path)
    if out_path.is_file() and not force:
        return "skipped"

    v, f, uvs, tex_inds, tex = _load_mesh_for_render(mesh_path, device)
    imgs = render_mesh_360(
        v,
        f,
        uvs,
        tex_inds,
        tex,
        n_step=n_step,
        radius=radius,
        img_height=img_size,
        img_width=img_size,
    )
    Image.fromarray(np.concatenate(imgs, axis=1)).save(out_path)
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="Deform run dir, batch root (mesh-*), or deformed.obj path",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-step", type=int, default=4)
    parser.add_argument("--radius", type=float, default=1.5)
    parser.add_argument(
        "--img-size",
        type=int,
        default=512,
        help="Per-view square resolution (default 512 → 2048x512 four-view strip)",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    total_ok = total_skip = total_fail = 0

    for root in args.roots:
        cases = discover_cases(root)
        print(f"\n[root] {root} ({len(cases)} meshes, img_size={args.img_size})")
        for out_dir, mesh_path in cases:
            try:
                status = render_case(
                    out_dir,
                    mesh_path,
                    device,
                    force=args.force,
                    n_step=args.n_step,
                    radius=args.radius,
                    img_size=args.img_size,
                )
                label = out_dir.name if out_dir.name != root.name else mesh_path.name
                if status == "ok":
                    total_ok += 1
                    print(f"  OK {label}")
                else:
                    total_skip += 1
            except Exception as exc:  # noqa: BLE001
                total_fail += 1
                label = out_dir.name if out_dir.name != root.name else mesh_path.name
                print(f"  FAIL {label}: {exc}")

    print(f"\nDone: ok={total_ok} skipped={total_skip} failed={total_fail}")
    if total_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
