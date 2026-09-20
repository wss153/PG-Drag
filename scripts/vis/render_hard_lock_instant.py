#!/usr/bin/env python3
"""One-shot hard-handle Poisson snapshot: no SDS / no iteration, render only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.geometry.poisson_system import PoissonSystem
from src.utils.geometry_utils import load_obj, save_obj
from src.utils.vis_utils import render_mesh_360, render_mesh_with_markers_360


def load_keypoints(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inds, pos = [], []
    with open(path) as f:
        for line in f:
            i, x, y, z = line.split()
            inds.append(int(i))
            pos.append([float(x), float(y), float(z)])
    return (
        torch.tensor(inds, device=device, dtype=torch.long),
        torch.tensor(pos, device=device, dtype=torch.float32),
    )


def parse_apap_line(line: str) -> tuple[str, str, str]:
    line = line.replace(" ", "")
    mesh_path = line.split(",")[0]
    name = mesh_path.split("/processed/")[1].split("/mesh.obj")[0]
    kp = line.split("keypoints/")[1].split("/user_single_keypoints.txt")[0]
    lora = line.split(",")[-1].strip().split("/")[-1]
    return name, kp, lora


def snapshot_case(
    name: str,
    kp: str,
    lora: str,
    out_root: Path,
    device: torch.device,
    *,
    radius: float,
    img_size: int,
    virtual_seams: bool,
) -> dict:
    mesh_file = ROOT / f"data/apap_3d/processed/{name}/mesh.obj"
    handle_file = ROOT / f"data/apap_3d/processed/{name}/keypoints/{kp}/user_single_keypoints.txt"
    anchor_file = ROOT / f"data/apap_3d/processed/{name}/keypoints/{kp}/constraint_single_keypoints.txt"
    out_dir = out_root / f"{name}_handle-{kp}_anchor-{kp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    v, f, _, uvs, _, tex_inds, _, tex = load_obj(mesh_file)
    v = torch.from_numpy(v).to(device)
    f = torch.from_numpy(f).to(device)
    uvs = torch.from_numpy(uvs).to(device)
    tex_inds = torch.from_numpy(tex_inds).to(device)
    tex = torch.from_numpy(tex).to(device)

    handle_inds, handle_pos = load_keypoints(handle_file, device)
    anchor_inds, anchor_pos = load_keypoints(anchor_file, device)

    poisson_inds = torch.cat([anchor_inds, handle_inds], dim=0)
    poisson_pos = torch.cat([anchor_pos, handle_pos], dim=0)

    rel_i = rel_j = rel_off = rel_w = None
    if virtual_seams:
        from src.geometry.virtual_seams import build_virtual_seams

        coupling = build_virtual_seams(
            v.detach().cpu().numpy(),
            f.detach().cpu().numpy(),
            handle_inds.detach().cpu().numpy(),
            anchor_inds.detach().cpu().numpy(),
        )
        if len(coupling.aux_pin_inds) > 0:
            aux_inds = torch.tensor(coupling.aux_pin_inds, dtype=torch.long, device=device)
            aux_pos = torch.tensor(coupling.aux_pin_pos, dtype=torch.float32, device=device)
            poisson_inds = torch.cat([poisson_inds, aux_inds], dim=0)
            poisson_pos = torch.cat([poisson_pos, aux_pos], dim=0)
        if coupling.n_pairs > 0:
            rel_i = torch.tensor(coupling.pair_i, dtype=torch.long, device=device)
            rel_j = torch.tensor(coupling.pair_j, dtype=torch.long, device=device)
            rel_off = torch.tensor(coupling.offset0, dtype=torch.float64, device=device)
            rel_w = torch.tensor(coupling.poisson_pair_w, dtype=torch.float64, device=device)

    poisson = PoissonSystem(
        v,
        f,
        device,
        train_J=False,
        anchor_inds=poisson_inds,
        rel_pair_i=rel_i,
        rel_pair_j=rel_j,
        rel_offsets=rel_off,
        rel_lambda=1.0 if rel_i is not None else 0.0,
        rel_pair_w=rel_w,
    )
    with torch.no_grad():
        curr_v, curr_f = poisson.get_current_mesh(poisson_pos)
        handle_err = torch.sum((curr_v[handle_inds] - handle_pos) ** 2).sqrt() / len(handle_inds)

    mesh_dir = out_dir / "mesh"
    mesh_dir.mkdir(exist_ok=True)
    save_obj(
        mesh_dir / "hard_lock.obj",
        curr_v.detach().cpu().numpy(),
        curr_f.detach().cpu().numpy(),
        uvs=uvs.detach().cpu().numpy(),
        tex_inds=tex_inds.detach().cpu().numpy(),
        tex=tex.detach().cpu().numpy(),
    )

    imgs = render_mesh_360(
        curr_v, curr_f, uvs, tex_inds, tex,
        n_step=4, radius=radius, img_height=img_size, img_width=img_size,
    )
    strip = np.concatenate(imgs, axis=1)
    Image.fromarray(strip).save(out_dir / "hard_lock_instant.png")

    marker_imgs = render_mesh_with_markers_360(
        curr_v, curr_f,
        torch.cat([handle_inds, anchor_inds], dim=0),
        torch.cat([handle_pos, anchor_pos], dim=0),
        n_step=4, radius=radius, img_height=img_size, img_width=img_size,
    )
    summary = np.concatenate([strip, np.concatenate(marker_imgs, axis=1)], axis=0)
    Image.fromarray(summary).save(out_dir / "hard_lock_instant_summary.png")

    metrics = {"object": name, "handle_err": float(handle_err.item())}
    with open(out_dir / "metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apap-txt",
        type=Path,
        default=ROOT.parent / "JAPAP/configs/deform_meshes/data/apap_3d.txt",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "outputs/hard_handle_ablation/hard_lock_instant",
    )
    parser.add_argument("--cases", nargs="*", help="subset object names")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--radius", type=float, default=1.5)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--no-virtual-seams", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out_root.mkdir(parents=True, exist_ok=True)

    cases: list[tuple[str, str, str]] = []
    with open(args.apap_txt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, kp, lora = parse_apap_line(line)
            if args.cases and name not in args.cases:
                continue
            cases.append((name, kp, lora))

    print(f"Rendering {len(cases)} hard-lock snapshots -> {args.out_root}")
    for name, kp, lora in cases:
        m = snapshot_case(
            name, kp, lora, args.out_root, device,
            radius=args.radius,
            img_size=args.img_size,
            virtual_seams=not args.no_virtual_seams,
        )
        print(f"  OK {name} handle_err={m['handle_err']:.3e}")

    print(f"Done. See {args.out_root}/<case>/hard_lock_instant.png")


if __name__ == "__main__":
    main()
