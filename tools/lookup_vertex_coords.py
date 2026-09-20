#!/usr/bin/env python3
"""Look up mesh vertex coordinates by index.

APAP keypoint files use 0-based vertex indices (same as NumPy / this script).
OBJ face indices are 1-based; pass --one-based if you copied numbers from an OBJ.

Examples:
    python tools/lookup_vertex_coords.py mesh.obj 3673 8719
    python tools/lookup_vertex_coords.py mesh.obj --file keypoints/000/user_single_keypoints.txt
    python tools/lookup_vertex_coords.py mesh.obj --one-based 3674
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_obj_vertices(path: Path) -> np.ndarray:
    verts: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"{path}: malformed vertex line: {line.rstrip()!r}")
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        raise ValueError(f"{path}: no vertices found")
    return np.asarray(verts, dtype=np.float64)


def parse_indices(tokens: list[str]) -> list[int]:
    out: list[int] = []
    for token in tokens:
        for part in token.replace(",", " ").split():
            if not part:
                continue
            out.append(int(part))
    return out


def parse_keypoint_file(path: Path) -> tuple[list[int], list[list[float] | None]]:
    ids: list[int] = []
    targets: list[list[float] | None] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            ids.append(int(parts[0]))
            if len(parts) >= 4:
                targets.append([float(parts[1]), float(parts[2]), float(parts[3])])
            else:
                targets.append(None)
    if not ids:
        raise ValueError(f"{path}: no vertex indices found")
    return ids, targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mesh", type=Path, help="Mesh .obj path")
    parser.add_argument(
        "indices",
        nargs="*",
        help="Vertex indices (0-based by default). Also accepts comma-separated values.",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        default=None,
        help="Keypoint txt: each line 'vid [x y z]'. Looks up vid on the mesh.",
    )
    parser.add_argument(
        "--one-based",
        action="store_true",
        help="Treat input indices as OBJ 1-based (internally converted to 0-based).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of a table.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mesh_path = args.mesh
    if not mesh_path.exists():
        raise SystemExit(f"Mesh not found: {mesh_path}")

    vertices = parse_obj_vertices(mesh_path)
    n = len(vertices)

    targets: list[list[float] | None] = []
    if args.file is not None:
        ids, targets = parse_keypoint_file(args.file)
    else:
        ids = parse_indices(args.indices)
        if not ids and not sys.stdin.isatty():
            ids = parse_indices(sys.stdin.read().split())
        if not ids:
            raise SystemExit("Provide vertex indices or --file")
        targets = [None] * len(ids)

    rows = []
    errors = []
    for raw_id, target in zip(ids, targets):
        idx = raw_id - 1 if args.one_based else raw_id
        if idx < 0 or idx >= n:
            errors.append(f"index {raw_id} -> {idx} is out of range [0, {n - 1}]")
            continue
        xyz = vertices[idx]
        rows.append(
            {
                "input_index": raw_id,
                "vertex_index": int(idx),
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
                "target": target,
            }
        )

    if args.json:
        import json

        print(
            json.dumps(
                {"mesh": str(mesh_path), "n_vertices": n, "one_based": args.one_based, "vertices": rows},
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(f"{mesh_path}  V={n}  index={'1-based' if args.one_based else '0-based'}")
        has_target = any(r["target"] is not None for r in rows)
        if has_target:
            print(
                f"{'input':>8s} {'idx':>8s} {'x':>14s} {'y':>14s} {'z':>14s} "
                f"{'tx':>14s} {'ty':>14s} {'tz':>14s} {'|v-t|':>10s}"
            )
            for r in rows:
                t = r["target"]
                dist = float(np.linalg.norm(np.array([r["x"], r["y"], r["z"]]) - np.array(t)))
                print(
                    f"{r['input_index']:8d} {r['vertex_index']:8d} "
                    f"{r['x']:14.8f} {r['y']:14.8f} {r['z']:14.8f} "
                    f"{t[0]:14.8f} {t[1]:14.8f} {t[2]:14.8f} {dist:10.6f}"
                )
        else:
            print(f"{'input':>8s} {'idx':>8s} {'x':>14s} {'y':>14s} {'z':>14s}")
            for r in rows:
                print(
                    f"{r['input_index']:8d} {r['vertex_index']:8d} "
                    f"{r['x']:14.8f} {r['y']:14.8f} {r['z']:14.8f}"
                )

    if errors:
        for err in errors:
            print(f"error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
