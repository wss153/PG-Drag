#!/usr/bin/env python3
"""Aggregate SD ablation results into SYMMETRIC_DIRICHLET_ABLATION.md."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def load_metrics(run_dir: Path) -> dict:
    p = run_dir / "metrics" / "eval_metrics.json"
    if not p.is_file():
        return {}
    with open(p) as f:
        return json.load(f)


def main() -> None:
    ab_root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs/sd_ablation/latest"
    if not ab_root.is_dir():
        print(f"Not found: {ab_root}")
        sys.exit(1)

    lines = [
        "# Symmetric Dirichlet Ablation (PG-Drag / HAPAP)",
        "",
        f"Results root: `{ab_root}`",
        "",
        "## Implementation",
        "",
        "Per-face energy from top-2 singular values of reconstructed mesh Jacobian:",
        "",
        "```",
        "E_f = 0.5 * (σ1² + σ2² + σ1⁻² + σ2⁻²) - 2",
        "L_SD = Σ(A0_f * E_f) / Σ(A0_f)",
        "```",
        "",
        "Stage 2 only; SSP disabled when SD reg enabled.",
        "",
        "## Results",
        "",
        "| Variant | Case | handle_err | SD-P99 | SD mean | σ_min | λ_sd |",
        "|---------|------|------------|--------|---------|-------|------|",
    ]

    for variant_dir in sorted(ab_root.iterdir()):
        if not variant_dir.is_dir():
            continue
        for case_dir in sorted(variant_dir.iterdir()):
            m = load_metrics(case_dir)
            if not m:
                continue
            lines.append(
                f"| {variant_dir.name} | {case_dir.name} | "
                f"{m.get('final_handle_error', 'NA'):.4e} | "
                f"{m.get('sd_p99', 'NA'):.4f} | "
                f"{m.get('sd_mean', 'NA'):.4f} | "
                f"{m.get('sigma_min_global', 'NA'):.4f} | "
                f"{m.get('lambda_sd', 'NA')} |"
            )

    lines += [
        "",
        "## Renderings",
        "",
        "See `deformed_mesh_summary.png` under each case directory.",
        "",
        "## Recommended λ_sd",
        "",
        "TBD after reviewing sweep (choose globally, not per-model).",
    ]

    out = ROOT / "SYMMETRIC_DIRICHLET_ABLATION.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
