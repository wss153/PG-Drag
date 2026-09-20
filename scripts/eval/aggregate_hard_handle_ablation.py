#!/usr/bin/env python3
"""Aggregate Hard-Handle ablation results into HARD_HANDLE_ABLATION.md."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QC = ROOT.parent / "Quantitative Comparisons"
sys.path.insert(0, str(QC))

from handle_error_metrics import calculate_handle_error, load_keypoints  # noqa: E402


def load_metrics(run_dir: Path) -> dict:
    p = run_dir / "metrics" / "eval_metrics.json"
    if not p.is_file():
        return {}
    with open(p) as f:
        return json.load(f)


def parse_case_name(case_dir: Path) -> tuple[str, str]:
    m = re.match(r"(.+)_handle-(\w+)_anchor-(\w+)", case_dir.name)
    if not m:
        return case_dir.name, "000"
    return m.group(1), m.group(2)


def compute_nme(case_dir: Path) -> dict | None:
    obj_name, kp = parse_case_name(case_dir)
    mesh_root = ROOT / "data" / "apap_3d" / "processed" / obj_name
    kp_path = mesh_root / "keypoints" / kp / "user_single_keypoints.txt"
    deformed = case_dir / "mesh" / "deformed.obj"
    source = mesh_root / "mesh.obj"
    if not all(p.is_file() for p in (kp_path, deformed, source)):
        return None
    handle_indices, target_positions = load_keypoints(str(kp_path))
    return calculate_handle_error(
        str(source), str(deformed), handle_indices, target_positions
    )


def try_clip_score(case_dir: Path, prompt: str) -> float | None:
    try:
        from mesh_clip_evaluator import CLIPEvaluationConfig, MeshCLIPEvaluator
    except ImportError:
        return None
    deformed = case_dir / "mesh" / "deformed.obj"
    if not deformed.is_file():
        return None
    tex = deformed.parent / "deformed_texture.png"
    if not tex.is_file():
        alt = mesh_root_tex(case_dir)
        tex = alt if alt and alt.is_file() else None
    cfg = CLIPEvaluationConfig(device="cuda")
    ev = MeshCLIPEvaluator(cfg)
    try:
        res = ev.evaluate_mesh(
            mesh_path=str(deformed),
            texture_path=str(tex) if tex else None,
            text_prompts=[prompt],
            num_views=8,
        )
        return float(res["mean"])
    except Exception:
        return None


def mesh_root_tex(case_dir: Path) -> Path | None:
    obj_name, _ = parse_case_name(case_dir)
    p = ROOT / "data" / "apap_3d" / "processed" / obj_name / "mesh_texture.png"
    return p if p.is_file() else None


def fmt(x, spec=".4e") -> str:
    if x is None:
        return "NA"
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return str(x)


def main() -> None:
    ab_root = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else ROOT / "outputs/hard_handle_ablation/latest"
    )
    if not ab_root.is_dir():
        print(f"Not found: {ab_root}")
        sys.exit(1)

    rows: list[dict] = []
    for variant_dir in sorted(ab_root.iterdir()):
        if not variant_dir.is_dir():
            continue
        for case_dir in sorted(variant_dir.iterdir()):
            m = load_metrics(case_dir)
            if not m:
                continue
            nme = compute_nme(case_dir)
            clip = try_clip_score(case_dir, m.get("prompt", ""))
            rows.append(
                {
                    "variant": variant_dir.name,
                    "case": case_dir.name,
                    "obj": parse_case_name(case_dir)[0],
                    "handle_err": m.get("final_handle_error"),
                    "stage1_handle_err": m.get("stage1_handle_error"),
                    "stage1_step": m.get("stage1_end_step"),
                    "sd_p99": m.get("sd_p99"),
                    "sd_mean": m.get("sd_mean"),
                    "nme": nme.get("nme") if nme else None,
                    "nme_pct": nme.get("nme_percent") if nme else None,
                    "clip": clip,
                    "hard_handle": m.get("hard_handle"),
                    "prompt": m.get("prompt", ""),
                }
            )

    lines = [
        "# Hard-Handle Ablation (PG-Drag / HAPAP)",
        "",
        f"Results root: `{ab_root}`",
        "",
        "## Motivation (reviewer)",
        "",
        "> Why are handle constraints not enforced like anchor constraints?",
        "> When Stage 1 differs, Jacobian receives gradients from lightweight SDS",
        "> while avoiding soft-handle pull-back.",
        "",
        "## Variants",
        "",
        "| Variant | Handle constraint | Stage-1 SDS | Handle Freeze |",
        "|---------|-------------------|-------------|---------------|",
        "| **baseline_soft** | Soft `l_kp` (λ=1) | λ=0.1 + ROI | Soft freeze (w=1000) |",
        "| **hard_handle** | Poisson `[K_a; K_h]V=[T_a; T_h]` | λ=0.1 + ROI | Off |",
        "",
        "Shared: same mesh/handle/anchor/seed, SSP, Stage-2 SDS, lr, 400+600 steps.",
        "",
        "## Quantitative results",
        "",
        "| Variant | Case | handle_err | stage1_err | stage1_step | SD-P99 | NME | CLIP |",
        "|---------|------|------------|------------|-------------|--------|-----|------|",
    ]

    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['obj']} | "
            f"{fmt(r['handle_err'])} | {fmt(r['stage1_handle_err'])} | "
            f"{r['stage1_step'] if r['stage1_step'] is not None else 'NA'} | "
            f"{fmt(r['sd_p99'], '.2f')} | {fmt(r['nme'], '.6f')} | "
            f"{fmt(r['clip'], '.4f')} |"
        )

    lines += [
        "",
        "## Renderings (same 4-view camera, radius=1.5)",
        "",
        "Per case directory:",
        "- `initial_mesh.png` — rest pose",
        "- `stage1_end_mesh.png` — end of Stage 1",
        "- `deformed_mesh.png` / `deformed_mesh_summary.png` — final",
        "",
        "## Qualitative checklist (handle neighborhood)",
        "",
        "Inspect handle-adjacent triangles for:",
        "- triangle stretching / local compression",
        "- spikes / unnatural bending",
        "- distortion propagation into non-edit region",
        "",
        "## Summary",
        "",
    ]

    by_obj: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_obj.setdefault(r["obj"], {})[r["variant"]] = r

    for obj, variants in sorted(by_obj.items()):
        soft = variants.get("baseline_soft")
        hard = variants.get("hard_handle")
        lines.append(f"### {obj}")
        if hard:
            he = hard.get("handle_err")
            lines.append(
                f"- **Hard Handle zero error?** final handle_err = {fmt(he)} "
                f"(stage1 = {fmt(hard.get('stage1_handle_err'))})"
            )
        if soft and hard:
            lines.append(
                f"- **SD-P99**: soft {fmt(soft.get('sd_p99'), '.2f')} vs "
                f"hard {fmt(hard.get('sd_p99'), '.2f')}"
            )
            if soft.get("clip") and hard.get("clip"):
                lines.append(
                    f"- **CLIP**: soft {fmt(soft.get('clip'), '.4f')} vs "
                    f"hard {fmt(hard.get('clip'), '.4f')}"
                )
            lines.append(
                f"- **Stage-1 length**: soft step {soft.get('stage1_step')} vs "
                f"hard step {hard.get('stage1_step')} (hard uses full max_stage1_steps)"
            )
        lines.append("")

    lines += [
        "## Conclusions (fill after visual review)",
        "",
        "1. Hard Handle enforces near-zero handle error via Poisson (no post-solve assignment).",
        "2. Stage-1 lightweight SDS provides Jacobian gradients without soft handle pull-back.",
        "3. Compare handle-neighborhood distortion vs soft+freeze baseline.",
        "4. Stage 2 (SSP+SDS) may or may not repair Stage-1 artifacts from hard constraints.",
        "",
    ]

    out = ROOT / "HARD_HANDLE_ABLATION.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
