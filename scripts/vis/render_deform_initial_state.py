"""
render_deform_initial_state.py

从形变实验输出目录生成「形变开始前」的示意图，与 optim.mp4 第一帧一致。

优先使用 optim_imgs/ 中最早一帧（即视频第一帧）；若无则尝试从视频提取；
最后回退为用 config.txt 中的原始 mesh 重新渲染。
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def parse_mesh_file(config_path: Path) -> Path:
    text = config_path.read_text()
    m = re.search(r"^mesh_file:\s*(.+)$", text, re.MULTILINE)
    if not m:
        raise ValueError(f"无法在 {config_path} 中找到 mesh_file")
    return Path(m.group(1).strip())


def copy_first_optim_img(run_dir: Path, out_path: Path) -> Path:
    optim_dir = run_dir / "optim_imgs"
    imgs = sorted(optim_dir.glob("*.png"))
    if not imgs:
        raise FileNotFoundError(f"未找到 optim_imgs: {optim_dir}")
    shutil.copy2(imgs[0], out_path)
    return imgs[0]


def extract_first_video_frame(run_dir: Path, out_path: Path) -> None:
    import imageio
    from PIL import Image

    video_path = run_dir / "video" / "optim.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"未找到视频: {video_path}")
    reader = imageio.get_reader(video_path, format="FFMPEG")
    frame = reader.get_data(0)
    reader.close()
    Image.fromarray(frame).save(out_path)


def render_from_mesh(mesh_file: Path, out_path: Path, device: str) -> None:
    import numpy as np
    import torch
    from PIL import Image

    from src.utils.geometry_utils import load_obj
    from src.utils.vis_utils import render_mesh_360

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    v, f, _, uvs, _, tex_inds, _, tex = load_obj(mesh_file)
    v_t = torch.from_numpy(v).to(dev)
    f_t = torch.from_numpy(f).to(dev)
    uvs_t = torch.from_numpy(uvs).to(dev)
    tex_inds_t = torch.from_numpy(tex_inds).to(dev)
    tex_t = torch.from_numpy(tex).to(dev)

    imgs = render_mesh_360(
        v_t, f_t, uvs_t, tex_inds_t, tex_t,
        n_step=4, radius=1.5,
    )
    Image.fromarray(np.concatenate(imgs, axis=1)).save(out_path)


def process_run_dir(
    run_dir: Path,
    source: str = "auto",
    device: str = "cuda:0",
    overwrite: bool = False,
) -> Path | None:
    run_dir = run_dir.resolve()
    out_path = run_dir / "initial_mesh.png"
    if out_path.exists() and not overwrite:
        print(f"[skip] 已存在: {out_path}")
        return out_path

    config_path = run_dir / "config.txt"
    if not config_path.exists():
        print(f"[skip] 无 config.txt: {run_dir}")
        return None

    if source in ("auto", "optim_imgs"):
        try:
            src = copy_first_optim_img(run_dir, out_path)
            print(f"[ok] {out_path} <- {src.name} (视频第一帧)")
            return out_path
        except FileNotFoundError:
            if source == "optim_imgs":
                raise

    if source in ("auto", "video"):
        try:
            extract_first_video_frame(run_dir, out_path)
            print(f"[ok] {out_path} <- video/optim.mp4 第 0 帧")
            return out_path
        except FileNotFoundError:
            if source == "video":
                raise

    mesh_file = parse_mesh_file(config_path)
    if not mesh_file.is_absolute():
        mesh_file = (Path.cwd() / mesh_file).resolve()
    render_from_mesh(mesh_file, out_path, device)
    print(f"[ok] {out_path} <- 重新渲染 {mesh_file}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="生成形变实验的初始状态示意图")
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="形变输出目录，或包含 mesh-* 子目录的 batch 根目录",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "optim_imgs", "video", "mesh"),
        default="auto",
        help="图片来源：optim_imgs(默认/与视频第一帧一致) / video / mesh(重渲原始 mesh)",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_dirs: list[Path] = []
    for p in args.paths:
        p = p.resolve()
        if (p / "config.txt").exists():
            run_dirs.append(p)
        else:
            run_dirs.extend(sorted(d for d in p.glob("mesh-*") if d.is_dir()))

    if not run_dirs:
        raise SystemExit("未找到任何形变输出目录")

    for run_dir in run_dirs:
        process_run_dir(
            run_dir,
            source=args.source,
            device=args.device,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
