#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MSD 分割微调 — GASP full pretrain（GeoPrior + HybridSparse 全开）→ newFineTune/segmentation。"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT_DEFAULT = SCRIPT_PATH.parents[1]
SEG_REL = Path("newFineTune") / "segmentation"

# 复用 baseline 脚本的数据检查与 ckpt 封装
sys.path.insert(0, str(SCRIPT_PATH.parent))
from msd_arssl_seg_train import (  # noqa: E402
    REPO_ROOT_DEFAULT as _,
    cuda_diag_lines,
    resolve_msd_json_list,
    summarize_msd_task,
    wrap_pretrain_ckpt,
)


def run_gasp_training_main(
    repo_root: Path,
    msd_task_dir: Path,
    save_base: Path,
    workers: int,
    skip_summary: bool,
    extra_main_args: list[str],
    *,
    pretrain_ckpt: Path | None = None,
    resume_ckpt: Path | None = None,
    epochs: int | None = None,
    val_every: int | None = None,
) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "需要 CUDA。\n" + "\n".join(cuda_diag_lines()) + "\n运行: python scripts/msd_gasp_seg_train.py env-check"
        )

    msd_task_dir = msd_task_dir.resolve()
    repo_root = repo_root.resolve()
    seg_dir = repo_root / SEG_REL
    if not (seg_dir / "main.py").is_file():
        raise SystemExit(f"未找到 {seg_dir / 'main.py'}")

    json_list = resolve_msd_json_list(msd_task_dir)
    if not skip_summary and not summarize_msd_task(msd_task_dir, json_list):
        raise SystemExit("MSD 数据检查失败")

    argv_list = [
        "main.py",
        "--MSD_data_base",
        str(msd_task_dir.parent),
        "--save_base",
        str(save_base),
        "--task_name",
        msd_task_dir.name,
        "--json_list",
        json_list,
        "--workers",
        str(workers),
        "--network",
        "base_vit_gasp",
    ]
    if resume_ckpt is not None:
        argv_list += ["--resume", str(resume_ckpt.resolve())]
    else:
        if pretrain_ckpt is None:
            raise SystemExit("需要 --pretrain 或 --resume")
        argv_list += ["--pretrain_path", str(wrap_pretrain_ckpt(pretrain_ckpt.resolve()))]
    if epochs is not None:
        argv_list += ["--max_epochs", str(epochs)]
    if val_every is not None:
        argv_list += ["--val_every", str(val_every)]
    argv_list += extra_main_args

    save_base.mkdir(parents=True, exist_ok=True)
    old_argv, old_cwd = sys.argv.copy(), os.getcwd()
    sys.argv = argv_list
    os.chdir(str(seg_dir))
    sys.path.insert(0, str(seg_dir))
    print("[GASP] 调用 newFineTune/segmentation:", " ".join(argv_list[1:]))
    spec = importlib.util.spec_from_file_location("gasp_seg_main", seg_dir / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        mod.main()
    finally:
        sys.argv, os.chdir(old_cwd)


def main() -> None:
    p = argparse.ArgumentParser(description="MSD + GASP full-pretrain 分割微调")
    p.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env-check", help="同 msd_arssl_seg_train env-check")
    tr = sub.add_parser("train")
    tr.add_argument("--msd-task-dir", type=Path, required=True)
    tr.add_argument("--pretrain", type=Path, default=None)
    tr.add_argument("--resume", type=Path, default=None)
    tr.add_argument("--save-base", type=Path, required=True)
    tr.add_argument("--workers", type=int, default=0)
    tr.add_argument("--epochs", type=int, default=None)
    tr.add_argument("--val-every", type=int, default=None, dest="val_every")
    tr.add_argument("--skip-dataset-summary", action="store_true")
    tr.add_argument("main_rest", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if args.cmd == "env-check":
        from msd_arssl_seg_train import cmd_env_check

        cmd_env_check()
        return
    if args.cmd == "train":
        if args.pretrain is None and args.resume is None:
            raise SystemExit("需要 --pretrain 或 --resume")
        extra = args.main_rest[1:] if args.main_rest[:1] == ["--"] else args.main_rest
        run_gasp_training_main(
            repo_root=args.repo_root,
            msd_task_dir=args.msd_task_dir,
            save_base=args.save_base,
            workers=args.workers,
            skip_summary=args.skip_dataset_summary,
            extra_main_args=extra,
            pretrain_ckpt=args.pretrain,
            resume_ckpt=args.resume,
            epochs=args.epochs,
            val_every=args.val_every,
        )


if __name__ == "__main__":
    main()
