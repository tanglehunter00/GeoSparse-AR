"""
Task03 等论文对齐 MSD 微调：在调用 newFineTune/segmentation/main 前注入 paper_data_utils。
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GASP_SEG_REL = Path("newFineTune") / "segmentation"
_DOWNSTREAM_SEG_REL = Path("downstream") / "segmentation"

# slug → 是否使用 paper_data_utils（Task03 肝+肿瘤合并为器官）
PAPER_DATA_UTILS_SLUGS = frozenset({"task03"})


def run_training_with_paper_data(
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
    run_test_after_train: bool = False,
    test_json_list: str = "",
) -> None:
    """GASP newFineTune 入口 + paper_data_utils（Task03 肝+肿瘤合并）。"""
    from scripts.msd_arssl_seg_train import (
        resolve_msd_json_list,
        summarize_msd_task,
        wrap_pretrain_ckpt,
    )

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA")

    msd_task_dir = msd_task_dir.resolve()
    repo_root = repo_root.resolve()
    seg_dir = repo_root / _GASP_SEG_REL
    downstream_seg_dir = repo_root / _DOWNSTREAM_SEG_REL
    if not (seg_dir / "main.py").is_file():
        raise SystemExit(f"未找到 {seg_dir / 'main.py'}")

    json_list = resolve_msd_json_list(msd_task_dir)
    msd_data_base = msd_task_dir.parent
    task_name = msd_task_dir.name

    if not skip_summary:
        ok = summarize_msd_task(msd_task_dir, json_list)
        if not ok:
            raise SystemExit("MSD 数据检查失败")

    if resume_ckpt is not None:
        resume_ckpt_resolved = resume_ckpt.resolve()
        pretrain_wrapped = None
        if not resume_ckpt_resolved.is_file():
            raise SystemExit(f"--resume 无效: {resume_ckpt_resolved}")
    else:
        resume_ckpt_resolved = None
        if pretrain_ckpt is None:
            raise SystemExit("需要 pretrain_ckpt")
        pretrain_wrapped = wrap_pretrain_ckpt(pretrain_ckpt.resolve())

    save_base.mkdir(parents=True, exist_ok=True)

    argv_list = [
        "main.py",
        "--MSD_data_base",
        str(msd_data_base),
        "--save_base",
        str(save_base),
        "--task_name",
        task_name,
        "--json_list",
        json_list,
        "--workers",
        str(workers),
        "--network",
        "base_vit_gasp",
    ]
    if pretrain_wrapped is not None:
        argv_list += ["--pretrain_path", str(pretrain_wrapped)]
    else:
        argv_list += ["--resume", str(resume_ckpt_resolved)]
    if epochs is not None:
        argv_list += ["--max_epochs", str(epochs)]
    if val_every is not None:
        argv_list += ["--val_every", str(val_every)]
    if run_test_after_train:
        argv_list += ["--run_test_after_train"]
    if (test_json_list or "").strip():
        argv_list += ["--test_json_list", test_json_list.strip()]
    argv_list += extra_main_args

    old_argv = sys.argv.copy()
    old_cwd = os.getcwd()
    sys.argv = argv_list
    os.chdir(str(seg_dir))
    if str(seg_dir) not in sys.path:
        sys.path.insert(0, str(seg_dir))
    if str(downstream_seg_dir) not in sys.path:
        sys.path.insert(0, str(downstream_seg_dir))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import utils.data_utils as du  # noqa: WPS433

    from newFineTune.segmentation import paper_data_utils as pdu  # noqa: WPS433

    du.get_loader = pdu.get_loader
    du.get_eval_split_dataloader = pdu.get_eval_split_dataloader
    print("[GASP+paper] 已启用 paper_data_utils（Task03: cancer→liver 器官分割）")

    print("[GASP+paper] 调用 newFineTune/segmentation:", " ".join(argv_list[1:]))
    spec = importlib.util.spec_from_file_location("gasp_seg_main_paper", seg_dir / "main.py")
    if spec is None or spec.loader is None:
        raise SystemExit("无法加载 main.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        mod.main()
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

    print("完成（GASP + paper data utils）。日志与权重在 save_base / task_name 下。")
