"""
Colab MSD 分割微调入口（独立任务、均从同一 0.pth 冷启动）。

不修改 downstream/；复用 scripts/msd_arssl_seg_train.py 的 run_training_main。
"""
from __future__ import annotations

import sys
from pathlib import Path

_NEW_FT_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _NEW_FT_ROOT.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.msd_arssl_seg_train import run_training_main  # noqa: E402
from newFineTune.msd_paper_train import (  # noqa: E402
    PAPER_DATA_UTILS_SLUGS,
    run_training_with_paper_data,
)

# slug（输出目录名）→ MSD 任务文件夹名（dataset/unzip 下）
MSD_FINETUNE_TASKS: dict[str, str] = {
    "task03": "Task03_Liver",
    "task06": "Task06_Lung",
    "task07": "Task07_Pancreas",
    "task08": "Task08_HepaticVessel",
    "task09": "Task09_Spleen",
    "task10": "Task10_Colon",
}


def run_one_task(
    *,
    task_slug: str,
    repo_root: Path | str,
    dataset_unzip_base: Path | str,
    pretrain_ckpt: Path | str,
    output_root: Path | str,
    epochs: int = 150,
    val_every: int = 10,
    workers: int = 0,
    skip_dataset_summary: bool = False,
    extra_main_args: list[str] | None = None,
) -> Path:
    """
    从 pretrain_ckpt（如 rawCheckPoint/0.pth）冷启动微调单个 MSD 任务。

    输出目录::
        {output_root}/{task_slug}/{TaskXX_Name}/base_vit_.../log.txt, model.pt, ...

    各任务互不影响；勿对其它 task 使用 --resume。
    """
    if task_slug not in MSD_FINETUNE_TASKS:
        raise KeyError(f"未知 task_slug={task_slug!r}，可选: {list(MSD_FINETUNE_TASKS)}")

    repo_root = Path(repo_root).resolve()
    dataset_unzip_base = Path(dataset_unzip_base).resolve()
    pretrain_ckpt = Path(pretrain_ckpt).resolve()
    output_root = Path(output_root).resolve()

    msd_task_name = MSD_FINETUNE_TASKS[task_slug]
    msd_task_dir = dataset_unzip_base / msd_task_name
    save_base = output_root / task_slug

    if not pretrain_ckpt.is_file():
        raise FileNotFoundError(f"预训练权重不存在: {pretrain_ckpt}")
    if not msd_task_dir.is_dir():
        raise FileNotFoundError(f"MSD 任务目录不存在: {msd_task_dir}")

    print("=" * 72)
    print(f"task_slug      : {task_slug}")
    print(f"msd_task_name  : {msd_task_name}")
    print(f"msd_task_dir   : {msd_task_dir}")
    print(f"pretrain_ckpt  : {pretrain_ckpt}")
    print(f"save_base      : {save_base}")
    print(f"epochs         : {epochs}")
    print(f"val_every      : {val_every}")
    if task_slug in PAPER_DATA_UTILS_SLUGS:
        print("data mode      : paper（Task03 label2 cancer 合并入 liver 器官类）")
    print("=" * 72)

    train_kw = dict(
        repo_root=repo_root,
        msd_task_dir=msd_task_dir,
        save_base=save_base,
        workers=workers,
        skip_summary=skip_dataset_summary,
        extra_main_args=extra_main_args or [],
        pretrain_ckpt=pretrain_ckpt,
        resume_ckpt=None,
        epochs=epochs,
        val_every=val_every,
    )
    if task_slug in PAPER_DATA_UTILS_SLUGS:
        run_training_with_paper_data(**train_kw)
    else:
        run_training_main(
            demo_quick=False,
            run_test_after_train=False,
            test_json_list="",
            **train_kw,
        )

    return save_base / msd_task_name


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Colab 单任务 MSD finetune（从 0.pth 冷启动）")
    p.add_argument("--task-slug", required=True, choices=list(MSD_FINETUNE_TASKS))
    p.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    p.add_argument(
        "--dataset-unzip",
        type=Path,
        default=Path("/content/drive/MyDrive/dataset/unzip"),
    )
    p.add_argument(
        "--pretrain",
        type=Path,
        default=Path("/content/drive/MyDrive/finetune/rawCheckPoint/0.pth"),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("/content/drive/MyDrive/finetune/output"),
    )
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--val-every", type=int, default=10)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--skip-dataset-summary", action="store_true")
    args = p.parse_args()

    run_one_task(
        task_slug=args.task_slug,
        repo_root=args.repo_root,
        dataset_unzip_base=args.dataset_unzip,
        pretrain_ckpt=args.pretrain,
        output_root=args.output_root,
        epochs=args.epochs,
        val_every=args.val_every,
        workers=args.workers,
        skip_dataset_summary=args.skip_dataset_summary,
    )
