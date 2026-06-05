#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSD（Decathlon）分割微调 + AR-SSL 预训练 checkpoint — 控制台入口。

与笔记本流程等价：解析/生成 dataset json →（可选）MONAI 数据检查 → 封装预训练权重 →
以单卡方式调用 downstream/segmentation/main.py（勿加 --distributed）。

示例（按本机路径修改）::

    python scripts/msd_arssl_seg_train.py env-check

    python scripts/msd_arssl_seg_train.py install-pytorch

    python scripts/msd_arssl_seg_train.py install-seg-deps

续训示例（需与首次训练相同的任务名与学习率日程，checkpoint 常为…/experiment_dir/model.pt）::

    python scripts/msd_arssl_seg_train.py train ^
        --msd-task-dir "D:\\finetune\\MSD\\…\\Task06_Lung" ^
        --resume "D:\\finetune\\msd_seg_ssl_runs\\Task06_Lung\\base_vit_Task06_Lung_0.0003_warmup_cosine\\model.pt" ^
        --save-base "D:\\finetune\\msd_seg_ssl_runs" ^
        --epochs 50 --val-every 2

    python scripts/msd_arssl_seg_train.py train ^
        --msd-task-dir "D:\\finetune\\MSD\\Task06_Lung-001\\Task06_Lung" ^
        --pretrain "D:\\finetune\\ar-ssl4m\\ar-ssl4m\\checkpoints\\0\\0.pth" ^
        --save-base "D:\\finetune\\msd_seg_ssl_runs" ^
        --epochs 50 --val-every 5 --run-test-after-train ^
        [--test-json-list dataset_with_test.json]

使用 `--run-test-after-train` 时，须在用于训练的同一份 `--json_list`（或 `--test-json-list`
指向的另一个 Decathlon JSON）里提供 **`"test"`**：每项含 **`image` + `label`**
（仅用训练集划出、需有标注；官方无标注 imagesTs 无法算 Dice）。

.repo 根目录默认为本脚本上一级目录。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT_DEFAULT = SCRIPT_PATH.parents[1]
SEG_REL = Path("downstream") / "segmentation"

PIP_SEG_DEPS = (
    "monai",
    "tensorboardX",
    "timm",
    "transformers",
    "scipy",
    "h5py",
    "batchgenerators",
    "nibabel",
    "SimpleITK",
)


def _pip_exec(args: list[str], env: dict | None = None) -> None:
    subprocess.check_call([sys.executable, "-m", "pip"] + args, env=env)


def resolve_msd_json_list(task_dir: Path) -> str:
    """与笔记本一致：优先 dataset_withVal.json；已有 autogen；否则从 dataset.json 划 validation。"""
    task_dir = task_dir.resolve()
    pref = task_dir / "dataset_withVal.json"
    if pref.is_file():
        print("使用已有:", pref.name)
        return "dataset_withVal.json"
    autogen = task_dir / "dataset_withVal_autogen.json"
    if autogen.is_file():
        print("使用已有:", autogen.name)
        return "dataset_withVal_autogen.json"
    base = task_dir / "dataset.json"
    if not base.is_file():
        return "dataset_withVal.json"
    obj = json.loads(base.read_text(encoding="utf-8"))
    train = obj.get("training")
    if not train or len(train) < 3:
        raise ValueError("dataset.json 缺少 training 或样本过少")
    if "validation" in obj:
        print("dataset.json 已含 validation，直接使用")
        return "dataset.json"
    n_val = max(1, len(train) // 5)
    obj["validation"] = train[-n_val:]
    obj["training"] = train[:-n_val]
    obj["numValidation"] = len(obj["validation"])
    obj["numTraining"] = len(obj["training"])
    out = task_dir / "dataset_withVal_autogen.json"
    out.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print("已从 dataset.json 生成:", out.name, "（约 20%% 原 training → validation）")
    return out.name


def summarize_msd_task(task_dir: Path, json_filename: str) -> bool:
    from monai.data import load_decathlon_datalist, load_decathlon_properties

    task_dir = task_dir.resolve()
    jp = task_dir / json_filename
    print("\n===== MSD 目录检查 =====")
    print("路径:", task_dir)
    if not jp.is_file():
        print(f"❌ 缺少 `{json_filename}`。")
        cands = [p.name for p in task_dir.glob("*.json")]
        print("当前目录下的 .json:", cands or "（无）")
        return False

    listing = sorted([p.name for p in task_dir.iterdir()])
    print("条目（节选）:", listing[:25], "…" if len(listing) > 25 else "")

    props_keys = ["name", "modality", "labels", "numTraining", "numValidation"]
    try:
        prop = load_decathlon_properties(str(jp), props_keys)
    except Exception as e:
        print("❌ load_decathlon_properties 失败:", e)
        return False

    print("dataset 名称:", prop.get("name"))
    print("modality:", prop.get("modality"))
    print("labels (类别字典):", prop.get("labels"))
    print("numTraining / numValidation:", prop.get("numTraining"), "/", prop.get("numValidation"))

    ok = True
    for split_key in ("training", "validation"):
        try:
            rows = load_decathlon_datalist(str(jp), True, split_key, base_dir=str(task_dir))
        except Exception as e:
            print(f"⚠️ 无法读取 split `{split_key}`:", e)
            ok = False
            continue
        print(f"[{split_key}] 条数: {len(rows)}")
        if not rows:
            print("⚠️ 列表为空")
            ok = False
            continue
        ex = rows[0]
        pi = Path(ex["image"]) if Path(ex["image"]).is_absolute() else task_dir / ex["image"]
        pl = Path(ex["label"]) if Path(ex["label"]).is_absolute() else task_dir / ex["label"]
        print("  首条 image:", pi, "exists" if pi.is_file() else "❌ MISSING")
        print("  首条 label:", pl, "exists" if pl.is_file() else "❌ MISSING")
        ok = ok and pi.is_file() and pl.is_file()
    print("========================================\n")
    return ok


def wrap_pretrain_ckpt(pretrain_path: Path) -> Path:
    import torch

    tmp_root = tempfile.mkdtemp(prefix="seg_ssl_ckpt_")
    out = Path(tmp_root) / "pretrain_segmentation.pth"
    if not pretrain_path.is_file():
        raise FileNotFoundError(pretrain_path)
    try:
        try:
            raw = torch.load(pretrain_path, map_location="cpu", weights_only=False)
        except TypeError:
            raw = torch.load(pretrain_path, map_location="cpu")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torch.load 失败（PyTorch 安装可能损坏或与 conda/User 目录混装）。请先：\n"
            f'  "{sys.executable}" -m pip uninstall -y torch torchvision torchaudio\n'
            '  "%s" -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url <cu索引>\n'
            % sys.executable
        ) from exc

    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        torch.save(raw, out)
    elif isinstance(raw, dict):
        torch.save({"state_dict": raw}, out)
    else:
        raise TypeError("不支持的 checkpoint 类型")
    print("供 segmentation/build.py 读取的封装文件:", out)
    return out


def cuda_diag_lines() -> list[str]:
    import torch

    try:
        n = torch.cuda.device_count()
    except Exception as e:
        n = f"(error: {e})"
    return [
        f"  torch.__version__ = {torch.__version__!r}",
        f"  torch.version.cuda = {torch.version.cuda!r}",
        f"  torch.cuda.is_available() = {torch.cuda.is_available()}",
        f"  torch.cuda.device_count() = {n}",
        f"  sys.executable = {sys.executable!r}",
        f"  torch.__file__ = {getattr(torch, '__file__', '?')!r}",
    ]


def cmd_env_check() -> None:
    import shutil
    import site

    import torch

    print("===== 环境自检 =====")
    print("Python:", sys.version.split()[0], "| 解释器:", sys.executable)
    ug = getattr(site, "getusersitepackages", None)
    if callable(ug):
        print("用户 site-packages:", ug())
    print("\n" + "\n".join(cuda_diag_lines()))
    try:
        import torchvision

        print("torchvision:", torchvision.__version__)
    except ImportError:
        print("torchvision: 未安装")
    try:
        import torchaudio

        print("torchaudio:", torchaudio.__version__)
    except (ImportError, OSError, RuntimeError) as e:
        print("torchaudio: （可选）", repr(e))

    exe = shutil.which("nvidia-smi")
    if exe:
        r = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=20)
        print("----- nvidia-smi -L -----")
        print((r.stdout or r.stderr).strip())
    print("=========================\n")


def cmd_install_pytorch(cu_url: str) -> None:
    env = os.environ.copy()
    env.pop("PIP_USER", None)
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"],
        env=env,
        check=False,
    )
    _pip_exec(
        [
            "install",
            "--upgrade",
            "--force-reinstall",
            "--no-cache-dir",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            cu_url,
        ],
        env=env,
    )
    subprocess.check_call(
        [sys.executable, "-c", "import torch; print('OK', torch.__version__, torch.__file__)"],
        env=env,
    )


def cmd_install_seg_deps() -> None:
    _pip_exec(["install", "-q"] + list(PIP_SEG_DEPS))
    print("已安装/更新:", ", ".join(PIP_SEG_DEPS))


def run_training_main(
    repo_root: Path,
    msd_task_dir: Path,
    save_base: Path,
    workers: int,
    demo_quick: bool,
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
    import torch

    if not torch.cuda.is_available():
        msg = (
            "`main.py` 需要 CUDA，当前 torch.cuda.is_available() 为 False。\n\n"
            "【诊断】\n"
            + "\n".join(cuda_diag_lines())
            + '\n\n可运行: python scripts/msd_arssl_seg_train.py env-check'
        )
        raise SystemExit(msg)

    msd_task_dir = msd_task_dir.resolve()
    if not msd_task_dir.is_dir():
        raise SystemExit(f"无效 MSD 任务目录: {msd_task_dir}")

    repo_root = repo_root.resolve()
    seg_dir = repo_root / SEG_REL
    if not (seg_dir / "main.py").is_file():
        raise SystemExit(f"未找到 {seg_dir / 'main.py'}，请检查 --repo-root")

    json_list = resolve_msd_json_list(msd_task_dir)
    msd_data_base = msd_task_dir.parent
    task_name = msd_task_dir.name

    if not skip_summary:
        ok = summarize_msd_task(msd_task_dir, json_list)
        if not ok:
            raise SystemExit("MSD 数据检查失败，请修正数据集或路径。")

    resume_ckpt_resolved: Path | None = None
    pretrain_wrapped: Path | None = None
    if resume_ckpt is not None:
        resume_ckpt_resolved = resume_ckpt.resolve()
        if not resume_ckpt_resolved.is_file():
            raise SystemExit(f"--resume 不是有效文件: {resume_ckpt_resolved}")
    else:
        if pretrain_ckpt is None:
            raise SystemExit("run_training_main: 需要提供 pretrain_ckpt 或使用 resume_ckpt")
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
    ]
    if pretrain_wrapped is not None:
        argv_list += ["--pretrain_path", str(pretrain_wrapped)]
    else:
        assert resume_ckpt_resolved is not None
        argv_list += ["--resume", str(resume_ckpt_resolved)]
    if demo_quick:
        argv_list += ["--max_epochs", "2", "--val_every", "1", "--use_normal_dataset"]
    else:
        if epochs is not None:
            argv_list += ["--max_epochs", str(epochs)]
        if val_every is not None:
            argv_list += ["--val_every", str(val_every)]
    if run_test_after_train:
        argv_list += ["--run_test_after_train"]
    _tjl = (test_json_list or "").strip()
    if _tjl:
        argv_list += ["--test_json_list", _tjl]
    argv_list += extra_main_args

    old_argv = sys.argv.copy()
    old_cwd = os.getcwd()
    sys.argv = argv_list
    os.chdir(str(seg_dir))
    if str(seg_dir) not in sys.path:
        sys.path.insert(0, str(seg_dir))

    print("调用 segmentation main，参数:", " ".join(argv_list[1:]))
    spec = importlib.util.spec_from_file_location("seg_ssl_main", seg_dir / "main.py")
    if spec is None or spec.loader is None:
        raise SystemExit("无法加载 main.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        mod.main()
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

    print("完成。日志与权重在 SAVE_BASE / task_name 下对应实验目录（见 log.txt）。")


def build_parser() -> argparse.ArgumentParser:
    # --repo-root 若只挂在顶层，写在 `train` 之后时子解析器无法识别，会与 main_rest(REMAINDER) 冲突，
    # 导致后续 --msd-task-dir 等被吞掉。此处用 parents 并入 train。
    repo_parent = argparse.ArgumentParser(add_help=False)
    repo_parent.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT_DEFAULT,
        help=f"仓库根目录（默认: {REPO_ROOT_DEFAULT}）",
    )

    p = argparse.ArgumentParser(description="MSD AR-SSL 分割训练控制台脚本")
    p.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT_DEFAULT,
        help=f"[env-check/install 等非 train 子命令用] 仓库根目录（默认: {REPO_ROOT_DEFAULT}）",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("env-check", help="打印 Python / torch / CUDA / nvidia-smi")

    ipp = sub.add_parser("install-pytorch", help="用 pip 重装 torch+torchvision+torchaudio（当前解释器）")
    ipp.add_argument(
        "--index-url",
        default="https://download.pytorch.org/whl/cu130",
        help="PyTorch CUDA 轮子索引",
    )

    sub.add_parser("install-seg-deps", help=f"pip 安装下游依赖: {', '.join(PIP_SEG_DEPS)}")

    tr = sub.add_parser("train", parents=[repo_parent], help="运行 MSD 微调（单卡）")
    tr.add_argument("--msd-task-dir", type=Path, required=True, help="MSD 任务根（含 dataset.json、imagesTr 等）")
    tr.add_argument(
        "--pretrain",
        type=Path,
        default=None,
        help="AR-SSL 预训练 .pth（新训必填；配合 --resume 续训时可省略）",
    )
    tr.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="已有训练 checkpoint（如 …/experiment_dir/model.pt），从 epoch+1 续训且恢复 optimizer/best_acc；与 --pretrain 二选一",
    )
    tr.add_argument("--save-base", type=Path, required=True, help="保存 run 的根目录")
    tr.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader 进程数；3D 肺部全卷解压易占 RAM，OOM 时请保持 0 或减小",
    )
    tr.add_argument("--demo-quick", action="store_true", help="max_epochs=2, val_every=1, monai Dataset")
    tr.add_argument("--epochs", type=int, default=None, help="传给 main.py --max_epochs（不与 --demo-quick 同用）")
    tr.add_argument("--val-every", type=int, default=None, dest="val_every", help="传给 main.py --val_every")
    tr.add_argument(
        "--run-test-after-train",
        action="store_true",
        dest="run_test_after_train",
        help="训练结束后用最优权重对 JSON 中的 test split 跑一次 Dice（需 image+label）",
    )
    tr.add_argument(
        "--test-json-list",
        default="",
        dest="test_json_list",
        help="含 test 的 json 文件名（相对任务目录）；默认同 dataset json_list",
    )
    tr.add_argument("--skip-dataset-summary", action="store_true", help="跳过 MONAI datalist 检查")
    tr.add_argument(
        "main_rest",
        nargs=argparse.REMAINDER,
        help="传给 main.py 的额外参数（建议以 -- 开头，例如：-- --noamp）",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "env-check":
        cmd_env_check()
        return
    if args.cmd == "install-pytorch":
        cmd_install_pytorch(args.index_url)
        return
    if args.cmd == "install-seg-deps":
        cmd_install_seg_deps()
        return
    if args.cmd == "train":
        if args.pretrain is None and args.resume is None:
            raise SystemExit("train：请指定 --pretrain（新训）或 --resume（续训）。二者至少其一。")
        if args.pretrain is not None and args.resume is not None:
            raise SystemExit("train：`--pretrain` 与 `--resume` 勿同时使用（续训用 --resume 即可）。")
        extra = args.main_rest
        if extra[:1] == ["--"]:
            extra = extra[1:]
        run_training_main(
            repo_root=args.repo_root,
            msd_task_dir=args.msd_task_dir,
            pretrain_ckpt=args.pretrain,
            resume_ckpt=args.resume,
            save_base=args.save_base,
            workers=args.workers,
            demo_quick=args.demo_quick,
            skip_summary=args.skip_dataset_summary,
            extra_main_args=extra,
            epochs=args.epochs,
            val_every=args.val_every,
            run_test_after_train=args.run_test_after_train,
            test_json_list=args.test_json_list,
        )
        return
    raise SystemExit("未知子命令")


if __name__ == "__main__":
    main()
