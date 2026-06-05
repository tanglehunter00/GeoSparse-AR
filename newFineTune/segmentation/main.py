"""
GASP full-pretrain 匹配的分割微调入口。
复用 downstream/segmentation 的数据与 trainer；编码器见 models/ssl_encoder.py。
"""
import argparse
import os
import random
import sys
import warnings
from functools import partial
from pathlib import Path

warnings.filterwarnings("ignore")

_SEG_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _SEG_ROOT.parents[1]
_DOWNSTREAM_SEG = _REPO_ROOT / "downstream" / "segmentation"

sys.path.insert(0, str(_SEG_ROOT))
sys.path.insert(0, str(_DOWNSTREAM_SEG))
sys.path.append(".models")

import os.path as osp

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from models.build import build_model
from trainer import run_training, val_epoch
from utils.data_laseg import get_loader_LA_seg
from utils.data_utils import get_eval_split_dataloader, get_loader
from utils.utils import DiceCELoss, LayerDecayValueAssigner, SimpleLogger, get_parameter_groups

parser = argparse.ArgumentParser(description="GASP-matched segmentation (GeoPrior + HybridSparse pretrain)")

parser.add_argument("--MSD_data_base", default="", type=str)
parser.add_argument("--LA_Seg_data_base", default="", type=str)
parser.add_argument("--save_base", default="", type=str)
parser.add_argument("--json_list", default="dataset_withVal.json", type=str)
parser.add_argument("--logdir", default="", type=str)
parser.add_argument("--data_dir", default="", type=str)
parser.add_argument("--max_epochs", default=None, type=int)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--sw_batch_size", default=4, type=int)
parser.add_argument("--optim_lr_base", default=1e-4, type=float)
parser.add_argument("--optim_lr", default=3e-4, type=float)
parser.add_argument("--reg_weight", default=1e-5, type=float)
parser.add_argument("--noamp", action="store_true")
parser.add_argument("--val_every", default=50, type=int)
parser.add_argument("--distributed", action="store_true")
parser.add_argument("--world_size", default=1, type=int)
parser.add_argument("--rank", default=0, type=int)
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str)
parser.add_argument("--dist-backend", default="nccl", type=str)
parser.add_argument("--norm_name", default="instance", type=str)
parser.add_argument("--workers", default=0, type=int)
parser.add_argument(
    "--network",
    default="base_vit_gasp",
    type=str,
    help="实验名前缀；须使用 newFineTune 的 GASP 编码器",
)
parser.add_argument("--in_channels", default=1, type=int)
parser.add_argument("--out_channels", default=14, type=int)
parser.add_argument("--use_normal_dataset", action="store_true")
parser.add_argument("--num_samples", default=4, type=int)
parser.add_argument("--cache_rate", default=1.0, type=float)
parser.add_argument("--ratio", default=1.0, type=float)
parser.add_argument("--a_min", default=-175.0, type=float)
parser.add_argument("--a_max", default=250.0, type=float)
parser.add_argument("--b_min", default=0.0, type=float)
parser.add_argument("--b_max", default=1.0, type=float)
parser.add_argument("--space_x", default=1.5, type=float)
parser.add_argument("--space_y", default=1.5, type=float)
parser.add_argument("--space_z", default=2.0, type=float)
parser.add_argument("--roi_x", default=96, type=int)
parser.add_argument("--roi_y", default=96, type=int)
parser.add_argument("--roi_z", default=96, type=int)
parser.add_argument("--RandFlipd_prob", default=0.2, type=float)
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float)
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float)
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float)
parser.add_argument("--infer_overlap", default=0.75, type=float)
parser.add_argument("--lrschedule", default="warmup_cosine", type=str)
parser.add_argument("--pos_type", default="sincos3d", type=str)
parser.add_argument("--warmup_epochs", default=50, type=int)
parser.add_argument("--pretrain_path", default=None, help="GASP full pretrain ckpt（须含 geo_prior_gen）")
parser.add_argument("--pretrain_load_nonstrict", action="store_true", help="非严格加载预训练（不推荐）")
parser.add_argument("--resume", default="", type=str)
parser.add_argument("--smooth_dr", default=1e-6, type=float)
parser.add_argument("--smooth_nr", default=0.0, type=float)
parser.add_argument("--squared_dice", action="store_true")
parser.add_argument("--sample_ratios", default=None, type=str)
parser.add_argument("--patch_size", default=16, type=int)
parser.add_argument("--cache_num", default=1, type=int)
parser.add_argument("--layer_decay", default=0.75, type=float)
parser.add_argument("--task_name", default=None)
parser.add_argument("--run_test_after_train", action="store_true")
parser.add_argument("--test_json_list", default="", type=str)


def _normalize_seg_training_state_dict(state_dict):
    out = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod.") :]
        out[nk] = v
    return out


def main():
    args = parser.parse_args()
    if args.network != "base_vit_gasp":
        print(f"警告: network={args.network}，建议使用 base_vit_gasp 以区分 GASP 实验目录。")

    spacing_dict = {
        "Task01_BrainTumour": (1.0, 1.0, 1.0),
        "Task03_Liver": (1.0, 1.0, 1.0),
        "Task06_Lung": (1.0, 1.0, 1.0),
        "Task07_Pancreas": (1.0, 1.0, 1.5),
        "Task08_HepaticVessel": (1.0, 1.0, 1.5),
        "Task09_Spleen": (1.0, 1.0, 1.0),
        "Task10_Colon": (1.0, 1.0, 1.5),
        "LA_Seg": (1.0, 1.0, 1.0),
    }
    max_epochs_dict = {
        "Task01_BrainTumour": 1000,
        "Task03_Liver": 1000,
        "Task06_Lung": 1000,
        "Task07_Pancreas": 500,
        "Task08_HepaticVessel": 500,
        "Task09_Spleen": 1000,
        "Task10_Colon": 1000,
        "LA_Seg": 1000,
    }

    data_path = osp.join(args.MSD_data_base, args.task_name)
    save_root = osp.join(args.save_base, args.task_name)
    exp_name = f"{args.network}_{args.task_name}_{args.optim_lr}_{args.lrschedule}"

    if args.task_name == "LA_Seg":
        data_path = osp.join(args.LA_Seg_data_base, "2018LA_Seg_Training_Set")
        args.roi_x, args.roi_y, args.roi_z = 192, 192, 64
        args.layer_decay = 0.9

    _resume = (getattr(args, "resume", None) or "").strip()
    if args.pretrain_path is None and not _resume:
        args.optim_lr = args.optim_lr_base

    args.data_dir = data_path
    args.space_x, args.space_y, args.space_z = spacing_dict[args.task_name]
    if args.max_epochs is None:
        args.max_epochs = max_epochs_dict[args.task_name]
    args.save_root = save_root
    args.amp = not args.noamp
    args.logdir = os.path.join(args.save_root, exp_name)
    os.makedirs(args.logdir, exist_ok=True)

    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        port = int(args.dist_url.split(":")[-1])
        args.dist_url = args.dist_url.replace(str(port), str(port - random.randint(1, 100)))
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    logger = SimpleLogger(os.path.join(args.logdir, "log.txt"), verbose=True)

    if "LA_Seg" in args.data_dir:
        loader = get_loader_LA_seg(args, logger)
        properties = {"name": "LA_Seg", "labels": {"0": "background", "1": "left_atrium"}}
    else:
        loader, properties = get_loader(args, logger)

    args.out_channels = len(properties["labels"]) if "labels" in properties else args.out_channels
    args.in_channels = len(properties["modality"]) if "modality" in properties else args.in_channels

    if args.rank == 0:
        logger.info(
            f"[GASP] rank={args.rank} gpu={args.gpu} network={args.network} "
            f"pretrain={args.pretrain_path} roi={args.roi_x}_{args.roi_y}_{args.roi_z} patch={args.patch_size}"
        )

    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    resume_path = (getattr(args, "resume", None) or "").strip()
    if resume_path:
        resume_path = osp.abspath(resume_path)
        if not osp.isfile(resume_path):
            raise FileNotFoundError(resume_path)
        args.resume = resume_path
        args.pretrain_path = None
        logger.info(f"resume from {resume_path}")

    model = build_model(args, inf_size)

    loss_func = DiceCELoss(
        to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr,
    )
    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    model.cuda(args.gpu)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
        )

    use_layer_decay_optim = (args.pretrain_path is not None) or bool(resume_path)
    vit_prefix = "module.vit." if args.distributed else "vit."
    if not use_layer_decay_optim:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    else:
        num_layers = 12
        args.beta1, args.beta2, args.weight_decay = 0.9, 0.95, 0.05
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
        optim_params = get_parameter_groups(
            args,
            model,
            get_layer_id=partial(assigner.get_layer_id, prefix=vit_prefix),
            get_layer_scale=assigner.get_scale,
            verbose=args.rank == 0,
        )
        optimizer = torch.optim.AdamW(
            optim_params, lr=args.optim_lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay
        )

    start_epoch, resume_best_tracking = 0, None
    if resume_path:
        ck_blob = torch.load(resume_path, map_location="cpu", weights_only=False)
        m = model.module if hasattr(model, "module") else model
        m.load_state_dict(_normalize_seg_training_state_dict(ck_blob["state_dict"]), strict=False)
        if ck_blob.get("optimizer") is not None:
            optimizer.load_state_dict(ck_blob["optimizer"])
        start_epoch = int(ck_blob["epoch"]) + 1
        resume_best_tracking = (float(ck_blob.get("best_acc", 0)), int(ck_blob["epoch"]))

    scheduler = (
        LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs)
        if args.lrschedule == "warmup_cosine"
        else None
    )

    run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=loss_func,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
        dataset_props=properties,
        logger=logger,
        resume_best_tracking=resume_best_tracking,
    )


if __name__ == "__main__":
    main()
