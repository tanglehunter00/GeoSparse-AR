import argparse
import os
import pdb
import warnings
warnings.filterwarnings('ignore')
import random
import sys
sys.path.append('.models')

import os.path as osp
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed

from monai.inferers import sliding_window_inference
# from monai.losses import DiceCELoss
from utils.utils import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from functools import partial

from models.build import build_model
from trainer import run_training, val_epoch
from utils.data_utils import get_loader, get_eval_split_dataloader
from utils.data_laseg import get_loader_LA_seg
from utils.utils import SimpleLogger, LayerDecayValueAssigner, get_parameter_groups


parser = argparse.ArgumentParser(description="Segmentation pipeline")

parser.add_argument("--MSD_data_base", default="", type=str)
parser.add_argument("--LA_Seg_data_base", default="", type=str)
parser.add_argument("--save_base", default="", type=str, help="directory to save")
parser.add_argument("--json_list", default="dataset_withVal.json", type=str, help="dataset json file")
parser.add_argument("--logdir", default="", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--data_dir", default="", type=str, help="dataset directory")
parser.add_argument(
    "--max_epochs",
    default=None,
    type=int,
    help="max epochs；默认按 MSD 任务表；命令行传入时覆盖（便于 demo / 脚本短训）",
)
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr_base", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_lr", default=3e-4, type=float, help="optimization learning rate")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=50, type=int, help="validation frequency")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--workers", default=4, type=int, help="number of workers")
parser.add_argument("--network", default='base_vit', type=str, help="feature size")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument("--num_samples", default=4, type=int, help="sample number in each volume")
parser.add_argument("--cache_rate", default=1.0, type=float, help="cache rate of CacheDataset")
parser.add_argument("--ratio", default=1.0, type=float, help="training data ratio between 0-1")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.75, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--pos_type", default="sincos3d", type=str, help="position embedding")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--pretrain_path", default=None, help="training from pretrained checkpoint")
parser.add_argument(
    "--resume",
    default="",
    type=str,
    help="从此 checkpoint 恢复（如 …/model.pt）：加载 state_dict、若存在则恢复 optimizer/best_acc，并从 epoch+1 继续训练；通常会跳过 --pretrain_path 的初始化加载",
)
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")
parser.add_argument('--sample_ratios', default=None, type=str, help='ratios in RandCropByLabelClassesd')
parser.add_argument("--patch_size", default=16, type=int, help="patch_size")
parser.add_argument("--cache_num", default=1, type=int)
parser.add_argument("--layer_decay", default=0.75, type=float, help="layer-wise learning rate decay rate")
parser.add_argument("--task_name", default=None)
parser.add_argument(
    "--run_test_after_train",
    action="store_true",
    help="训练完成后加载验证集最优 model.pt（或 model_final.pt），对 JSON 中 test 列表跑 DiceMetric；test 每项需含 image+label。",
)
parser.add_argument(
    "--test_json_list",
    default="",
    type=str,
    help="从该文件名（相对于 data_dir）读取 test split；默认为 --json_list 同一文件内需含 test。",
)


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

    spacing_dict = {
        'Task01_BrainTumour': (1.0, 1.0, 1.0),
        'Task03_Liver': (1.0, 1.0, 1.0),
        'Task06_Lung': (1.0, 1.0, 1.0),
        'Task07_Pancreas': (1.0, 1.0, 1.5),
        'Task08_HepaticVessel': (1.0, 1.0, 1.5),
        'Task09_Spleen': (1.0, 1.0, 1.0),
        'Task10_Colon': (1.0, 1.0, 1.5),
        'LA_Seg': (1.0, 1.0, 1.0),
    }

    max_epochs_dict = {
        'Task01_BrainTumour': 1000,
        'Task03_Liver': 1000,
        'Task06_Lung': 1000,
        'Task07_Pancreas': 500,
        'Task08_HepaticVessel': 500,
        'Task09_Spleen': 1000,
        'Task10_Colon': 1000,
        'LA_Seg': 1000,
    }

    data_path = osp.join(args.MSD_data_base, args.task_name)
    save_root = osp.join(args.save_base, args.task_name)
    exp_name = f'{args.network}_{args.task_name}_{args.optim_lr}_{args.lrschedule}'

    if args.task_name == 'LA_Seg':
        data_path = osp.join(args.LA_Seg_data_base, '2018LA_Seg_Training_Set')
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
        print("Found total gpus", args.ngpus_per_node)
        port = int(args.dist_url.split(':')[-1])
        x = random.randint(-100, 100)
        new_port = port-x
        args.dist_url = args.dist_url.replace(str(port), str(new_port))
        print('dist_url', args.dist_url)
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
    logger = SimpleLogger(os.path.join(args.logdir, 'log.txt'), verbose=True)

    if "LA_Seg" in args.data_dir:
        args.in_channels = 1
        # args.out_channels = 1
        loader = get_loader_LA_seg(args, logger)
        properties = {'name': 'LA_Seg', 'labels':{'0': 'background', '1': 'left_atrium'}}
    else:
        loader, properties = get_loader(args, logger)

    args.out_channels = len(properties['labels']) if 'labels' in properties else args.out_channels
    args.in_channels = len(properties['modality']) if 'modality' in properties else args.in_channels
    logger.info(f"rank={args.rank}, gpu={args.gpu}")
    if args.rank == 0:
        logger.info(f"Batch size is: {args.batch_size}, epochs {args.max_epochs}, network: {args.network} "
                          f"pretrain_path: {args.pretrain_path}, image_size: {args.roi_x}_{args.roi_y}_{args.roi_z}, "
                          f"infer_overlap: {args.infer_overlap}, lr: {args.optim_lr}, layer_decay: {args.layer_decay}")

    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    rois_size = inf_size

    resume_path = (getattr(args, "resume", None) or "").strip()
    if resume_path:
        resume_path = osp.abspath(resume_path)
        if not osp.isfile(resume_path):
            raise FileNotFoundError(f"--resume 文件不存在: {resume_path}")
        args.resume = resume_path
        if args.rank == 0:
            logger.info(
                f"--resume={resume_path}（跳过 ViT --pretrain_path 初值加载，完整权重由 checkpoint 恢复）"
            )
        args.pretrain_path = None

    model = build_model(args, rois_size)

    dice_loss = DiceCELoss(
        to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr,
    )

    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)

    loss_func = dice_loss

    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)

    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters count: {pytorch_total_params/1e6:.2f}M")

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)

    use_layer_decay_optim = (args.pretrain_path is not None) or bool(resume_path)
    vit_prefix = "module.vit." if args.distributed else "vit."
    if not use_layer_decay_optim:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    else:
        num_layers = 12
        args.beta1 = 0.9
        args.beta2 = 0.95
        args.weight_decay = 0.05
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
        optim_params = get_parameter_groups(args, model, get_layer_id=partial(assigner.get_layer_id, prefix=vit_prefix),
                                                 get_layer_scale=assigner.get_scale,
                                                 verbose=True)
        optimizer = torch.optim.AdamW(optim_params,
                                            lr=args.optim_lr,
                                            betas=(args.beta1, args.beta2),
                                            weight_decay=args.weight_decay)

    start_epoch = 0
    resume_best_tracking = None

    if resume_path:
        try:
            ck_blob = torch.load(resume_path, map_location="cpu", weights_only=False)
        except TypeError:
            ck_blob = torch.load(resume_path, map_location="cpu")
        if not isinstance(ck_blob, dict) or "state_dict" not in ck_blob:
            raise RuntimeError(f"{resume_path} 不是此处保存的训练 checkpoint（需含 state_dict、epoch）")
        sd_norm = _normalize_seg_training_state_dict(ck_blob["state_dict"])
        m = model.module if hasattr(model, "module") else model
        inc = m.load_state_dict(sd_norm, strict=False)
        missing = getattr(inc, "missing_keys", None) or inc[0]
        unexpected = getattr(inc, "unexpected_keys", None) or inc[1]
        logger.info(f"resume: load missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
        if ck_blob.get("optimizer") is not None:
            try:
                optimizer.load_state_dict(ck_blob["optimizer"])
                logger.info("resume: restored optimizer.state_dict")
            except Exception as e:
                logger.info(f"resume: optimizer restore skipped ({e}); using freshly built optimizer.")
        ck_ep = ck_blob.get("epoch")
        if ck_ep is None:
            raise RuntimeError("checkpoint 缺少 epoch")
        ck_ep_int = int(ck_ep)
        start_epoch = ck_ep_int + 1
        best_acc_ck = float(ck_blob.get("best_acc", 0.0))
        resume_best_tracking = (best_acc_ck, ck_ep_int)
        logger.info(f"resume: saved epoch={ck_ep_int}, best_acc={best_acc_ck}, continue from epoch {start_epoch}")

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        ckpt_last = max(-1, start_epoch - 1) if resume_path else -1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epochs, last_epoch=ckpt_last
        )
    else:
        scheduler = None

    accuracy = run_training(
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

    if args.run_test_after_train:
        _run_holdout_test_if_configured(
            args=args,
            logger=logger,
            model=model,
            dice_metric=dice_acc,
            properties=properties,
            model_inferer=model_inferer,
            post_label=post_label,
            post_pred=post_pred,
        )

    return accuracy


def _run_holdout_test_if_configured(args, logger, model, dice_metric, properties, model_inferer, post_label, post_pred):
    if args.rank != 0:
        return
    if "LA_Seg" in args.data_dir:
        logger.info("[test] skip LA_Seg in this runner")
        return
    if args.distributed:
        logger.info("[test] skip under --distributed")
        return
    jr = (args.test_json_list or "").strip() or None
    test_dl = get_eval_split_dataloader(args, logger, split="test", json_relative=jr)
    if test_dl is None:
        return

    ck_candidates = [osp.join(args.logdir, "model.pt"), osp.join(args.logdir, "model_final.pt")]
    ck_path = next((p for p in ck_candidates if osp.isfile(p)), None)
    if ck_path is None:
        logger.info("[test] skip: neither model.pt nor model_final.pt in logdir")
        return

    try:
        blob = torch.load(ck_path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(ck_path, map_location="cpu")
    state = blob.get("state_dict", blob)
    m = model.module if hasattr(model, "module") else model
    m.load_state_dict(state, strict=False)
    model.cuda(args.gpu)
    model.eval()

    labels = properties.get("labels")
    te = args.max_epochs - 1 if args.max_epochs > 0 else 0
    dice_acc_batch = val_epoch(
        model,
        test_dl,
        epoch=te,
        acc_func=dice_metric,
        args=args,
        model_inferer=model_inferer,
        post_label=post_label,
        post_pred=post_pred,
        labels=labels,
        logger=logger,
    )
    dice_acc_batch = np.asarray(dice_acc_batch).reshape(-1)
    lbl = labels or {}
    if len(dice_acc_batch) > 1:
        msg_each = ",".join(
            [f"{lbl.get(str(i_), str(i_))}={dice_acc_batch[i_]:.4f}" for i_ in range(len(dice_acc_batch))]
        )
        mean_nb = np.mean(dice_acc_batch[1:])
    else:
        msg_each = "n/a"
        mean_nb = float(dice_acc_batch[0])
    logger.info(
        "[test] FINAL (best ckpt %s): mean_dice(no_bg)=%.4f, per-class: (%s)",
        osp.basename(ck_path),
        mean_nb,
        msg_each,
    )


if __name__ == "__main__":
    main()
