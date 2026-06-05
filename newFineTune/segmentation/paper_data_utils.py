"""
MSD 论文对齐的数据加载：Task03 将 cancer(2) 合并入 liver(1) 做器官分割。
复用 downstream 的 ScaleIntensityRanged_select / Sampler，不修改原 data_utils.py。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from monai import data, transforms
from monai.data import load_decathlon_datalist, load_decathlon_properties

_SEG_ROOT = Path(__file__).resolve().parents[2] / "downstream" / "segmentation"
if str(_SEG_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEG_ROOT))

from utils.data_utils import (  # noqa: E402
    ScaleIntensityRanged_select,
    Sampler,
    _dataloader_parallel_kwargs,
)

from .label_transforms import MergeLiverTumorIntoOrgand  # noqa: E402

TASK03_ORGAN_NAME = "Task03_Liver"
MERGE_LIVER_TUMOR = MergeLiverTumorIntoOrgand(keys=["label"])


def _is_task03_organ(args) -> bool:
    task = getattr(args, "task_name", "") or ""
    data_dir = getattr(args, "data_dir", "") or ""
    return task == TASK03_ORGAN_NAME or TASK03_ORGAN_NAME in data_dir


def _organ_labels_task03():
    return {"0": "background", "1": "liver"}


def _inject_liver_merge(pipeline: list) -> list:
    out: list = []
    for t in pipeline:
        out.append(t)
        if isinstance(t, transforms.EnsureChannelFirstd):
            out.append(MERGE_LIVER_TUMOR)
    return out


def get_loader(args, logger=None):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    roi_size = [args.roi_x, args.roi_y, args.roi_z]
    sample_ratios = args.sample_ratios
    if sample_ratios is not None:
        sample_ratios = [int(x) for x in sample_ratios.split(",")]

    task03_organ = _is_task03_organ(args)

    property_keys = ["name", "modality", "labels", "numTraining", "numValidation"]
    properties = load_decathlon_properties(datalist_json, property_keys)
    properties["labels"] = {str(int(k)): v for k, v in properties["labels"].items()}

    if task03_organ:
        properties["labels"] = _organ_labels_task03()
        if logger is not None and getattr(args, "rank", 0) == 0:
            logger.info(
                "Task03 论文器官模式: label 2 (cancer) 合并入 label 1 (liver)，"
                "2 类分割 bg / liver(含肿瘤)"
            )

    is_multi_class = len(properties["labels"]) > 2

    train_base = [
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.Spacingd(
            keys=["image", "label"],
            pixdim=(args.space_x, args.space_y, args.space_z),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged_select(
            keys=["image"],
            a_min=args.a_min,
            a_max=args.a_max,
            b_min=args.b_min,
            b_max=args.b_max,
            clip=True,
        ),
        transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
        transforms.SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
    ]
    train_crop = (
        transforms.RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_size,
            ratios=sample_ratios,
            num_classes=len(properties["labels"]),
            num_samples=args.num_samples,
            image_key="image",
            image_threshold=0,
        )
        if is_multi_class
        else transforms.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_size,
            pos=1,
            neg=1,
            num_samples=args.num_samples,
            image_key="image",
            image_threshold=0,
        )
    )
    train_tail = [
        transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
        transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
        transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
        transforms.ToTensord(keys=["image", "label"]),
    ]
    train_transform = transforms.Compose(_inject_liver_merge(train_base) + [train_crop] + train_tail)

    val_base = [
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.Spacingd(
            keys=["image", "label"],
            pixdim=(args.space_x, args.space_y, args.space_z),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged_select(
            keys=["image"],
            a_min=args.a_min,
            a_max=args.a_max,
            b_min=args.b_min,
            b_max=args.b_max,
            clip=True,
        ),
        transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
        transforms.ToTensord(keys=["image", "label"]),
    ]
    val_transform = transforms.Compose(_inject_liver_merge(val_base))
    test_transform = transforms.Compose(_inject_liver_merge(list(val_base)))

    if args.test_mode:
        test_files = load_decathlon_datalist(datalist_json, True, "test", base_dir=data_dir)
        for item in test_files:
            item.update({"name": item["image"]})
        test_ds = data.Dataset(data=test_files, transform=test_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=False,
            **_dataloader_parallel_kwargs(args.workers),
        )
        return test_loader, properties

    datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
    for item in datalist:
        item.update({"name": item["image"]})
    data_ratio = args.ratio
    datalist = datalist[: int(len(datalist) * data_ratio)]
    if args.rank == 0 and logger is not None:
        logger.info("number of train subjects: {}, ratio: {}".format(len(datalist), data_ratio))

    if args.use_normal_dataset:
        train_ds = data.Dataset(data=datalist, transform=train_transform)
    else:
        train_ds = data.CacheDataset(
            data=datalist,
            transform=train_transform,
            cache_num=args.cache_num,
            cache_rate=args.cache_rate,
            num_workers=args.workers,
        )
    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True,
        **_dataloader_parallel_kwargs(args.workers),
    )

    val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
    for item in val_files:
        item.update({"name": item["image"]})
    val_ds = data.Dataset(data=val_files, transform=val_transform)
    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        sampler=val_sampler,
        pin_memory=False,
        **_dataloader_parallel_kwargs(args.workers),
    )
    return [train_loader, val_loader], properties


def get_eval_split_dataloader(args, logger, *, split: str = "test", json_relative=None):
    if json_relative is None or not str(json_relative).strip():
        json_relative = args.json_list
    datalist_json = os.path.join(args.data_dir, json_relative)

    val_base = [
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.Spacingd(
            keys=["image", "label"],
            pixdim=(args.space_x, args.space_y, args.space_z),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged_select(
            keys=["image"],
            a_min=args.a_min,
            a_max=args.a_max,
            b_min=args.b_min,
            b_max=args.b_max,
            clip=True,
        ),
        transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
        transforms.ToTensord(keys=["image", "label"]),
    ]
    val_like = transforms.Compose(_inject_liver_merge(val_base))

    if logger is None or args.rank != 0:
        return None
    if not os.path.isfile(datalist_json):
        logger.info(f"[{split}] skip: JSON not found: {datalist_json}")
        return None
    try:
        files = load_decathlon_datalist(datalist_json, True, split, base_dir=args.data_dir)
    except KeyError:
        logger.info(f"[{split}] skip: no `{split}` list in JSON")
        return None
    except Exception as e:
        logger.info(f"[{split}] skip: load_decathlon failed: {e}")
        return None
    if not files or any("label" not in it for it in files):
        logger.info(f"[{split}] skip: empty or missing labels")
        return None

    for item in files:
        item.update({"name": item["image"]})
    ds = data.Dataset(data=files, transform=val_like)
    val_sampler = Sampler(ds, shuffle=False) if args.distributed else None
    dl = data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        sampler=val_sampler,
        pin_memory=False,
        **_dataloader_parallel_kwargs(args.workers),
    )
    logger.info(f"[{split}] size={len(ds)} json={datalist_json}")
    return dl
