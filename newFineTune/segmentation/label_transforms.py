"""论文对齐的标签变换（不修改 downstream/）。"""
from __future__ import annotations

import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform


class MergeLiverTumorIntoOrgand(MapTransform):
    """
    Task03 器官分割：MSD label 1=liver, 2=cancer → 统一为 1=liver（含肿瘤区域）。

    0 保持 background；任一前景 (1 或 2) 变为 1。
    """

    def __init__(self, keys: KeysCollection = "label", allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            label = np.asarray(d[key])
            merged = (label > 0).astype(label.dtype, copy=False)
            d[key] = merged
        return d
