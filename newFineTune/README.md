# newFineTune — 与 GASP Full Pretrain 匹配的分割下游

## 设计目标（为何有“理论保证”）

预训练（`newFullPretrain/newNewModel.py`，`GeoPriorGen3DOn=1` 且 `HybridSparseAttnOn=1`）与旧下游（`downstream/segmentation`）不一致之处：

| 项目 | 旧下游 | GASP 预训练 | 本目录 `newFineTune` |
|------|--------|-------------|----------------------|
| 序列 | BOS + patch | BOS + patch + **EOS** | **BOS + patch + EOS**（与预训练一致） |
| 注意力 | 稠密因果 | 因果 + **Hybrid 稀疏** + **Geo bias** | **相同构造** |
| 模块 | 无 `geo_prior_gen` | 有 | **有** |
| 输出给分割头 | `[:,1:,:]` | 重建用 `[:,1:-1,:]` | **`[:,1:-1,:]`**（与重建监督区间一致） |

因此：

1. **结构同构**：`PretrainMatchedEncoder` 的 `state_dict` 键与 `ReconModel.model.*` 一一对应（含 `geo_prior_gen.decay`）。
2. **前向对齐**：在相同 `(t,h,w)` 与 `patch_size` 下，进入 `layers` 的 `attention_mask` 与预训练相同；仅最后去掉 `decoder_pred`，改为 UNETR 卷积解码。
3. **表示一致性**：finetune 时 encoder 在 patch token 上的动态与预训练 MAE 目标所依赖的 token 一致，避免“在错误注意力图上学出的权重被当成通用 ViT”的分布偏移。

**仍需 finetune 的部分**：UNETR 的 `encoder1`–`decoder2` 与 `out`（随机初始化），与论文中“冻结/微调 encoder + 训练分割头”相同。

## 目录

```
newFineTune/segmentation/
  main.py              # 入口（复用 downstream 的 data_utils / trainer）
  models/
    ssl_encoder.py     # PretrainMatchedEncoder（GASP 全开）
    build.py           # 严格加载 + 校验 geo_prior_gen
    unetr.py
```

## 预训练 checkpoint 要求

- 必须由 **`GeoPriorGen3DOn=1` 且 `HybridSparseAttnOn=1`** 训练得到。
- `state_dict` 中应含 **`geo_prior_gen.decay`**（或完整 `model.geo_prior_gen.*`）。
- **不要**使用开关关闭的 `ssl_checkpoint.pth`（会触发 `build.py` 报错）。

## 运行示例（PowerShell）

```powershell
cd D:\Cursor\AR-SSL4M-DEMO\newFineTune\segmentation

python main.py `
  --MSD_data_base D:\finetune\MSD\Task06_Lung-001 `
  --save_base D:\finetune\msd_seg_gasp_runs `
  --task_name Task06_Lung `
  --json_list dataset_withVal_autogen.json `
  --pretrain_path D:\finetune\你的GASP全开权重.pth `
  --max_epochs 150 `
  --val_every 5 `
  --workers 0
```

或使用仓库脚本：

```powershell
python scripts/msd_gasp_seg_train.py train `
  --msd-task-dir D:\finetune\MSD\Task06_Lung-001\Task06_Lung `
  --pretrain D:\finetune\你的GASP全开权重.pth `
  --save-base D:\finetune\msd_seg_gasp_runs `
  --epochs 150 --val-every 5
```

## 与旧下游对比

| 场景 | 使用 |
|------|------|
| baseline / `ssl_checkpoint`（Geo 关） | `downstream/segmentation` |
| **GASP full pretrain（Geo+Hybrid 开）** | **`newFineTune/segmentation`** |

## 超参建议

- **`patch_size=16`**：与预训练一致（`configs/datasets.py` 默认 16³）。
- **`roi_*`**：可与预训练 `img_size` 不同（如 96 vs 128）；`patchifier` 卷积核仍为 16³，**Geo/Sparse mask 按当前 crop 的 `(t,h,w)` 动态生成**，与预训练在可变分辨率下的行为一致。
- **`pos_type=sincos3d`**：与默认 full pretrain 一致。
