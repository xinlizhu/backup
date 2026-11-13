# CLIP-L CIR-QDiag / CIRR / PinPoint 实验总结

更新日期：2026-08-07

本文档总结当前围绕标准 CLIP-B/L、CIRR composed image retrieval、CIR-QDiag
negative-text supervision、Full/LoRA backbone tuning 和 PinPoint 官方全量评估完成的实验。

## 1. 统一实验与评估口径

### 训练数据

- CIRR train：28,225 个训练样本。
- Hard-negative 训练样本中，25,225 个样本具有可用 negative text。
- ADD negatives：32,012 条。
- SUBSTITUTE negatives：53,284 条。
- 当前正式 hard-negative 实验不使用 DELETE。

### 训练样本如何构建

每个 CIRR 训练样本的基本三元组为：

```text
(reference image, positive modification text, target image)
```

`data/cirr_train_with_negatives.jsonl` 在三元组上额外挂载 CIR-QDiag 生成的
negative modification text。构建命令：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train

python scripts/prepare_train_manifest.py \
  --annotations /mnt/data/datasets/search_dataset/zxl/CIR_new/datasets/CIRR/cirr/captions/cap.rc2.train.json \
  --image-split /mnt/data/datasets/search_dataset/zxl/CIR_new/datasets/CIRR/cirr/image_splits/split.rc2.train.json \
  --image-root /mnt/data/datasets/search_dataset/zxl/CIR_new/datasets/CIRR \
  --negative-jsonl ../cir_qdiag/outputs/train_500_v5/final.jsonl \
  --output data/cirr_train_with_negatives.jsonl
```

Baseline 和 Hard 使用同一份 manifest：

- Baseline 设置 `negative_mode: none`，不 tokenize、不编码、不使用 negative text。
- 当前正式 Hard 设置 `negative_mode: add_substitute`，不使用 DELETE。
- 每个样本每个 epoch 最多取 6 条 ADD/SUBSTITUTE negative。若原始数量更多，会按
  `seed + epoch + sample_id + negative_id` 的确定性哈希顺序选取，因此不同 epoch
  可以看到不同 negative，但同一 seed 可复现。
- 样本不需要拥有相同数量的 negative。没有 negative 的样本仍参与普通 batch
  InfoNCE，但不参与 hard loss。
- `max_delete_negatives_per_sample: 3` 只在 `add_substitute_delete` 模式下生效；正式
  Full/LoRA Hard 实验中它不生效。

### 模型和 composed query

正式模型使用本地标准 OpenAI CLIP ViT-L/14，不使用 BGE。对 reference image 和
modification text 分别编码为 `r` 和 `t`，query 由以下方式合成：

```text
residual = MLP([r, t, r * t])
q = normalize(r + t + residual)
```

MLP 是 `2304 -> 1024 -> 768`，使用 GELU 和 dropout 0.1。最后一层零初始化，所以所有
实验都从普通 `normalize(r + t)` CLIP composition 出发，而不是从随机 query 出发。
Target image 使用同一 CLIP image encoder 得到归一化 embedding。

### 训练损失

Baseline 使用对称 batch InfoNCE：

```text
L_batch = 0.5 * (CE(q -> target) + CE(target -> q))
```

每条 query 的正例是同一个 batch 中配对的 target，其他 63 个 target 是 in-batch negatives。
当前正式训练为单卡 batch 64，所以每条正 query 在 `L_batch` 中对比 1 个正例和
63 个 in-batch negatives。多卡时会 all-gather query/target，候选数变为全局 batch。

Hard 训练另外构造同 reference 的 negative queries `q-`，并让正 query `q+` 在相同
target image 上的分数高于自己的 1--6 个 `q-`：

```text
L_hard = CE([sim(q+, target), sim(q-_1, target), ..., sim(q-_m, target)], label=0)
L_total = L_batch + hard_negative_weight * L_hard
```

这里的 hard negatives 是错误文本 query，不是 negative image。它们不会把 batch 样本数从 64
扩大为 `64 * 7`，而是为每个拥有 negative 的样本增加一个局部 hard-query
分类项。`hard_negative_weight=0.5` 表示该项以 0.5 的系数加到普通 InfoNCE。

### Full/LoRA 正式训练参数

| 参数 | Full | LoRA r16 |
|---|---:|---:|
| Epochs | 10 | 10 |
| 单卡 batch size | 64 | 64 |
| Gradient accumulation | 1 | 1 |
| Head learning rate | `1e-4` | `1e-4` |
| Backbone/LoRA learning rate | `1e-6` | `1e-4` |
| Weight decay | 0.01 | 0.01 |
| Warmup | 5% | 5% |
| LR schedule | warmup + cosine decay | warmup + cosine decay |
| Max grad norm | 1.0 | 1.0 |
| Mixed precision | BF16 | BF16 |
| Initial temperature | 0.07，可学习 | 0.07，可学习 |
| Gradient checkpointing | non-reentrant | non-reentrant |
| Checkpoint interval | 每 2 epochs | 每 2 epochs |
| Seed | 42 | 42 |

Full 更新全部 vision/text encoder、projection、fusion head 和 temperature。LoRA 冻结原始
backbone weight，只在 vision/text 每层 attention 的 `q_proj`/`v_proj` 注入 rank-16 LoRA，
`alpha=32`、dropout 0.05，同时训练 fusion head 和 temperature。第一个 batch 会执行
gradient audit；声明可训练却没有梯度，或梯度不是有限值时，训练直接报错。

### 如何复现正式训练

安装项目：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
python -m pip install -e '.[dev]'
```

复现最早的 Frozen CLIP-B/L baseline、ADD/SUBSTITUTE 和包含 DELETE 的六组对照：

```bash
CUDA_VISIBLE_DEVICES=0 GPUS=1 VARIANTS="clip_b clip_l" \
  bash scripts/run_clip_comparison.sh
```

复现后续的 Frozen CLIP-L batch-64 hard-weight 0/0.5/1.0 对照：

```bash
CUDA_VISIBLE_DEVICES=0 VARIANTS=clip_l \
  bash scripts/run_clip_batch64_weights.sh
```

顺序训练 Full baseline 和 Full Hard-0.5，并在训练后运行 CIRR validation：

```bash
CUDA_VISIBLE_DEVICES=0 METHODS=full bash scripts/run_clip_l_full_lora.sh
```

顺序训练 LoRA baseline 和 LoRA Hard-0.5：

```bash
CUDA_VISIBLE_DEVICES=0 METHODS=lora_r16 bash scripts/run_clip_l_full_lora.sh
```

单独训练一个模型：

```bash
# Full baseline
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  scripts/train.py --config configs/clip_l_full_baseline.yaml

# Full Hard-0.5
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  scripts/train.py --config configs/clip_l_full_hard_w0_5.yaml
```

从 `last.pt` 断点续训：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  scripts/train.py \
  --config configs/clip_l_full_hard_w0_5.yaml \
  --resume-from outputs/clip_l/full/hard_w0_5/last.pt
```

运行 Full hard-weight 网格（默认补训 0.1/0.25/0.75/1.0，复用已有 0/0.5）：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_clip_l_full_hard_grid.sh
```

输出结构：

```text
outputs/clip_l/full/{baseline,hard_w0_1,hard_w0_25,hard_w0_5,hard_w0_75,hard_w1_0}/
outputs/clip_l/lora_r16/{baseline,hard_w0_5}/
  effective_config.json       # 实际生效配置
  run_summary.json            # 样本数、negative 数、参数量
  gradient_audit.json         # 首批梯度审计
  train_metrics.jsonl         # 每 20 step 训练指标
  checkpoint_epoch_002.pt     # 每 2 epoch 的 model-only checkpoint
  ...
  checkpoint_epoch_010.pt
  last.pt                     # model + optimizer + scheduler，用于续训
  val_metrics.json            # CIRR validation 结果
```

### CIRR validation

- Query 数量：4,181。
- `Recall@K`：在完整 validation gallery 中检索目标图。
- `Group Recall@K`：只在每条 query 对应的高度相似候选组中检索目标图。

### PinPoint 官方全量评估

- 官方 query：7,635 条。
- 官方 index signatures：109,600。
- 本地可用候选图：109,598。
- 缺失的两张图不属于 reference/positive，但出现在少量 negative 标注中。因此绝对
  `NegRecall@10` 可能有极小低估，但所有模型使用相同图库，模型间对比公平。
- 主要指标：官方 `mAP@10`（越高越好）和 `NegRecall@10`（越低越好）。

## 2. 最早的 Frozen CLIP-B/L 实验

这组实验冻结 CLIP image/text encoder，只训练 composed-query fusion head 和 temperature。

| Backbone | 训练方式 | CIRR R@1 | R@10 | Group R@1 |
|---|---|---:|---:|---:|
| CLIP-B | baseline | **14.11** | **54.72** | **35.85** |
| CLIP-B | ADD/SUBSTITUTE | 13.78 | 54.22 | 35.30 |
| CLIP-B | +DELETE | 13.80 | 54.15 | 35.33 |
| CLIP-L | baseline | **14.26** | **57.43** | **37.67** |
| CLIP-L | ADD/SUBSTITUTE | 13.80 | 56.42 | 36.81 |
| CLIP-L | +DELETE | 13.59 | 56.35 | 36.76 |

结论：

- CLIP-L 略优于 CLIP-B。
- Frozen encoder 下，hard-negative supervision 没有改善 CIRR。
- 加入 DELETE 没有带来稳定收益，因此后续正式实验只保留 ADD/SUBSTITUTE。

## 3. Frozen CLIP-L batch-64 hard-weight 实验

| Hard weight | CIRR R@1 | R@5 | R@10 | R@50 | Group R@1 |
|---:|---:|---:|---:|---:|---:|
| 0.0 | **19.40** | **50.73** | **66.11** | 91.10 | 44.27 |
| 0.5 | 19.13 | 50.59 | 65.70 | **91.44** | **44.39** |
| 1.0 | 18.42 | 49.63 | 65.08 | 90.98 | 43.41 |

结论：

- batch=64 的结果明显优于早期 Frozen 配置。
- Frozen 条件下，hard weight 增大主要损害普通 CIRR Recall。
- Weight 0.5 只带来非常小的 Group R@1 改善；weight 1.0 明显过强。

## 4. Pilot negative-text 方向诊断

该诊断使用 PinPoint pilot 500 组生成的 negative text：1,933 条正指令、
5,576 条 negative。测量：

```text
Delta = sim(q+, positive image) - sim(q-, positive image)
Raw win = Delta > 0
```

每条 negative 先在该 group 的 positive images 上统计，然后在 instruction 内平均、
group 内平均，最后对 group 等权平均。这样 negative 数量多的 group 不会获得更大权重。
`strict` 只包含 ADD/SUBSTITUTE，排除语义方向较特殊的 DELETE。

### 4.1 Frozen CLIP-L

| Frozen 模型 | 平均 Delta | Raw win rate |
|---|---:|---:|
| baseline | 0.01005 | 68.14% |
| Hard-0.5 | **0.02223** | **78.86%** |

在 strict 子集上，Hard-0.5 确实增强了正确文本与错误文本的 embedding 方向分离。
Robust win rate 仍只有约 2%，Recall/NDCG 和官方 negative-image discrimination 没有同步提升。

原始结果：

```text
../cir_qdiag_openai/outputs/pinpoint_pilot_500_v7/
  clip_l_pinpoint_diagnostics/comparison.csv
```

### 4.2 Full CLIP-L baseline vs Hard-0.5

这次复用了 PinPoint 官方全量评估中已缓存的、两个 Full 模型各自的 109,598 张
candidate image embedding 与 7,635 条正 query embedding，只重新编码 pilot 中的
negative text。因此 baseline 和 Hard-0.5 均使用自己微调后的 image/text backbone 与
fusion head，没有在两个模型之间共享 image embedding。

| 子集 | Full baseline Delta | Full Hard-0.5 Delta | Baseline win | Hard-0.5 win | Hard 胜率变化 |
|---|---:|---:|---:|---:|---:|
| strict（ADD + SUBSTITUTE） | 0.03059 | **0.07101** | 72.17% | **80.25%** | **+8.08 pp** |
| all（包含 DELETE） | 0.02175 | **0.04046** | 65.40% | 65.96% | +0.56 pp |
| ADD | 0.02152 | **0.08896** | 70.52% | **86.23%** | **+15.71 pp** |
| SUBSTITUTE | 0.03582 | **0.05190** | 72.21% | 73.92% | +1.71 pp |
| DELETE | **0.01194** | 0.00674 | **58.16%** | 51.65% | **-6.51 pp** |

按相同 group 配对 bootstrap 的 95% 置信区间：

- strict Delta 提升：`+0.04042`，95% CI `[+0.03682, +0.04413]`。
- strict 胜率提升：`+8.08 pp`，95% CI `[+6.59, +9.58] pp`。
- all Delta 提升：`+0.01870`，95% CI `[+0.01608, +0.02133]`。
- all 胜率变化：`+0.56 pp`，95% CI `[-0.84, +1.92] pp`，不能认为显著提升。

结论：

- Full Hard-0.5 在 strict negative 上的分离提升是稳定的，而且大幅强于 Full baseline。
- 提升主要来自 ADD；SUBSTITUTE 的 Delta 明显提升，但胜率只小幅提升。
- DELETE 的 Delta 和胜率反而下降，因此将 DELETE 混入 all 后，整体胜率提升几乎被抵消。
- 后续报告 negative-text 方向分离时应以 strict 为主，同时单独列出 DELETE。
- Negative-text Delta 与真实 negative-image retrieval 是两个不同的目标；这些数字不代替
  PinPoint 官方 `mAP@10`/`NegRecall@10`。

可复现脚本与原始结果：

```text
../diag_model/compute_pilot_delta_from_official_cache.py
../cir_qdiag_openai/outputs/pinpoint_pilot_500_v7/
  full_delta_from_official_cache/comparison.csv
  full_delta_from_official_cache/negative_summary.csv
  full_delta_from_official_cache/<model>/negative_pair_metrics.csv
```

## 5. Last-4 实验无效说明

曾运行以下实验：

- `CLIP-L-last4-baseline`
- `CLIP-L-last4-hard-w0.5`

事后检查发现，原实现使用 reentrant gradient checkpointing，同时 checkpoint block 的输入
来自 frozen 前层，导致声明解冻的 Transformer block 实际没有收到梯度。真正更新的主要是
projection、norm、fusion head 和 temperature。

因此：

- 这两组结果不能作为“最后四层解冻”的证据。
- 只能将其理解为 projection/norm tuning。
- 后续 Full/LoRA 已改用 non-reentrant checkpointing，并加入首批梯度审计。

## 6. Full 与 LoRA 正式训练

### 训练参数规模

| 模式 | 总可训练参数 | 可训练 backbone 参数 |
|---|---:|---:|
| Full | 430.76M | 427.62M |
| LoRA r16 | 5.31M | 2.16M |

LoRA 设置：

- Vision/text attention 的 `q_proj`、`v_proj`。
- Rank 16。
- Alpha 32。
- Dropout 0.05。

Full/LoRA 梯度审计均通过，没有缺失梯度或非有限梯度。

### CIRR validation

| 模型 | R@1 | R@5 | R@10 | R@50 | Group R@1 | Group R@2 | Group R@3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full baseline | **40.28** | **75.13** | 85.94 | **97.63** | 64.48 | 85.10 | 93.37 |
| Full Hard-0.5 | 39.92 | 73.59 | 84.48 | 96.63 | **65.56** | **85.51** | **93.97** |
| LoRA baseline | 39.87 | 75.03 | **86.37** | 97.39 | 64.46 | 84.76 | 93.61 |
| LoRA Hard-0.5 | 39.23 | 73.50 | 84.41 | 96.51 | 64.79 | 84.84 | 93.64 |

结论：

- 解冻 encoder 是整个实验中最大的提升来源。
- Full/LoRA 将 CIRR R@1 从 Frozen 的约 19% 提升到约 40%。
- LoRA 只训练约 531 万参数，但普通 CIRR Recall 几乎追平 Full。
- Hard-0.5 会降低普通全图库 Recall，但 Full 上的 Group Recall 得到改善。

## 7. PinPoint 官方全量结果

| 模型 | mAP@10 ↑ | NegRecall@10 ↓ |
|---|---:|---:|
| Frozen baseline | 0.0521 | 0.2012 |
| Frozen Hard-0.5 | 0.0521 | 0.2027 |
| Full baseline | **0.0677** | 0.1719 |
| Full Hard-0.5 | 0.0673 | 0.1636 |
| LoRA baseline | 0.0589 | 0.1569 |
| LoRA Hard-0.5 | 0.0583 | **0.1514** |

结果目录：

```text
../cir_qdiag_openai/outputs/pinpoint_official_full/
../cir_qdiag_openai/outputs/pinpoint_official_full_lora/
```

主要结论：

- Frozen Hard-0.5 没有改善 mAP，NegRecall 反而变差。
- Full baseline 相比 Frozen baseline：
  - mAP 相对提升约 29.7%；
  - NegRecall 相对降低约 14.5%。
- LoRA baseline 相比 Frozen baseline：
  - mAP 相对提升约 12.9%；
  - NegRecall 相对降低约 22.0%。
- Full Hard-0.5 相比 Full baseline：
  - mAP 下降 0.00033，配对置信区间包含零；
  - NegRecall 降低 0.00832，相对改善约 4.84%。
- LoRA Hard-0.5 相比 LoRA baseline：
  - mAP 小幅下降；
  - NegRecall 相对改善约 3.50%。

当前单模型选择：

- 最高 mAP：Full baseline。
- 最低 NegRecall：LoRA Hard-0.5。
- mAP 与 NegRecall 最均衡：Full Hard-0.5。

## 8. PinPoint 单图与双图

| Full 模型 | 单图 mAP@10 | 双图 mAP@10 |
|---|---:|---:|
| baseline | 0.07313 | 0.03156 |
| Hard-0.5 | 0.07317 | 0.02881 |

双图 query 是当前模型的明显弱项。CIRR 训练主要使用单 reference；PinPoint 双图推理只是
将两个 reference embedding 平均并归一化，没有对应的双图训练监督。

## 9. Full baseline + Full Hard-0.5 reranking

两阶段流程：

```text
Full baseline 从 109k 图库召回候选
    -> Full Hard-0.5 对候选重新计算相似度并排序
    -> PinPoint 官方 evaluator
```

| Pipeline | mAP@10 ↑ | NegRecall@10 ↓ |
|---|---:|---:|
| Full baseline | 0.06766 | 0.17195 |
| Full Hard-0.5 独立全库检索 | 0.06733 | **0.16363** |
| Baseline Top-10 -> Hard rerank | **0.06842** | 0.17195 |
| Baseline Top-20 -> Hard rerank | 0.06768 | 0.16544 |
| Baseline Top-50 -> Hard rerank | 0.06751 | 0.16419 |

结论：

- 只追求最高 mAP：Top-10 rerank，`mAP@10=0.0684`。
- 兼顾 mAP 与 negatives：Top-20 rerank。
- 更重视 negative rejection：Hard-0.5 独立全库检索。
- Top-10 rerank 的 mAP 相对提升约 1.13%，但配对置信区间包含零，属于小幅正向结果。
- Top-10 rerank 不改变候选集合，因此 `NegRecall@10` 严格不变。

结果目录：

```text
../cir_qdiag_openai/outputs/pinpoint_official_full_lora/
  rerank_full_baseline_by_full_hard_w0.5/
```

## 10. Full hard-negative weight 网格

目前完成的正式 Full 权重为 `0、0.1、0.25、0.5、0.75`。

| Weight | R@1 | R@5 | R@10 | R@50 | Group R@1 | Group R@2 | Group R@3 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 40.28 | 75.13 | 85.94 | **97.63** | 64.48 | 85.10 | 93.37 |
| 0.1 | **41.38** | **75.70** | **86.10** | 97.32 | **66.23** | 85.58 | 93.97 |
| 0.25 | 40.83 | 75.10 | 85.58 | 97.01 | 65.94 | **85.82** | **94.12** |
| 0.5 | 39.92 | 73.59 | 84.48 | 96.63 | 65.56 | 85.51 | 93.97 |
| 0.75 | 38.72 | 72.69 | 83.57 | 96.24 | 64.89 | 84.98 | 93.66 |

汇总文件：

```text
outputs/clip_l/full/hard_weight_grid_cirr_partial.csv
```

当前结论：

- CIRR 综合最佳权重：0.1。
- Group R@2/R@3 最佳权重：0.25。
- 0.5 开始明显牺牲普通全图库 Recall。
- 0.75 已经过强，普通 Recall 和 Group Recall 都比 0.1/0.25 差。
- 0.1、0.25、0.75 尚未进行官方 PinPoint 全量评估，因此最终最佳权重尚未确定。

## 11. Epoch 数量与训练曲线

原先保存的 epoch 2/4/6/8 中间 checkpoint 已于 2026-08-07 按磁盘清理要求删除，
每个有效 Full/LoRA 实验只保留 epoch-10 最终推理 checkpoint。因此现在无法再做
epoch 2/4/6/8/10 的 validation 曲线；若要回答 10 epochs 是否过多，需要重新训练并
保留中间 checkpoint。

后续训练会把每 20 step 的 loss、accuracy、temperature、head LR 和 backbone LR 保存到
`train_metrics.jsonl`。旧实验无法恢复终端中的历史 loss，但仍可通过 checkpoint 比较
validation 曲线。

## 12. 当前未完成项

### BGE-VL-large Full Hard-0.1

已新增 BGE-VL-large Full finetune 配置，使用与 CLIP-L Full 一致的 CIRR manifest、
batch 64、10 epochs、BF16、head LR `1e-4`、backbone LR `1e-6`，只使用 ADD/SUBSTITUTE
negative，`hard_negative_weight=0.1`。训练前已验证 residual 为零时与原版 BGE native
`normalize(raw_image + raw_text)` composition 完全一致。

训练与 CIRR validation：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
CUDA_VISIBLE_DEVICES=0 bash scripts/run_bge_vl_large_hard_w0_1.sh
```

训练完成后，对比零训练原版 BGE-VL-large 与 Full Hard-0.1 的 PinPoint 官方
`mAP@10`/`NegRecall@10`：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/diag_model
CUDA_VISIBLE_DEVICES=0 bash run_bge_vl_large_pinpoint_official.sh
```

当前执行环境无可见 CUDA GPU，因此尚未实际启动训练和全量 PinPoint 编码。该对比
同时混合 Full CIRR finetune 与 hard loss 效果；若要单独测量 weight 0.1，后续需补
BGE-VL-large Full weight-0 baseline。

### Full Hard-1.0

- 已开始训练。
- 当前只有第 1 epoch 的 `train_metrics.jsonl`。
- 尚无 epoch-2 checkpoint 和最终 CIRR validation。

### Full 0.1/0.25/0.75 PinPoint

尚未运行。训练完成后执行：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/diag_model
CUDA_VISIBLE_DEVICES=0 python run_clip_pinpoint_official.py \
  --models CLIP-L-full-hard-w0.1,CLIP-L-full-hard-w0.25,CLIP-L-full-hard-w0.75 \
  --output-root ../cir_qdiag_openai/outputs/pinpoint_official_full_hard_grid \
  --batch-size 64
```

每个 Full 模型的 image encoder 不同，必须分别编码完整 candidate corpus。

## 13. 总体结论

1. 最大的性能提升来自解冻 CLIP encoder，而不是单独增加 hard loss。
2. LoRA 参数效率很高，CIRR 普通 Recall 几乎追平 Full。
3. Negative-text hard loss 能增强 `q+`/`q-` 方向分离；在 Full 模型的 strict pilot 子集上，
   Hard-0.5 将 Delta 从 0.03059 提高到 0.07101，胜率从 72.17% 提高到 80.25%。
4. Encoder 可更新后，Hard-0.5 能稳定降低 PinPoint `NegRecall@10`，但会牺牲一部分 CIRR
   全图库 Recall；negative-text Delta 与官方 negative-image retrieval 仍应作为两类指标分开报告。
5. Hard weight 不是越大越好；CIRR 当前最合理区间是 0.1--0.25。
6. Full Hard-0.5 是目前 PinPoint 验证最充分的平衡模型，但很可能不是最终最佳权重。
7. Full Hard-0.1 是当前最值得关注的候选：它同时提升 CIRR 普通 Recall 和 Group Recall。
   如果其 PinPoint NegRecall 也能下降，它可能取代 Full Hard-0.5。
8. 当前最高观测 PinPoint `mAP@10` 来自 Full baseline Top-10 + Full Hard-0.5 reranking：
   `0.0684`。

## 14. 推荐的下一步顺序

1. 等待 Full Hard-1.0 完成，补齐 CIRR weight grid。
2. 评估 0.1、0.25、0.75 的官方 PinPoint mAP@10/NegRecall@10。
3. 若需要重新研究 epoch 数，单独重训最有希望的配置并保留 2/4/6/8/10 checkpoint。
4. 根据 CIRR 与 PinPoint 的 Pareto frontier 选出最终 standalone model。
5. 如果最终部署允许两阶段检索，比较 standalone model 与 Top-20 reranking pipeline。
6. 后续如要进一步降低 NegRecall，应将真实 negative candidate image 加入训练 loss，而不只
   使用 negative text query。
