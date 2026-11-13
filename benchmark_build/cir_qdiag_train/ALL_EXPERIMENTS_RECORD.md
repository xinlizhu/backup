# CIR-QDiag 全部实验记录

更新日期：2026-08-10

本文按磁盘上的真实产物记录已经实际运行过的训练、评估和数据构建，包括实验方案、
实际生效配置、数据、checkpoint 状态和最终数值。不提供结果分析、模型选择建议或后续
实验建议。只有脚本而没有结果文件的方案不算已完成实验；中断或仅完成部分阶段的运行会
明确标为不完整。

## 1. 统一定义

### 1.1 数据集

| 数据 | 用途 | 规模 |
|---|---|---:|
| CIRR train manifest | 标准 CLIP/BGE 训练 | 28,225 samples |
| CIRR train 中可用 ADD/SUBSTITUTE negative 的样本 | Hard loss | 25,225 samples |
| CIRR train ADD negative | Hard loss | 32,012 texts |
| CIRR train SUBSTITUTE negative | Hard loss | 53,284 texts |
| CIRR train DELETE negative | 早期 DELETE 对照 | 37,815 texts |
| CIRR validation | 标准检索评估 | 4,181 queries；2,297 gallery images |
| PinPoint official full | 官方全量评估 | 7,635 queries；本地 109,598 images |
| PinPoint V7 pilot | negative-text 诊断 | 500 groups；1,933 positive queries |
| PinPoint V7 strict negatives | ADD + SUBSTITUTE 诊断 | 3,242 negative texts；1,580 eligible queries |
| PinPoint V7 all edits | ADD + SUBSTITUTE + DELETE | 5,576 edited texts |
| CIRR train pilot 500 V8 | Hard-only continuation | 500 samples；2,446 usable edited texts |
| CIRR train all V5 no-VA | O/A/P residual adapter | 26,086 samples；205,375 ADD/SUBSTITUTE texts |

### 1.2 标准 CLIP composed query

所有 `cir_qdiag_train` 中的 CLIP-B/CLIP-L 实验使用：

```text
r = reference image embedding
t = modification text embedding
residual = MLP([r, t, r * t])
q = normalize(r + t + residual)
```

Fusion MLP：`2304 -> 1024 -> 768`（CLIP-L），GELU，dropout 0.1，末层零初始化。

### 1.3 标准训练损失

Baseline：

```text
L_batch = 0.5 * [CE(q -> target) + CE(target -> q)]
```

Hard：

```text
L_hard = CE([sim(q+, target), sim(q-_1, target), ..., sim(q-_m, target)], label=0)
L_total = L_batch + hard_negative_weight * L_hard
```

`max_non_delete_negatives_per_sample=6`；仅在包含 DELETE 时额外使用
`max_delete_negatives_per_sample=3`。样本不要求具有相同数量的 negative。

### 1.4 指标

| 指标 | 数据集 | 口径 |
|---|---|---|
| Recall@K | CIRR val | 在完整 gallery 中检索 target |
| Group Recall@K | CIRR val | 在 query 自带的相似候选组中检索 target |
| mAP@10 | PinPoint official | 官方正例检索指标 |
| NegRecall@10 | PinPoint official | 官方 negative 进入 Top-10 的比例；越低越好 |
| Delta | PinPoint V7 | `sim(q+, positive image) - sim(q-, positive image)` |
| Raw win | PinPoint V7 | `Delta > 0` 的比例 |
| nDCG@K | PinPoint | 二值 positive relevance 的排序指标 |
| Top-K Jaccard | 两模型排名 | 两个 Top-K candidate set 的交并比 |

## 2. 训练实验总表

### 2.1 BGE-VL-base：早期 ADD/SUBSTITUTE 与 DELETE 对照

共同配置：BGE-VL-base，冻结 backbone，只训练 fusion head 与 temperature，2 epochs，
batch 512，BF16，head LR `1e-4`，hard weight 1.0，seed 42。

| 实验 | Negative mode | 训练样本 | 有 usable negative 的样本 | 最终 checkpoint | 最终评估 |
|---|---|---:|---:|---|---|
| without-delete | ADD + SUBSTITUTE | 28,225 | 25,225 | `outputs/without_delete/last.pt` | 未生成 CIRR/PinPoint 结果 |
| with-delete | ADD + SUBSTITUTE + DELETE | 28,225 | 25,695 | `outputs/with_delete/last.pt` | 未生成 CIRR/PinPoint 结果 |

配置：`configs/without_delete.yaml`、`configs/with_delete.yaml`。

### 2.2 Frozen CLIP-B/CLIP-L：最早六组对照

共同配置：冻结 image/text encoder，只训练 fusion head 与 temperature；10 epochs；
batch 1024；BF16；head LR `1e-4`；seed 42；每 epoch 保存一次。

| Backbone | 实验 | Negative mode | Hard weight | 最终 checkpoint |
|---|---|---|---:|---|
| CLIP-B/16 | baseline | none | 0.0 | `outputs/clip_b/baseline/last.pt` |
| CLIP-B/16 | ADD/SUBSTITUTE | add_substitute | 1.0 | `outputs/clip_b/add_substitute/last.pt` |
| CLIP-B/16 | ADD/SUBSTITUTE/DELETE | add_substitute_delete | 1.0 | `outputs/clip_b/add_substitute_delete/last.pt` |
| CLIP-L/14 | baseline | none | 0.0 | `outputs/clip_l/baseline/last.pt` |
| CLIP-L/14 | ADD/SUBSTITUTE | add_substitute | 1.0 | `outputs/clip_l/add_substitute/last.pt` |
| CLIP-L/14 | ADD/SUBSTITUTE/DELETE | add_substitute_delete | 1.0 | `outputs/clip_l/add_substitute_delete/last.pt` |

CIRR validation：

| Backbone | 实验 | R@1 | R@5 | R@10 | R@50 | Group R@1 | Group R@2 | Group R@3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP-B | baseline | 14.1115 | 40.6601 | 54.7238 | 84.5731 | 35.8527 | 60.6075 | 77.5413 |
| CLIP-B | ADD/SUBSTITUTE | 13.7766 | 39.8230 | 54.2215 | 83.8077 | 35.3026 | 60.0574 | 77.0868 |
| CLIP-B | ADD/SUBSTITUTE/DELETE | 13.8005 | 40.1818 | 54.1497 | 83.9034 | 35.3265 | 60.2009 | 77.3738 |
| CLIP-L | baseline | 14.2550 | 42.4540 | 57.4265 | 86.2234 | 37.6704 | 62.0187 | 79.7417 |
| CLIP-L | ADD/SUBSTITUTE | 13.8005 | 41.2342 | 56.4219 | 85.7450 | 36.8094 | 60.8467 | 79.1437 |
| CLIP-L | ADD/SUBSTITUTE/DELETE | 13.5853 | 41.7125 | 56.3502 | 85.9603 | 36.7615 | 61.2055 | 79.1437 |

### 2.3 Frozen CLIP-L：batch 64 与 hard-weight 对照

共同配置：CLIP-L/14 backbone frozen，10 epochs，batch 64，BF16，head LR `1e-4`，
ADD/SUBSTITUTE，最多 6 negatives/sample，seed 42。

| 实验 | Negative mode | Hard weight | 最终 checkpoint |
|---|---|---:|---|
| Frozen baseline B64 | none | 0.0 | `outputs/clip_l/batch64/baseline/last.pt` |
| Frozen Hard-0.5 B64 | add_substitute | 0.5 | `outputs/clip_l/batch64/hard_w0_5/last.pt` |
| Frozen Hard-1.0 B64 | add_substitute | 1.0 | `outputs/clip_l/batch64/hard_w1_0/last.pt` |

CIRR validation：

| Hard weight | R@1 | R@5 | R@10 | R@50 | Group R@1 | Group R@2 | Group R@3 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 19.3973 | 50.7295 | 66.1086 | 91.1026 | 44.2717 | 68.3090 | 83.7838 |
| 0.5 | 19.1342 | 50.5860 | 65.7020 | 91.4375 | 44.3913 | 68.1655 | 83.8555 |
| 1.0 | 18.4166 | 49.6293 | 65.0801 | 90.9830 | 43.4107 | 67.9742 | 83.3533 |

### 2.4 CLIP-L Last-4 实验（已运行但实现无效）

计划配置：解冻最后 4 个 Transformer blocks；10 epochs；batch 64；BF16；baseline 与
Hard-0.5。

实际状态：训练使用了 reentrant gradient checkpointing，frozen 前层输出未携带梯度，
声明解冻的最后 4 blocks 实际未得到梯度。以下结果只记录运行事实，不作为 Last-4
finetuning 结果。

| 实验 | Hard weight | R@1 | R@5 | R@10 | R@50 | Group R@1 | Group R@2 | Group R@3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Last4 baseline | 0.0 | 19.8278 | 51.6862 | 66.1564 | 91.3418 | 44.4630 | 68.8113 | 84.1425 |
| Last4 Hard-0.5 | 0.5 | 19.4690 | 51.2078 | 65.9172 | 91.3657 | 44.5348 | 68.5721 | 83.9990 |

### 2.5 CLIP-L Full finetuning hard-weight 网格

共同配置：CLIP-L/14，全量更新 image encoder、text encoder、projections、fusion head 和
temperature；10 epochs；batch 64；gradient accumulation 1；BF16；head LR `1e-4`；
backbone LR `1e-6`；weight decay 0.01；5% warmup + cosine；non-reentrant gradient
checkpointing；seed 42；每 2 epochs 保存一次。Hard 使用 ADD/SUBSTITUTE。

可训练参数：430,764,033；其中可训练 backbone 参数 427,616,512。

| 实验 | Negative mode | Hard weight | 训练状态 | 最终推理 checkpoint |
|---|---|---:|---|---|
| Full baseline | none | 0.0 | 完成 10 epochs | `outputs/clip_l/full/baseline/checkpoint_epoch_010.pt` |
| Full Hard-0.1 | add_substitute | 0.1 | 完成 10 epochs | `outputs/clip_l/full/hard_w0_1/checkpoint_epoch_010.pt` |
| Full Hard-0.25 | add_substitute | 0.25 | 完成 10 epochs | `outputs/clip_l/full/hard_w0_25/checkpoint_epoch_010.pt` |
| Full Hard-0.5 | add_substitute | 0.5 | 完成 10 epochs | `outputs/clip_l/full/hard_w0_5/checkpoint_epoch_010.pt` |
| Full Hard-0.75 | add_substitute | 0.75 | 完成 10 epochs | `outputs/clip_l/full/hard_w0_75/checkpoint_epoch_010.pt` |
| Full Hard-1.0 | add_substitute | 1.0 | 未完成；只有部分 `train_metrics.jsonl` | 无最终 checkpoint |

CIRR validation：

| Hard weight | R@1 | R@5 | R@10 | R@50 | Group R@1 | Group R@2 | Group R@3 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 40.2774 | 75.1256 | 85.9364 | 97.6321 | 64.4822 | 85.0993 | 93.3748 |
| 0.1 | 41.3777 | 75.6996 | 86.1038 | 97.3212 | 66.2282 | 85.5776 | 93.9727 |
| 0.25 | 40.8276 | 75.1017 | 85.5776 | 97.0103 | 65.9412 | 85.8168 | 94.1162 |
| 0.5 | 39.9187 | 73.5948 | 84.4774 | 96.6276 | 65.5585 | 85.5059 | 93.9727 |
| 0.75 | 38.7228 | 72.6860 | 83.5685 | 96.2449 | 64.8888 | 84.9797 | 93.6618 |
| 1.0 | — | — | — | — | — | — | — |

### 2.6 CLIP-L LoRA r16

共同配置：冻结原始 backbone weights；在 vision/text 每层 attention 的 `q_proj`、
`v_proj` 加 LoRA；rank 16；alpha 32；dropout 0.05；同时训练 fusion head 与 temperature；
10 epochs；batch 64；BF16；head/LoRA LR `1e-4`；每 2 epochs 保存一次；seed 42。

总可训练参数 5,310,209；可训练 backbone/LoRA 参数 2,162,688。

| 实验 | Hard weight | Negative mode | 最终 checkpoint |
|---|---:|---|---|
| LoRA baseline | 0.0 | none | `outputs/clip_l/lora_r16/baseline/checkpoint_epoch_010.pt` |
| LoRA Hard-0.5 | 0.5 | add_substitute | `outputs/clip_l/lora_r16/hard_w0_5/checkpoint_epoch_010.pt` |

CIRR validation：

| 实验 | R@1 | R@5 | R@10 | R@50 | Group R@1 | Group R@2 | Group R@3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoRA baseline | 39.8708 | 75.0299 | 86.3669 | 97.3930 | 64.4583 | 84.7644 | 93.6140 |
| LoRA Hard-0.5 | 39.2251 | 73.4992 | 84.4056 | 96.5080 | 64.7931 | 84.8362 | 93.6379 |

### 2.7 BGE-VL-large Full Hard-0.1

配置：BGE-VL-large；Full finetuning；CIRR 28,225 samples；ADD/SUBSTITUTE；hard weight
0.1；10 epochs；batch 64；BF16；head LR `1e-4`；backbone LR `1e-6`；weight decay
0.01；5% warmup + cosine；每 2 epochs 保存；seed 42。

| 项目 | 数值/路径 |
|---|---|
| 完成 epoch | 10 |
| 最终推理 checkpoint | `outputs/bge_vl_large/full/hard_w0_1/checkpoint_epoch_010.pt` |
| CIRR R@1 / R@5 / R@10 / R@50 | 50.0598 / 81.0093 / 89.0696 / 97.5126 |
| CIRR Group R@1 / R@2 / R@3 | 79.6460 / 92.5855 / 97.0103 |

本实验没有训练 matched BGE-VL-large Full weight-0 baseline；对照模型是零训练的原版
BGE-VL-large。

### 2.8 Full baseline 后续 Hard-only 500 V8

初始化：`CLIP-L Full baseline epoch 10`。训练只使用 CIRR V8 的 500 个样本，只优化
`L_hard`，不再加入普通 batch InfoNCE。

配置：Full finetuning；ADD/SUBSTITUTE/DELETE；2,446 usable edited texts（ADD 719、
SUBSTITUTE 976、DELETE 751）；hard weight 1.0；10 epochs；batch 64；80 optimizer
steps；BF16；head LR `1e-4`；backbone LR `1e-6`；每 2 epochs 保存；seed 42。

评估 checkpoints：epoch 2、6、10；对照为原始 CLIP-L 与 Full baseline。

CIRR validation：

| 模型 | R@1 | R@5 | R@10 | R@50 | Group R@1 | Group R@2 | Group R@3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP-L original | 12.7960 | 37.7661 | 51.1600 | 78.4023 | 34.5133 | 58.8854 | 76.6324 |
| Full baseline | 40.2774 | 75.1256 | 85.9364 | 97.6321 | 64.4822 | 85.0993 | 93.3748 |
| Hard-only E2 | 40.4688 | 75.6518 | 85.8407 | 97.5126 | 65.6063 | 85.9842 | 94.0206 |
| Hard-only E6 | 39.4882 | 74.5037 | 84.7405 | 96.1492 | 65.8933 | 86.4626 | 94.0445 |
| Hard-only E10 | 38.9381 | 73.5948 | 84.0947 | 95.7187 | 66.2282 | 86.3191 | 94.0206 |

PinPoint official full：

| 模型 | mAP@10 | NegRecall@10 | Queries |
|---|---:|---:|---:|
| CLIP-L original | 0.0462 | 0.2056 | 7,635 |
| Full baseline（本次评估 cache） | 0.0677 | 0.1722 | 7,635 |
| Hard-only E2 | 0.0643 | 0.1661 | 7,635 |
| Hard-only E6 | 0.0594 | 0.1537 | 7,635 |
| Hard-only E10 | 0.0584 | 0.1509 | 7,635 |

### 2.9 Target-Conditioned Harmful Edit Routing 六通道实验

项目：`benchmark_build/diag_model_training_new_method`。

配置：BGE-VL-base；训练数据 CIRR train pilot 500 V8；500 samples；493 samples 有 harmful
edit；1,695 harmful edits；embedding 为 `256 shared + 6 × 64 edit = 640`；fusion hidden
1024；dropout 0.1；前 2 epochs 冻结 backbone，从 epoch 3 全量训练；10 epochs；batch
16；gradient accumulation 2；BF16；head LR `1e-4`；fusion LR `5e-5`；backbone LR
`1e-5`；weight decay 0.05；ordering temperature 0.07；margin 0.1；seed 42。

| 项目 | 结果 |
|---|---|
| 最终 checkpoint | `diag_model_training_new_method/outputs/six_channel/last.pt` |
| CIRR R@1 | 4.5204 |
| CIRR R@5 | 16.4554 |
| CIRR R@10 | 27.4097 |
| CIRR R@50 | 61.5164 |
| Routing/repair metrics | 未生成 `routing_metrics.json` |
| PinPoint official | 未运行 |

### 2.10 O/A/P Semantic Residual Adapter

两个实验都冻结既有 global CIR checkpoint，只训练 Object、Attribute、Predicate 三组
query/target adapter。实际推理分数为：

```text
S_final = S_global + beta / 3 * (S_object + S_attribute + S_predicate)
```

global embedding 为 768 维，每个 O/A/P embedding 为 64 维，packed embedding 为 960 维；
packed query/target 内积与 `S_final` 完全相同。训练数据为
`data/cirr_train_all_v5_no_va.jsonl`，26,086 samples；每个样本、每个 semantic、每个 epoch
最多抽取 2 条 ADD/SUBSTITUTE negative；DELETE 不参与训练。

共同配置：5 epochs；batch 64；BF16；adapter LR `1e-4`；weight decay 0.01；5% warmup；
ordering temperature 0.07；retrieval margin 0；patch margin 0.01；seed 42。每组有 640,896
个可训练参数；gradient audit 均为 36/36 个 trainable tensors 有有限梯度，冻结 base 没有梯度。

| 模型 | 冻结 global checkpoint | 训练 beta | 保存的推理 checkpoint |
|---|---|---:|---|
| CLIP-L OAP | `outputs/clip_l/full/baseline/checkpoint_epoch_010.pt` | 0.1 | epoch 2、4、5 |
| BGE-VL-large OAP | `outputs/bge_vl_large/full/hard_w0_1/checkpoint_epoch_010.pt` | 0.1 | epoch 2、4、5 |

CIRR 全库 `S_final` sweep；Delta 均相对同一次 OAP evaluator 中的冻结 global score：

| 模型 | Checkpoint | 推理 beta | R@1 | R@5 | R@10 | Group R@1 | ΔR@1 | ΔGroup R@1 | corrected | regressed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BGE OAP | E2 | 0.1 | 49.4379 | 80.4114 | 89.0935 | 74.1689 | -0.2392 | +0.1435 | 16 | 26 |
| BGE OAP | E4 | 0.1 | 49.6054 | 80.3396 | 89.1174 | 74.1210 | -0.0717 | +0.0956 | 20 | 23 |
| BGE OAP | E5 | 0.025 | 49.7489 | 80.4353 | 89.2609 | 74.0732 | +0.0718 | +0.0478 | 5 | 2 |
| BGE OAP | E5 | 0.05 | 49.5814 | 80.3635 | 89.2131 | 74.0732 | -0.0957 | +0.0478 | 10 | 14 |
| BGE OAP | E5 | 0.1 | 49.5097 | 80.4114 | 89.1414 | 74.0493 | -0.1674 | +0.0239 | 15 | 22 |
| BGE OAP | E5 | 0.2 | 49.4619 | 80.2200 | 88.9261 | 74.1689 | -0.2152 | +0.1435 | 38 | 47 |
| CLIP-L OAP | E2 | 0.1 | 40.3970 | 75.3169 | 86.0320 | 64.7453 | +0.1196 | +0.2631 | 29 | 24 |
| CLIP-L OAP | E4 | 0.1 | 40.3492 | 75.1495 | 86.0320 | 64.7453 | +0.0718 | +0.2631 | 29 | 26 |
| CLIP-L OAP | E5 | 0.025 | 40.3014 | 75.1973 | 85.9125 | 64.5779 | +0.0240 | +0.0957 | 6 | 5 |
| CLIP-L OAP | E5 | 0.05 | 40.2296 | 75.2212 | 86.0081 | 64.6018 | -0.0478 | +0.1196 | 12 | 14 |
| CLIP-L OAP | E5 | 0.1 | 40.4449 | 75.1973 | 86.0081 | 64.7692 | +0.1675 | +0.2870 | 31 | 24 |
| CLIP-L OAP | E5 | 0.2 | 40.2057 | 75.0538 | 85.6733 | 64.8649 | -0.0717 | +0.3827 | 51 | 54 |

另运行了 global 全库召回后、只在原 global Top-K 内按 OAP 分数重排的 CIRR reranker。
冻结 global baseline 分别为：CLIP-L `R@1=40.2774`、BGE `R@1=49.6771`。

| 模型 | beta | Top-K | R@1 | R@5 | R@10 | Group R@1 | ΔR@1 | ΔR@5 | ΔR@10 | ΔGroup R@1 | corrected | regressed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BGE E5 | 0.025 | 10 | 49.8685 | 80.5071 | 89.2849 | 74.1449 | +0.1914 | +0.0479 | 0.0000 | +0.0478 | 28 | 20 |
| BGE E5 | 0.025 | 50 | 49.7249 | 80.5071 | 89.2609 | 74.1210 | +0.0478 | +0.0479 | -0.0240 | +0.0239 | 18 | 16 |
| BGE E5 | 0.025 | 100 | 49.7249 | 80.5071 | 89.2609 | 74.1449 | +0.0478 | +0.0479 | -0.0240 | +0.0478 | 18 | 16 |
| BGE E5 | 0.1 | 10 | 49.7728 | 80.4592 | 89.2849 | 74.1210 | +0.0957 | 0.0000 | 0.0000 | +0.0239 | 37 | 33 |
| BGE E5 | 0.1 | 50 | 49.6054 | 80.4592 | 89.2370 | 74.0971 | -0.0717 | 0.0000 | -0.0479 | 0.0000 | 25 | 28 |
| BGE E5 | 0.1 | 100 | 49.6054 | 80.4592 | 89.2370 | 74.1210 | -0.0717 | 0.0000 | -0.0479 | +0.0239 | 25 | 28 |
| BGE E5 | 0.2 | 10 | 49.5814 | 80.3635 | 89.2849 | 74.0971 | -0.0957 | -0.0957 | 0.0000 | 0.0000 | 50 | 54 |
| BGE E5 | 0.2 | 50 | 49.4379 | 80.3875 | 88.9739 | 74.1210 | -0.2392 | -0.0717 | -0.3110 | +0.0239 | 37 | 47 |
| BGE E5 | 0.2 | 100 | 49.4379 | 80.3875 | 88.9739 | 74.1449 | -0.2392 | -0.0717 | -0.3110 | +0.0478 | 37 | 47 |
| CLIP E2 | 0.05 | 10 | 39.9904 | 75.1256 | 85.9364 | 64.5061 | -0.2870 | 0.0000 | 0.0000 | -0.0239 | 25 | 37 |
| CLIP E2 | 0.05 | 50 | 40.1579 | 75.1256 | 86.0320 | 64.6257 | -0.1195 | 0.0000 | +0.0956 | +0.0957 | 22 | 27 |
| CLIP E2 | 0.05 | 100 | 40.1579 | 75.1256 | 86.0320 | 64.6257 | -0.1195 | 0.0000 | +0.0956 | +0.0957 | 22 | 27 |
| CLIP E2 | 0.1 | 10 | 40.0144 | 75.2930 | 85.9364 | 64.5539 | -0.2630 | +0.1674 | 0.0000 | +0.0239 | 35 | 46 |
| CLIP E2 | 0.1 | 50 | 40.2057 | 75.2691 | 86.0081 | 64.7453 | -0.0717 | +0.1435 | +0.0717 | +0.2153 | 31 | 34 |
| CLIP E2 | 0.1 | 100 | 40.2057 | 75.2691 | 86.0081 | 64.7453 | -0.0717 | +0.1435 | +0.0717 | +0.2153 | 31 | 34 |
| CLIP E2 | 0.2 | 10 | 40.1100 | 75.2212 | 85.9364 | 64.8649 | -0.1674 | +0.0956 | 0.0000 | +0.3349 | 61 | 68 |
| CLIP E2 | 0.2 | 50 | 40.2774 | 75.2452 | 85.7690 | 65.0562 | 0.0000 | +0.1196 | -0.1674 | +0.5262 | 55 | 55 |
| CLIP E2 | 0.2 | 100 | 40.2774 | 75.2452 | 85.7690 | 65.0562 | 0.0000 | +0.1196 | -0.1674 | +0.5262 | 55 | 55 |
| CLIP E5 | 0.05 | 10 | 40.0861 | 75.1256 | 85.9364 | 64.6735 | -0.1913 | 0.0000 | 0.0000 | +0.1435 | 28 | 36 |
| CLIP E5 | 0.05 | 50 | 40.1579 | 75.1256 | 86.0560 | 64.6974 | -0.1195 | 0.0000 | +0.1196 | +0.1674 | 21 | 26 |
| CLIP E5 | 0.05 | 100 | 40.1579 | 75.1256 | 86.0560 | 64.6974 | -0.1195 | 0.0000 | +0.1196 | +0.1674 | 21 | 26 |
| CLIP E5 | 0.1 | 10 | 40.0383 | 75.2212 | 85.9364 | 64.6974 | -0.2391 | +0.0956 | 0.0000 | +0.1674 | 36 | 46 |
| CLIP E5 | 0.1 | 50 | 40.1579 | 75.2212 | 86.0799 | 64.7214 | -0.1195 | +0.0956 | +0.1435 | +0.1914 | 28 | 33 |
| CLIP E5 | 0.1 | 100 | 40.1579 | 75.2212 | 86.0799 | 64.7214 | -0.1195 | +0.0956 | +0.1435 | +0.1914 | 28 | 33 |
| CLIP E5 | 0.2 | 10 | 40.0622 | 75.1256 | 85.9364 | 64.8409 | -0.2152 | 0.0000 | 0.0000 | +0.3109 | 59 | 68 |
| CLIP E5 | 0.2 | 50 | 40.0861 | 75.1256 | 85.8168 | 64.9605 | -0.1913 | 0.0000 | -0.1196 | +0.4305 | 47 | 55 |
| CLIP E5 | 0.2 | 100 | 40.0861 | 75.1256 | 85.8168 | 64.9605 | -0.1913 | 0.0000 | -0.1196 | +0.4305 | 47 | 55 |

## 3. PinPoint 官方全量评估

### 3.1 Frozen、Full 与 LoRA CLIP-L

| 模型 | 训练方案 | mAP@10 | NegRecall@10 | Queries |
|---|---|---:|---:|---:|
| Frozen baseline B64 | encoder frozen，weight 0 | 0.0521 | 0.2012 | 7,635 |
| Frozen Hard-0.5 B64 | encoder frozen，weight 0.5 | 0.0521 | 0.2027 | 7,635 |
| Full baseline | full，weight 0 | 0.067656 | 0.171946 | 7,635 |
| Full Hard-0.5 | full，weight 0.5 | 0.067326 | 0.163631 | 7,635 |
| LoRA baseline | LoRA r16，weight 0 | 0.0589 | 0.1569 | 7,635 |
| LoRA Hard-0.5 | LoRA r16，weight 0.5 | 0.0583 | 0.1514 | 7,635 |

Full Hard-0.1、0.25、0.75 未运行 PinPoint official full。

### 3.2 BGE-VL-large 原版与 Full Hard-0.1

| 模型 | 训练方案 | mAP@10 | NegRecall@10 | Queries |
|---|---|---:|---:|---:|
| BGE-VL-large original | 无训练 | 0.1103 | 0.1017 | 7,635 |
| BGE-VL-large Full Hard-0.1 | Full；ADD/SUBSTITUTE；weight 0.1 | 0.0917 | 0.1355 | 7,635 |

### 3.3 Full CLIP-L 单 reference / 双 reference 分层

| 模型 | 单图 mAP@10 | 双图 mAP@10 |
|---|---:|---:|
| Full baseline | 0.07313 | 0.03156 |
| Full Hard-0.5 | 0.07317 | 0.02881 |

### 3.4 O/A/P packed embedding

评估使用 7,635 条 official queries 和 109,598 张本地可用图片；index 中另外 2 张图片
本地缺失。OAP query/target 使用 960 维 packed embedding 做精确 inner-product 全库检索。

| 模型 | 对照 global | mAP@10 | ΔmAP@10 | NegRecall@10 | ΔNegRecall@10 |
|---|---|---:|---:|---:|---:|
| CLIP-L OAP E2 beta 0.1 | CLIP-L Full baseline | 0.0680 | +0.000320 | 0.1714 | -0.000550 |
| CLIP-L OAP E5 beta 0.1 | CLIP-L Full baseline | 0.0680 | +0.000340 | 0.1713 | -0.000681 |
| BGE OAP E5 beta 0.025 | BGE Full Hard-0.1 | 0.0917 | +0.000055 | 0.1353 | -0.000170 |

官方 CSV 以 4 位小数输出 mAP/NegRecall；表中 Delta 由每条 query 的未舍入指标现场重算。
NegRecall Delta 为负表示进入 Top-10 的 official negative 减少。

## 4. PinPoint V7 negative-text 方向实验

### 4.1 Frozen CLIP-L baseline 与 Hard-0.5

配置：PinPoint V7 500 groups；`strict=ADD+SUBSTITUTE`；group-balanced；模型使用各自的
positive/negative query encoder。下表为 strict 子集。

| 模型 | Groups | Negatives | Mean Delta | Raw win | Robust win |
|---|---:|---:|---:|---:|---:|
| Frozen baseline | 471 | 3,242 | 0.010046 | 0.681432 | 0.022266 |
| Frozen Hard-0.5 | 471 | 3,242 | 0.022231 | 0.788615 | 0.023440 |

### 4.2 Last-4 baseline 与 Hard-0.5

这两项对应第 2.4 节的无效 Last-4 checkpoint，只记录输出。

| 模型 | Groups | Negatives | Mean Delta | Raw win | Robust win |
|---|---:|---:|---:|---:|---:|
| Last4 baseline | 471 | 3,242 | 0.010852 | 0.686577 | 0.023066 |
| Last4 Hard-0.5 | 471 | 3,242 | 0.023827 | 0.794258 | 0.023577 |

### 4.3 Full baseline 与 Full Hard-0.5

配置：复用两个 Full 模型各自的 PinPoint official corpus/query embedding cache，只重新编码
V7 edited texts；group-balanced。

| Subset | 模型 | Groups | Instructions | Negatives | Mean Delta | Raw win |
|---|---|---:|---:|---:|---:|---:|
| strict | Full baseline | 471 | 1,580 | 3,242 | 0.030589 | 0.721686 |
| strict | Full Hard-0.5 | 471 | 1,580 | 3,242 | 0.071011 | 0.802464 |
| all | Full baseline | 499 | 1,862 | 5,576 | 0.021754 | 0.653964 |
| all | Full Hard-0.5 | 499 | 1,862 | 5,576 | 0.040457 | 0.659581 |
| ADD | Full baseline | 423 | 1,162 | 1,598 | 0.021515 | 0.705229 |
| ADD | Full Hard-0.5 | 423 | 1,162 | 1,598 | 0.088961 | 0.862323 |
| SUBSTITUTE | Full baseline | 432 | 1,214 | 1,644 | 0.035820 | 0.722084 |
| SUBSTITUTE | Full Hard-0.5 | 432 | 1,214 | 1,644 | 0.051897 | 0.739152 |
| DELETE | Full baseline | 495 | 1,632 | 2,334 | 0.011939 | 0.581645 |
| DELETE | Full Hard-0.5 | 495 | 1,632 | 2,334 | 0.006735 | 0.516547 |

### 4.4 O/A/P negative-text Delta / Raw win

配置：复用 OAP official corpus 和 positive-query packed embedding cache，只重新编码 V7 的
5,576 条 edited query。每个 negative 先对同组 positive images 计算
`Delta = sim(q+, target) - sim(q-, target)`，再按 instruction 和 group 做 balanced 汇总。
packed inner product 即实际 OAP `S_final`。Raw win 为 `Delta > 0`；strict 只含 ADD 与
SUBSTITUTE。

| 模型 | Subset | Groups | Negatives | Mean Delta | Raw win | 95% CI |
|---|---|---:|---:|---:|---:|---:|
| CLIP-L Full baseline | all | 499 | 5,576 | 0.021754 | 0.653964 | [0.639655, 0.669111] |
| CLIP-L OAP E2 beta 0.1 | all | 499 | 5,576 | 0.029121 | 0.650299 | [0.634950, 0.666370] |
| CLIP-L OAP E5 beta 0.1 | all | 499 | 5,576 | 0.028877 | 0.648838 | [0.633347, 0.664381] |
| CLIP-L Full baseline | strict | 471 | 3,242 | 0.030589 | 0.721686 | [0.704676, 0.737429] |
| CLIP-L OAP E2 beta 0.1 | strict | 471 | 3,242 | 0.052865 | 0.792880 | [0.774387, 0.807908] |
| CLIP-L OAP E5 beta 0.1 | strict | 471 | 3,242 | 0.052919 | 0.795055 | [0.777621, 0.810325] |
| BGE Full Hard-0.1 | all | 499 | 5,576 | 0.075782 | 0.792520 | [0.771381, 0.812228] |
| BGE OAP E5 beta 0.025 | all | 499 | 5,576 | 0.078771 | 0.796003 | [0.775079, 0.815046] |
| BGE Full Hard-0.1 | strict | 471 | 3,242 | 0.103264 | 0.862528 | [0.844276, 0.879510] |
| BGE OAP E5 beta 0.025 | strict | 471 | 3,242 | 0.109907 | 0.872244 | [0.854375, 0.888488] |

相对各自冻结 global checkpoint：

| OAP 模型 | ΔMean Delta（all） | ΔRaw win（all） | ΔMean Delta（strict） | ΔRaw win（strict） |
|---|---:|---:|---:|---:|
| CLIP-L E2 | +0.007366 | -0.3665 pp | +0.022276 | +7.1193 pp |
| CLIP-L E5 | +0.007123 | -0.5126 pp | +0.022330 | +7.3368 pp |
| BGE E5 | +0.002989 | +0.3483 pp | +0.006644 | +0.9716 pp |

按 edit operation：

| 模型 | Operation | Mean Delta | Raw win | 相对 global Raw win |
|---|---|---:|---:|---:|
| CLIP-L Full baseline | ADD | 0.021515 | 0.705229 | — |
| CLIP-L OAP E5 | ADD | 0.062523 | 0.858249 | +15.3020 pp |
| CLIP-L Full baseline | SUBSTITUTE | 0.035820 | 0.722084 | — |
| CLIP-L OAP E5 | SUBSTITUTE | 0.040732 | 0.729794 | +0.7711 pp |
| CLIP-L Full baseline | DELETE | 0.011939 | 0.581645 | — |
| CLIP-L OAP E5 | DELETE | 0.003034 | 0.495158 | -8.6487 pp |
| BGE Full Hard-0.1 | ADD | 0.104704 | 0.874635 | — |
| BGE OAP E5 | ADD | 0.116830 | 0.894155 | +1.9520 pp |
| BGE Full Hard-0.1 | SUBSTITUTE | 0.100960 | 0.856500 | — |
| BGE OAP E5 | SUBSTITUTE | 0.102297 | 0.856322 | -0.0178 pp |
| BGE Full Hard-0.1 | DELETE | 0.045834 | 0.717837 | — |
| BGE OAP E5 | DELETE | 0.045000 | 0.715266 | -0.2571 pp |

CLIP-L OAP E5 的 semantic × operation 结果：

| Cell | Mean Delta | Raw win | 相对 Full baseline Raw win |
|---|---:|---:|---:|
| object × ADD | 0.072662 | 0.890503 | +18.3869 pp |
| attribute × ADD | 0.044529 | 0.776768 | +8.8514 pp |
| predicate × ADD | 0.031374 | 0.780447 | +14.7553 pp |
| object × SUBSTITUTE | 0.040461 | 0.761377 | +1.6667 pp |
| attribute × SUBSTITUTE | 0.037506 | 0.687272 | -2.5906 pp |
| predicate × SUBSTITUTE | 0.018826 | 0.667806 | +3.9290 pp |
| object × DELETE | 0.014167 | 0.548381 | -8.8002 pp |
| attribute × DELETE | -0.002001 | 0.461437 | -9.8618 pp |
| predicate × DELETE | -0.011180 | 0.397391 | -9.3452 pp |

### 4.5 Frozen / Last-4 的 positive retrieval 与 paraphrase 稳定性

第 4.1、4.2 节对应的诊断运行不只输出 Delta；还在 500 groups、1,933 条 positive
queries 上生成了 positive retrieval 与正向 paraphrase 排名稳定性。这里的 Recall/nDCG
是 V7 pilot 的 pair-balanced 指标，不是 PinPoint official full 的 mAP/NegRecall。

| 模型 | Hit@10 | Recall@10 | nDCG@10 | Paraphrase Top-10 Jaccard |
|---|---:|---:|---:|---:|
| Frozen baseline | 0.348400 | 0.110103 | 0.089903 | 0.554608 |
| Frozen Hard-0.5 | 0.343233 | 0.108171 | 0.088840 | 0.554507 |
| Last-4 baseline（实现无效） | 0.348700 | 0.110353 | 0.090701 | 0.552418 |
| Last-4 Hard-0.5（实现无效） | 0.343033 | 0.108049 | 0.089315 | 0.553959 |

## 5. PinPoint V7 BGE 诊断基线

本实验没有训练模型。BGE-VL-base 与 BGE-VL-MLLM-S1 在同一 V7 500-group 数据上编码
1,933 条正 query、5,576 条 edited query，并在 109,598 张图像上做 Top-50 检索。

Strict 子集结果：

| 模型 | Mean Delta | Raw win | Robust win | Positive nDCG@10 | Edited-query residual nDCG@10 | Delta nDCG@10 | Positive NegRecall@10 | Edited NegRecall@10 | Positive/edited Top10 Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BGE-VL-base | 0.014867 | 0.607720 | 0.033684 | 0.154654 | 0.080711 | 0.073943 | 0.115382 | 0.086138 | 0.348048 |
| BGE-VL-MLLM-S1 | 0.023290 | 0.699734 | 0.082585 | 0.207969 | 0.115548 | 0.092421 | 0.089310 | 0.073457 | 0.410383 |

正向 paraphrase 检索一致性：

| 模型 | Positive query pairs | Top1 Jaccard | Top5 Jaccard | Top10 Jaccard | Top50 Jaccard |
|---|---:|---:|---:|---:|---:|
| BGE-VL-base | 3,341 | 0.249800 | 0.237960 | 0.251381 | 0.293334 |
| BGE-VL-MLLM-S1 | 3,341 | 0.385800 | 0.364135 | 0.377878 | 0.413085 |

## 6. Full baseline 与 Full Hard-0.5 原生 Top-K 对照

配置：两个模型都独立对 109,598 张图库做标准 positive-query 检索；7,635 official queries。

| K | Jaccard | 平均交集数 | 交集/K | Baseline positive recall | Hard positive recall | Baseline NegRecall | Hard NegRecall | Baseline nDCG | Hard nDCG |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.858415 | 0.8584 | 0.8584 | 0.038376 | 0.048330 | 0.466798 | 0.448199 | 0.038376 | 0.048330 |
| 5 | 0.598202 | 3.6073 | 0.7215 | 0.127077 | 0.127404 | 0.230203 | 0.218356 | 0.109216 | 0.110653 |
| 10 | 0.592695 | 7.2605 | 0.7261 | 0.145839 | 0.143090 | 0.171946 | 0.163631 | 0.121422 | 0.120515 |
| 20 | 0.594774 | 14.6466 | 0.7323 | 0.187563 | 0.182928 | 0.131801 | 0.125182 | 0.140645 | 0.138635 |
| 50 | 0.606046 | 37.2155 | 0.7443 | 0.269977 | 0.261955 | 0.130230 | 0.123117 | 0.173739 | 0.170516 |

## 7. 跨模型 rerank 实验

所有有效 rerank 都只使用正常 positive query 与两个模型各自的 image embedding，不使用
negative text，不使用 OOF、线性 probe 或其他机器学习分类器。

### 7.1 Full baseline 召回，Full Hard-0.5 rerank

候选由 Full baseline 原生排名召回；候选内部由 Full Hard-0.5 的 positive-query cosine
score 重排。未进入 rerank depth 的候选保持 baseline 顺序。

| Rerank depth | mAP@10 | NegRecall@10 | Queries |
|---:|---:|---:|---:|
| Native Full baseline | 0.067656 | 0.171946 | 7,635 |
| Top-10 | 0.0684 | 0.1719 | 7,635 |
| Top-20 | 0.0677 | 0.1654 | 7,635 |
| Top-50 | 0.067512 | 0.164194 | 7,635 |

Top-50 相对 baseline native 排名：

| mAP@10 | PositiveRecall@10 | PositiveHitRate@10 | nDCG@10 | NegRecall@10 | Top1 Jaccard | Top5 Jaccard | Top10 Jaccard | Top20 Jaccard | Top50 Jaccard |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.067512 | 0.143881 | 0.401441 | 0.120970 | 0.164194 | 0.859201 | 0.600718 | 0.600363 | 0.622457 | 1.000000 |

### 7.2 Full Hard-0.5 召回，Full baseline Top-50 rerank

候选由 Full Hard-0.5 原生 Top-50 召回；候选内部由 Full baseline 的 positive-query cosine
score 重排。

| mAP@10 | PositiveRecall@10 | PositiveHitRate@10 | nDCG@10 | NegRecall@10 | Top1 Jaccard | Top5 Jaccard | Top10 Jaccard | Top20 Jaccard | Top50 Jaccard |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.067623 | 0.144881 | 0.405501 | 0.121083 | 0.169920 | 0.860511 | 0.603250 | 0.604754 | 0.630118 | 1.000000 |

对应 Hard native：mAP@10 `0.067326`，PositiveRecall@10 `0.143090`，
PositiveHitRate@10 `0.399607`，nDCG@10 `0.120515`，NegRecall@10 `0.163631`。

### 7.3 V7 strict 1,580-query 跨模型 Top-50 rerank

| Retriever -> Reranker | mAP@10 | PositiveRecall@10 | PositiveHitRate@10 | nDCG@10 | NegRecall@10 | Top10 Jaccard vs retriever native |
|---|---:|---:|---:|---:|---:|---:|
| Full baseline -> Full Hard-0.5 | 0.066838 | 0.148591 | 0.408228 | 0.122128 | 0.183102 | 0.611351 |
| Full Hard-0.5 -> Full baseline | 0.066558 | 0.151442 | 0.416456 | 0.123036 | 0.187749 | 0.616318 |

## 8. Simple no-ML counterfactual profile 实验

配置：Full baseline 与 Full Hard-0.5；PinPoint V7；候选池使用两个模型 Top-50 union；
edited text 只用于离线 Delta/profile 统计；缺失类别保留 NaN 并按可用类别平均；不训练
probe，不做 OOF，不使用 negative text rerank。

### 8.1 Native Top-50 profile

| Contrast | 模型 | d_available | Queries | Groups |
|---|---|---:|---:|---:|
| retrieved positive - retrieved pool nonpositive | Full baseline | 0.010577 | 951 | 313 |
| retrieved positive - retrieved pool nonpositive | Full Hard-0.5 | 0.012439 | 933 | 307 |
| all positive - retrieved pool nonpositive | Full baseline | -0.002033 | 1,580 | 471 |
| all positive - retrieved pool nonpositive | Full Hard-0.5 | -0.007946 | 1,580 | 471 |
| retrieved positive - appended positive | Full baseline | 0.015333 | 817 | 282 |
| retrieved positive - appended positive | Full Hard-0.5 | 0.023964 | 810 | 277 |

### 8.2 Shared-pool Hard-minus-baseline paired difference

| Contrast | Estimate | 95% CI | Queries | Groups |
|---|---:|---:|---:|---:|
| retrieved positive - retrieved pool nonpositive | 0.002868 | [0.001472, 0.004264] | 992 | 326 |
| all positive - retrieved pool nonpositive | -0.003477 | [-0.004627, -0.002303] | 1,580 | 471 |
| retrieved positive - appended positive | 0.008019 | [0.006283, 0.009846] | 849 | 292 |

## 9. 早期 QDiag 数据构建运行

下面是三个 QDiag 数据项目中实际留下中间产物或最终 JSONL 的构建运行。`source / candidates /
final` 均为文件实际行数；`—` 表示没有生成该阶段文件，而不是 0 条样本。

| 项目与运行目录 | source/groups | candidates | final | 状态 |
|---|---:|---:|---:|---|
| `cir_qdiag/outputs/pilot_20_v4_1` | 20 | 20 | 20 | 完成 |
| `cir_qdiag/outputs/pilot_20_v5` | 20 | 20 | 20 | 完成 |
| `cir_qdiag/outputs/pilot_50_v5` | — | — | — | 只保留 50 组 paraphrase cache；不完整 |
| `cir_qdiag/outputs/pilot_500_v5` | 500 | 465 | 465 | 完成；另有三模型诊断 |
| `cir_qdiag/outputs/pilot_50_v6_openai_terra` | 50 | 48 | 48 | 完成；另有三模型诊断 |
| `cir_qdiag/outputs/train_500_v5` | 28,225 | 26,080 | 26,080 | 完成；目录名中的 500 不是最终行数 |
| `cir_qdiag/outputs/train_all_v5` | 28,225 | 26,086 | 3,068 / 26,086 no-VA | 完成；OAP 使用 `final_no_va.jsonl` |
| `cir_qdiag_openai/outputs/pilot_50_v6_openai` | 5 | 5 | 5 | 仅实际跑了 5 条 |
| `cir_qdiag_openai/outputs/pilot_50_v6_openai_terra` | 50 | 48 | 48 | 上述 V6 Terra 的镜像；诊断核心文件哈希一致 |
| `cir_qdiag_openai/outputs/pinpoint_pilot_50_v7` | 5 groups | 5 | 5 | 仅实际跑了 5 组 |
| `cir_qdiag_openai/outputs/pinpoint_pilot_500_v7` | 500 groups | 500 | 500 | 完成；第 3 至 8 节使用的数据 |
| `cir_qdiag_openai/outputs/pinpoint_full_v7` | 1,683 groups | — | — | 只完成 group 构建；未完成 candidate/final |
| `cir_qdiag_openai_cirr/outputs/cirr_train_pilot_50_v8` | 50 | — | — | 只生成 source；不完整 |
| `cir_qdiag_openai_cirr/outputs/cirr_train_pilot_500_v8` | 500 | 500 | 500 | 完成；另有七模型诊断 |

## 10. 早期与 CIRR V8 多模型诊断

### 10.1 Pilot 500 V5 三模型方向诊断

这次运行使用 465 个生成成功的样本，共输出 3,401 条 negative score。下表直接按
negative 行计算，未做 pair/group balancing，因此不能与第 4 节的 group-balanced V7 数字
直接比较。

| 模型 | Negative rows | Raw mean Delta | Raw win |
|---|---:|---:|---:|
| BGE-VL-base | 3,401 | 0.025467 | 0.710379 |
| BGE-VL-MLLM-S1 | 3,401 | 0.032817 | 0.793002 |
| Qwen3-VL-Embedding-2B | 3,401 | 0.034157 | 0.793884 |

原始结果：`../cir_qdiag/outputs/pilot_500_v5/model_diagnostics/`。其中还实际生成了
factorial report、reference ablation、confusion tables、OOF probe 和 delta-mode 图表；这些
属于旧诊断产物，不作为当前模型训练结论。

### 10.2 Pilot 50 V6 三模型方向诊断

实际完成 48 个样本、331 条 negative score。`cir_qdiag` 与 `cir_qdiag_openai` 下的
`pilot_50_v6_openai_terra` 核心 `scores.csv` 和 `probe_metrics.csv` 哈希一致，因此计为同一次
实验的两份产物，不重复计数。

| 模型 | Negative rows | Raw mean Delta | Raw win |
|---|---:|---:|---:|
| BGE-VL-base | 331 | 0.023225 | 0.712991 |
| BGE-VL-MLLM-S1 | 331 | 0.027983 | 0.758308 |
| Qwen3-VL-Embedding-2B | 331 | 0.034964 | 0.797583 |

### 10.3 CIRR train pilot 500 V8 七模型方向诊断

这次诊断不是训练实验。数据为 500 pairs、2,446 negatives；strict 子集为 493 pairs、
1,695 条 ADD/SUBSTITUTE negatives。表中是 comparison 输出的 pair-balanced 结果。

| 模型 | All Mean Delta | All win | Strict Mean Delta | Strict win |
|---|---:|---:|---:|---:|
| BGE-VL-base | 0.024863 | 0.705444 | 0.027614 | 0.727823 |
| BGE-VL-large | 0.027556 | 0.728473 | 0.032866 | 0.774239 |
| BGE-VL-MLLM-S1 | 0.027564 | 0.776236 | 0.027133 | 0.786680 |
| BGE-VL-MLLM-S2 | 0.026953 | 0.801960 | 0.029231 | 0.817241 |
| Qwen3-VL-Embedding-2B | 0.034442 | 0.780665 | 0.041625 | 0.837289 |
| LinCIR | 0.013179 | 0.717119 | 0.017842 | 0.805308 |
| Pic2Word | 0.013062 | 0.716957 | 0.017005 | 0.784178 |

### 10.4 CIRR V8 的 493-image target-pool retrieval

同一次七模型诊断还运行了 500 queries 对 493 张 target images 的检索，reference image 被
排除。这个候选池远小于标准 CIRR val gallery，因此下表不是官方 CIRR Recall，不能与第 2
节的 2,297-image CIRR val 数字直接比较。

| 模型 | R@1 | R@5 | R@10 | R@50 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| BGE-VL-base | 0.664 | 0.932 | 0.964 | 0.994 | 0.827119 |
| BGE-VL-large | 0.712 | 0.932 | 0.972 | 0.998 | 0.849385 |
| BGE-VL-MLLM-S1 | 0.742 | 0.958 | 0.978 | 1.000 | 0.867391 |
| BGE-VL-MLLM-S2 | 0.806 | 0.970 | 0.990 | 0.998 | 0.905263 |
| Qwen3-VL-Embedding-2B | 0.724 | 0.938 | 0.982 | 1.000 | 0.856915 |
| LinCIR | 0.522 | 0.850 | 0.934 | 0.988 | 0.732916 |
| Pic2Word | 0.530 | 0.824 | 0.918 | 0.986 | 0.721025 |

原始结果：
`../cir_qdiag_openai_cirr/outputs/cirr_train_pilot_500_v8/model_diagnostics/comparison/`。

## 11. 历史 OOF/probe 与无效 rerank 记录

| 项目 | 状态 | 当前是否作为有效结果 |
|---|---|---|
| Logistic Regression / OOF probe | 曾运行，文件仍保留在 `counterfactual_candidate_profiles/*/probe_*` | 否；默认分析已关闭，需显式 `--enable-oof-probes` 才运行 |
| 基于 negative-text profile 的直接公式 rerank | 曾生成过结果 | 否；该路径已从默认代码移除，直接 rerank artifacts 已删除 |
| 当前 rerank | 两个检索模型相互召回/重排，positive-query cosine only | 是 |

## 12. 配置与原始结果索引

### 12.1 训练配置

```text
configs/clip_b_baseline.yaml
configs/clip_b_add_substitute.yaml
configs/clip_b_add_substitute_delete.yaml
configs/clip_l_b64_baseline.yaml
configs/clip_l_b64_hard_w0_5.yaml
configs/clip_l_b64_hard_w1_0.yaml
configs/clip_l_last4_baseline.yaml
configs/clip_l_last4_hard_w0_5.yaml
configs/clip_l_full_baseline.yaml
configs/clip_l_full_hard_w0_1.yaml
configs/clip_l_full_hard_w0_25.yaml
configs/clip_l_full_hard_w0_5.yaml
configs/clip_l_full_hard_w0_75.yaml
configs/clip_l_full_hard_w1_0.yaml
configs/clip_l_lora_r16_baseline.yaml
configs/clip_l_lora_r16_hard_w0_5.yaml
configs/clip_l_full_baseline_then_hard_only_500_v8.yaml
configs/bge_vl_large_full_hard_w0_1.yaml
configs/oap_residual_clip_l_beta0_1.yaml
configs/oap_residual_bge_vl_large_beta0_1.yaml
../diag_model_training_new_method/configs/six_channel.yaml
```

### 12.2 主要结果

```text
outputs/**/effective_config.json
outputs/**/run_summary.json
outputs/**/val_metrics.json
outputs/clip_l/full/hard_weight_grid_cirr_partial.csv
outputs/clip_l/full/baseline_then_hard_only_500_v8/comparison/
outputs/oap_residual/clip_l/beta0_1/
outputs/oap_residual/bge_vl_large/beta0_1/
outputs/oap_residual/eval_sweep_summary.csv
outputs/oap_residual/cirr_reranker/

../cir_qdiag_openai/outputs/pinpoint_official_full/
../cir_qdiag_openai/outputs/pinpoint_official_full_lora/
../cir_qdiag_openai/outputs/pinpoint_official_bge_vl_large/
../cir_qdiag_openai/outputs/pinpoint_official_oap/
../cir_qdiag_openai/outputs/pinpoint_pilot_500_v7/clip_l_pinpoint_diagnostics/
../cir_qdiag_openai/outputs/pinpoint_pilot_500_v7/clip_l_last4_pinpoint_diagnostics/
../cir_qdiag_openai/outputs/pinpoint_pilot_500_v7/full_delta_from_official_cache/
../cir_qdiag_openai/outputs/pinpoint_pilot_500_v7/oap_delta_from_official_cache/
../cir_qdiag_openai/outputs/pinpoint_pilot_500_v7/bge_full_hard_delta_from_official_cache/
../cir_qdiag_openai/outputs/pinpoint_pilot_500_v7/full_retrieval_diagnostics/
../cir_qdiag_openai/outputs/pinpoint_pilot_500_v7/counterfactual_candidate_profiles/

../cir_qdiag/outputs/pilot_500_v5/model_diagnostics/
../cir_qdiag/outputs/pilot_50_v6_openai_terra/model_diagnostics/
../cir_qdiag_openai_cirr/outputs/cirr_train_pilot_500_v8/model_diagnostics/comparison/
```
