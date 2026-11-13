# CIR Semantic Ability Distillation：四组消融实验

## 实验定义

四组使用同一个 Student 起点：

```text
CLIP-L Full baseline epoch 10
outputs/clip_l/full/baseline/checkpoint_epoch_010.pt
```

Teacher 为冻结的 `BGE-VL-MLLM-S2`。训练集为
`data/cirr_train_all_v5_no_va.jsonl`，语义监督只使用 Object ADD/SUBSTITUTE。

| 组别 | Positive Rank KD | Semantic Response KD | 是否继续训练 |
|---|---:|---:|---:|
| Baseline | 0 | 0 | 否 |
| Positive Rank KD | 1 | 0 | 是 |
| Semantic Response KD | 0 | 1 | 是 |
| Rank + Response KD | 1 | 1 | 是 |

所有 KD 训练均满足：

- 不使用 InfoNCE；
- CLIP-L vision encoder 冻结；
- 训练 CLIP-L text encoder 和 composition head；
- 16 epochs；
- batch size 128；
- 每 4 epochs 保存一次：4、8、12、16；
- 三个训练实验拥有完全相同的数据顺序、初始化和优化器设置。

## N×M Response Matrix

每条 CIRR 样本使用：

- `N` 条同类 Object ADD/SUBSTITUTE negative query，`N` 可以不同；
- `M=5` 张 group candidate image：正确 target 加四张相似干扰图。

Teacher 与 Student 分别计算：

```text
D[n, m] = sim(q+, candidate[m]) - sim(q-[n], candidate[m])
```

因此 Semantic Response KD 对每条样本提供 `N×5` 个 response constraints。实现使用
variable-N flattened batch 和 owner index，不重复 negative，也不把 padding 当成训练关系。

## 两个 Loss

Positive Rank KD 在五张 group candidates 上蒸馏 Teacher 的 soft ranking：

```text
KL(softmax(S_teacher / 0.05), softmax(S_student / 0.05))
```

只保留 Teacher 把真实 target 排在 group Top-1 的样本。Teacher cache 完成后会现场输出这项
coverage，而不是预先假定 coverage。

Semantic Response KD 使用：

```text
SmoothL1(D_student, D_teacher, beta=0.01)
```

只保留 Teacher 自己满足 `D_teacher[target] > 0` 的 negative relation，避免蒸馏 Teacher
方向错误的关系；但一条被保留的 relation 会监督全部五张 candidate，而不只是 target。

联合组的总损失为：

```text
L = L_positive_rank + L_semantic_response
```

## 缓存策略

为了避免16轮中重复运行 Teacher 和冻结的 Student vision encoder，一键脚本先生成两份缓存：

1. BGE-VL-MLLM-S2 的 positive ranking 与 Object `N×5` response 所需分数；
2. Full baseline CLIP-L 的所有 reference/group image features。

Teacher cache 和三组训练均可断点续跑。新 checkpoint 只保存可训练的 text/fusion delta，
并通过 `base_checkpoint` 继承 Full baseline 的完整权重；评估加载时会自动恢复这条继承链。

## 一键运行

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train

CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_semantic_distillation_ablation.sh
```

默认一条命令依次运行：

1. Teacher 与 Student image cache；
2. 三个16-epoch KD 实验；
3. baseline 和所有 4/8/12/16 checkpoint 的 CIRR val/Group Recall；
4. 四个最终模型的 PinPoint official mAP@10、NegRecall@10；
5. 四个最终模型的 Delta、raw win、校准后 Stable Win 和 random-10 gap；
6. 自动生成 `ABLATION_SUMMARY.md` 和 `ablation_summary.csv`。

如果只想先完成训练与 CIRR，而暂时跳过两项 PinPoint：

```bash
CUDA_VISIBLE_DEVICES=0 \
RUN_PINPOINT_OFFICIAL=0 \
RUN_PINPOINT_DIAGNOSTICS=0 \
  bash scripts/run_semantic_distillation_ablation.sh
```

## 输出目录

```text
outputs/semantic_distillation/clip_l_full_baseline/
├── baseline/
├── positive_rank/
├── semantic_response/
├── joint/
├── pinpoint_official/
├── pinpoint_diagnostics/
├── ablation_summary.csv
└── ABLATION_SUMMARY.md
```

每个训练目录包含：

- `checkpoint_epoch_004.pt`
- `checkpoint_epoch_008.pt`
- `checkpoint_epoch_012.pt`
- `checkpoint_epoch_016.pt`
- `last.pt`，用于断点续训；
- `train_metrics.jsonl`；
- `training_time.json`；
- `gradient_audit.json`；
- CIRR checkpoint curve 指标。
