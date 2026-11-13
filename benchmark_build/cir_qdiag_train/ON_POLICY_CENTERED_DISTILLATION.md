# CLIP-L On-Policy Centered Distillation

## 实验目的

这组实验不再从任何 CIRR baseline、Hard-negative 或其他后训练 checkpoint 继续训练。
三个训练组都直接从本地原版 OpenAI CLIP-L/14 权重开始：

```text
models/clip-vit-large-patch14
```

训练只使用原始 CIRR train：

- 28,225 条 `(reference image, modification text, target image)`；
- 16,939 张 train gallery 图片；
- 不读取 CIR-QDiag 生成的 `q-`；
- 不使用 InfoNCE。

Teacher 仍为冻结的 `BGE-VL-MLLM-S2`。

## 四组消融

| 实验 | Rank KD | Centered Response KD | 是否训练 |
|---|---:|---:|---:|
| Original CLIP-L | 0 | 0 | 否 |
| Positive Rank KD | 1 | 0 | 是 |
| Centered Response KD | 0 | 1 | 是 |
| Rank + Centered KD | 1 | 1 | 是 |

三个训练实验的初始化、数据顺序、优化器和训练轮数一致，但每个实验会根据自己的当前
Student 独立检索候选，不能跨实验共享 on-policy Top-32。

## Epoch-level on-policy Top-32

每个 epoch 开始时，使用该实验此时的 Student：

1. 重新编码全部 16,939 张 CIRR train 图片；
2. 重新编码全部 28,225 条 composed query；
3. 排除该 query 自己的 reference image；
4. 在 GPU 上检索当前 Student Top-32；
5. 构造最终的、恰好包含32张图片的候选集合。

最终候选布局固定为：

```text
candidate[0]    = ground-truth target
candidate[1:32] = Student 当前最高分的31张 non-target 图片
```

如果 target 原本就在 Top-32，删除原位置后移到索引0；如果不在，则用 target 替换原本的
第32名。代码同时记录替换前真实的 `raw_target_recall_at_k`，因此强制插入不会掩盖 Student
原始召回失败。

这是每个 epoch 更新一次的 on-policy / iterative hard-candidate distillation，不是每个
optimizer step 都重建全图库的严格同步 on-policy。

## 两个 loss

### Positive Rank KD

Teacher 和 Student 在同一组当前 Top-32 candidates 上计算相似度。Rank KD 为：

```text
KL(softmax(S_teacher / 0.05), softmax(S_student / 0.05))
```

### Centered Response KD

对每条 query，分别减去 Teacher/Student 自己在32张候选上的平均分：

```text
S_centered = S - mean(S)
```

然后使用：

```text
SmoothL1(S_student_centered, S_teacher_centered, beta=0.01)
```

中心化去掉了不同模型的整体 similarity offset，只蒸馏候选之间的相对响应结构。它不再是
旧实验中的 `q+ - q-` response。

默认只训练 Teacher 把真实 target 排为这32张候选 Top-1 的样本，并记录每轮 Teacher
coverage。

## 全量微调与候选缓存

CLIP-L 的 vision encoder、text encoder 和 residual composition head 全部训练。为了避免
每个训练 batch 对 `32 queries × 32 candidates` 的约1,024张图片反向传播：

- reference image：训练时现场编码，有梯度；
- target image：训练时现场编码，有梯度；
- 其余31张 hard candidates：使用本 epoch 开始时的 Student 缓存，stop-gradient；
- 每个新 epoch 重新编码全图库并更新缓存。

因此视觉编码器确实得到训练，正确 target 也参与 loss 和反向传播；缓存只用于当前 hard
non-target candidates。

## 训练参数

```text
epochs                    = 20
training batch size       = 32 queries
candidate count/query     = 32 images, including target
save epochs               = 4, 8, 12, 16, 20
gallery image batch       = 512
mining query batch        = 1024
Teacher score batch       = 512
mixed precision           = BF16
```

`image batch=512` 和 `query batch=1024` 只用于无梯度建库/检索，不是训练 batch。

## 一键运行

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train

CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_on_policy_centered_ablation.sh
```

脚本依次执行：

1. 一次性缓存 BGE Teacher 对全部原始 CIRR train 图片和正问句的 embedding；
2. 从原版 CLIP-L 独立训练 Rank、Centered、Joint 三组，各20 epochs；
3. 评估原版模型及各训练组 epoch 4/8/12/16/20 的 CIRR Recall 和 Group Recall；
4. 对四组最终模型运行 PinPoint 官方 mAP@10、NegRecall@10；
5. 运行 Delta、Raw Win、Stable Win 和 random-10 separation，并生成总表。

若第一次缓存 BGE Teacher 时显存不足，可只降低 Teacher 编码 batch：

```bash
CUDA_VISIBLE_DEVICES=0 \
TEACHER_IMAGE_BATCH_SIZE=8 \
TEACHER_QUERY_BATCH_SIZE=8 \
  bash scripts/run_on_policy_centered_ablation.sh
```

若 CLIP-L 建库阶段显存不足，修改三个 `configs/on_policy_clip_l_*.yaml` 中的
`image_batch_size` 或 `query_batch_size`；不要改变训练 `batch_size_per_gpu: 32`。

## 输出

```text
outputs/on_policy_centered/clip_l_original/
├── baseline/
├── positive_rank/
├── centered_response/
├── joint/
├── pinpoint_official/
├── pinpoint_diagnostics/
├── ablation_summary.csv
└── ABLATION_SUMMARY.md
```

每个训练组的 `on_policy/epoch_XXX/metadata.json` 会记录：

- 插入 target 前的真实 Target Recall@32；
- Teacher 在当前候选上的 Target Top-1 coverage；
- 建库和打分耗时；
- candidate 数量及 target 强制插入协议。

`training_time.json` 分别记录每轮 mining、反向训练和总耗时。
