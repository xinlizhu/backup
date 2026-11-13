# Rank candidate-policy ablation

这组实验只保留 Rank KD 主方法，不运行 Centered 或 Joint。所有模型均从同一个原始
CLIP ViT-L/14 初始化，全量微调 4 epochs，K=32，每个 epoch 保存 checkpoint。

## 实验矩阵

| Run | 候选图片 | 刷新方式 | 训练目标 | Teacher 筛选 |
|---|---|---|---|---|
| Dynamic Student Rank（四轮受控主方法） | target + 当前 Student Top-31 | 每个 epoch | Teacher→Student KL | target rank ≤ 10 |
| Teacher Top-K Rank | target + Teacher Top-31 | Teacher 冻结，因此只建一次 | Teacher→Student KL | target rank ≤ 10 |
| Random Rank | target + 31张随机非目标图 | 每个 epoch 重新随机 | Teacher→Student KL | target rank ≤ 10 |
| Static Student Rank | target + 初始 Student Top-31 | epoch 1 建一次并固定 | Teacher→Student KL | target rank ≤ 10 |
| Dynamic Student InfoNCE | target + 当前 Student Top-31 | 每个 epoch | 32-way target cross-entropy | 不使用 Teacher |
| Dynamic Student InfoNCE + Teacher Top-10 filter | target + 当前 Student Top-31 | 每个 epoch | 32-way target cross-entropy | target rank ≤ 10 |

Random 会严格排除 reference 与 target，并在每条样本内无放回采样。所有候选集都把
ground-truth target 固定在索引 0。候选 ID 可以缓存；进入训练 batch 后，reference 和全部
候选图片仍由当前 Student 实时重新编码，因此全量视觉骨干能够获得梯度。

额外重训 `Dynamic Student Rank` 是为了得到严格的四轮学习率计划对照。旧 epoch-4
checkpoint 来自更长的训练计划，其学习率状态与这组四轮实验不同。

## 一键运行

前台运行完整训练与评估：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train

CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_rank_candidate_policy_ablation.sh
```

后台运行：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
mkdir -p outputs/rank_candidate_policy_e4

nohup env CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_rank_candidate_policy_ablation.sh \
  > outputs/rank_candidate_policy_e4/nohup.log 2>&1 &

echo $!
```

默认流程包括：

1. 检查或复用 BGE Teacher 的 CIRR train embedding cache；
2. 顺序训练六个受控模型；
3. 评估所有 epoch 1–4 checkpoint 的 CIRR val 与 Group Recall；
4. 评估六个 epoch-4 checkpoint 的 PinPoint 官方 mAP@10/NegRecall@10；
5. 评估 QDiag Delta、Raw Win、Stable Win 和 random-10 separation；
6. 评估 FashionIQ val 跨数据集迁移；
7. 生成统一结果表 `outputs/rank_candidate_policy_e4/RESULTS.md`。

只训练并评估 CIRR 时：

```bash
RUN_EXTENDED_EVAL=0 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_rank_candidate_policy_ablation.sh
```

脚本会跳过已经存在的最终 checkpoint 和评估文件；若某个训练只有 `last.pt`，会自动从
该 checkpoint 断点续跑。Static Student/Teacher Top-K 的 epoch-1 candidate cache 会在续跑时
继续复用。

## 输出结构

```text
outputs/rank_candidate_policy_e4/
├── dynamic_student_rank/
├── teacher_topk_rank/
├── random_rank/
├── static_student_rank/
├── dynamic_student_infonce/
├── dynamic_student_infonce_teacher_top10/
├── pinpoint_official/
├── pinpoint_diagnostics/
├── fashioniq/
└── RESULTS.md
```

每个训练目录保存：

```text
checkpoint_epoch_001.pt
checkpoint_epoch_002.pt
checkpoint_epoch_003.pt
checkpoint_epoch_004.pt
last.pt
train_metrics.jsonl
training_time.json
val_epoch_001.json
val_epoch_002.json
val_epoch_003.json
val_epoch_004.json
on_policy/epoch_*/metadata.json
```
