# CIR-QDiag Standard CLIP Training

本项目使用标准 OpenAI CLIP ViT-B/16 和 ViT-L/14 进行 CIRR composed image retrieval 训练，不再使用 BGE 权重。

当前所有 CLIP-L、Full/LoRA、hard-weight、CIRR、PinPoint 和 reranking 实验的统一总结见
[`EXPERIMENT_SUMMARY.md`](EXPERIMENT_SUMMARY.md)。

每个 backbone 运行三种严格对照：

| mode | 损失 | CIR-QDiag negative |
|---|---|---|
| `baseline` | 对称 batch InfoNCE | 无 |
| `add_substitute` | 对称 batch InfoNCE + hard-query InfoNCE | Add + Substitute |
| `add_substitute_delete` | 同上 | Add + Substitute + Delete |

baseline 会设置 `negative_mode: none` 和 `hard_negative_weight: 0`，不会 tokenize 或编码任何 negative text。三组共享 train triples、seed、数据顺序、模型结构和其他优化参数。

## Backbone

- CLIP-B：`openai/clip-vit-base-patch16`，ViT-B/16，512 维 projection。
- CLIP-L：`openai/clip-vit-large-patch14`，ViT-L/14，768 维 projection。

加载器会校验 projection 维度、vision hidden size、层数和 patch size，并拒绝声明 `auto_map` 自定义代码的 checkpoint，因此 BGE checkpoint 不能被误当成普通 CLIP 加载。

标准 CLIP 没有预训练 composed-query head。三组都从 `normalize(image_embedding + text_embedding)` 开始，训练同一个零初始化 residual combiner。默认冻结 CLIP，只训练 combiner 和 temperature；如需端到端实验，三组必须同时设置 `freeze_backbone: false`。

## 安装

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
python -m pip install -e '.[dev]'
```

本工作区已经从 Hugging Face 下载官方权重到：

```text
models/clip-vit-base-patch16
models/clip-vit-large-patch14
```

六份配置均使用这些本地目录并设置 `local_files_only: true`，训练时不访问网络。若在另一台机器重新部署，可运行：

```bash
hf download openai/clip-vit-base-patch16 --local-dir models/clip-vit-base-patch16
hf download openai/clip-vit-large-patch14 --local-dir models/clip-vit-large-patch14
```

## 构建数据 manifest

```bash
python scripts/prepare_train_manifest.py \
  --annotations /mnt/data/datasets/search_dataset/zxl/CIR_new/datasets/CIRR/cirr/captions/cap.rc2.train.json \
  --image-split /mnt/data/datasets/search_dataset/zxl/CIR_new/datasets/CIRR/cirr/image_splits/split.rc2.train.json \
  --image-root /mnt/data/datasets/search_dataset/zxl/CIR_new/datasets/CIRR \
  --negative-jsonl ../cir_qdiag/outputs/train_500_v5/final.jsonl \
  --output data/cirr_train_with_negatives.jsonl
```

同一个 manifest 同时用于三组实验。baseline 忽略其中所有 negatives。

## 六组配置

```text
configs/clip_b_baseline.yaml
configs/clip_b_add_substitute.yaml
configs/clip_b_add_substitute_delete.yaml
configs/clip_l_baseline.yaml
configs/clip_l_add_substitute.yaml
configs/clip_l_add_substitute_delete.yaml
```

## 推荐的 batch-64 hard-weight 对照

为避免覆盖已经完成的 batch-1024 结果，新的实验使用独立配置和输出目录。它暂时排除 Delete，
只比较标准训练、Add/Substitute weight 0.5 和 Add/Substitute weight 1.0：

```text
configs/clip_b_b64_baseline.yaml
configs/clip_b_b64_hard_w0_5.yaml
configs/clip_b_b64_hard_w1_0.yaml
configs/clip_l_b64_baseline.yaml
configs/clip_l_b64_hard_w0_5.yaml
configs/clip_l_b64_hard_w1_0.yaml
```

单张 GPU 默认只顺序完成 CLIP-L 的三组：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_clip_batch64_weights.sh
```

它会依次运行 CLIP-L baseline、weight 0.5、weight 1.0，然后生成比较结果。输出位于：

```text
outputs/clip_l/batch64/comparison.json
```

之后如需只跑 CLIP-B：

```bash
CUDA_VISIBLE_DEVICES=0 VARIANTS=clip_b bash scripts/run_clip_batch64_weights.sh
```

如需顺序跑 B 和 L 全部六组：

```bash
CUDA_VISIBLE_DEVICES=0 VARIANTS="clip_b clip_l" bash scripts/run_clip_batch64_weights.sh
```

如果有六张 GPU，可以一张卡并行一组：

```bash
GPU_IDS=0,1,2,3,4,5 bash scripts/run_clip_batch64_weights_parallel.sh
```

每组仍是独立单卡训练，因此全局 InfoNCE batch 保持 64。

## CLIP-L 最后四层解冻对照

为测试冻结 backbone 是否限制 Recall/NDCG，新增一组严格的两模型对照：

```text
configs/clip_l_last4_baseline.yaml
configs/clip_l_last4_hard_w0_5.yaml
```

两组都更新 vision/text encoder 最后 4 个 Transformer block、vision/text projection、
fusion head 和 temperature；前面的层保持冻结。两组都保持单卡 batch=64、seed=42 和相同
学习率，唯一实验差异仍是 hard-negative loss。实测总参数约 430.8M，其中可训练参数约
83.3M（可训练 backbone 参数约 80.1M）。

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
CUDA_VISIBLE_DEVICES=0 bash scripts/run_clip_l_last4.sh
```

脚本顺序训练、评估 baseline 和 hard-0.5，输出到：

```text
outputs/clip_l/last4/baseline/
outputs/clip_l/last4/hard_w0_5/
outputs/clip_l/last4/comparison.json
```

每两轮保留一个 epoch checkpoint，减少 optimizer state 带来的磁盘占用。若 batch=64
仍然 OOM，可以先改为每卡 32；注意 gradient accumulation 只能保持优化器有效 batch，
不能把单次 InfoNCE 的候选数量恢复为 64，因此正式对比时两组必须采用相同设置。

## CLIP-L Full 与 LoRA 对照

新的正式实验同时比较两种 backbone 训练方式，每种都包含 baseline 和 hard-weight=0.5：

```text
Full:    全量更新 vision/text encoder 与 projection，backbone LR=1e-6
LoRA:    vision/text 每层 attention 的 q_proj、v_proj，rank=16，alpha=32，LR=1e-4
Common:  fusion head、temperature、单卡 InfoNCE batch=64、seed=42
```

一键顺序运行四组（默认先 LoRA，再 Full）：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
CUDA_VISIBLE_DEVICES=0 bash scripts/run_clip_l_full_lora.sh
```

也可以单独运行一种方法：

```bash
CUDA_VISIBLE_DEVICES=0 METHODS=lora_r16 bash scripts/run_clip_l_full_lora.sh
CUDA_VISIBLE_DEVICES=0 METHODS=full bash scripts/run_clip_l_full_lora.sh
```

输出位于：

```text
outputs/clip_l/lora_r16/{baseline,hard_w0_5}/
outputs/clip_l/full/{baseline,hard_w0_5}/
```

训练首批数据后会写入 `gradient_audit.json`。所有声明为可训练的参数都必须具有有限梯度，
否则训练立即失败，避免再次出现配置上解冻但实际没有更新的问题。Gradient checkpointing
明确使用 non-reentrant 实现，以支持冻结 base weights 的 LoRA。

每两轮的 `checkpoint_epoch_*.pt` 只保存模型，用于评估；`last.pt` 额外包含 optimizer 和
scheduler，用于断点续训。Full 模式文件很大，两个 Full 实验建议至少预留约 30GB 磁盘。

### Full hard-negative weight 网格

在已有的 `0.0` baseline 和 `0.5` 基础上，新增四个严格对照权重：

```text
0.1, 0.25, 0.75, 1.0
```

除 `hard_negative_weight` 和独立输出目录外，所有配置都与 Full Hard-0.5 相同：CLIP-L
全量更新、backbone LR=1e-6、单卡 batch=64、10 epochs、seed=42，只使用 ADD 与
SUBSTITUTE negative。顺序训练并进行 CIRR validation：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
CUDA_VISIBLE_DEVICES=0 bash scripts/run_clip_l_full_hard_grid.sh
```

默认只训练新增的四组，已有 baseline 和 0.5 不会重跑；最终六档权重汇总到：

```text
outputs/clip_l/full/hard_weight_grid_cirr.csv
```

脚本检测到 `checkpoint_epoch_010.pt` 会跳过训练，检测到未完成的 `last.pt` 会自动断点
续训。也可以只跑部分权重：

```bash
CUDA_VISIBLE_DEVICES=0 WEIGHTS="0.1 0.25" bash scripts/run_clip_l_full_hard_grid.sh
```

四组新增训练完成后，在 `diag_model` 运行官方全量 PinPoint：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/diag_model
CUDA_VISIBLE_DEVICES=0 python run_clip_pinpoint_official.py \
  --models CLIP-L-full-hard-w0.1,CLIP-L-full-hard-w0.25,CLIP-L-full-hard-w0.75,CLIP-L-full-hard-w1.0 \
  --output-root ../cir_qdiag_openai/outputs/pinpoint_official_full_hard_grid \
  --batch-size 64
```

每个 Full 模型的 image encoder 不同，不能共享 candidate embedding。

## CLIP-L baseline 后续接 500 条 hard-only 训练

这组实验以已经完成普通 batch InfoNCE 训练的 `CLIP-L Full baseline epoch-10`
为初始化，只在 CIRR train pilot v8 的 500 条样本上继续训练。优化目标严格为同一目标图像下
正文本与 hard-negative 文本之间的交叉熵；普通 batch InfoNCE 只计算监控指标，不参与反向
传播。500 条里有 7 条只有 DELETE negative，因此本实验使用全部 ADD、SUBSTITUTE 和 DELETE，
并在 hard-only 模式保留最后一个不完整 batch，确保每轮都覆盖 500 条。

默认训练 10 epochs、batch 64、每 2 epochs 保存，随后选择 epoch 2/6/10，并与 Full
baseline 和原始未训练 CLIP-L 一起评估 CIRR val 及官方全量 PinPoint：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
CUDA_VISIBLE_DEVICES=0 bash scripts/run_clip_l_hard_only_500_v8.sh
```

只训练并运行 CIRR val、暂不运行耗时较长的 PinPoint：

```bash
CUDA_VISIBLE_DEVICES=0 RUN_PINPOINT=0 \
  bash scripts/run_clip_l_hard_only_500_v8.sh
```

主要输出：

```text
outputs/clip_l/full/baseline_then_hard_only_500_v8/
  checkpoint_epoch_{002,004,006,008,010}.pt
  comparison/cirr_val/cirr_val_comparison.csv
  comparison/pinpoint_official/official_map10_negrecall10.csv
```

`--initialize-from` 与 `--resume-from` 语义不同：前者只载入 Full baseline 模型权重并从
epoch 1 创建新的 optimizer/scheduler，后者用于恢复本次 hard-only 实验的 `last.pt`。

## BGE-VL-large Full Hard-0.1

这组实验从本地原版 `BGE-VL-large` 权重出发，使用与 CLIP-L Full 一致的
CIRR 训练流程，全量更新 image/text encoder、projection、fusion head 和 temperature。
差异是 hard-negative weight 设为 0.1，且 composition 保留 BGE 原生的 branch 尺度：

```text
q0 = normalize(raw_image_feature + raw_text_feature)
q  = normalize(raw_image_feature + raw_text_feature + residual)
```

Residual 最后一层零初始化，已验证训练前 `q0` 与原版 BGE native composition 完全
一致。其他主要设置为单卡 batch 64、10 epochs、BF16、head LR `1e-4`、backbone LR
`1e-6`、每 2 epochs 保存。训练与 CIRR validation：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
CUDA_VISIBLE_DEVICES=0 bash scripts/run_bge_vl_large_hard_w0_1.sh
```

脚本检测到最终 checkpoint 会跳过训练；只有 `last.pt` 时会自动断点续训。输出：

```text
outputs/bge_vl_large/full/hard_w0_1/
```

训练完成后，在 PinPoint 全部 7,635 条官方 query 和 109,598 张本地可用候选图上，
对比零训练原版 BGE-VL-large 与 Full Hard-0.1：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/diag_model
CUDA_VISIBLE_DEVICES=0 bash run_bge_vl_large_pinpoint_official.sh
```

最终主表：

```text
../cir_qdiag_openai/outputs/pinpoint_official_bge_vl_large/
  official_map10_negrecall10.csv
```

注意：该对比同时包含 CIRR Full finetune 和 hard loss 的影响。若要单独归因
`hard_negative_weight=0.1`，还需要训练一个其他设置完全相同的
`BGE-VL-large Full baseline (weight=0)`。

## O/A/P Semantic Residual Adapter

[`cir_oap_two_main_schemes.md`](cir_oap_two_main_schemes.md) 的第一优先级方案已经实现。
它冻结既有 global CIR 模型，只训练 Object、Attribute、Predicate 三组 query/target
adapter：

```text
S_final = S_global + beta / 3 * (S_object + S_attribute + S_predicate)
beta = 0.1
```

训练同时使用批内正常 retrieval ordering 和 typed counterfactual patching。Add 与
Substitute 按 semantic target 合并；Delete 不进入 O/A/P 训练。每个 semantic 每个样本
每轮最多选择 2 条 negative，因此三个分支的监督量基本平衡。

同一套训练、checkpoint 和评估接口现在支持两种架构，通过配置中的
`model.architecture` 区分：

```text
residual_adapter  从最终 768-d global embedding 提取 O/A/P
attention_tokens  从 vision patch tokens + text tokens 提取 O/A/P
```

原始训练数据来自：

```text
../cir_qdiag/outputs/train_all_v5/final_no_va.jsonl
```

转换后的 manifest 为 `data/cirr_train_all_v5_no_va.jsonl`，包含 26,086 条 train pair、
257,136 条原始 negative。文件名中的 `no_va` 表示这些 negative 没有经过视觉验证，结果
解释时必须保留这一数据质量限制。

第一轮使用两个冻结 global checkpoint：

- CLIP-L：`outputs/clip_l/full/baseline/checkpoint_epoch_010.pt`；
- BGE-VL-large：`outputs/bge_vl_large/full/hard_w0_1/checkpoint_epoch_010.pt`。

BGE 的初始化是已有 Full Hard-0.1，而不是纯 Full baseline，因此 CLIP/BGE 之间不作严格
架构归因；每个 OAP 结果只与其自身冻结的 global score 对比。

顺序训练并评估 CLIP-L 和 BGE-VL-large：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
CUDA_VISIBLE_DEVICES=0 bash scripts/run_oap_residual.sh
```

只跑其中一个：

```bash
CUDA_VISIBLE_DEVICES=0 MODELS=clip_l bash scripts/run_oap_residual.sh
CUDA_VISIBLE_DEVICES=0 MODELS=bge_vl_large bash scripts/run_oap_residual.sh
```

两组都训练 5 epochs、batch 64，只更新约 64.1 万个 adapter 参数。epoch checkpoint
只包含 adapter，避免再次保存数 GB 的冻结 backbone。结果目录为：

```text
outputs/oap_residual/clip_l/beta0_1/
outputs/oap_residual/bge_vl_large/beta0_1/
```

`oap_cirr_metrics.json` 在同一次推理中报告冻结 `global_baseline` 与 `oap_final` 的 Recall、
Group Recall、MRR、mean rank，以及 adapter 导致的 Top-1 corrected/regressed query 数量。
这能直接发现 semantic residual 是净增益还是破坏原有检索。

### OAP reranker 与 PinPoint 实验

CIRR reranker 实验先用冻结的 global 分数检索全库，再只用 OAP 分数重排原始
Top-10/50/100。它同时扫描多个 inference beta，不重新训练模型：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
CUDA_VISIBLE_DEVICES=0 bash scripts/run_oap_cirr_reranker.sh
```

结果写入 `outputs/oap_residual/cirr_reranker/summary.csv`。脚本比较 CLIP-L epoch 2/5
与 BGE-VL-large epoch 5；可以用 `TOP_KS=10,50` 改重排深度。Top-K 外的候选顺序完全
不变，因此 Top-10 rerank 的 Recall@10 必须与 global baseline 一致。

PinPoint 实验使用相同 OAP checkpoint，把 global 与 O/A/P 分量打包为单个 960 维向量，
其内积严格等价于 `S_final`，再走官方全库检索和官方 evaluator：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/diag_model
CUDA_VISIBLE_DEVICES=0 bash run_oap_pinpoint_official.sh
```

结果写入：

```text
../cir_qdiag_openai/outputs/pinpoint_official_oap/
  official_map10_negrecall10.csv
  oap_vs_global.csv
```

`oap_vs_global.csv` 会把 CLIP OAP 与 `CLIP-L-full-baseline`、BGE OAP 与其实际冻结的
`BGE-VL-large-full-hard-w0.1` 分别比较，报告 mAP@10 和 NegRecall@10 的绝对差值。

断点续训示例：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  scripts/train_oap.py \
  --config configs/oap_residual_clip_l_beta0_1.yaml \
  --resume-from outputs/oap_residual/clip_l/beta0_1/last.pt
```

## O/A/P Semantic Attention Tokens

[`cir_oap_two_main_schemes.md`](cir_oap_two_main_schemes.md) 的第二个方案也已经实现。Global
branch 与方案一相同，始终来自冻结的既有 CIR checkpoint；新增分支直接读取最后一层
vision patch tokens 和有效 text tokens：

```text
reference patches + modification tokens
                  ↓
        [OBJ] [ATTR] [PRED]
                  ↓
   1-layer cross-attention + FFN
                  ↓
          three normalized 64-d vectors
```

Target 侧使用另一组独立的 `[OBJ]/[ATTR]/[PRED]` token 和 cross-attention，从 candidate
patch tokens 生成三个 target vectors。Query/target attention 参数在初始化时数值对齐，
之后独立训练。配置为 attention hidden 256、8 heads、FFN 512、dropout 0.1，新增可训练
参数 1,814,144；最终仍打包为 960 维向量，内积严格等于：

```text
S_global + beta / 3 * (S_object + S_attribute + S_predicate)
```

训练数据、counterfactual patching、5 epochs、batch 64、beta 0.1 和两个 frozen global
checkpoint 均与 Residual Adapter 实验保持一致。顺序训练并执行 CIRR val：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
CUDA_VISIBLE_DEVICES=0 bash scripts/run_oap_attention.sh
```

只跑一个 backbone：

```bash
CUDA_VISIBLE_DEVICES=0 MODELS=clip_l bash scripts/run_oap_attention.sh
CUDA_VISIBLE_DEVICES=0 MODELS=bge_vl_large bash scripts/run_oap_attention.sh
```

输出目录：

```text
outputs/oap_attention/clip_l/beta0_1/
outputs/oap_attention/bge_vl_large/beta0_1/
```

训练完成后可运行 PinPoint official full：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/diag_model
CUDA_VISIBLE_DEVICES=0 bash run_oap_attention_pinpoint_official.sh
```

PinPoint 双 reference query 会先分别编码两个 reference，再对相同 ViT patch position 的
token 与 global feature 做均值；单 reference query 不做额外 pooling。结果写入
`../cir_qdiag_openai/outputs/pinpoint_official_oap_attention/`，其中
`official_map10_negrecall10.csv` 是官方指标，`oap_attention_vs_global.csv` 是相对各自
冻结 global checkpoint 的差值。

### 判断 epoch 数量是否过多

> 2026-08-07 清理磁盘后，历史 Full 实验的 epoch 2/4/6/8 中间 checkpoint 已删除，
> 只保留 epoch 10 最终推理权重。下面的曲线命令只适用于未来重新保留了中间
> checkpoint 的训练。

训练过程现在会把每 20 step 的 `loss`、`batch_loss`、`hard_loss`、训练准确率、温度、head
LR 和 backbone LR 持久化到每个输出目录的 `train_metrics.jsonl`。旧实验只向终端打印过
这些信息，无法事后恢复 loss 曲线，但保存的 epoch 模型仍可直接比较 validation 曲线。

评估当前已经完成的 Full baseline、0.1、0.25、0.5 在 epoch 2/4/6/8/10 的 CIRR 表现：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
CUDA_VISIBLE_DEVICES=0 bash scripts/run_clip_l_full_epoch_curves.sh
```

每个模型输出：

```text
outputs/clip_l/full/<run>/checkpoint_curve_cirr/
  epoch_002.json
  epoch_004.json
  epoch_006.json
  epoch_008.json
  epoch_010.json
  curve.csv
  best_by_metric.json
```

评估结果会缓存，中断后重新运行只补未完成的 epoch。只检查一个模型可设置：

```bash
CUDA_VISIBLE_DEVICES=0 RUNS=hard_w0_25 bash scripts/run_clip_l_full_epoch_curves.sh
```

如果 CIRR Recall 和 Group Recall 在 epoch 6/8 达到峰值后持续下降，就说明 10 epochs
存在过拟合；如果到 epoch 10 仍上升或持平，则 10 epochs 并不多。最终模型若以 PinPoint
为主，还应对候选 epoch 额外比较官方 mAP@10 和 NegRecall@10，而不能只按 CIRR 选点。

单独训练一组：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 scripts/train.py \
  --config configs/clip_b_baseline.yaml
```

完整运行 B 和 L 的三组训练、评估与比较：

```bash
CUDA_VISIBLE_DEVICES=0 GPUS=1 bash scripts/run_clip_comparison.sh
```

只运行 CLIP-B：

```bash
CUDA_VISIBLE_DEVICES=0 GPUS=1 VARIANTS=clip_b bash scripts/run_clip_comparison.sh
```

四卡运行时三组必须使用相同卡数：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 bash scripts/run_clip_comparison.sh
```

CLIP-B/L 默认都使用每卡 batch 32，以保持单次 InfoNCE candidate 数量可比。跨实验比较时不要改变 batch size、GPU 数或 gradient accumulation；梯度累积不会扩大单次 InfoNCE 的 in-batch negative 集合。

## 输出与指标比较

```text
outputs/clip_b/
  baseline/
  add_substitute/
  add_substitute_delete/
  comparison.json

outputs/clip_l/
  ...
```

`comparison.json` 对每个 CIRR Recall 和 Group Recall 报告：

- `add_substitute_minus_baseline`：严格 harmful negatives 相对正常训练的增益。
- `add_substitute_delete_minus_baseline`：全部 negatives 相对正常训练的增益。
- `delete_increment`：在相同 Add/Substitute 基础上额外加入 Delete 的增量。

单独评估：

```bash
python scripts/evaluate.py \
  --config configs/clip_b_baseline.yaml \
  --checkpoint outputs/clip_b/baseline/last.pt
```

断点续训：

```bash
torchrun --standalone --nproc_per_node=1 scripts/train.py \
  --config configs/clip_b_baseline.yaml \
  --resume-from outputs/clip_b/baseline/last.pt
```

## 验收

```bash
PYTHONPATH=src pytest -q
ruff check .
python -m compileall -q src scripts tests
```
