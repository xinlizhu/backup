# Typed Candidate Counterfactual Training 实验记录

## 1. 实验目的

CIR-QDiag 的出发点是：现有 composed image retrieval（CIR）模型对 object、attribute 和 predicate 等细粒度修改理解不足。以前的文本 hard-negative 训练可以提高正负 query 的向量分离，但与最终的图像检索排序仍然存在目标差距。

本实验把训练闭环改为：

1. 使用带有 O/A/P 类型的反事实 negative text 构造错误 query；
2. 从该 CIRR 样本的真实 `image_set` 中取出组内候选图；
3. 在 negative query 与候选图之间在线挖掘最可能的反事实配对；
4. 同时训练正向候选排序和 2×2 双向反事实排序；
5. 检查 CIRR 正向检索、target-to-query 反向检索以及 PinPoint 官方检索是否实际改善。

这次实验验证的是整个 **typed candidate-aware counterfactual objective**。它不是一个新模型结构，而是一种针对 CIR 的训练算法。

## 2. 实际训练的模型

本次已经完成并评估的模型是：

- Backbone：CLIP ViT-L/14；
- 本地预训练模型：`models/clip-vit-large-patch14`；
- CIR query composer：reference image embedding、modification text embedding 及两者逐元素乘积经过两层 MLP，作为 `image + text` 的 residual；
- 微调方式：Full fine-tuning，不冻结 image encoder 或 text encoder，不使用 LoRA；
- 总参数量：430,764,034；
- 可训练参数量：430,764,033；
- 初始化 checkpoint：`outputs/clip_l/full/hard_w0_1/checkpoint_epoch_010.pt`。

该初始化 checkpoint 是此前完成的 CLIP-L Full Hard-0.1 模型。新方法和严格对照都从同一个 checkpoint 出发，以排除初始模型不同造成的影响。

项目中也准备了 BGE-VL-large 的对应配置，但截至本文档生成时，BGE-VL-large 的 control/typed 实验尚未运行，因此本文结果只代表 CLIP ViT-L/14。

## 3. 数据与样本构造

训练清单：

```text
data/cirr_train_all_v5_no_va.jsonl
```

数据统计：

- 训练样本：26,086 条；
- 至少具有一个可用 text negative 的样本：26,085 条；
- text negative 使用 `add` 和 `substitute`，不使用 `delete`；
- 每条样本每个 epoch 最多采样 6 个非 delete text negatives；
- 每条样本从 CIRR `image_set` 中排除 reference 和 target 后取 4 张候选图；
- negative text 保留 `object`、`attribute` 或 `predicate` 类型标签。

记：

- `q+`：reference image 与正确 modification text 合成的 query embedding；
- `z+`：正确 target image embedding；
- `q-`：reference image 与某条 O/A/P negative text 合成的错误 query embedding；
- `z-`：从同一条 CIRR 样本的组内 candidate image 中挖掘出的反事实候选。

当前数据只直接标注了 negative text 的 O/A/P 类型，没有直接标注每张 candidate image 对应哪一种语义错误。因此 `q- ↔ z-` 的关系是在线挖掘得到的弱配对，而不是真实人工配对。

## 4. 模型与 query composition

CLIP image/text feature 经过归一化后，正例 query 表示为：

```text
r = normalize(CLIP_image(reference))
t = normalize(CLIP_text(modification))
h = MLP([r; t; r ⊙ t])
q = normalize(r + t + h)
```

其中 MLP 的最后一层在最初构建 composer 时使用零初始化，使模型从普通的 `normalize(image + text)` composition 开始学习 residual。本实验从已经训练好的 Full Hard-0.1 checkpoint 继续训练，因此 composer 和 CLIP backbone 都继承该 checkpoint 的权重。

target image 和 candidate images 使用同一个可训练 CLIP image encoder 编码。因此这次训练不仅改变 query space，也会更新 image embedding space。

## 5. 训练算法

### 5.1 Batch InfoNCE

对 batch 中的正例 query 和 target image 进行双向 InfoNCE：

```text
L_batch = 0.5 × [CE(q+ → z+) + CE(z+ → q+)]
```

其他 batch 样本的 target/query 充当 in-batch negatives。

### 5.2 Text hard-negative loss

对每条样本，要求正确 query 在其 target 上的相似度高于同一 reference 对应的错误 query：

```text
s+ = scale × sim(q+, z+)
s- = scale × sim(q-, z+)
L_text-hard = CE([s+, s-1, ..., s-n], label=positive)
```

本实验中该项权重为 `0.1`。

### 5.3 Group candidate ranking loss

对每个正确 query，从同一 CIRR `image_set` 的 4 张候选图中选择得分最高的候选：

```text
z_hard = argmax_z sim(q+, z)
L_candidate = softplus(scale × [sim(q+, z_hard) - sim(q+, z+) + margin])
```

实验使用 `margin = 0`，该项直接要求 target image 排在最困难的组内候选图之前。

### 5.4 O/A/P typed candidate mining

对于每条样本和每一种存在的 O/A/P 类型，在该类型的 negative queries 与 4 张 candidate images 的笛卡尔积中选择：

```text
(q-, z-) = argmax [sim(q-, z-) - sim(q+, z-)]
```

只有当：

```text
sim(q-, z-) - sim(q+, z-) >= mining_min_advantage
```

时才接受该配对。本实验的 `mining_min_advantage = 0`。

这表示被选中的 candidate 至少要比正确 query 更偏向该 negative query。

### 5.5 2×2 双向反事实排序

对于挖掘出的 `(q-, z-)`，构造以下 2×2 匹配矩阵：

| | 正确 target `z+` | 反事实 candidate `z-` |
|---|---:|---:|
| 正确 query `q+` | `sim(q+, z+)` | `sim(q+, z-)` |
| 错误 query `q-` | `sim(q-, z+)` | `sim(q-, z-)` |

要求正确对角线得分高于交叉对角线：

```text
diagonal = sim(q+, z+) + sim(q-, z-)
crossed  = sim(q+, z-) + sim(q-, z+)

L_square = softplus(scale × [margin - (diagonal - crossed)])
```

本实验同样使用 `margin = 0`。

### 5.6 总损失

Typed 模型的最终损失为：

```text
L_total = L_batch
        + 0.1 × L_text-hard
        + 0.1 × L_candidate
        + 0.1 × L_square
```

严格续训对照使用：

```text
L_control = L_batch + 0.1 × L_text-hard
```

因此 control 和 typed 的初始化、数据、训练轮数、batch size、优化器及学习率完全相同，唯一实验变量是 candidate ranking 与 2×2 counterfactual loss。

## 6. 训练超参数

| 参数 | 设置 |
|---|---:|
| Epoch | 3 |
| Micro batch size | 16 |
| Gradient accumulation | 4 |
| Effective batch size | 64 |
| Composer/head learning rate | 1e-5 |
| CLIP backbone learning rate | 5e-7 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| LR schedule | warmup + cosine decay |
| Warmup ratio | 0.05 |
| Max gradient norm | 1.0 |
| Initial temperature | 0.07，可学习 |
| Mixed precision | BF16 |
| Gradient checkpointing | 开启 |
| Random seed | 42 |
| Checkpoint 保存 | epoch 2 和 epoch 3 |

Typed 分支在一个 micro batch 中最多编码：

```text
16 references + 16 targets + 16 × 4 candidates = 96 images
```

## 7. 严格对照设计

两个实验分支如下：

| 分支 | 初始化 | v5 续训 | Text hard | Candidate ranking | 2×2 square |
|---|---|---:|---:|---:|---:|
| `hard_w0_1_continue_v5` | Full Hard-0.1 epoch 10 | 3 epoch | 0.1 | 0 | 0 |
| `typed_candidate_w0_1` | Full Hard-0.1 epoch 10 | 3 epoch | 0.1 | 0.1 | 0.1 |

这样可以避免把“额外训练 3 epoch”本身误认为新算法的收益。

## 8. 训练过程指标

Typed 分支按 epoch 汇总：

| Epoch | Total loss | Hard accuracy | Candidate accuracy | Square accuracy | Mining coverage |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.1880 | 0.6888 | 0.7354 | 0.9605 | 0.4859 |
| 2 | 0.1719 | 0.7301 | 0.7649 | 0.9652 | 0.4436 |
| 3 | 0.1653 | 0.7402 | 0.7800 | 0.9665 | 0.4333 |

第三个 epoch 的分类别 mining coverage：

- Object：0.3030；
- Attribute：0.5230；
- Predicate：0.4721。

Candidate accuracy 从 73.54% 提升到 78.00%，说明模型更能把正确 target 排在组内困难候选之前。

Square accuracy 从训练开始就约为 96%，说明大部分被挖掘出来的 2×2 关系原本已经满足正确对角线约束。因此当前收益可能更多来自 candidate ranking，而不是完全来自 square loss；必须通过后续消融确认。

## 9. CIRR validation 正向检索

评估集包含 4,181 条 query。结果单位为百分比。

| 指标 | 原始 Hard-0.1 | v5 续训对照 | Typed | Typed − 对照 | Typed − 原始 |
|---|---:|---:|---:|---:|---:|
| Recall@1 | 41.3777 | 40.4927 | **42.4779** | **+1.9852** | **+1.1002** |
| Recall@5 | 75.6996 | 74.4080 | **75.6996** | **+1.2916** | 0.0000 |
| Recall@10 | 86.1038 | 85.5059 | **85.9125** | +0.4066 | -0.1913 |
| Recall@50 | 97.3212 | **96.9864** | 96.9385 | -0.0479 | -0.3827 |
| Group Recall@1 | 66.2282 | 65.5346 | **68.3808** | **+2.8462** | **+2.1526** |
| Group Recall@2 | 85.5776 | 85.6494 | **87.4432** | **+1.7938** | **+1.8656** |
| Group Recall@3 | 93.9727 | 93.6857 | **94.7142** | **+1.0285** | **+0.7415** |

结果说明：

- 普通 v5 续训导致整体退化；
- Typed 相对严格对照显著改善 R@1、R@5 和所有 Group Recall；
- Typed 相对初始化 checkpoint 也提高了 R@1 和 Group Recall；
- 收益集中在排名前部，R@50 略有下降，更像是困难候选的重排改善，而不是扩大整体召回范围。

## 10. Target image → composed query 反向检索

反向评估包含 2,206 个唯一 target、4,181 个 candidate queries 和 4,181 个正例 pair。

| 指标 | v5 续训对照 | Typed | 绝对变化 | 相对变化 |
|---|---:|---:|---:|---:|
| Mean best positive rank ↓ | 13.8939 | **13.4266** | -0.4673 | -3.36% |
| Mean positive rank ↓ | 24.2430 | **23.3444** | -0.8986 | -3.71% |
| MRR ↑ | 0.3574 | **0.3875** | +0.0300 | **+8.41%** |
| Target Hit Rate@1 ↑ | 16.0018 | **19.5830** | +3.5812 | **+22.38%** |
| Target Hit Rate@5 ↑ | 60.2448 | **63.5993** | +3.3545 | +5.57% |
| Target Hit Rate@10 ↑ | 76.9719 | **79.0571** | +2.0852 | +2.71% |
| Pair Recall@10 ↑ | 60.3923 | **62.9036** | +2.5113 | +4.16% |
| mAP@10 ↑ | 0.2803 | **0.3066** | +0.0264 | **+9.42%** |
| NDCG@10 ↑ | 0.3912 | **0.4181** | +0.0268 | **+6.85%** |

反向检索是本实验最明显的提升，说明 image-aware 双向训练确实增强了 target image 与正确 composed query 的互相对应关系。

## 11. PinPoint 官方评估

使用 PinPoint 全量 7,635 条官方 query，执行全库检索，并调用 PinPoint 官方 evaluator。这里：

- `mAP@10` 越高越好；
- `NegRecall@10` 表示被明确标记为 negative 的图片进入 Top-10 的比例，因此越低越好。

| 模型 | mAP@10 ↑ | NegRecall@10 ↓ |
|---|---:|---:|
| v5 续训对照 | 0.0684 | 0.1653 |
| Typed | **0.0692** | **0.1552** |
| 绝对变化 | **+0.0008** | **-0.0101** |
| 相对变化 | +1.17% | -6.11% |

逐 query 配对 bootstrap（20,000 次，seed 42）结果：

| 指标 | Typed − Control | 95% bootstrap CI |
|---|---:|---:|
| AP@10 | +0.000861 | `[+0.000142, +0.001583]` |
| NegRecall@10 | -0.010092 | `[-0.011171, -0.009014]` |
| Positive Precision@1 | +0.009692 | `[+0.006942, +0.012574]` |

三个区间均未跨过零。在 7,635 条 query 中：

- AP@10：1,221 条改善，1,115 条下降，5,299 条不变；
- NegRecall@10：1,021 条改善，359 条下降，6,255 条不变；
- Top-1 正例命中：96 条从错变对，22 条从对变错，其余不变。

按 PinPoint `query_category` 划分：

| Category | Query 数 | AP@10 变化 | NegRecall@10 变化 |
|---|---:|---:|---:|
| Complement | 307 | +0.001015 | -0.006515 |
| Composition | 894 | -0.001893 | -0.013982 |
| Context Fit | 2,520 | **+0.004350** | **-0.014524** |
| Explore | 188 | -0.000647 | -0.004255 |
| Negation | 328 | -0.003367 | 0.000000 |
| Swap | 3,398 | -0.000525 | -0.007403 |

PinPoint 的主要收益来自：

- 更强的 Top-1 正例排序；
- 更少的官方 negative candidate 进入 Top-10；
- `Context Fit` 类别的正例 AP 提升。

它尚未表现为所有 query 类别上的统一 mAP 提升。

## 12. 当前结论

本实验支持以下结论：

1. 只继续使用 text hard negatives 训练会让 CIRR validation 性能下降；
2. 加入真实 group candidate images 后，可以阻止这种退化并改善低 K 和 Group Recall；
3. target-to-query 反向检索得到明显提升，说明训练目标与双向 image-query 对齐更加一致；
4. PinPoint mAP@10 获得较小但配对统计显著的提升；
5. PinPoint NegRecall@10 明显下降，说明模型更少把诊断负例检索到 Top-10；
6. 当前结果验证了 candidate ranking 与 2×2 square 的组合，但还不能把收益单独归因于 2×2 反事实训练。

## 13. 局限与下一步消融

当前实验仍有以下缺口：

1. **Candidate 没有 O/A/P 真值标签。** Text negative 有类型，但 candidate image 与具体错误类型之间只是在线相似度挖掘关系。
2. **Square constraint 大部分原本已经成立。** 约 96% 的已选关系满足正确对角线，因此 square loss 可能不是主要收益来源。
3. **只完成一个随机种子。** CIRR aggregate metrics 还需要 3 个或更多 seeds 报告均值和方差。
4. **只完成 CLIP-L。** BGE-VL-large 配置已准备，但尚未运行验证。
5. **缺少损失消融。** 下一步至少要运行：
   - text hard + candidate ranking，不使用 square；
   - text hard + square，不使用普通 candidate ranking；
   - 完整 typed candidate objective。
6. **需要评估 epoch 2。** 现有 epoch 2 checkpoint 已保存，不需要重新训练，可用于判断第 3 个 epoch 是否必要。

只有完成 candidate-only 与 square-only 消融后，才能严谨判断论文贡献应该表述为“真实候选图 hard-negative training”，还是更强的“typed bidirectional counterfactual training”。

## 14. 复现实验

训练 CLIP-L control 与 typed 分支，并运行 CIRR、反向检索和 PinPoint：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train

CUDA_VISIBLE_DEVICES=0 \
PYTHON=/mnt/data/datasets/search_dataset/zxl/multi_modal_py3.12/bin/python \
MODELS=clip_l \
RUNS="control typed" \
RUN_PINPOINT=1 \
bash scripts/run_typed_candidate_training.sh
```

主要实现文件：

- `src/cir_qdiag_train/data.py`：CIRR group candidate image 加载；
- `src/cir_qdiag_train/model.py`：candidate image 编码；
- `src/cir_qdiag_train/losses.py`：candidate ranking 与 2×2 counterfactual loss；
- `src/cir_qdiag_train/trainer.py`：总损失和训练指标接入；
- `configs/clip_l_full_hard_w0_1_continue_v5.yaml`：严格续训对照；
- `configs/clip_l_full_typed_candidate_w0_1.yaml`：完整新方法；
- `scripts/run_typed_candidate_training.sh`：一键训练和评估脚本。

主要结果文件：

- `outputs/clip_l/full/hard_w0_1_continue_v5/val_metrics.json`；
- `outputs/clip_l/full/typed_candidate_w0_1/val_metrics.json`；
- `outputs/clip_l/full/hard_w0_1_continue_v5/reverse_val_metrics.json`；
- `outputs/clip_l/full/typed_candidate_w0_1/reverse_val_metrics.json`；
- `outputs/typed_candidate_pinpoint_official/official_map10_negrecall10.csv`。
