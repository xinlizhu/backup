# CIR-QDiag / OPD 项目 Codex 交接文档

更新时间：2026-09-02

这份文件用于 Codex 重装、对话丢失或更换 Agent 后恢复上下文。新 Agent 应先完整阅读本文，
再按“权威结果文件”章节读取专项报告。不要只根据 README 判断当前研究状态；README 包含大量
早期训练方案，当前论文主线已经转向 CIR-QDiag benchmark 与 rank-based OPD。

## 1. 项目位置与环境

项目根目录：

```text
/mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
```

常用 Python 环境：

```text
/mnt/data/datasets/search_dataset/zxl/multi_modal_py3.12
```

重新安装项目依赖：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
source /mnt/data/datasets/search_dataset/zxl/multi_modal_py3.12/bin/activate
python -m pip install -e '.[dev]'
```

CLIP 本地权重：

```text
models/clip-vit-base-patch16
models/clip-vit-large-patch14
```

Qwen3-VL 本地权重：

```text
/mnt/data/datasets/search_dataset/zxl/CIR_PLAY/models/Qwen3-VL-Embedding-2B
/mnt/data/datasets/search_dataset/zxl/CIR_PLAY/models/Qwen3-VL-Embedding-8B
```

当前项目目录未检测到可用的独立 Git 仓库，因此不要假设代码改动可以从 Git 恢复。Codex
重装通常不应删除 `/mnt/data`，但如果环境重装会清理该挂载，必须先备份本文件、`src/`、
`configs/`、`scripts/` 和需要保留的最终 checkpoint。

## 2. 研究目标与当前论文故事

### 2.1 CIR 任务

Composed Image Retrieval（CIR）的输入是一张 reference image 和一条修改文本，模型需要
检索满足修改文本的 target image：

```text
(reference image, modification text) -> target image
```

标准评估包括 CIRR Recall、CIRR Group Recall、FashionIQ Recall 和 PinPoint 官方检索指标。

### 2.2 CIR-QDiag benchmark 贡献

核心假设：现有 CIR embedding 模型不能稳定理解修改文本中的细粒度 Object、Attribute 和
Predicate 语义。CIR-QDiag 将一条正确修改文本局部编辑为反事实文本，检查同一个 target 对
正确 query 与错误 query 的响应方向：

$$
s(q^+,t)>s(q^-,t).
$$

语义类型和编辑操作构成最多 $3\times3$ 的诊断网格：

```text
semantic target: Object / Attribute / Predicate
edit operation:  ADD / DELETE / SUBSTITUTE
```

ADD/SUBSTITUTE 是严格反事实；DELETE 是移除约束后的更弱 query，应单独解释。

### 2.3 OPD 方法贡献

第二个贡献是在 Student 当前检索到的 hard candidate 集合上进行 on-policy rank
distillation。Student 负责暴露自己的决策边界，Teacher 负责提供候选排序分布。主要目标是
同时提高标准 CIR retrieval 和 CIR-QDiag 细粒度语义能力。

当前论文范围应继续以 CIR 为中心，不要轻易扩张为“通用 embedding 理论”。FashionIQ 是
跨 CIR 数据集迁移证据，而不是通用 embedding 任务声明。

论文/导师汇报只讲 benchmark、rank OPD、必要消融和成功结果。不要把 centered、joint、
hard-negative、OAP、失败 RL 等历史探索塞进主故事，除非用户明确要求写实验分析或附录。

## 3. 用户明确偏好与禁区

后续 Agent 必须遵守：

1. 当前方法只讨论 rank，不把 centered response 和 joint 作为主方法。
2. 用户在 2026-08-31 明确要求：后续新 OPD 设计不要使用 Forward KL，统一采用 Reverse
   KL。历史结果文件中确实存在 Forward/Reverse 两组已完成实验，必须保留真实标签，不能
   追溯性改名或伪造口径。
3. 不要主动把 q- negative candidates 拉回当前 RL/OPD 讨论。用户曾明确说“不要再和我聊
   q- 之类的东西”。只有用户主动询问时再分析。
4. 不要建议 RL 冻结 image encoder。用户认为冻结特定层容易被 challenge。
5. 不要建议每个 minibatch 都重新做完整 gallery 检索；用户认为代价不可接受。
6. 不需要传统机器学习 probe/六通道分类器来做主要诊断。优先报告 Recall、Group Recall、
   mAP@10、NegRecall@10、NDCG、Delta、Raw Win、Stable Win。
7. 不要把 reranker 说成自己检索 Top-K。正确流程是 Student/embedding retriever 召回候选，
   reranker 只对候选打分。
8. Agent 环境不要假设有可用 GPU。实现代码、做 CPU 测试，然后把 GPU 命令交给用户运行。
9. 用户常有两个彼此独立但共享 `/mnt/data` 的单卡容器。并行实验必须使用不同输出目录和
   不同日志，不能两边原样执行会写同一 checkpoint 的脚本。
10. 用户重视论文可辩护性：避免随意冻结层、隐含过滤、手工挑选单个超参却不说明、改变
    评估口径或把不同候选集合直接比较。

## 4. 数据与缓存

### 4.1 CIRR

```text
/mnt/data/datasets/search_dataset/zxl/CIR_new/datasets/CIRR
```

训练 annotation：

```text
CIR_new/datasets/CIRR/cirr/captions/cap.rc2.train.json
```

验证 annotation：

```text
CIR_new/datasets/CIRR/cirr/captions/cap.rc2.val.json
```

### 4.2 FashionIQ

```text
/mnt/data/datasets/search_dataset/zxl/CIR_new/datasets/fashion-iq
```

FashionIQ 标准摘要使用 dress、shirt、toptee 三个类别的宏平均，主要报告 R@10/R@50。

### 4.3 PinPoint

原始数据项目：

```text
/mnt/data/datasets/search_dataset/zxl/pinpoint-dataset
```

本地诊断/评估代码：

```text
/mnt/data/datasets/search_dataset/zxl/benchmark_build/diag_model
```

常用 QDiag 数据：

```text
/mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_openai/outputs/pinpoint_pilot_500_v7/final.jsonl
/mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_openai/outputs/pinpoint_pilot_500_v7/groups.jsonl
```

PinPoint 论文指标和本地 BGE-VL-MLLM-S2 结果曾出现口径差异。未来 Agent 在引用论文数字前
必须重新核对 `pinpoint-dataset` 的协议、checkpoint、gallery 和 query 定义，不能直接断言
本地结果复现论文。

### 4.4 CIRR train 的 q- 生成结果（当前主方法不使用）

```text
/mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag/outputs/train_all_v5/final_no_va.jsonl
```

### 4.5 BGE-VL-MLLM-S2 Teacher cache

```text
outputs/on_policy_centered/cache/bge_mllm_s2_cirr_train
```

缓存已经包含 CIRR train gallery/query embeddings、target ranks 和 negative-query features。
使用 embedding-cache Teacher 的实验不需要重新运行 S2 模型。

## 5. CIR-QDiag 构造统计

完整构造说明的权威来源是 `ALL_MODELS_RESULTS.md` 顶部。摘要如下：

- 原始 PinPoint：7,635 条 query。
- 按 Reference 1、可选 Reference 2、完整 Positive IDs、完整 official Negative IDs 精确分组。
- 保留 2–6 条 instruction 的组：1,683 组、6,553 条 instruction。
- seed 42 抽取500组，包含1,933条原始正向 instruction。
- 候选生成模型：`gpt-5.6-terra`；生成时不看图像。
- group-level vision-grounded 审核时看 reference、最多4张 positive 和文本，但不看官方
  negative image，避免标签泄漏。
- 共生成11,743条编辑候选，审核通过5,576条，平均每条原 instruction 2.89条。

通过样本分布：

| Semantic target | ADD | DELETE | SUBSTITUTE | Total |
|---|---:|---:|---:|---:|
| Object | 947 | 643 | 767 | 2,357 |
| Attribute | 411 | 1,383 | 567 | 2,361 |
| Predicate | 240 | 308 | 310 | 858 |
| Total | 1,598 | 2,334 | 1,644 | 5,576 |

严格 ADD+SUBSTITUTE 集合共3,242条。

## 6. 指标口径

### 6.1 CIRR

- `R@K`：target 是否进入完整 validation gallery 的 Top-K。
- `Group R@K`：只在 CIRR 为该 query 提供的小型语义相近 group 内排序。它比全图库 Recall
  更聚焦细粒度区分，不能和完整 gallery R@K 混写。

### 6.2 PinPoint

- `mAP@10`：官方 positive image 集合在前10中的平均精度。
- `NegRecall@10`：官方 negative image 是否被模型前10检索结果覆盖的协议指标；引用前应
  以本项目评估脚本定义为准。

### 6.3 QDiag

- `Delta`：$s(q^+,t)-s(q^-,t)$ 的平均差。
- `Raw Win`：Delta 大于0的比例。
- `tau95`：同义改写造成分数波动的95%阈值。
- `Stable Win`：Delta 超过 `tau95` 的比例。
- `Own-Random`：模型自己的 query-target 分数相对随机不相关图片的间隔。

## 7. 原始 OPD 实现和历史口径

候选集合：

$$
\mathcal C_\theta^K(x)
=
\{y^+\}
\cup
\operatorname{TopK}_{K-1}
\{s_\theta(x,y):y\in\mathcal G\setminus\{y^+\}\}.
$$

通常使用 Student 每个 epoch 动态检索、强制 target、BGE-VL-MLLM-S2 Teacher、Teacher
target Top-10 过滤、temperature 0.05、CLIP-L full finetune。

历史主 checkpoint：

```text
outputs/rank_candidate_policy_e4/dynamic_student_rank/checkpoint_epoch_004.pt
```

历史代码和结果表把该 checkpoint 标为 Teacher-to-Student KL；K/tau/KL direction 消融也
真实包含 Forward 与 Reverse 两个方向。用户当前要求未来方法只使用 Reverse KL，因此：

- 读取历史结果时按文件中的真实 direction 报告；
- 新实验配置必须显式写 `rank_kl_direction: reverse`；
- 不要把历史 Forward 数字重新命名为 Reverse；
- 论文最终采用哪个历史 checkpoint，需要用户最终确认后再统一叙述。

## 8. 权威结果文件

按优先级读取：

1. `ALL_MODELS_RESULTS.md`：CIR-QDiag 构造、OPD 主结果、PinPoint、QDiag、FashionIQ、
   Student/Teacher 矩阵。
2. `RL_MODELS_RESULTS.md`：旧 RL 方法公式、CIRR/FashionIQ、OPD K/temperature/KL
   direction 消融。
3. `outputs/reward_tilted_reverse_beta0_1_e4/RESULTS.md`：最新 NDCG-tilted reverse
   OPD/GSPO。
4. `outputs/live_reservoir_512_anchor_kl_e4_cached/RESULTS.md`：Top-512 reservoir、
   GRPO/GSPO、adaptive anchor KL。
5. `ALL_EXPERIMENTS_RECORD.md`：早期 hard-negative、Full/LoRA、OAP、rerank、BGE 等完整
   历史记录。
6. `writing/检索RankKD的TopK_OPD公式推导.md`：OPD 推导。
7. `writing/CIR_QDiag_OPD_导师汇报.html`：导师汇报页面。

不要手工从多个旧 JSON 拼出结论而忽略上述汇总；如数字冲突，以对应输出目录的原始 JSON
为准，并重新运行该目录的 summarize 脚本。

## 9. 当前关键基线数字

### 9.1 CIRR val / FashionIQ

| Model | CIRR R@1 | CIRR R@10 | Group R@1 | FashionIQ R@10 | FashionIQ R@50 |
|---|---:|---:|---:|---:|---:|
| CLIP-L original | 12.7960 | 51.1600 | 34.5133 | 20.2010 | 36.4484 |
| Dynamic Student Rank E4 | 42.4540 | 87.1562 | 67.7828 | 31.6529 | 52.4326 |
| Dynamic Student Rank E8 | 44.0804 | 88.4238 | 70.1507 | 31.3227 | 52.3614 |
| BGE-VL-MLLM-S2 Teacher | 49.4140 | 89.2849 | 75.1017 | 34.1685 | 55.1285 |
| K32 Reverse, T=0.05 E4 | 42.6214 | 87.1083 | 67.3762 | 31.5293 | 52.2481 |
| K8 historical best E4 | 43.7694 | 87.4193 | 68.8591 | 32.2379 | 52.9462 |

### 9.2 PinPoint / QDiag 关键行

| Model | PinPoint mAP@10 | PinPoint NegRecall@10 | QDiag Delta | Raw Win | Stable Win |
|---|---:|---:|---:|---:|---:|
| CLIP-L original | 0.0462 | 0.2056 | 0.006177 | 0.661461 | 0.020157 |
| Dynamic Student Rank E4 | 0.0886 | 0.1740 | 0.025387 | 0.780768 | 0.098707 |
| BGE-VL-MLLM-S2 | 0.1262 | 0.1219 | 0.021749 | 0.742554 | 0.091780 |

Student 在某些 QDiag 指标上超过 Teacher 并不矛盾：训练只蒸馏 Teacher 在 Student hard
candidates 上的局部排序，Student 的架构、归纳偏置、数据分布和评估目标都不同。Teacher
不是 QDiag 指标的直接最优解。

## 10. OPD K / temperature / direction 消融

完整表：`outputs/opd_k_tau_kl_ablation/RESULTS.md`。

已完成：

```text
K=8,16,32 with T=0.05
K=32 with T=1
K=32 Forward/Reverse
```

K=64 未产生有效 checkpoint，不应放进完成结果表。

历史数值上 K=8、T=0.05 的 CIRR/FashionIQ 最好；K32 Reverse T=0.05 与历史 Forward
基线非常接近。T=1 整体稍差。注意这些是历史消融事实，不代表未来新方法允许 Forward KL。

## 11. 旧 RL 结论

`RL_MODELS_RESULTS.md` 记录 Full-sequence REINFORCE、causal-prefix REINFORCE、GRPO、
GSPO。主要结论：

- 从原始 CLIP-L 做 target-only RL 能学到 CIRR 信号，但仍远低于 OPD。
- 从 OPD E4 继续 target-only RL，CIRR 和 FashionIQ 均退化。
- causal-prefix 比 full-sequence credit assignment 更合理，但不能消除退化。
- 局部强制包含 target 的 Top-32 reward 会过度优化局部 target rank，破坏完整语义排序和
  跨数据集泛化。

Top-512 reservoir 实验：

```text
outputs/live_reservoir_512_anchor_kl_e4_cached/RESULTS.md
```

- Teacher-free GRPO/GSPO 随 epoch 继续退化。
- adaptive anchor KL 的 epoch 1 一度达到 CIRR R@1 43.48/43.60、FashionIQ R@10
  32.10/32.30，但继续训练后下降。
- 用户后来拒绝把固定冻结或昂贵的每-batch完整图库重检索作为核心修复方案。
- `scripts/run_live_reservoir_512_anchor_kl_e4.sh` 已支持 `RUN_INDICES` 和
  `RUN_SUMMARY`，供两个容器安全拆分运行。

## 12. 最新 NDCG-Tilted Reverse-KL 实验

输出：

```text
outputs/reward_tilted_reverse_beta0_1_e4
```

两组均从原始 CLIP-L 开始、full finetune 4 epochs、Student 动态 Top-32、强制 target、
16条 target-stopped Top-10 rollout、BGE-VL-MLLM-S2 cache、T=0.05、tilt beta=0.1，
不加额外 anchor KL。

### 12.1 方法 A：Reward-Tilted Reverse-OPD

在16条 Student rollout 的 group 支持集上构造：

$$
q_g\propto
\frac{\pi_T(\tau_g)}{\pi_{\mathrm{old}}(\tau_g)}
\exp(R_g/\beta),
$$

直接最小化 group reverse KL：

$$
D_{\mathrm{KL}}(p_\theta^{group}\|q^{group}).
$$

保留 Teacher target Top-10 过滤，每批只更新一次。

### 12.2 方法 B：Reward-Tilted Reverse-KL GSPO

统一 shaped reward：

$$
\widetilde R_g
=R_g-\beta\log\frac{\pi_{\mathrm{old}}(\tau_g)}{\pi_T(\tau_g)}.
$$

对同一 query 的16条 shaped rewards 做组内标准化，然后使用 sequence-level GSPO ratio、
clip 0.2 和两次 rollout reuse 更新。无 Teacher 过滤，无额外 KL。

### 12.3 最终结果

| Model E4 | CIRR R@1 | R@5 | R@10 | R@50 | Group R@1 | FashionIQ R@10 | R@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tilted Reverse-OPD | 39.8708 | 74.3124 | 85.8885 | 97.1060 | 64.4343 | 30.6912 | 51.5295 |
| Tilted Reverse-KL GSPO | 39.3207 | 74.9103 | 86.0320 | 97.2016 | 64.8170 | 29.0882 | 50.0857 |
| Original K32 Reverse T=.05 | 42.6214 | 76.2497 | 87.1083 | 97.4169 | 67.3762 | 31.5293 | 52.2481 |

结论：两种 tilted 方法都显著优于原始 CLIP-L，但都没有超过最简单的 K32 Reverse OPD。
直接 Reverse-OPD 的迁移表现优于 GSPO。

诊断：

- Reverse-OPD 的16条 rollout ESS 长期只有约2.5，最大单条权重约60–62%，目标分布过尖。
- GSPO 四轮 `clip_fraction=0`；importance ratio 始终未越过 `[0.8,1.2]`，clipping 没有
  实际触发。
- 两者 CIRR train Raw Target R@32 最终约87–88%，但 val 仍低于原始 OPD，证明训练集
  target coverage 不等于可泛化的细粒度排序。
- 不加 anchor KL 没有导致崩溃，因此失败不能简单归因于缺少防漂移 KL。
- 总运行时间约29,795秒，即8小时16分钟。

最新代码文件：

```text
src/cir_qdiag_train/on_policy_config.py
src/cir_qdiag_train/on_policy_loss.py
src/cir_qdiag_train/on_policy_trainer.py
configs/clip_l_reward_tilted_reverse_opd_e4.yaml
configs/clip_l_reward_tilted_reverse_gspo_e4.yaml
scripts/run_reward_tilted_reverse_e4.sh
scripts/summarize_reward_tilted_reverse.py
```

新增 loss type：

```text
reward_tilted_reverse_opd
reward_tilted_reverse_gspo
```

日志中 `reward_tilt_beta=0.1000977` 是 BF16 表示 `0.1` 的量化误差，不是配置被改变。

## 13. 最新实验运行与恢复

两组已经完成，无需重跑。若结果文件需要重新汇总：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
python scripts/summarize_reward_tilted_reverse.py \
  --output-root outputs/reward_tilted_reverse_beta0_1_e4
```

脚本支持断点续跑和两个容器拆分：

```bash
# container A: direct Reverse-OPD
RUN_INDICES=0 RUN_SUMMARY=0 bash scripts/run_reward_tilted_reverse_e4.sh

# container B: Reverse-KL GSPO
RUN_INDICES=1 RUN_SUMMARY=0 bash scripts/run_reward_tilted_reverse_e4.sh
```

两边共享同一 `/mnt/data` 时输出目录不冲突。完成后只在一边运行 summarize。

## 14. 工程注意事项

1. CIRR 中存在超大图，会出现 PIL `DecompressionBombWarning`。这是警告，不一定是错误；
   GPU 利用率低通常来自图片解码、I/O、候选建库和 CPU->GPU 传输。
2. 用户容器曾发生 DataLoader worker 被 `Killed`。cgroup 历史 `oom_kill=12`，但容器实际
   memory limit 约128 GiB，不是32 GiB。K=64 高并发图片预取容易触发内存峰值。
3. 稳定设置曾使用 `num_workers=4, prefetch_factor=1`，但速度较慢；之后放弃 K=64，改做
   K=8，并恢复更高并发。不要把 OOM 简单归因于 GPU 显存。
4. Top-512 pixel cache 参数曾使用：12 workers、build batch128、prefetch3、transfer
   prefetch2、float16。不要把不用的缓存持久化到硬盘；用户明确要求临时缓存用完释放。
5. `PYTHONUNBUFFERED=1` 只保证 nohup 日志及时刷新，不影响算法，可以保留。
6. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 用于缓解 CUDA allocator 碎片，不是
   训练必需条件。
7. checkpoint 体积很大。当前主要输出目录大致：最新 tilted 24 GiB、Top-512 RL 46 GiB、
   rank candidate ablation 72 GiB。删除前必须确认每个有效实验只保留哪个最终推理
   checkpoint；不要擅自删除用户文件。

## 15. 测试状态

截至最新改动：

- 新 Reverse-OPD / Reverse-KL GSPO 损失的 CPU 梯度测试通过。
- causal REINFORCE/GRPO/GSPO 和 anchor-KL 回归测试通过。
- `tests/test_config.py`：24项通过。
- Ruff 静态检查通过。
- `bash -n scripts/run_reward_tilted_reverse_e4.sh` 通过。

相关测试位于：

```text
tests/test_loss_and_model.py
tests/test_config.py
```

## 16. 下一位 Agent 的启动清单

1. 完整阅读本文件。
2. 阅读 `ALL_MODELS_RESULTS.md` 和 `RL_MODELS_RESULTS.md`，不要只看 README。
3. 检查用户最新问题是否要求分析、改代码、跑评估或写论文；不要自动启动 GPU 任务。
4. 若继续方法设计，先明确是在比较 candidate-level Reverse OPD、rollout-level direct
   Reverse KL，还是 GSPO policy surrogate，不能把三者都简称为“KL”。
5. 若提出新 loss，必须说明 Teacher similarity 如何转为 probability、NDCG 在何处进入、
   KL direction、proposal correction、Teacher filtering、候选来源和是否有固定 anchor。
6. 所有新模型默认从用户指定 checkpoint 开始；若用户说“原始 CLIP-L”，配置中的
   `experiment.initial_checkpoint` 必须是 `null`。
7. 新实验至少自动评估 CIRR val 和 FashionIQ；PinPoint/QDiag 是否运行由用户确认或按已有
   脚本显式配置。
8. 分析时同时报告最终 checkpoint 和最佳 epoch，尤其关注 CIRR 提升但 FashionIQ 下降的
   dataset-specific overfitting。
9. 不要仅根据 train Raw Target R@32 判断成功；必须看 val R@1/R@10、Group Recall 和
   FashionIQ。

## 17. 给新 Codex 的建议首条提示词

重装后用户可以把下面这段直接发给新 Agent：

```text
请先完整阅读
/mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train/CODEX_HANDOFF.md
再阅读其中列出的权威结果文件。不要重跑任何实验，也不要修改历史实验口径。读完后告诉我：
当前论文故事、当前主方法、最新两组 NDCG-tilted 实验结论，以及你认为下一步最需要确认的
一个问题。
```
