# CLIP-L Reverse Rank 五组实验

五组均从原始 CLIP-L 开始（`initial_checkpoint: null`），S2 embedding-cache Teacher，
全量微调 4 epochs，seed 42，batch 32，T=0.05，候选级
`KL(softmax(student/T) || softmax(teacher/T))`。只使用 rank loss。

| RUN_INDICES | 名称 | 候选来源与刷新 | Teacher target 筛选 |
|---|---|---|---|
| 0 | static | 初始化 Student 的 target + Top-31，固定四轮 | 最终 32 中 Top-10 |
| 1 | teacher | Teacher 的 target + Top-31，固定四轮 | 最终 32 中 Top-10 |
| 2 | random | 每 epoch 重采样 target + 31 张随机图 | 最终 32 中 Top-10 |
| 3 | no_filter | 每 epoch Student 建 128；每步当前 Student 选 32 | 无 |
| 4 | opd | 每 epoch Student 建 128；每步当前 Student 选 32 | 最终 32 中 Top-10 |

`no_filter` 与 `opd` 除实验名、输出目录和 Teacher 筛选外配置完全相同。
Static/Teacher/Random 是 32 候选对照；因此与 OPD 的比较还涉及候选池大小和刷新方式，
不应称为仅改变刷新频率的单变量对照。历史 epoch-level Reverse Top-32 可作为额外已有参照。

## 128 → 32 的具体计算

1. 每个 epoch 开始，用当前 Student 编码图库与 query；排除 reference，构造
   `target + 当前 Student Top-127 non-target`，共 128 张。
2. 使用已有冻结 S2 gallery/query embeddings，计算这 128 张图的 Teacher 分数。
   Teacher 分数与候选 ID 对齐，保存在当前 epoch 的内存中。
3. 每个 minibatch 开始，用更新后的 Student 在 eval/no-grad 模式下重新编码该 batch
   的 reference、文本和 128 张候选图；选择 `target + 当前 Top-31 non-target`。
   此处只缓存像素，不复用旧 Student 图像特征。
4. 按相同索引 gather Teacher 分数；OPD 在这 32 张候选上计算 Teacher target rank，
   保留 rank ≤ 10 的 query。无筛选组使用全部 query。
5. Student 恢复 train 模式，对选中的 32 张图重新做带梯度的前向，在该集合上重新
   softmax 并计算 Reverse KL；每 batch 更新一次，全量更新图像与文本编码器。

沿用 RL 的候选选择工程实现；训练中不采样排序 rollout、不使用 NDCG reward、
不使用 GRPO/GSPO 或 anchor KL。候选池包含强制 target，因此日志同时报告强制注入前的
raw target coverage，不能把候选内 target 恒在集合中当作完整图库 Recall。

128 候选池、对应 Teacher 分数与筛选中间特征只在内存中存在；epoch 结束释放。
像素缓存覆盖训练 gallery，在进程中复用，训练结束释放。Static/Teacher/Random 的
候选缓存写入各自目录，Static/Teacher 断点恢复依赖首次候选缓存。

## 运行

新启动的进程默认对 CIRR / FashionIQ 评估及普通 Student 建库启用 8 个 CPU worker，
并行读图与预处理。可用 `CIR_INFERENCE_NUM_WORKERS` 单独调整，不改变训练配置。
共享内存要求、断点续评和回退方式见 [INFERENCE_WORKERS.md](INFERENCE_WORKERS.md)。

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
source /mnt/data/datasets/search_dataset/zxl/multi_modal_py3.12/bin/activate

# 只检查五组配置，不加载模型或启动训练
bash scripts/run_reverse_rank_r128_e4.sh --dry-run

# 单卡顺序运行全部五组
nohup bash scripts/run_reverse_rank_r128_e4.sh > reverse_rank_r128.log 2>&1 &
```

默认每个模型训练完成后，评估 E1–E4 的 CIRR val 与 FashionIQ val，再评估 E4 的
PinPoint official 和 QDiag，随后进入下一模型。FashionIQ 始终评估；如暂时跳过最终
PinPoint/QDiag，使用 `RUN_EXTENDED_EVAL=0`。以后用默认设置重跑脚本即可补评估。

两台共享 `/mnt/data` 的单卡容器可拆开执行：

```bash
# 容器 A：Static、Teacher、无筛选
RUN_INDICES="0 1 3" RUN_SUMMARY=0 nohup bash scripts/run_reverse_rank_r128_e4.sh > reverse_rank_r128_A.log 2>&1 &

# 容器 B：Random、OPD
RUN_INDICES="2 4" RUN_SUMMARY=0 nohup bash scripts/run_reverse_rank_r128_e4.sh > reverse_rank_r128_B.log 2>&1 &

# 两边完成后，在任意一边汇总
python scripts/run_reverse_rank_r128.py --summarize-only
```

支持 `RUN_INDICES="4"` 单独运行 OPD。每组训练、评估、日志均写自己的子目录，
同组有进程锁防止重复启动。脚本检测 E4 checkpoint、逐轮评估文件和最终诊断文件，
已经完成的阶段自动跳过；存在 `last.pt` 时从最近完成的 epoch 恢复，未完成的 epoch
重新执行。恢复时检查有效配置，避免将新配置写入旧实验目录；恢复并非 minibatch 级精确续跑。

## 输出与调参

输出根目录：`outputs/reverse_rank_r128_e4/`。

每个 `{static,teacher,random,no_filter,opd}` 子目录包含：

- `runner.log`、`train_metrics.jsonl`、`effective_config.json`、`gradient_audit.json`。
- E1–E4 推理 checkpoint、带优化器状态的 `last.pt`。
- `val_epoch_*.json`、`fashioniq/epoch_*/metrics.json`。
- `pinpoint_official/`、`qdiag/`：E4 评估与各自缓存。

总表为 `all_metrics.csv`、`RESULTS.md`，报告全部 epoch、最终值及 CIRR/FashionIQ 各自
最佳 epoch；尚未完成的评估标为缺失。

配置文件：`configs/clip_l_reverse_rank_{名称}_r128_e4.yaml`。若需要控制候选筛选时的
显存峰值，调整 `mining.live_selection_image_batch_size`（默认 512）；选中图像的
梯度前向分块由 `optimizer.live_image_batch_size`（默认 256）控制。筛选分块不改变 K。
建议在首次启动前调整并保持 `no_filter` 与 `opd` 的这些配置一致。

脚本只读取现有 S2 Teacher 缓存，不自动重建 Teacher：
`outputs/on_policy_centered/cache/bge_mllm_s2_cirr_train`。
