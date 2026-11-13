# 原始 epoch Top-32：过滤、target 注入与 CMSE

复用 `outputs/opd_k_tau_kl_ablation/k32_reverse_t005` 的 Reverse KL / Top-10 / 强制注入
基线，不修改该目录。新增四组，均原始 CLIP-L、S2 Teacher、full FT、seed 42、4 epoch、
batch 32、head LR 1e-5 / backbone LR 5e-7。每 epoch 重新检索 Top-32，无 128→32。

| RUN_INDICES | 名称 | 损失 | 教师过滤 | 注入 target |
|---|---|---|---|---|
| 0 | kl_all_inject | Reverse KL，T=.05 | None | 是 |
| 1 | kl_top10_natural | Reverse KL，T=.05 | Top-10 | 否 |
| 2 | kl_all_natural | Reverse KL，T=.05 | None | 否 |
| 3 | cmse_top10_inject | Centered MSE | Top-10 | 是 |

## 无注入组的过滤定义

先排除 reference，按当前学生排名取自然 Top-32。实际 KL 只在这 32 个候选上计算，
target 可以缺席或位于任意位置；不为缺席 target 加入学生前向或额外监督项。
教师过滤另用虚拟集合 `target + Student Top-31 non-target`，独立计算 target 教师排名。
该集合仅用于资格筛选，不替换 KL 支持集。target 缺席不自动丢弃该 query。
None 组也保存该排名供诊断，但不用它筛选。教师分数使用与旧实验相同的缓存精度。

自然候选组的 `teacher_filter_target_*` 表示虚拟集合上的排名，
`teacher_target_top1_rate` 表示真实 KL 候选集合中 target 是否为教师第一名（缺席为 false）。
不要把两种统计混写。两个训练得到的学生会分化，因此候选策略相同不保证后续候选 ID 相同。

CMSE 为 `mean(((s-t)-mean_candidates(s-t))**2)`；不是 Smooth L1、不做 softmax、
不按温度缩放。先运行原始尺度 1 的匹配对照，不预先承诺已完成最优损失尺度搜索。

## 启动与恢复

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
source /mnt/data/datasets/search_dataset/zxl/multi_modal_py3.12/bin/activate
bash scripts/run_epoch_target_filter_e4.sh --dry-run
nohup bash scripts/run_epoch_target_filter_e4.sh > epoch_target_filter.log 2>&1 &
```

两容器可分别指定 `RUN_INDICES="0 1"` 与 `RUN_INDICES="2 3"`，使用不同重定向日志。
每个实验有独立锁和输出目录。同组不要重复启动。
脚本自动从 `last.pt` 恢复已完成 epoch；当前未完成 epoch 重跑，不是逐 batch 恢复。
默认评估 CIRR/FashionIQ E1–E4，PinPoint/QDiag E4；支持跳过已完成阶段。
推理加载继续支持 `CIR_INFERENCE_NUM_WORKERS=8`。`RUN_EXTENDED_EVAL=0` 仅跳过最终
PinPoint/QDiag，FashionIQ 不跳过。`RUN_SUMMARY=0` 可暂时不汇总。

输出为 `outputs/epoch_target_filter_e4/{name}`，不与旧实验共用候选缓存。
所有结果可用以下命令汇总（只读旧基线结果）：

```bash
python scripts/run_epoch_target_filter.py --summarize-only
```

候选缓存明确记录注入模式，无注入缓存还包含 `teacher_filter_target_ranks.npy`。
训练代码禁止将无注入模式用于 InfoNCE、RL、HTTP Teacher、混合 q- 或 live reservoir。
