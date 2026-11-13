# 补齐主表：九组训练

已有 OPRD 不因 KL 方向重训；唯一新增 OPRD 使用 reverse KL。
本次不修改 PAPER_MAIN_TABLES.md 中的已有结果，不启动训练。

| 编号 | 学生 | 方法 | 教师 |
|---|---|---|---|
| 0 | CLIP-L | Random InfoNCE | 无 |
| 1 | CLIP-B | Random InfoNCE | 无 |
| 2 | Qwen3-2B | Random InfoNCE | 无 |
| 3 | CLIP-L | Random CMSE | Qwen-8B |
| 4 | CLIP-B | Random CMSE | S2 |
| 5 | CLIP-B | Random CMSE | Qwen-8B |
| 6 | Qwen3-2B | Random CMSE | S2 |
| 7 | Qwen3-2B | Random CMSE | Qwen-8B |
| 8 | Qwen3-2B | OPRD，reverse KL | S2 |

所有实验 seed=42、4 epochs、每轮保存；均从原始预训练学生初始化。
CLIP full FT，batch=32，head LR=1e-5、backbone LR=5e-7。
Qwen 保留现有 query-LoRA：rank=16、alpha=32、LR=2e-5、microbatch=8、累积4步；
图库是原始 Qwen-2B 的冻结编码，与已有 OPRD 一致。

## 候选与损失

- InfoNCE 和 CMSE：target 在位置0，加31个不重复随机负例；排除 reference 和 target，
  每 epoch 使用 seed+epoch 重采样。**不做学生召回，不做教师资格筛选**。
- InfoNCE：在这32个显式候选上做 target one-hot CE，温度0.07；不是 in-batch InfoNCE。
  不读取教师缓存。Qwen 配置中的 teacher_cache_dir 是兼容旧 schema 的占位路径，不参与此训练。
- CMSE：直接计算 `mean(((s-t)-mean_candidates(s-t))**2)`，尺度1，无 softmax 或温度缩放。
- OPRD：每 epoch 当前学生 Top-32，强制 target 注入，Teacher target Top-10过滤，reverse KL，T=0.05。
- 无128→32、无RL、无reranker。所有 baseline 随机候选由同一采样函数生成。

## 启动

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
source /mnt/data/datasets/search_dataset/zxl/multi_modal_py3.12/bin/activate
python scripts/run_main_comparison_e4.py --dry-run
nohup python scripts/run_main_comparison_e4.py > main_comparison_e4.log 2>&1 &
```

按编号分组执行，例如两个容器分别运行（每个容器应分配独立 GPU）：

```bash
nohup python scripts/run_main_comparison_e4.py --run-indices 0 1 3 4 5 > main9_clip.log 2>&1 &
nohup python scripts/run_main_comparison_e4.py --run-indices 2 6 7 8 > main9_qwen.log 2>&1 &
```

输出：`outputs/main_comparison_e4/{run_name}`。每组独立日志、缓存、checkpoint和进程锁。
启动前检查所选组需要的教师／冻结图库缓存是否 complete；这些缓存复用原有矩阵，
不自动修改或重建共享缓存。缺失时停止并报告路径。
重复执行从 `last.pt` 恢复已完成 epoch（未完成 epoch 重跑；不是逐 batch 精确恢复），
E4 checkpoint 已存在则跳过训练。仅允许调整 `data.num_workers` / `data.prefetch_factor` 后续跑；其余训练配置变更仍拒绝混用原输出目录。

### Worker 设置（2026-09-10）

九组 `main9_*.yaml` 的 `data.num_workers` 已从16回调为8，`prefetch_factor` 保持2不变。
CLIP 使用现有DataLoader；Qwen 新增spawn DataLoader，worker只接收CPU processor与配置，
并行完成query图像读取／解码和文本／图像预处理，GPU模型仍仅在训练主进程。
Qwen训练及OPRD候选召回均使用此加载路径；随机InfoNCE／CMSE不增加学生召回。
保持候选规则、batch顺序、batch大小、学习率及训练预算不变。

已在运行的训练进程不会动态应用改动；需要重启训练进程才生效。
只保证epoch级恢复，当前未完成epoch会重跑；请先考虑已有进度。
增加worker会增加CPU内存与共享内存使用，不能解决GPU被其他进程占满的问题。
独立评估脚本的加载配置不在本次训练worker调整范围内。

## 自动评估和汇总

- CLIP：CIRR/FashionIQ E1–E4，PinPoint/QDiag E4。
- Qwen：保存 E1–E4 checkpoint，自动评估 E4 的 CIRR/FashionIQ/PinPoint/QDiag。
- 已有最终评估文件自动跳过；`--skip-extended-eval` 仅跳过 PinPoint/QDiag。
- Qwen 默认评估batch=16，可用 `QWEN2_EVAL_BATCH_SIZE` 调整；CLIP 推理继续使用现有多worker流程。
- 每组完成后生成 `all_metrics.csv` / `all_metrics.json`：保存 CIRR 全Recall、FashionIQ
  各类别和宏／微平均、PinPoint全部字段及QDiag全部子集字段，不仅保留R@1/R@10。
  原始评估仍各自保存在实验目录。尚未完成的指标留空。

```bash
python scripts/run_main_comparison_e4.py --summarize-only
```

这九组是单seed的缺口补齐，不含调参和多seed重复；旧 CLIP-L ← S2 Random CMSE直接复用。
