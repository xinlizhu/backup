# Qwen3-VL-2B-Instruct：双侧 all-linear LoRA

## 当前推荐：CLIP 式直接反传

使用 `run_qwen_shared_lora_direct_e4.py`，配置前缀 `qwen2_shared_direct_`，输出 `outputs/qwen_shared_lora_direct_e4/`。原慢版的配置、日志、缓存和 checkpoint 均保留；此版本不自动导入旧权重或候选缓存。

与 CLIP 对齐的流程：每轮刷新学生候选（随机基线不做学生召回）→ 当前模型实时编码 query 和去重候选 → 相似度与同一套损失函数 → 一次直接 backward → 梯度累积／裁剪／AdamW 更新。**关闭额外的 embedding 梯度缓存和梯度回放**，保留 CLIP 也使用的模型梯度检查点。两侧 all-linear LoRA、K=32、强制插入、无教师过滤、4 epochs 等不变。

| 参数 | CLIP 主实验 | 原 Qwen 慢版 | Qwen 直接反传版 |
|---|---:|---:|---:|
| Query micro-batch | 32 | 1 | 16 |
| 梯度累积 | 1 | 32 | 2 |
| 有效 batch | 32 | 32 | 32 |
| 图像编码分块 | 256 | 2 | 32 |
| 额外 embedding 梯度缓存 | 无 | 有 | 无 |
| 模型梯度检查点 | 有 | 有 | 有 |
| CPU workers／prefetch | 8／2 | 4／2 | 8／2 |

最初按 CLIP 的 `32×1、图像分块256` 运行后发生 OOM；用户实测 `8×4、图像分块32` 通过。现按用户要求将四组正式配置更新为 `16×2、图像分块32`，有效 batch 保持32，其他参数不变，不重新开启梯度缓存；`16×2` 的显存占用尚待实测。

召回 gallery batch=512、query batch=128；CIRR／FashionIQ 编码 batch=256；PinPoint／QDiag batch=128。Gallery 编码已通过；query batch=1024 在第一批返回前发生 DataLoader worker 被杀，降至256后在77/111批时再次出现同类错误。现按用户要求降至128，尚待实测。训练 `16×2`、workers=8、prefetch=2 及其他参数不变。模型结构、图文输入融合、LoRA 和之前确认的 LoRA 学习率 `2e-5` 保留 Qwen 设置，不声称所有模型参数均与 CLIP 相同。评估不在训练冒烟测试覆盖范围内；若实际 OOM，再单独调整对应批量。

先停止同卡旧任务，再在原训练环境运行：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
source /mnt/data/datasets/search_dataset/zxl/multi_modal_py3.12/bin/activate
python scripts/run_qwen_shared_lora_direct_e4.py --dry-run

# 完整的一个优化步：16 queries × 累积2次，共32 queries；不保存权重
python scripts/smoke_qwen_shared_lora.py --num-workers 8

# 0 OPRD、1 Random KL、2 InfoNCE、3 CMSE；去掉 --run-indices 0 跑四组
nohup python scripts/run_qwen_shared_lora_direct_e4.py --run-indices 0 \
  > qwen_shared_lora_direct_e4.log 2>&1 &
```

冒烟输出包含实际 query 数、累积次数、数据等待／前向反传／优化器耗时及峰值显存。默认读取直接反传版配置，测试完整的32-query优化步；若显式指定其他配置，应结合 `queries_in_update` 与 `micro_steps_in_update` 解读耗时。该测试使用随机候选 InfoNCE，不包含整库召回，含首次 DataLoader 启动开销；不能直接代替整轮测速。

2026-09-12 用户提供的 H100 单步实测（历史 `8×4、图像分块32` 配置）：32 queries 耗时93.774秒，其中数据等待49.966秒、前向反传43.513秒、优化器更新0.117秒；峰值 allocated 34.583 GiB、reserved 36.965 GiB。梯度无缺失、无非有限值。这是含首次 DataLoader 启动的单步结果，不代表稳定吞吐，也不保证所有图像尺寸组合均不会 OOM。

该次输出的 `matches_configured_accumulation=false` 是因为命令行指定累积4次，而当时 YAML 仍是1次；当前正式 YAML 为 `16×2`，默认冒烟测试也读取这一配置。冒烟命令的覆盖参数仍只作用于本次测试，不会自动修改正式 YAML。

与 CLIP 运行器一致：训练结束后依次评估 E1–E4 的 CIRR／FashionIQ，E4 再评估 PinPoint／QDiag。前三轮结果位于 `cirr_val/epoch_00N/` 和 `fashioniq/epoch_00N/`；最终轮仍用 `cirr_val/` 和 `fashioniq/`，供汇总读取。若环境预设了 `QWEN2_EVAL_BATCH_SIZE` 或 `QWEN2_DIAG_BATCH_SIZE`，它们会覆盖对应评估批量；默认分别为256和128。整轮吞吐与峰值仍需在正式运行中观察。

## 历史：梯度缓存版（保留复现）

新实验独立保存在 `outputs/qwen_shared_lora_e4/`，不覆盖旧的 query-only Embedding 实验或论文表格。

## 设置

- 初始化：`/mnt/data/datasets/search_dataset/zxl/qw3/Qwen3-VL-2B-Instruct`。
- 视觉、连接层、语言模型的所有线性层：`target_modules="all-linear"`，`r=16`、`alpha=32`、`dropout=0.05`；原始权重冻结。
- 按实际2B结构检查：300个适配器模块（语言196、视觉96、连接层8），可训练参数 **24,641,536**；双侧共享，不翻倍。
- Query 和 Gallery 共享同一个模型和同一套适配器，两侧均反传。非线性层、embedding、patch卷积等不增加适配器。生成 LM head 不参与检索。
- 最后有效 token pooling + L2 归一化；query 是参考图＋指令，gallery 是候选图像＋固定模板。
- 4 epochs，seed42，学习率 `2e-5`，BF16，query micro-batch 1 × 梯度累积32，K=32，强制插入 target；不使用 Teacher Top-10 过滤。
- 所有模块启用梯度检查点；候选图像每块2张，以梯度缓存重计算反传，并恢复 dropout 的随机状态。训练 loss 不使用陈旧的 gallery 向量。
- CPU workers=4，prefetch=2；召回／评估 batch=4。新配置先保守设置，不修改旧配置。

| 编号 | 配置名 | 方法 | 候选 | 教师 | 温度 |
|---|---|---|---|---|---|
| 0 | qwen2_shared_oprd_s2 | OPRD，reverse KL | 每轮学生 Top-32 | S2 | 0.05 |
| 1 | qwen2_shared_random_kl_s2 | Random KL，reverse KL | 随机 | S2 | 0.05 |
| 2 | qwen2_shared_infonce | InfoNCE | 随机 | 无 | 0.07 |
| 3 | qwen2_shared_cmse_s2 | Centered MSE | 随机 | S2 | 不使用 |

## 运行

使用原训练虚拟环境，在项目目录运行：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
source /mnt/data/datasets/search_dataset/zxl/multi_modal_py3.12/bin/activate

# 只检查配置，不训练
python scripts/run_qwen_shared_lora_e4.py --dry-run

# 先在 H100 做一个真实模型更新，报告峰值显存，不保存权重
python scripts/smoke_qwen_shared_lora.py --config configs/qwen2_shared_infonce.yaml --micro-steps 1

# 可选：同时检查多进程图像预处理
python scripts/smoke_qwen_shared_lora.py --config configs/qwen2_shared_infonce.yaml --micro-steps 1 --num-workers 4

# 先跑 OPRD 一组；去掉 --run-indices 0 就依次跑四组
nohup python scripts/run_qwen_shared_lora_e4.py --run-indices 0 \
  > qwen_shared_lora_e4.log 2>&1 &
```

冒烟测试固定使用随机候选 InfoNCE，仅检查完整图像／查询梯度、优化器初始化和显存，不代表 OPRD 训练结果。

运行器自动检查所需 S2 教师缓存。学生不需要事先生成固定 gallery 缓存：OPRD 每轮用当前适配器重新编码训练 gallery；随机候选方法不执行学生召回。

每组训练完成后自动评估 E4 的 CIRR、FashionIQ、PinPoint 和 QDiag；候选图像侧始终启用当前适配器。评估结果隔离在各自实验目录，可设置 `QWEN2_EVAL_BATCH_SIZE=2` 降低评估批量。`--skip-extended-eval` 仅跳过 PinPoint／QDiag。

重复运行同一命令会从 `last.pt` 恢复；保存粒度为每个 epoch，不是每个 step。中途停止会重跑未完成 epoch。配置不匹配会拒绝覆盖旧实验；改模型、rank或训练预算时使用新输出目录。可查看 `adapter_audit.json` 确认实际覆盖层及可训练参数，`gradient_audit.json` 检查梯度。

若显存不足，先把四个新配置的 `candidate_encode_batch_size` 改为1；CPU内存不足时降低 `num_workers`。不要通过停用候选侧梯度或复用固定学生特征规避显存问题，否则不再是本实验设置。
