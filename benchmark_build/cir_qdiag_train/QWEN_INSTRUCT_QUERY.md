# Qwen3-VL-2B-Instruct：沿用旧 Embedding 训练配置

新版双侧 all-linear LoRA（包含 MLP、取消固定学生 gallery）见 [QWEN_INSTRUCT_SHARED.md](QWEN_INSTRUCT_SHARED.md)，使用 `scripts/run_qwen_instruct_shared_e4.py`。本文及原入口保留旧 query-only 实验，不会自动切换正在运行的任务。

仅替换学生初始化为 `/mnt/data/datasets/search_dataset/zxl/qw3/Qwen3-VL-2B-Instruct`。独立输出 `outputs/qwen_instruct_query_e4/`；不修改双侧 LoRA 实验的配置、日志、缓存或 checkpoint。

## 保留的训练设置

| 设置 | 值 |
|---|---|
| LoRA | 仅 query 侧语言 attention 的 q/k/v/o_proj |
| rank / alpha / dropout | 16 / 32 / 0.05 |
| 可训练参数 | 6,422,528；不使用 all-linear |
| 图像侧 | 原始模型冻结，禁用 LoRA，训练 gallery 一次性缓存 |
| 训练 batch / 梯度累积 | 8 / 4，有效 batch=32 |
| Epoch / seed | 4 / 42 |
| 学习率 / weight decay | 2e-5 / 0.01 |
| warmup / 梯度裁剪 | 0.05 / 1.0 |
| 精度 / 模型梯度检查点 | BF16 / 开启 |
| K / 强制目标插入 | 32 / 开启 |
| 召回 image / query batch | 16 / 16 |
| 评估 batch | 16 |
| max_length / min_pixels / max_pixels | 1024 / 4096 / 262144 |

学生候选每轮刷新；InfoNCE 和 CMSE 使用随机候选。训练损失直接与冻结的候选向量计算，不再对候选图像实时反传；`candidate_encode_batch_size` 和额外 embedding 梯度缓存不参与这个分支。

## 五组实验

| 编号 | 配置后缀 | 对应旧配置 | 教师过滤 | workers / prefetch |
|---|---|---|---|---|
| 0 | oprd_s2 | main9_qwen2_oprd_s2 | Top-10，reverse KL | 8 / 2 |
| 1 | oprd_qwen8 | opd_qwen2_qwen8_rank | Top-10，forward KL | 0 / 2（不预取） |
| 2 | infonce | main9_qwen2_infonce | 无，T=0.07 | 8 / 2 |
| 3 | cmse_s2 | main9_qwen2_cmse_s2 | 无 | 8 / 2 |
| 4 | cmse_qwen8 | main9_qwen2_cmse_qwen8 | 无 | 8 / 2 |

新配置前缀统一为 `configs/qwen2_instruct_query_`。旧 OPRD 的 Top-10、KL 方向和 Qwen8 组的 workers=0 原样保留，不在本轮对照中另行统一。启动时逐字段对比旧配置，只允许模型初始化/类型、实验名称、输出路径和新学生缓存路径不同。

## 运行

等待当前同卡任务结束后，在原虚拟环境运行。不要与今晚的双侧 LoRA 任务同时占用同一张 H100。

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
source /mnt/data/datasets/search_dataset/zxl/multi_modal_py3.12/bin/activate

# 只核对配置，不加载模型、不训练
python scripts/run_qwen_instruct_query_e4.py --dry-run

# 自动准备新模型 gallery 缓存，依次训练并评估五组
nohup python scripts/run_qwen_instruct_query_e4.py \
  >> qwen_instruct_query_e4.log 2>&1 &
```

只跑一组可加 `--run-indices 0`，或用 `--run-indices 0 2 3` 跑 S2 OPRD、InfoNCE、S2 CMSE。默认每组 E4 完成后评估 CIRR、FashionIQ、PinPoint official 和 QDiag；`--skip-extended-eval` 跳过后两项。新入口固定评估 batch=16，避免继承双侧任务的大批量环境变量。

## 缓存与恢复

新学生 gallery 位于 `outputs/qwen_instruct_query_e4/cache/qwen3_vl_instruct_2b_fixed_gallery/`，自动生成并检查模型来源，不能使用旧 Embedding 学生缓存。教师缓存仍沿用原实验。缓存预处理与训练子进程串行执行，不同时保留两个 GPU 模型。

每轮保存 checkpoint 和 `last.pt`；重跑入口可按旧逻辑恢复，未完成的 epoch 重跑。首次 gallery 编码支持进度恢复。新结果汇总为 `outputs/qwen_instruct_query_e4/all_metrics.csv` 和 `all_metrics.json`，不自动修改论文主表。

## 验证

已做旧/新配置逐字段比对、真实处理器的短文本/长文本/图像输入一致性检查、真实2B结构的 LoRA 参数计数，以及小型 Qwen 的梯度、冻结图像侧和 checkpoint 加载测试。CPU 模拟训练覆盖 KL、InfoNCE、CMSE。尚未启动 H100 实际训练，不能保证与旧实验耗时完全一致。

加载适配仅解决 Instruct checkpoint 的原生权重映射与检索 pooling；不改变模型层数。有效图像的处理参数沿用旧实现；无效图像明确报错，不采用上游静默替换成 NULL 文本的行为。
