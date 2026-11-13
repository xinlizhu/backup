# Qwen3.5-0.8B：all-linear LoRA 与全参微调

本地模型：`/mnt/data/datasets/search_dataset/zxl/qwen3.5/Qwen3.5-0.8B`。

两套均为 **query＋candidate 双侧更新**，共享同一个模型。不是 query-only，不使用固定学生 gallery 向量。全参版更新检索主干所有参数；未使用的生成 LM head 和 MTP 不进入检索计算。

## 两个正式实验：CLIP-L batch 对齐版

| 编号 | 配置 | 更新范围 | 可训练参数 |
|---|---|---|---:|
| 0 | `configs/qwen35_clip_batch_lora_oprd_s2.yaml` | 语言、视觉、merger 的全部236个线性模块加 LoRA | 13,345,792 |
| 1 | `configs/qwen35_clip_batch_full_oprd_s2.yaml` | 检索主干全部参数，含 embedding、norm、卷积、注意力与 MLP | 852,985,920 |

对齐 CLIP-L 的训练 batch 和候选编码分块；Qwen 的模型结构、图像分辨率、学习率、召回及评估 batch 不变。原 `qwen35_{lora,full}_oprd_s2.yaml` 与 `run_qwen35_e4.py` 保留16×2、候选分块16，供已有任务继续使用；新版本使用独立配置和输出，不影响当前训练。

LoRA：`target_modules="all-linear"`，r=16、alpha=32、dropout=0.05。原权重冻结；卷积、embedding、norm 不加 LoRA。全参版不创建 LoRA，配置中的 LoRA 专用字段不参与计算。

两组目前都用 **OPRD、S2教师、reverse KL**，仅训练范围不同。训练引擎也支持 InfoNCE、CMSE，但没有擅自加入额外正式实验。

| 公共设置 | 值 |
|---|---|
| 训练轮数 / seed | 4 / 42 |
| Query batch / 梯度累积 | 32 / 1，有效batch=32 |
| 候选K / 温度T | 32 / 0.05 |
| 强制目标插入 / Teacher Top-10 | 开启 / 关闭 |
| 候选刷新 | 每个epoch用当前学生重新编码gallery并召回 |
| 候选编码分块 | 256，分块计算后统一反传 |
| 额外embedding梯度缓存 | 关闭 |
| 学习率 / weight decay | 2e-5 / 0.01 |
| Warmup / 梯度裁剪 | 0.05 / 1.0 |
| 计算精度 / 模型梯度检查点 | BF16 autocast / 开启 |
| CPU workers / prefetch | 8 / 2，训练、召回及评估 |
| 召回 image/query batch | 16 / 16 |
| 四项评估编码batch | 16 |
| max_length / min_pixels / max_pixels | 1024 / 4096 / 262144 |

全参版使用 FP32 主权重、梯度与 Adam 状态配合 BF16 autocast；LoRA版采用 BF16 基座和 FP32 adapter。两组损失、数据、候选和更新预算相同，但不保证这是各自最优学习率。

训练时对候选图像去重，但保留完整K候选的损失。候选编码分块256是每次编码的图像数上限，不是把K改成256。新配置增加单次计算规模，尚未验证H100峰值显存，可能OOM；不承诺GPU利用率达到100%。未改 Qwen3-VL-2B 或 CLIP 的配置。

## 环境

已单独创建 `.venv_qwen35/`，Transformers 5.3.0、PEFT 0.20.0，复用原 PyTorch 2.10.0+cu128。原 `multi_modal_py3.12` 仍是 Transformers 4.57.6，不要在原环境升级。

如需重建：

```bash
python scripts/setup_qwen35_env.py
```

依赖版本固定在 `requirements/qwen35.txt`。使用 FLA 0.5.2，不另装或升级 PyTorch；H100 上当前 FLA 会为部分反向操作选择 TileLang。没有安装需编译的 `causal-conv1d` 扩展，卷积分支允许使用 PyTorch 实现。CPU检查不启用CUDA kernel；GPU smoke会打印实际 FLA/conv kernel 可用状态。

`qwen35_full.log` 的首次失败发生在第一批反向传播：TileLang 自动选中了 Python 包内的 nvcc 13.3，但该目录运行时头文件是13.0。下面命令显式指定已有且版本配套的 `/usr/local/cuda-12.9`，只作用于本次进程及子进程，不安装、升级或删除任何依赖。工具链选择依据 [TileLang CUDA_HOME 解析代码](https://github.com/tile-ai/tilelang/blob/v0.1.9/tilelang/env.py)。原始失败内核已用 CUDA 12.9 编译通过；重启后的原16×2全参任务也已实际完成多个GPU优化步。

模型与混合注意力实现参考 [Transformers Qwen3.5 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)；线性注意力加速参考 [FLA 官方项目](https://github.com/fla-org/flash-linear-attention)。输入保留多模态 token 类型及位置编码信息，关闭思考生成和KV缓存，使用最后有效token pooling与L2归一化。

## 运行

等待同一张 H100 上的旧任务结束后运行，不要两套同时抢同一卡。

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train

# 只核对配置，不加载模型或训练
.venv_qwen35/bin/python scripts/run_qwen35_clip_batch_e4.py --dry-run

# 各测一个完整优化步：32 queries × 累积1，候选分块256，不保存权重
CUDA_HOME=/usr/local/cuda-12.9 .venv_qwen35/bin/python scripts/smoke_qwen35.py \
  --mode lora --config configs/qwen35_clip_batch_lora_oprd_s2.yaml
CUDA_HOME=/usr/local/cuda-12.9 .venv_qwen35/bin/python scripts/smoke_qwen35.py \
  --mode full --config configs/qwen35_clip_batch_full_oprd_s2.yaml

# 单独运行 LoRA，完成后评估
nohup env CUDA_HOME=/usr/local/cuda-12.9 .venv_qwen35/bin/python scripts/run_qwen35_clip_batch_e4.py --run-indices 0 \
  >> qwen35_clip_batch_lora.log 2>&1 &

# 单独运行全参，完成后评估；不要与上面一组同时启动
nohup env CUDA_HOME=/usr/local/cuda-12.9 .venv_qwen35/bin/python scripts/run_qwen35_clip_batch_e4.py --run-indices 1 \
  >> qwen35_clip_batch_full.log 2>&1 &
```

只跑LoRA加 `--run-indices 0`；只跑全参加 `--run-indices 1`。Smoke使用随机候选InfoNCE以检查梯度和容量，不是正式OPRD训练结果，也不包括整库召回。支持 `--batch-size`、`--micro-steps`、`--candidate-batch-size`、`--num-workers`，这些覆盖仅作用于冒烟测试。

## 输出、恢复与评估

新输出隔离在 `outputs/qwen35_clip_batch_e4/qwen35_clip_batch_lora_oprd_s2/` 和 `outputs/qwen35_clip_batch_e4/qwen35_clip_batch_full_oprd_s2/`。

旧 `outputs/qwen35_e4/`、日志和召回缓存均保留。新配置首次运行从预训练模型开始，不自动接续旧16×2实验的checkpoint；后续重跑新命令可恢复新实验自己的checkpoint。原命令继续使用原配置及原输出。

每轮保存 checkpoint，`last.pt` 包含对应 adapter或完整主干、优化器、scheduler、scaler和随机状态。同一命令重跑可按epoch恢复，未完成的epoch重跑；全参与LoRA的checkpoint不能混用。全参checkpoint会明显大于LoRA，四轮权重加last约需24GB磁盘，写临时文件时还需额外空间。

每组训练完成后评估E4的 CIRR、FashionIQ、PinPoint official、QDiag。`--skip-extended-eval` 跳过后两项。新汇总为 `outputs/qwen35_clip_batch_e4/all_metrics.csv` 和 `all_metrics.json`，不自动改论文主表。

`adapter_audit.json` 记录训练范围与参数量，`gradient_audit.json` 记录梯度检查，`run_summary.json` 记录有效batch和训练模式。全参若出现非有限梯度，直接报错停止，不静默保存异常训练结果。

## 已验证范围

已验证本地0.8B完整权重加载及真实图像/图文query的CPU前向；真实模型输出为1024维且有限、归一化正常。小型Qwen3.5覆盖线性注意力、标准注意力、视觉和merger的两侧反传，验证LoRA冻结范围、全参无冻结、checkpoint与优化器恢复；配置与旧实验回归测试已执行。

Qwen3.5的23项测试已通过，覆盖新旧配置、独立启动、评估与恢复。原16×2全参任务已记录20个优化步，473个可训练张量均有非零梯度、无缺失或非有限梯度。新增32×1、候选分块256配置尚未执行GPU smoke或正式训练，可先用上述smoke命令确认容量。
