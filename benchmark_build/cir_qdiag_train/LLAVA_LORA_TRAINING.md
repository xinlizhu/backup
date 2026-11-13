# LLaVA-1.6 Mistral 7B：双卡、双侧语言 Attention LoRA

原始模型已下载到 `/mnt/data/datasets/search_dataset/zxl/llava/llava-v1.6-mistral-7b-hf`。
来源：`llava-hf/llava-v1.6-mistral-7b-hf`，revision `2424fdd47412fccc66d91719126b420e9fbd7065`。
四个权重分片共15,133,587,296 bytes，687个张量已按索引检查。不是 BGE S1/S2 权重。

## 配置

| 设置 | 值 |
|---|---|
| GPU / 每卡 Query batch / 累积 | **2 / 16 / 1**，总 batch=32 |
| 候选 K / 候选图像编码分块 | **32 / 8** |
| 训练与召回 workers / prefetch | **每卡8 / 1**，双卡训练共16个 worker |
| 评估 workers / prefetch | 16 / 2（不变） |
| LoRA | **语言 attention，r=16，alpha=32，dropout=0.05** |
| 更新方式 | query＋candidate 共享同一个 adapter，两侧反传 |
| LoRA 覆盖 | 仅语言 self-attention 的 q/k/v/o_proj；视觉、连接层和 MLP 不加 adapter |
| 训练参数 | 以启动审计为准；原始权重冻结；不使用生成 LM head |
| 训练轮数 / seed | 2 / 42 |
| 学习率 / 调度 | 5e-6 / 原训练引擎的 warmup＋cosine |
| OPRD 温度 / KL | 0.05 / reverse，两位教师均如此 |
| InfoNCE 温度 | 0.07，沿用已有 candidate InfoNCE 对照 |
| 强制 target / Teacher Top-10 | 开启 / 关闭 |
| 候选刷新 | OPRD 每轮由 rank 0 重新编码当前学生 gallery，再召回；另一卡等待 |
| 训练候选编码 | 每步重新编码候选图像并反传，不固定学生 gallery |
| 额外 embedding 梯度缓存 | 开启；先算 embedding 梯度，再逐块重算、反传 |
| 原始图片 resize | 固定512×512，随后使用原生 LLaVA AnyRes 处理器 |
| 最大输入长度 | 8192，超长报错，不截断图像 token |
| 召回 image/query batch | 16 / 16 |
| 评估 batch | 16 |
| 精度 / 梯度检查点 | BF16 autocast / non-reentrant 开启 |

这是我们的蒸馏配置，不是 S1 的完整复现。保持原模型层数及视觉取层方式。

原生 `vision_feature_layer=-2` 不变。当前仅语言 attention 加 LoRA，审计检查语言各层 q/k/v/o 覆盖，并确保视觉和连接层无 adapter。旧 all-linear 模式仍可加载历史配置。

输出目录不变。切换时不能续训旧 all-linear checkpoint：runner 仅在没有 checkpoint、缓存为训练前 global_step=0 的第1轮结果且其他训练设置一致时，自动归档旧配置与日志并复用初始候选缓存。后续 epoch 正常重新召回；运行中的旧任务须先停止。

训练、召回、评估使用同一个 CPU 预处理函数：BGE 风格 query/candidate 提示词、显式 EOS、最后一个有效 EOS hidden state pooling、L2归一化。右侧 padding 不参与 pooling；支持一条 query 多张参考图。512输入仍会产生5个336×336 crop，不能按单个512图估算所有显存。`min_pixels/max_pixels` 是共享配置的兼容字段，LLaVA 以 `llava_image_size` 为准。

双卡使用同步数据并行：每张卡保存一份模型、处理不同的 query，每步对 LoRA 梯度做一次 all-reduce 后共同更新。没有固定候选 embedding，也没有跨 query 扩充负例；每条 query 仍是K=32。末尾不足32条的 batch 不重复补样本、不丢弃，按实际 query 数加权。原生共享编码器有多次前向及无梯度视觉末层，因此这里显式同步梯度，不在每次前向外套 DDP reducer。

OOM后的配置：保持每卡16条 query、总 batch=32、K=32，候选编码分块改为8，并开启梯度缓存。每步用当前模型计算 embedding，得到损失对 embedding 的梯度，再恢复对应 dropout 随机状态逐块重算、反传；不会同时保留全部候选的计算图。query 和 candidate 两侧仍训练，不是固定 gallery。代价是额外前向计算；query 侧仍一次编码16条，GPU显存是否足够须用 smoke 确认。

## 命令

使用原 `/mnt/data/datasets/search_dataset/zxl/multi_modal_py3.12` 环境，不升级依赖。
在 `cir_qdiag_train` 目录执行，使用空闲的两张GPU；以下GPU命令不要同时运行。外层脚本自动启动 torchrun，不要再对外层 runner 执行 torchrun。

```bash
# 只打印五组配置，不加载模型、不训练
python scripts/run_llava_lora_e4.py --dry-run

# 双卡完整优化步测试：每卡16条query，各32个随机候选；输出每卡时间与显存
CUDA_VISIBLE_DEVICES=0,1 python scripts/smoke_llava_lora.py

# 未训练模型：同一编码协议，先看 CIRR；去掉 --only-cirr 可评估全部四项
python scripts/evaluate_llava_original.py --only-cirr

# 先正式运行 S2 OPRD 一组，完成后自动评估
nohup env CUDA_VISIBLE_DEVICES=0,1 python scripts/run_llava_lora_e4.py --run-indices 0 \
  >> llava_lora_2gpu_b32_gc8_e2.log 2>&1 &
```

Smoke使用随机候选 InfoNCE，只检查容量与梯度，不是正式 OPRD 结果，也不保存训练 checkpoint；第一步含 worker 启动与冷数据读取，不能当稳定吞吐。

| 编号 | 实验 | 配置 |
|---|---|---|
| 0 | OPRD ← BGE S2 | `configs/llava_oprd_s2.yaml` |
| 1 | OPRD ← Qwen8B | `configs/llava_oprd_qwen8.yaml` |
| 2 | InfoNCE，随机候选 | `configs/llava_infonce.yaml` |
| 3 | CMSE ← BGE S2，随机候选 | `configs/llava_cmse_s2.yaml` |
| 4 | CMSE ← Qwen8B，随机候选 | `configs/llava_cmse_qwen8.yaml` |

去掉 `--run-indices 0` 就会依次跑五组。`--skip-extended-eval` 仅跳过 PinPoint/QDiag。

## 输出与恢复

- 新输出：`outputs/llava_lora_2gpu_b32_gc8_e2/llava_*/`；原始基线：`outputs/llava_lora_2gpu_b32_gc8_e2/original/`。旧配置及四轮目录保留，不自动混用 checkpoint；本次从新目录开始。启动脚本沿用 `run_llava_lora_e4.py` 文件名，实际轮数由配置决定，现在是2轮。
- 每组两轮结束后评估 E2（`checkpoint_epoch_002.pt`）的 CIRR、FashionIQ、PinPoint official、QDiag。汇总到根目录 `all_metrics.csv/json`；原始基线单独保存，不混充训练轮次。不自动改论文主表。
- 仅 rank 0 每轮保存 adapter checkpoint 与 `last.pt`，包含优化器、scheduler、scaler、RNG；两卡重跑同一命令自动按 epoch 恢复。每轮使用固定的 epoch/rank seed，缓存命中不改变训练随机数。没有每步保存；未完成的 epoch 重跑。
- 新实验不读取旧 Qwen checkpoint，不使用旧固定学生 gallery；仍复用相应教师的向量缓存。
- `adapter_audit.json` / `gradient_audit.json` 记录训练范围及梯度；旧模型配置和日志不改。
- CIRR、FashionIQ、PinPoint、QDiag 仍在训练结束后单进程评估，不会由两个训练 rank 各跑一遍。

双卡验证：`tests/test_llava_parallel.py` 的13项测试通过，包含分片无重复、无dropout时双进程梯度与单进程等价、三种目标的缩小版原生 LLaVA 训练，以及重启两个进程后恢复到完全相同的参数。通信测试只使用CPU/Gloo；完整7B的双卡 GPU smoke 需在训练机器执行。

重新下载/补齐时可运行 `python scripts/download_llava.py`，默认固定上述revision并复用已下载文件。
