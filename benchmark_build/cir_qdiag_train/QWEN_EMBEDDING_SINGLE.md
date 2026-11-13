# Embedding-2B：一张卡一个独立实验

每组逐项对齐 `qwen2_instruct_shared_*.yaml`，仅替换模型初始化与输出身份。模型为本地 Qwen3-VL-Embedding-2B，不是 Instruct。

单卡 batch16、累积2、有效 batch32；all-linear LoRA rank16/alpha32/dropout0.05；双侧实时反传；候选分块32；额外梯度缓存关闭；无需跨卡通信。

| 编号 | 实验 | 轮数 | workers / prefetch |
|---|---|---:|---|
| 0 | OPRD-S2 | 4 | 8 / 2 |
| 1 | OPRD-Qwen8 | 2 | 8 / 4 |
| 2 | InfoNCE | 2 | 8 / 4 |
| 3 | CMSE-S2 | 2 | 8 / 4 |
| 4 | CMSE-Qwen8 | 2 | 8 / 4 |

轮数与 KL 方向沿用各自 Instruct 配置（S2 reverse、Qwen8 OPRD forward）。不复制旧入口的“只评估 S2 已有第2轮权重”特殊逻辑，五组均从头训练。每组完成后评估 CIRR、FashionIQ、PinPoint、QDiag。

先停止旧双卡任务，在 `cir_qdiag_train` 目录分别启动两条队列：

```bash
# GPU 0：OPRD-S2 → CMSE-S2
nohup env CUDA_VISIBLE_DEVICES=0 python scripts/run_qwen_embedding_single.py \
  --run-indices 0 3 >> qwen_embedding_single_gpu0.log 2>&1 &

# GPU 1：OPRD-Qwen8 → InfoNCE → CMSE-Qwen8
nohup env CUDA_VISIBLE_DEVICES=1 python scripts/run_qwen_embedding_single.py \
  --run-indices 1 2 4 >> qwen_embedding_single_gpu1.log 2>&1 &
```

两条命令可同时运行，每张卡内顺序训练/评估，不是双卡数据并行。可只指定一个编号独立尝试。进程里的 cuda:0 是 CUDA_VISIBLE_DEVICES 指定的那张物理卡。

输出位于 `outputs/qwen_embedding_single/`。不加载旧双卡或 query-only checkpoint，不复用旧学生 gallery；教师缓存可复用。已有旧实验不删除。同一单卡新实验重启按其 checkpoint 恢复；未保存的 epoch 重跑。

每个实验有独立运行锁，汇总写入另加短时锁，不会用全局队列锁阻止另一张卡工作。没有修改已有 Instruct 或双卡配置，也没有启动正式训练。
