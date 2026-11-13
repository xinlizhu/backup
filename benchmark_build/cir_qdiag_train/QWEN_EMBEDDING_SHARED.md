# Qwen3-VL-Embedding-2B：双卡双侧 all-linear 重训

## 短窗口性能诊断

停止旧任务后，用下面命令重启。每个训练进程只测第5–14步，然后继续正常训练；不改变模型、batch 或训练轮数。

```bash
nohup env CUDA_VISIBLE_DEVICES=0,1 QWEN_PROFILE_START=5 QWEN_PROFILE_STEPS=10 \
  python scripts/run_qwen_embedding_shared_e4.py \
  >> qwen_embedding_shared_2gpu_e1.log 2>&1 &
```

结果在每组输出目录的 `step_profile_rank0.jsonl`、`step_profile_rank1.jsonl`，并打印为 `[step profile]`。追加记录含 PID，可区分重启。

- `data_wait_s`：等待 DataLoader；`step_wall_s`：本卡完整样本步耗时（不含写日志与周期性跨卡指标汇总）。
- `forward_loss_cuda_stream_s`：输入搬运、前向和损失；`backward_replay_cuda_stream_s`：反向（开启重算时也含重算）。
- `gradient_finish_cuda_stream_s`：反向后剩余通信等待及梯度拷贝，不是全部通信总时间。
- `optimizer_cuda_stream_s`：unscale、裁剪、更新；首步还含梯度审计，应跳过首步分析。
- 同时记录各段 `*_host_s`，但主机和CUDA时间不能相加。CUDA Event 测的是计算流上两事件之间的经过时间，包含流空闲、CPU发射延迟和依赖等待，不等同于纯kernel执行时间。
- 仅在采样步开始与结束同步，不在阶段间同步，避免人为破坏通信重叠；采样仍有少量扰动。
- 每卡候选数量、候选分块数和 padded token 长度用于检查负载差异。只有真实GPU运行后才能判断瓶颈。

模型：本地 `CIR_PLAY/models/Qwen3-VL-Embedding-2B`，不是 Instruct。
默认顺序及 `--run-indices` 编号：0 OPRD-S2、1 InfoNCE、2 CMSE-S2、3 OPRD-Qwen8、4 CMSE-Qwen8。每组训练和评估完成后进入下一组。

- 两卡同步数据并行，每卡16条 query，累积1，总有效 batch32。
- all-linear LoRA：rank16、alpha32、dropout0.05，覆盖语言 attention/MLP、视觉与 merger 的线性层；基础权重冻结。
- 双侧实时编码和反传，不使用旧学生 gallery 缓存。OPRD 每轮刷新 Top32；InfoNCE/CMSE 使用随机候选。强制 target 插入，无 teacher 过滤。
- 五组统一1轮（含完整1轮 warmup/cosine 计划）；学习率2e-5；KL reverse、T=0.05，InfoNCE T=0.07。入口文件名保留 e4 以兼容原命令。
- 候选编码分块32，额外梯度缓存关闭。每卡 workers8、prefetch2；召回/评估 batch16。
- 双卡默认开启梯度通信重叠：单次 backward 中按8MiB桶异步求和，反向结束后等待通信完成，再裁剪和更新。保留损失的 local/global query 加权；不额外除以卡数。若开启候选梯度重算，自动使用原集中同步路径。设置 `QWEN_OVERLAP_GRADIENTS=0` 可对照原同步实现。
- 输出：`outputs/qwen_embedding_shared_2gpu_e1/`。首次从原始 Embedding 权重开始；只复用教师缓存。不加载旧2轮、4轮或单卡 checkpoint，旧结果不删除。重新运行可恢复本套1轮实验的 checkpoint。
- 每组训练后评估 CIRR、FashionIQ、PinPoint、QDiag。没有更改环境，也不会自动停止或启动其他任务。

在 `cir_qdiag_train` 目录，先停止占用这两张卡的其他训练。

```bash
# 配置检查
python scripts/run_qwen_embedding_shared_e4.py --dry-run

# 可选：双卡一更新 smoke，不跑完整 epoch，不保存 checkpoint
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/train_qwen_on_policy_distillation.py \
  --config configs/qwen2_embedding_shared_oprd_s2.yaml --smoke-only

# 五组依次重训并评估
nohup env CUDA_VISIBLE_DEVICES=0,1 python scripts/run_qwen_embedding_shared_e4.py \
  >> qwen_embedding_shared_2gpu_e1.log 2>&1 &
```

只跑第一组加 `--run-indices 0`。双侧直接反传的显存容量与吞吐需在真实双卡环境验证；CPU测试不代表GPU运行通过。
