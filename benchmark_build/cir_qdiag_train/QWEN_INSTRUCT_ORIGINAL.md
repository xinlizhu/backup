# Qwen3-VL-2B-Instruct Org 评估

不训练。使用与当前双侧 Instruct 相同的提示词、图像预处理、EOS/最后有效token pooling、L2归一化与评估流程。零初始化 LoRA B 使 adapter 无更新；保存为 initialization.pt 只是为了复用评估适配器，不是训练后权重。加载已有初始化时校验模型配置、epoch=-1、step=0 和 LoRA B 全零。

依次评估 CIRR、FashionIQ、PinPoint、QDiag；不需要教师缓存。不修改现有训练结果。默认 batch16，单卡即可。

在 cir_qdiag_train 目录，用空闲 GPU 运行（GPU 0 可改为 1）：

```bash
nohup env CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_qwen_instruct_original.py \
  >> qwen_instruct_original.log 2>&1 &
```

只评估 CIRR 加 `--only-cirr`；检查命令而不加载模型加 `--dry-run`。不要与该卡上的训练争用资源。

输出：`outputs/qwen_instruct_original/`，包含 initialization.pt、baseline_metadata.json，以及 cirr_val、fashioniq、pinpoint_official、qdiag。重新运行会跳过已有评估结果。请勿将训练后权重放入此目录冒充 Org。
