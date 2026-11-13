# TG-CIR：CLIP-L，4 epochs，batch size 32

只准备代码，未启动正式训练。当前默认使用本地公开 **CLIP ViT-L/14** 预训练权重初始化，**不使用作者的 TG-CIR checkpoint**。原 CLIP-B 配置保留，可显式指定；B/L 输出与断点隔离。

## 运行

在已有 `multi_modal_py3.12` 环境中执行：

当前环境已具备所需依赖，不需要安装官方旧版 PyTorch 或 `clip` 包。其他环境可使用 `python -m pip install -e '.[tg-cir]'` 安装此实现的依赖。

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train

# 只检查数据和配置，不训练
python scripts/train_tg_cir.py --dry-run

# 一键：训练、每 epoch 的 CIRR 验证、E4 的 FashionIQ / PinPoint / QDiag
nohup python -u scripts/run_tg_cir_e4.py --config configs/tg_cir_clip_l_e4.yaml > tg_cir_clip_l_e4.log 2>&1 &
```

只训练并做 CIRR 验证：

```bash
nohup python -u scripts/run_tg_cir_e4.py --train-only > tg_cir_clip_l_e4.log 2>&1 &
```

训练后单独补齐评估：

```bash
python scripts/run_tg_cir_e4.py --eval-only
```

可选 GPU 冒烟测试：只跑两步，不写 checkpoint，不进入正式实验目录。

```bash
python scripts/train_tg_cir.py --smoke-test
```

## 配置

当前默认配置：`configs/tg_cir_clip_l_e4.yaml`；可选旧配置：`configs/tg_cir_clip_b_e4.yaml`。

| 项目 | 设置 |
|---|---|
| 学生 / 教师共享 backbone | 本地 `models/clip-vit-large-patch14`，全量微调；没有第二个外部教师模型 |
| 图像 / 文本隐藏维度 → 属性维度 | 1024 / 768 → 768；组合头输入为 1536，最终检索向量为 768 维 |
| 数据 | CIRR train，28,225 queries |
| Epoch / batch | 4 / 32，drop_last，每 epoch 882 步，共 3,528 步 |
| 训练 / CIRR、FashionIQ 推理 workers | 8 / 8，prefetch=2，CPU spawn worker 不携带模型 |
| CLIP / 新模块学习率 | `1e-5` / `1e-4`，AdamW，weight_decay=0.01 |
| 属性数 | 全局 P=4，局部 Q=8 |
| KL temperature | 0.05；InfoNCE 使用官方可学习 logit multiplier，初值 10 |
| 六项损失权重 | student rank=1，teacher rank=1，teacher mask=1，orthogonal=.1，mask distillation=1，KL=.1 |
| 候选 | batch 内 target，**无 on-policy 召回、Teacher Top-10 筛选或 128→32** |
| 精度 / 显存 | FP16 AMP，CLIP gradient checkpointing |
| Seed | 42 |

官方使用 ViT-B/16、默认 batch=64、最多 30 epochs；本实验按要求改为 **TG-CIR 的 CLIP-L backbone 扩展，E4/B32**，并非官方原配置的严格复现。其余使用官方 CIRR 方法与损失设置，不声称复现论文原始分数。官方学习率在 E5/E10 开始衰减，当前四轮不触发，也不额外添加 warmup。

Teacher 是训练时额外看到正确目标图的组合分支：学习哪些参考图属性应保留、哪些修改文本属性应加入，再用 mask MSE 让只看参考图和文本的 student 模仿。Teacher 也有自己的检索损失以学习有效组合；另有目标图之间相似度分布构造的 KL 监督。两分支共享 CLIP-L 编码器、联合训练，不需要 S2/Qwen 或教师 checkpoint。评估只运行 student，不提供正确目标图。

## 输出与恢复

输出目录：`outputs/tg_cir_clip_l_e4/`；不会从 `outputs/tg_cir_clip_b_e4/` 恢复。

| 文件 / 目录 | 内容 |
|---|---|
| `last.pt` | 每 100 步及每 epoch 末保存；模型、AdamW、AMP scaler、RNG、epoch 和 batch 位置 |
| `checkpoint_epoch_001.pt` … `004.pt` | 四个 epoch 的固定模型 |
| `best.pt`、`best_metrics.json` | 按官方 CIRR 七项 Recall 之和选择最佳 epoch，立即固定权重 |
| `train_steps.jsonl` | 每 20 步记录六项损失、总损失、平均秒/步 |
| `val_epoch_001.json` … `004.json` | CIRR 全部七项 Recall |
| `fashioniq/epoch_004/metrics.json` | 三类别及 macro/micro 全部 Recall |
| `pinpoint_official/TG-CIR-CLIP-L-E4/official_metrics.csv` | PinPoint 官方指标 |
| `qdiag/TG-CIR-CLIP-L-E4/summary.csv` | QDiag 指标 |

中断后重新执行原命令，自动读取 `last.pt`，从保存的 epoch 内 batch 位置继续，已完成的 epoch 不重训。正常 SIGINT/SIGTERM 会请求当前训练步完成后保存；SIGKILL、OOM 或断电只能使用之前成功保存的断点。验证期间中断可以恢复该 epoch 的特征缓存。`last.pt` 原子替换，不会修改已固定的 epoch/best 文件。

主表与跨域评估统一使用 **E4**，不自动换成 `best.pt`。FashionIQ 只评估，不在该数据集训练。PinPoint/QDiag 使用各自现有评估协议，不修改主表或已有实验结果。

## 接入说明

- `tg_cir_model.py` 按[官方 model.py](https://github.com/iLearn-Lab/MM23-TG-CIR/blob/main/model.py)移植到现有 Transformers CLIP；局部视觉 token 包含 CLS，文本保持 77 长度、零补齐且不加 padding attention mask。超过 77 token 的文本保留 EOS 截断，便于跨域评估。
- 保留 teacher 的参考图/文本特征 detach、mask 蒸馏 detach，以及目标相似度 KL 的教师分布 detach；不是把我们的 OPRD 损失重命名为 TG-CIR。
- 去掉官方重复的参考图/文本编码，复用无 dropout 的 CLIP 特征；不影响师生分支梯度截断。将硬编码 CUDA 改为设备随输入，补上断点和真正的最佳权重保存。
- PinPoint 两参考图查询采用“对应属性取均值后进入学生组合分支”的显式扩展；单参考图路径与原 TG-CIR 相同。此多参考图规则记录在 adapter metadata，不属于论文原始训练设置。
- 注册名 `TG-CIR-CLIP-L-E4` 接入 `CIR_PLAY` 的 adapter registry；指定 B 配置时自动使用 B 标签。CIRR/FashionIQ 多进程预处理，PinPoint/QDiag 使用原评估器的并行预处理。
- 模型移植部分保留官方作者署名和 GPL-3.0-or-later 标记；[官方仓库](https://github.com/iLearn-Lab/MM23-TG-CIR)声明 GPL v3 或后续版本。

测试命令（随机小模型和临时图片，不运行正式训练）：

```bash
python -m pytest tests/test_tg_cir.py -q
```

验证记录：15 项 CPU 测试通过，包含两 worker 的实际张量传输、六项损失/梯度对照、断点恢复一致性、CLIP-L 维度适配及 B/L 输出标签隔离。本地完整 CLIP-L 预训练权重的查询/图片前向输出均为有限的 768 维单位向量；此前 CLIP-B 的 512 维前向及分词补齐、HTML 清理、长文本截断对照也已通过。未启动正式训练，未实测 GPU AMP 训练和完整四数据集指标。
