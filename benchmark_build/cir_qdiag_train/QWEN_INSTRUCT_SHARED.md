# Qwen3-VL-2B Instruct：双侧 all-linear LoRA

2026-09-14：五组最终均使用第2轮权重。当前 S2 OPRD 直接选用已有 `checkpoint_epoch_002.pt`，跳过训练、绝不恢复更晚的 `last.pt`；其原4轮训练配置和后续权重仅作为历史记录保留，因此该权重仍来自原4轮学习率计划的前两轮。S2评估独立保存在 `selected_epoch_002/`，总表自动读取该目录。若第2轮权重缺失，报错而不重训。后续四组按2轮学习率计划训练并评估。入口文件名仍保留 e4；用户停止旧队列后，重跑原命令即可。脚本不会主动停止其他进程。

替代 query-only＋固定学生 gallery 的五组对比，旧配置、任务及输出不修改。

- 共享一个模型与 adapter，query 和 candidate 两侧均参与反向传播；评估两侧也启用 adapter。
- `target_modules="all-linear"`：语言 attention 的 q/k/v/o、MLP 的 gate/up/down，以及视觉 attention、MLP、merger 的线性层均覆盖。r=16、alpha=32、dropout=0.05。不是全参训练，原始权重冻结，embedding/norm 不加 LoRA。
- OPRD 每轮用当前模型重新编码 gallery、召回 Top-32；每步用当前模型重新编码候选图像并反传。不是每步重编码整库。
- InfoNCE、CMSE 使用随机候选，不做学生预筛选。三种损失均强制包含 target，不做 Teacher Top-10 过滤。
- 训练 batch=16×2，有效 batch=32。轮数按上方说明；保留学习率2e-5、K=32、蒸馏温度0.05、图像像素设置和召回/评估 batch=16。候选图像分块32，关闭额外 embedding 梯度缓存。
- 训练及召回 workers 均为8；OPRD-Qwen8、InfoNCE、CMSE-S2、CMSE-Qwen8 的 prefetch 均为4。S2 OPRD 保留历史配置，仅评估已有权重。没有改环境。已启动的训练进程需重启才生效；预取增加会提高CPU内存占用。

| 编号 | 方法 | 教师 | KL方向 |
|---|---|---|---|
| 0 | OPRD | S2 | reverse |
| 1 | OPRD | Qwen8B | forward，沿用旧组 |
| 2 | InfoNCE | 无 | — |
| 3 | CMSE | S2 | — |
| 4 | CMSE | Qwen8B | — |

在原训练环境、`cir_qdiag_train` 目录运行；同卡先结束其他任务，不要并行争用显存。

```bash
# 核对配置，不启动训练
python scripts/run_qwen_instruct_shared_e4.py --dry-run

# 依次训练五组，每组完成后评估 CIRR、FashionIQ、PinPoint、QDiag
nohup python scripts/run_qwen_instruct_shared_e4.py \
  >> qwen_instruct_shared_e4.log 2>&1 &
```

仅试 S2 OPRD：添加 `--run-indices 0`。新输出在 `outputs/qwen_instruct_shared_e4/`，首次从本地预训练 Instruct 模型开始，不加载旧 query-only checkpoint 或固定学生 gallery 缓存。重跑同一新命令按已有新 checkpoint 恢复；未完成的 epoch 重跑。教师缓存仍可复用。

新配置经过 CPU 回归测试后仍需 GPU 实测；双侧反传比 query-only 占用更多显存，不能承诺速度、容量或效果。旧训练候选准确率与整库检索脱节的原因尚未定位，不能认定改为双侧就一定解决。
