# 并行评估与建库加载

本次只改变 CPU 数据供给：图片读取、RGB 解码、原有 processor 的 resize / normalize，
以及 query 的文本 tokenization 在 DataLoader worker 中执行；主进程继续执行原来的
GPU 前向。模型权重、精度、推理 batch、图片顺序、文本拼接、归一化和指标协议不变。
没有新增磁盘像素缓存，也没有跨 checkpoint 复用 Student 特征。

## 覆盖路径

- CIRR validation gallery 和 query：默认 `evaluation.num_workers=8`。
- FashionIQ gallery 和 query：同样默认 8 worker，保留特征缓存与断点续评。
- 普通 `Student train gallery`：接入已有 `data.num_workers`，这五组配置为 8。
- `Student train gallery from RAM`：保持已有缓存构建与 RAM 预取逻辑。
- PinPoint / QDiag：已有独立的并行加载实现，本次没有修改。

## 使用

现有五组 runner 的启动命令不变，新启动的进程默认启用并行加载。
可以只覆盖推理加载，不修改训练 DataLoader 或实验配置文件：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
source /mnt/data/datasets/search_dataset/zxl/multi_modal_py3.12/bin/activate
CIR_INFERENCE_NUM_WORKERS=8 CIR_INFERENCE_PREFETCH_FACTOR=2 \
  bash scripts/run_reverse_rank_r128_e4.sh
```

这两个环境变量仅覆盖上述新接入的 CIRR、FashionIQ 和普通建库路径，不覆盖原有的
RAM 像素缓存构建、训练加载或 PinPoint / QDiag。现有 `RUN_INDICES` 等开关仍然可用。
不要在同一实验仍运行时启动重复任务；已经运行的 Python 进程不会热更新。
本次没有主动停止任何任务、启动训练或重新计算已有结果。

单独运行 `evaluate_fashioniq_clip.py` 或 `evaluate_on_policy_distillation.py` 时，
也可追加 `--num-workers 8 --prefetch-factor 2`。
优先级是显式 CLI 参数 > 上述环境变量 > 配置中的 evaluation 默认值。
CIRR / FashionIQ 的 YAML 可使用：

```yaml
evaluation:
  batch_size: 256
  num_workers: 8
  prefetch_factor: 2
  pin_memory: true
```

日志会打印 `[inference loader] ... num_workers=8 prefetch_factor=2 ...`。
`num_workers=0` 为主进程串行回退。每个 worker 的 PyTorch CPU 线程数限制为 1，
使用 spawn，避免从已初始化 CUDA 的主进程 fork。

## 内存与测量

并行预取会增加 CPU 内存及 `/dev/shm` 占用。仅预取的 float32 像素张量，粗略占用为：

`workers × prefetch_factor × batch_size × 3 × H × W × 4 bytes`

例如 CLIP-L 224×224、batch=512、8 worker、prefetch=2，仅这部分约 4.6 GiB；
还要给解码中的原图、worker 进程和当前 batch 留出余量。
容器共享内存很小时可能出现 shared-memory / bus-error 错误。
先检查 `df -h /dev/shm`，必要时降低到 4 worker / prefetch=1，或使用 0 worker 回退；
容器启动时配置充足共享内存由用户决定，代码不会修改容器或共享内存配置。

先对照 0、4、8 worker 测同一 checkpoint 的总耗时。首批包含 worker 启动开销，
因此同时观察后续每批速度和全阶段时间。不要直接复用已经完成的 FashionIQ 缓存来
衡量加载加速，否则实际不会触发编码；测速可用单独的临时输出目录。
当前开发环境无 CUDA，只有 CPU 正确性检查，不宣称任何 GPU 实测加速倍率。
