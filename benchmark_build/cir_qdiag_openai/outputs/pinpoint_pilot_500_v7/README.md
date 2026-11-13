# PinPoint Hard-Negative Text（V7，500 组）

本目录不是原始 PinPoint 数据的副本，而是从 PinPoint 中抽取“共享完全相同检索集合的
instruction 改写组”，再使用 LLM 生成并审核 hard-negative text 得到的派生数据。

本次运行使用：

- Schema / prompt：V7
- 生成模型与审核模型：`gpt-5.6-terra`
- 抽样：随机 500 组，seed 42
- 最终状态：500/500 组生成及审核成功

## 1. 对 PinPoint 做了什么处理

### 1.1 构造 instruction 改写组

只有同时满足以下条件的 PinPoint query 才会被放入同一组：

1. Reference 图像完全一致（每组 1–2 张）；
2. Positive image ID 集合完全一致；
3. 官方 Negative image ID 集合完全一致。

随后仅保留含 **2–6 条 instruction** 的组。因此，本目录中的一组表示多条具有相同
Reference、Positive 集合和官方 Negative 集合的 query，而不是依赖文本相似度猜测出来的
改写关系。

当前 500 组共包含：

- 1,933 条原始 instruction；
- 11,743 条生成候选；
- 5,576 条审核通过的 hard-negative text。

没有进行额外的“PinPoint 源样本质量筛除”。官方 Negative 图像只保留其 ID，不参与
hard-negative text 的生成或审核。

### 1.2 Text-only 生成

每条 instruction 分三次调用 LLM：

1. `ADD`：增加一个语义约束；
2. `DELETE`：删除一个语义约束；
3. `SUBSTITUTE`：替换一个语义约束。

每次调用同时尝试三个语义对象：

- `object`：实体身份、类别或实体是否存在；
- `attribute`：已有实体的属性；
- `predicate`：动作或实体间关系。

因此目标空间为 **3 个 semantic target × 3 个 edit operation = 9 种候选**。生成阶段
完全不看图；当某个组合无法进行可靠的单语义编辑时，模型可以返回 `null`，所以每条
instruction 不一定恰好产生 9 条候选。

编辑约束如下：

- `ADD`：只插入一个连续的 `after_span`，`before_span=null`；
- `DELETE`：只删除一个连续的 `before_span`，`after_span=null`；
- `SUBSTITUTE`：用一个连续的 `after_span` 替换一个连续的 `before_span`；
- 除必要的空格和句末标点外，不能改写其他内容；
- 文本应自然、流畅、接近原 instruction，并只改变一个语义约束。

### 1.3 Vision-grounded LLM 审核

审核按组进行：一组内所有 instruction 的全部候选一次性提交给审核模型。模型看到：

- 该组全部 Reference 图像；
- 从完整 Positive 集合中确定性选择的至多 4 张 Positive 图像；
- 该组全部原 instruction 和全部候选文本。

模型**不看官方 Negative 图像**。

审核规则：

- `ADD` / `SUBSTITUTE`：新增或替换后的事实必须对所有送审 Positive 图像明确为假；
  若任一 Positive 仍然兼容该文本，则以 `positive_compatible` 拒绝；若视觉上无法可靠
  判断，则以 `not_visually_verifiable` 拒绝。
- `DELETE`：不要求删除后的弱化文本与 Positive 图像矛盾。只要它恰好删除一个有意义
  的约束、保留其他语义、语言流畅且严格弱于原 query，就可以接受。随被删约束一起
  删除仅用于语法连接的冠词、介词、逗号或 `and`，仍计为一次语义删除。
- 所有类型都必须满足单次编辑、语义类别正确、语言自然，并与原 query 足够接近。

## 2. 最终结果

`final.jsonl` 共 500 行，每行对应一个 group，而不是一条 hard negative。

| Semantic target | Operation | 生成候选 | 审核通过 | 通过率 |
|---|---:|---:|---:|---:|
| Object | Add | 1,674 | 947 | 56.6% |
| Object | Delete | 918 | 643 | 70.0% |
| Object | Substitute | 1,543 | 767 | 49.7% |
| Attribute | Add | 1,845 | 411 | 22.3% |
| Attribute | Delete | 1,612 | 1,383 | 85.8% |
| Attribute | Substitute | 1,342 | 567 | 42.3% |
| Predicate | Add | 1,094 | 240 | 21.9% |
| Predicate | Delete | 397 | 308 | 77.6% |
| Predicate | Substitute | 1,318 | 310 | 23.5% |
| **合计** |  | **11,743** | **5,576** | **47.5%** |

平均每条 instruction 有 2.89 条通过审核的 hard negative，平均每组有 11.15 条。

拒绝候选的主要原因：

| Rejection code | 数量 | 含义 |
|---|---:|---|
| `positive_compatible` | 3,693 | 至少一张送审 Positive 仍与候选兼容 |
| `not_visually_verifiable` | 1,029 | 无法从所给图像可靠判定 |
| `wrong_semantic_target` | 409 | 实际修改对象与声明的 object/attribute/predicate 不一致 |
| `invalid_delete` | 307 | 不是合法的单约束弱化删除 |
| `other_semantic_failure` | 243 | 其他语义问题 |
| `not_single_edit` | 215 | 未改变、跨度无法还原，或改变了多个语义约束 |
| `not_fluent` | 210 | 文本不自然或不完整 |
| `not_close_or_hard` | 61 | 与原 query 不够接近或不构成困难负例 |

## 3. JSONL 数据结构

### 3.1 Group 级字段

- `schema_version`：当前为 `7.0`。
- `group_id`：稳定的组 ID，由检索集合及 instruction IDs 计算得到。
- `source_dataset`：`PinPoint`。
- `references`：Reference 图像的 `image_id` 和本地 `image_path`。
- `instructions`：本组 2–6 条原 query，以及从 PinPoint 复制的类别、长度和兴趣字段。
- `positive_images`：该组完整的 Positive 图像集合。
- `official_negative_image_ids`：PinPoint 官方 Negative ID，仅保留，不参与本项目审核。
- `candidates`：所有生成候选，包括接受和拒绝的候选，用于审计。
- `negatives`：`candidates` 中所有且仅有 `status="accepted"` 的候选；这是下游应使用的
  最终 hard-negative text 集合。
- `metadata`：阶段状态、数量、模型信息、审核所用 Positive IDs、错误与调用追踪。

### 3.2 Instruction 字段

- `instruction_id`：原始 PinPoint query ID。
- `text`：原始正向 instruction。
- `query_category`、`token_count`、`length_category`、`is_question`、
  `is_imperative`、`l1_interest`：从 PinPoint 原记录复制的辅助标注。

### 3.3 Candidate / Negative 字段

- `negative_id`：由 group、instruction、semantic target 和 operation 组成的唯一 ID。
- `instruction_id`：该负例对应的原 instruction。
- `source_text`：原 instruction 文本。
- `text`：生成的 hard-negative text。
- `semantic_target`：`object`、`attribute` 或 `predicate`。
- `edit_operation`：`add`、`delete` 或 `substitute`。
- `before_span` / `after_span`：声明的局部删除、插入或替换跨度。
- `generation_attempts`：获得该结构化生成结果所用的请求次数。
- `status`：`generated`、`accepted`、`rejected` 或 `failed`。
- `rejection_code`：拒绝类别；通过时为 `null`。
- `selection_reason`：审核模型给出的简短接受/拒绝理由。

`before_span` 和 `after_span` 是生成模型提出的辅助定位信息，不被直接信任；审核模型会
重新核对它们能否从 `source_text` 还原出候选 `text`。

### 3.4 Metadata 中的辅助信息

- `operational_status`：流水线是否成功运行；它不同于语义是否通过。
- `semantic_status`：组内候选为全部通过、部分通过或全部拒绝。
- `num_generated_candidates` / `num_accepted_negatives`：组内候选和最终负例数量。
- `validated_positive_image_ids`：本次实际送入审核模型的 Positive 子集。
- `call_traces`：模型、prompt/input hash、缓存路径、request ID、token usage 和重试次数，
  用于复现及问题追踪。
- `build_fingerprint` / `selection_fingerprint`：代码、prompt、配置和抽样选择的版本指纹，
  用于防止混用旧逻辑输出。
- `errors` / `notes`：运行错误及补充说明。

## 4. 本目录各文件

- `groups.jsonl`：经过精确检索集合分组和随机抽样后的源数据。
- `candidates.jsonl`：Text-only 生成结果，尚未做最终视觉审核。
- `final.jsonl`：完整审核结果；正式使用时读取其中的 `negatives`。
- `audit.html`：可视化逐组审计页面，可查看图像、原 query、候选和审核理由。
- `group_summary.csv`：每组 instruction、图像、候选、通过数量和状态摘要。
- `generation_by_edit.csv`：3×3 类型的生成数量及平均生成尝试次数。
- `selection_by_edit.csv`：3×3 类型的接受、拒绝数量和通过率。
- `selection_reasons.csv`：逐候选的原文、负例、状态、拒绝码和审核理由。
- `cache/`：结构化 API 响应缓存和错误记录，用于断点续跑及复现。

本次运行配置见
[`../../configs/pilot_500.yaml`](../../configs/pilot_500.yaml)，V7 prompt 见
[`../../prompts/`](../../prompts/)。
