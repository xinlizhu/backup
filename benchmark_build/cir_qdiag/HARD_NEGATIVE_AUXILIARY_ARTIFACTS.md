# V5 单条 Hard Negative 保存的辅助数据

V5 不再保存 claim graph、mutation、positive transition 或多层 judgment。每个最终 negative 除文本
外只保留生成和复核真正需要的信息。

## Sample 级

- CIRR `pair_id`、reference/target 的 image id 与路径。
- 原始 `positive_text` 和 CIRR `image_set`。
- `source_screen_decision` 与 `source_screen_reason`：源三元组的保留/淘汰结论及依据。
- `candidates`：六条生成路由产生的全部非空、非重复候选。
- `negatives`：按操作分开的两次批量筛选后被保留的候选，是 `candidates` 的严格子集。
- `metadata`：阶段、运行状态、生成数、入选数、模型、版本、fingerprint、错误和调用 traces。

## Candidate 级

- `negative_id` 和完整 `text`。
- `semantic_target`：`object | attribute | predicate`。
- `edit_operation`：`add | delete | substitute`。
- `construction_mode`：`text_only_editable | vision_grounded`。
- `before_span` / `after_span`：可直接对照原文和候选的编辑锚点。
  - Add：`before_span=null`，`after_span` 是只出现在 negative 的新增短语。
  - Delete：`before_span` 是只出现在 positive 的被删短语，`after_span=null`。
  - Substitute：两者都非空且不同，分别是 positive 中的旧短语和 negative 中的新短语。
  生成后会在本地做这些存在性检查；失败的提议不会进入 `candidates`，也不会新增一次模型审核。
- `change_summary`：生成器用一句话说明做了什么改变。
- `visual_basis`：Vision-grounded 候选所依据的 Target 图像事实，说明 Target 为什么反驳该
  候选；Text-only 固定为 `null`。
- `generation_attempts`：结构化生成实际尝试次数。
- `status`：`generated | accepted | rejected | failed`。
- `selection_reason`：对应操作审核器给出的简短保留或拒绝原因。

## 原始调用证据

每条通过源筛查的样本正常情况下有 9 个 trace（某一审核分组没有候选时会少一次）：

```text
screen_cirr
generate_text_add
generate_text_delete
generate_text_substitute
generate_vision_add
generate_vision_delete
generate_vision_substitute
select_add_substitute
select_delete
```

Trace 保存模型、温度、随机种子、prompt/input hash、缓存路径和尝试次数。完整结构化响应保存在
`cache/<stage>/`，用于复现和错误分析，不再由更多判别器重复解释。

编辑类别由 positive text 和两个锚点决定；Vision-grounded 的唯一图像输入 Target 只提供反事实
证据，不允许决定“原来改了什么”。这会阻止把 Target 中可见但 positive 未写出的颜色/状态当成
Substitute 旧值，也会阻止把新增实体连同属性或动作塞进 Attribute/Predicate。
