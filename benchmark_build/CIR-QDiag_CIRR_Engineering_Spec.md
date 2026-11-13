# CIR-QDiag V5：CIRR Hard Negative 精简工程规格

## 1. 目标

给定 CIRR validation 三元组 `(R, m+, T)`，自动生成一组 query-side hard-negative text `m-`，
用于测试：

\[
s(R,m^+,T)>s(R,m^-,T)
\]

系统不修改图片、不训练模型、不使用 test ground truth，也没有人工审核环节。

## 2. 分类空间

每个候选由三个轴描述：

```text
semantic_target    = object | attribute | predicate
edit_operation     = add | delete | substitute
construction_mode  = text_only_editable | vision_grounded
```

- Object：实体存在、身份或类别。
- Attribute：颜色、材质、大小、数量、状态、视角或构图等一元性质。
- Predicate：动作、姿态或实体间关系，包括空间关系。
- Add：保留原意并增加一个约束。
- Delete：删除一个明确约束，不作替换。
- Substitute：只替换一个值、实体、动作或关系。

两种 construction mode 分开生成，因此理论上最多有 `3 × 3 × 2 = 18` 条候选。任何不可靠
的槽位都应返回 JSON `null`，18 不是配额。

## 3. 固定七次调用

```text
CIRR sample
  -> text-only add        -> object / attribute / predicate / null
  -> text-only delete     -> object / attribute / predicate / null
  -> text-only substitute -> object / attribute / predicate / null
  -> vision add           -> object / attribute / predicate / null
  -> vision delete        -> object / attribute / predicate / null
  -> vision substitute    -> object / attribute / predicate / null
  -> one batch review of every generated candidate
  -> final negatives
```

生成调用直接输出完整 negative text，不再经过 claim parser、mutation planner 或 sentence
realizer。批量筛选只调用一次，不再执行 sample context、text judge、visual judge 和 final judge。

Text-only 请求不传图片，只能依据 positive text。Vision-grounded 请求依次传
Image 1=reference、Image 2=target，并为非空候选输出具体 `visual_basis`。

## 4. CIRR 语义边界

CIRR 是 composed image retrieval，不是要求同一照片内完成局部编辑：

- Reference 和 Target 可以是不同照片、不同主体实例或不同场景。
- Positive text 可以是 target-oriented 描述，不必穷举两张图的全部差异。
- 不存在“主体被完全替换，所以 positive 无效”的全局拒绝规则。
- 只有文本明确使用 `more`、`another`、`farther`、`turn X into Y` 等相对表达时，才必须比较
  Reference 和 Target。

CIRR 提供的 positive 被当作有效查询输入。系统只审核生成的 negative，不先重新判定整个正样本
是否是合法的图像编辑过程。

## 5. 生成响应

每次生成的 operation 和 mode 由调用方固定，语义目标由三个响应槽位固定。模型不重复填写这些
标签。非空槽位结构为：

```json
{
  "negative_text": "Show the full syringes against a gray background.",
  "change_summary": "empty to full",
  "visual_basis": "Image 2 visibly contains empty rather than full syringes."
}
```

Text-only 的 `visual_basis` 固定为 `null`。本地代码只做非空、与 positive 不完全相同和规范化
去重，不用模板替换生成句子，也不进行语义判断。

## 6. 一次批量筛选

筛选器同时接收双图、positive text 和最多 18 个候选。它必须把每个 candidate ID 恰好放入
`selected` 或 `rejected`，每项附一条简短原因。它不能改写句子、改标签或创造新候选。

共同选择标准：

- 文本自然、与 positive 接近。
- 只有一个主要语义变化，其他明确含义被保留。
- semantic target 和 operation 标签与实际变化一致。
- 候选能形成有意义的诊断难例，而不是无语义变化的改写。

操作特定标准：

- Add/Substitute：新增或替换的命题应当与 Target 清楚不兼容。
- Delete：删除指定约束后句子仍自然且严格更弱；不要求与 Target 矛盾。
- Vision-grounded：`visual_basis` 必须受图片支持并真正解释候选的难度。
- Text-only：不要求反事实被 Reference 支持。

不确定、多个同时编辑、标签错误、视觉证据不足或语义重复的候选进入 `rejected`。不存在候选级
连续 `yes/no` 门控，也不存在样本级一票否决。

## 7. 数据与状态

唯一 schema 真相源是 `src/cir_qdiag/schemas.py`，版本为 `5.0`。

Candidate 只保存：

```text
negative_id, text,
semantic_target, edit_operation, construction_mode,
change_summary, visual_basis,
generation_attempts, status, selection_reason
```

状态只有：

```text
generated | accepted | rejected | failed
```

Sample 保存 CIRR 原始字段、全部 `candidates`、最终 `negatives` 和 metadata。`negatives` 必须等于
全部且仅有的 accepted candidates。

## 8. 工程可靠性

保留以下非语义设施：

- JSON Schema 结构化输出。
- 失败重试和原始响应缓存。
- 固定 seed、prompt/input hash 和 call trace。
- build fingerprint 与断点续跑。
- 单路由失败隔离、JSONL 原子写入和稳定顺序。
- 候选 ID partition 检查：未知、重复或遗漏 ID 会触发结构重试并最终 fail closed。

V5 使用独立 prompt version、schema version 和输出目录，禁止 resume 混入 V4 记录。

## 9. 验收

```bash
PYTHONPATH=src python scripts/export_schemas.py
ruff check src scripts tests
PYTHONPATH=src pytest -q
python -m compileall -q src scripts tests
```

关键回归条件：单样本正常路径恰好 6 次生成加 1 次筛选；Text-only 不传图；Vision 和筛选按
Reference、Target 顺序传图；空槽位合法；没有候选时跳过筛选；系统不存在 positive-transition
全局门控。
