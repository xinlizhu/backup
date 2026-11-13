# CIR Semantic Ability Distillation — Single-Teacher MVP Spec

## 1. Goal

验证一个核心假设：

> 如果某个 Teacher 模型在诊断中表现出更强的细粒度语义辨别能力（例如 Object error discrimination），那么是否可以将这种“局部语义扰动响应能力”蒸馏到一个更普通的 Student CIR 模型中。

当前阶段只做 **Single Teacher → Single Student**，不做多 Teacher router，不做复杂多任务 loss。

---

## 2. Data Structure

每个训练样本包含：

- `reference_image`
- `target_image`
- `q_pos`：正确的 composed query
- `q_negs`：多个错误 query，例如 6 个 `q-`
  - 可来自 Object / Attribute / Predicate
  - 当前 MVP 可以只选某一类，例如 Object
  - 每个样本固定 `M` 个 negative query，示例 `M=6`

一个 batch 大小为 `B`：

```text
reference_images: [B]
target_images:    [B]
q_pos:            [B]
q_negs:           [B, M]
```

---

## 3. Core Idea

不直接对齐 Teacher 和 Student 的 embedding vector。

原因：

- Teacher 和 Student embedding 维度可能不同
- 两个模型的 latent space 坐标系本身也不同
- 直接做 `v_student ≈ v_teacher` 没有稳定语义

因此蒸馏的是：

> query 从 `q-` 变成 `q+` 后，对一组 candidate images 的 similarity response pattern。

---

## 4. Semantic Perturbation Vector

对某个模型 `M`：

```text
q_pos_embedding = encoder(reference_image, q_pos)
q_neg_embedding = encoder(reference_image, q_neg)
```

先对 query embedding 和 image embedding 做 L2 normalization。

理论上可以定义：

\[
v = \hat q^+ - \hat q^-
\]

但实现时不需要显式构造 `v`。

直接计算：

\[
\Delta(I_j)
=
\cos(q^+, I_j)
-
\cos(q^-, I_j)
\]

即可。

这里的 `Δ(I_j)` 表示：

> 把错误 query 改成正确 query 后，该 candidate image 的相似度应该变化多少。

---

## 5. Candidate Images

第一版直接使用 **同一 batch 内所有 target images** 作为 candidate set。

对于 batch：

\[
\mathcal{I}
=
\{I_1, I_2, \dots, I_B\}
\]

对于第 `i` 个样本、第 `m` 个 negative query：

\[
F_{i,m}
=
[
\Delta_{i,m,1},
\Delta_{i,m,2},
\dots,
\Delta_{i,m,B}
]
\]

其中：

\[
\Delta_{i,m,j}
=
s(q_i^+, I_j)
-
s(q_{i,m}^-, I_j)
\]

因此：

```text
F_teacher: [B, M, B]
F_student: [B, M, B]
```

这里最后一个维度是 batch 内 candidate image 数。

---

## 6. Teacher Forward

Teacher 全程冻结：

```python
teacher.eval()

for p in teacher.parameters():
    p.requires_grad = False
```

Teacher 需要分别编码：

```text
q_pos_teacher:  [B, D_t]
q_neg_teacher:  [B, M, D_t]
img_teacher:    [B, D_t]
```

全部 L2 normalize。

计算：

```python
teacher_pos_sim = q_pos_teacher @ img_teacher.T
# [B, B]

teacher_neg_sim = einsum(
    q_neg_teacher,
    img_teacher
)
# [B, M, B]
```

更明确地：

```python
teacher_neg_sim = torch.einsum(
    "bmd,nd->bmn",
    q_neg_teacher,
    img_teacher
)
```

将 positive similarity 扩展：

```python
teacher_pos_sim = teacher_pos_sim[:, None, :]
# [B, 1, B]
```

得到：

```python
F_teacher = teacher_pos_sim - teacher_neg_sim
# [B, M, B]
```

Teacher 输出必须：

```python
F_teacher = F_teacher.detach()
```

---

## 7. Student Forward

Student 同样得到：

```text
q_pos_student:  [B, D_s]
q_neg_student:  [B, M, D_s]
img_student:    [B, D_s]
```

Teacher 和 Student 的维度：

```text
D_t != D_s
```

完全允许。

因为最终比较的是：

```text
F_teacher: [B, M, B]
F_student: [B, M, B]
```

而不是原始 embedding。

Student 计算：

```python
student_pos_sim = q_pos_student @ img_student.T
# [B, B]

student_neg_sim = torch.einsum(
    "bmd,nd->bmn",
    q_neg_student,
    img_student
)
# [B, M, B]

F_student = student_pos_sim[:, None, :] - student_neg_sim
# [B, M, B]
```

---

## 8. Distillation Loss

第一版使用：

```text
SmoothL1 / Huber Loss
```

直接对 raw semantic response 做蒸馏：

\[
L_{distill}
=
\operatorname{SmoothL1}
(
F_{student},
F_{teacher}
)
\]

实现：

```python
loss_distill = F.smooth_l1_loss(
    F_student,
    F_teacher.detach(),
    reduction="mean",
)
```

当前阶段：

- 不使用 KL
- 不使用 softmax
- 不使用 cosine loss
- 不做 z-score
- 不 normalize `v`
- 不额外设计复杂 weighting

原因是 `F` 本身已经是有正负号和幅度意义的 similarity difference。

---

## 9. Why SmoothL1

`F` 的每个元素：

\[
\Delta
=
\cos(q^+, I)
-
\cos(q^-, I)
\]

如果 cosine similarity 范围为：

\[
[-1,1]
\]

那么：

\[
\Delta \in [-2,2]
\]

Teacher 和 Student 已经位于同一个标量空间。

SmoothL1 相比 MSE：

- 对普通误差仍然能精细拟合
- 对 Teacher 的异常 response 更稳健
- 不会让少量极端值主导训练

---

## 10. Training Strategy

### MVP 推荐

第一版：

- Teacher：完全冻结
- Student image encoder：冻结
- Student query/composition side：训练

即：

```text
Teacher:
  query side  frozen
  image side  frozen

Student:
  image side  frozen
  query / composition side trainable
```

目的：

> 先验证 semantic query discrimination 能否被迁移，而不是同时让 image space 发生变化。

如果第一版有效，再做 image-side unfreeze ablation。

---

## 11. Important Implementation Detail

不要直接用：

```python
v = q_pos - q_neg
v = normalize(v)
```

然后再和 image 点积。

因为二次 normalize 会破坏：

\[
v^\top I
=
\cos(q^+,I)-\cos(q^-,I)
\]

的严格对应关系。

推荐始终直接写：

```python
delta = sim(q_pos, image) - sim(q_neg, image)
```

---

## 12. Minimal Training Pseudocode

```python
for batch in dataloader:

    ref_images = batch["reference_image"]
    target_images = batch["target_image"]
    q_pos = batch["q_pos"]
    q_negs = batch["q_negs"]   # [B, M]

    # --------------------------------
    # Teacher
    # --------------------------------
    with torch.no_grad():

        t_img = teacher.encode_image(target_images)
        t_q_pos = teacher.encode_query(ref_images, q_pos)
        t_q_neg = teacher.encode_query_multi(ref_images, q_negs)

        t_img = F.normalize(t_img, dim=-1)
        t_q_pos = F.normalize(t_q_pos, dim=-1)
        t_q_neg = F.normalize(t_q_neg, dim=-1)

        t_pos_sim = t_q_pos @ t_img.T
        # [B, B]

        t_neg_sim = torch.einsum(
            "bmd,nd->bmn",
            t_q_neg,
            t_img
        )
        # [B, M, B]

        F_teacher = (
            t_pos_sim[:, None, :]
            - t_neg_sim
        )
        # [B, M, B]

    # --------------------------------
    # Student
    # --------------------------------

    s_img = student.encode_image(target_images)
    s_q_pos = student.encode_query(ref_images, q_pos)
    s_q_neg = student.encode_query_multi(ref_images, q_negs)

    s_img = F.normalize(s_img, dim=-1)
    s_q_pos = F.normalize(s_q_pos, dim=-1)
    s_q_neg = F.normalize(s_q_neg, dim=-1)

    s_pos_sim = s_q_pos @ s_img.T
    # [B, B]

    s_neg_sim = torch.einsum(
        "bmd,nd->bmn",
        s_q_neg,
        s_img
    )
    # [B, M, B]

    F_student = (
        s_pos_sim[:, None, :]
        - s_neg_sim
    )
    # [B, M, B]

    # --------------------------------
    # Distillation
    # --------------------------------

    loss = F.smooth_l1_loss(
        F_student,
        F_teacher,
        reduction="mean"
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 13. Optional Baseline Comparison

建议至少跑：

### A. Student Baseline

原 Student，不做蒸馏。

### B. MSE Distillation

```python
loss = F.mse_loss(F_student, F_teacher)
```

### C. SmoothL1 Distillation

```python
loss = F.smooth_l1_loss(F_student, F_teacher)
```

主实验优先使用 C。

---

## 14. Evaluation

训练后不要只看 Recall。

必须重新跑现有 Reverse Query / Semantic Diagnostic。

重点比较：

### Retrieval

- Recall@1
- Recall@5
- Recall@10
- mAP@10
- nDCG@10
- MRR

### Semantic Ability

重点看训练目标对应的类别，例如 Object：

- raw win rate
- stable win rate
- mean target delta
- own-target delta
- random-target delta
- own-minus-random target
- positive / negative query separation

---

## 15. Desired Result

理想结果不是简单：

```text
所有 Δ 都变大
```

而是 Student 的 response pattern 更接近 Teacher。

例如 Teacher：

```text
[0.35, 0.28, 0.03, 0.01, -0.01, 0.02]
```

Student baseline：

```text
[0.12, 0.10, 0.08, 0.09, 0.07, 0.08]
```

训练后希望：

```text
[0.30, 0.24, 0.04, 0.02, 0.01, 0.03]
```

即：

- 对真正与 semantic perturbation 相关的 image response 增强
- 对无关 image response 保持接近 0
- 避免再次出现“所有 image 都一起被推开”的现象

---

## 16. Main Hypothesis

最终验证的是：

> 一个模型通过诊断得到的特定 semantic discrimination ability，能否通过 retrieval response distillation 被迁移到另一个 embedding / CIR model。

蒸馏对象不是 Teacher embedding 本身，而是：

\[
\boxed{
F(q^+,q^-,\mathcal I)
=
[
s(q^+,I_j)-s(q^-,I_j)
]_{j=1}^{B}
}
\]

即：

> semantic perturbation 在 retrieval space 中产生的 response pattern。

---

## 17. Current Scope

当前版本只实现：

```text
Single Teacher
→ Same q+ / q-
→ Batch target images as candidate set
→ Semantic response tensor F
→ SmoothL1 distillation
→ Frozen Teacher
→ Frozen Student image side
```

暂时不要加入：

- Multi-Teacher routing
- Teacher ensemble
- KL / softmax distillation
- projector between Teacher / Student embedding
- additional semantic classifiers
- additional complicated losses
- per-category adaptive weights
- z-score normalization
- explicit v-vector alignment

先验证最核心假设是否成立。
