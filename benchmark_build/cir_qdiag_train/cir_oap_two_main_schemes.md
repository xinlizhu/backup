# CIR 中基于 Object / Attribute / Predicate 的两种语义增强方案

## 1. 研究目标

当前 CIR（Composed Image Retrieval）模型通常将：

\[
(\text{reference image}, \text{modification text})
\]

编码成一个统一的 composed query embedding：

\[
q=F(I_r,t),
\]

再与 candidate image embedding：

\[
z=G(I_c)
\]

计算相似度：

\[
S_{\mathrm{base}}(q,z)=\operatorname{sim}(q,z).
\]

现有实验已经表明，基于 ADD / SUBSTITUTE negative modification text 的训练可以显著增强模型对错误文本的区分能力，但这种能力向最终 Recall / mAP 的转化并不充分。

因此，后续方法不再把目标定义为进一步提高 negative-text classification accuracy，而是尝试显式建模 CIR 中真正需要的三个语义匹配维度：

\[
\boxed{
\mathrm{Object},
\qquad
\mathrm{Attribute},
\qquad
\mathrm{Predicate}
}
\]

其中：

- **Object Matching**：目标对象是否正确；
- **Attribute Matching**：颜色、材质、大小、状态等属性是否正确；
- **Predicate Matching**：动作、关系、空间位置等 predicate 是否正确。

已有六类 harmful negatives：

\[
\{\mathrm{Object},\mathrm{Attribute},\mathrm{Predicate}\}
\times
\{\mathrm{Add},\mathrm{Substitute}\}
\]

不再对应六个模型通道。

Add / Substitute 只作为制造 counterfactual corruption 的方式：

\[
\{\text{Object-Add},\text{Object-Substitute}\}
\rightarrow
\text{Object supervision},
\]

\[
\{\text{Attribute-Add},\text{Attribute-Substitute}\}
\rightarrow
\text{Attribute supervision},
\]

\[
\{\text{Predicate-Add},\text{Predicate-Substitute}\}
\rightarrow
\text{Predicate supervision}.
\]

最终只保留三个真正参与 CIR inference 的 semantic matching component。

---

# 2. 统一设计原则

两个方案都遵循同一个总体框架：

\[
\boxed{
S_{\mathrm{final}}
=
S_{\mathrm{global}}
+
\beta S_{\mathrm{semantic}}
}
\]

其中：

\[
S_{\mathrm{semantic}}
=
S_O+S_A+S_P.
\]

Global branch 保留当前 CIR baseline 已经具备的全局 retrieval 能力。

Object / Attribute / Predicate branch 只负责提供一个较小的 semantic residual correction。

第一版建议：

\[
\beta \in [0.05,0.2],
\]

默认：

\[
\beta=0.1.
\]

核心设计原则是：

> **Global branch 负责找得到，O/A/P branch 负责找得更准。**

---

# 3. 统一训练机制：Counterfactual Semantic Patching

两个方案虽然生成 O/A/P representation 的方式不同，但使用相同的训练机制。

## 3.1 Positive Query

对于：

\[
(I_r,t^+,I_t),
\]

计算：

\[
q_g^+,
\quad
q_O^+,
\quad
q_A^+,
\quad
q_P^+.
\]

完整 positive query 表示为：

\[
Q^+
=
[
q_g^+,
q_O^+,
q_A^+,
q_P^+
].
\]

## 3.2 Object Corruption

对于 Object-Add 或 Object-Substitute negative：

\[
t_O^-,
\]

计算 negative query 的 Object representation：

\[
q_O^-.
\]

但不直接使用完整 negative query。

构造：

\[
\widetilde Q_O
=
[
q_g^+,
q_O^-,
q_A^+,
q_P^+
].
\]

也就是只将 Object component 从 clean representation 替换成 corrupted representation。

要求：

\[
S(Q^+,z^+)
>
S(\widetilde Q_O,z^+).
\]

## 3.3 Attribute Corruption

构造：

\[
\widetilde Q_A
=
[
q_g^+,
q_O^+,
q_A^-,
q_P^+
].
\]

要求：

\[
S(Q^+,z^+)
>
S(\widetilde Q_A,z^+).
\]

## 3.4 Predicate Corruption

构造：

\[
\widetilde Q_P
=
[
q_g^+,
q_O^+,
q_A^+,
q_P^-
].
\]

要求：

\[
S(Q^+,z^+)
>
S(\widetilde Q_P,z^+).
\]

## 3.5 为什么使用 Patching

普通 hard-negative loss：

\[
S(q^+,z^+)
>
S(q^-,z^+)
\]

允许整个 query representation 一起变化。

因此 Object negative 可能通过 global embedding、Attribute representation 和 Predicate representation 共同完成 separation。

Counterfactual patching 则固定其他 component：

\[
\boxed{
\text{Object corruption}
\rightarrow
\text{Object matching score}
}
\]

\[
\boxed{
\text{Attribute corruption}
\rightarrow
\text{Attribute matching score}
}
\]

\[
\boxed{
\text{Predicate corruption}
\rightarrow
\text{Predicate matching score}
}
\]

Add / Substitute 只是同一种 semantic corruption 的不同数据增强方式。

---

# 4. 方案一：Semantic Residual Adapters

## 4.1 核心思想

该方案是第一轮最推荐实现的方法。

它借鉴 Residual Learning、Adapter、LoRA、Low-rank feature correction 等思想。

核心目标：

> **保持原有 Global CIR embedding 不被破坏，只学习三个轻量语义 residual matcher。**

## 4.2 Base Representation

现有 CIR query：

\[
q_g=F(I_r,t).
\]

Target image：

\[
z_g=G(I_t).
\]

Global score：

\[
S_g
=
\operatorname{sim}(q_g,z_g).
\]

这一部分尽可能保留现有 Full baseline 的 retrieval geometry。

## 4.3 Query Semantic Adapters

增加三个独立小模块：

\[
q_O=A_O(q_g),
\]

\[
q_A=A_A(q_g),
\]

\[
q_P=A_P(q_g).
\]

推荐第一版：

```text
768
 ↓
Linear 768 -> 128
 ↓
GELU
 ↓
Linear 128 -> 64
 ↓
LayerNorm
 ↓
L2 Normalize
```

每个 adapter 输出：

\[
64\text{-d}.
\]

## 4.4 Target Semantic Adapters

Candidate image 使用三个对应模块：

\[
z_O=B_O(z_g),
\]

\[
z_A=B_A(z_g),
\]

\[
z_P=B_P(z_g).
\]

第一版建议 Query / Target adapter 不共享参数，因为 composed multimodal representation 和 image representation 的输入分布不同。

## 4.5 Semantic Scores

定义：

\[
S_O
=
\operatorname{sim}(q_O,z_O),
\]

\[
S_A
=
\operatorname{sim}(q_A,z_A),
\]

\[
S_P
=
\operatorname{sim}(q_P,z_P).
\]

最终：

\[
\boxed{
S_{\mathrm{Adapter}}
=
S_g
+
\frac{\beta}{3}
(
S_O+S_A+S_P
)
}
\]

第一版建议：

\[
\beta=0.1.
\]

## 4.6 ANN 检索兼容

可以拼接：

\[
Q=
[
q_g;
\sqrt{\beta/3}q_O;
\sqrt{\beta/3}q_A;
\sqrt{\beta/3}q_P
],
\]

\[
Z=
[
z_g;
\sqrt{\beta/3}z_O;
\sqrt{\beta/3}z_A;
\sqrt{\beta/3}z_P
].
\]

则：

\[
Q^\top Z
=
S_{\mathrm{final}}.
\]

因此不需要单独 reranker。

---

# 5. 方案一的第一轮训练方式

直接从当前 Full baseline checkpoint 初始化。

第一轮建议：

- 冻结 image encoder；
- 冻结 text encoder；
- 冻结 fusion head；
- 冻结 global projection；
- 只训练 6 个 semantic adapters。

这样可以回答一个干净问题：

> 在不改变原有 CIR retrieval geometry 的前提下，仅增加 O/A/P semantic matching component 能否改善 retrieval？

训练同时包含：

### 普通 CIR Ranking

\[
S(Q_i^+,z_i)
>
S(Q_i^+,z_j).
\]

### Object Patching

\[
S(Q^+,z^+)
>
S(\widetilde Q_O,z^+).
\]

### Attribute Patching

\[
S(Q^+,z^+)
>
S(\widetilde Q_A,z^+).
\]

### Predicate Patching

\[
S(Q^+,z^+)
>
S(\widetilde Q_P,z^+).
\]

---

# 6. 方案一的优缺点

## 优点

- 改动最小；
- 最容易稳定训练；
- 可以完全保护 Full baseline 的 global retrieval geometry；
- 推理成本小；
- 可以直接判断 O/A/P decomposition 是否有价值。

## 风险

Adapter 输入的是已经压缩后的：

\[
768\text{-d global embedding}.
\]

如果 Object / Attribute / Predicate 的细粒度信息在 global pooling 前已经丢失，adapter 无法恢复。

这正是方案二存在的主要原因。

---

# 7. 方案二：Semantic Attention Tokens

## 7.1 核心思想

方案二不从最终 global embedding 中再分解 O/A/P。

而是在更早的 token-level representation 上加入三个 learnable semantic query token：

```text
[OBJ]
[ATTR]
[PRED]
```

让它们直接从 reference image patches 和 modification text tokens 中提取对应语义。

该方法借鉴 Transformer attention、special token、query token、Q-Former、DETR object query 等经典结构。

---

# 8. Query-side Semantic Attention

假设 reference image 得到视觉 token：

\[
V_r=
[v_1^r,\ldots,v_N^r].
\]

Modification text 得到：

\[
T=
[t_1,\ldots,t_M].
\]

拼接：

\[
H_q=
[
V_r;
T
].
\]

加入三个 learnable query：

\[
e_O,
\quad
e_A,
\quad
e_P.
\]

经过 lightweight cross-attention block：

\[
q_O
=
\operatorname{CrossAttn}
(e_O,H_q),
\]

\[
q_A
=
\operatorname{CrossAttn}
(e_A,H_q),
\]

\[
q_P
=
\operatorname{CrossAttn}
(e_P,H_q).
\]

第一版只使用：

```text
1 Layer Cross-Attention
+
FFN
+
Residual
+
LayerNorm
```

不需要大 Transformer。

---

# 9. Attention 与自然语言指代

方案二最大的价值之一，是不需要显式解析文本。

例如：

```text
make it red
```

不需要标注：

```text
it -> 哪个 object
```

Attribute token：

\[
e_A
\]

可以同时 attend：

- `red` token；
- `it` token；
- reference image 中相关视觉 patch。

因此模型可以自己学习 soft binding：

\[
\text{textual reference}
\leftrightarrow
\text{visual entity}.
\]

不需要 object span、attribute span、dependency parsing、scene graph 或 explicit grounding label。

---

# 10. Target-side Semantic Attention

Candidate image 保留 patch tokens：

\[
V_c=
[v_1^c,\ldots,v_N^c].
\]

使用三个 target semantic queries：

\[
e_O^z,
\quad
e_A^z,
\quad
e_P^z.
\]

得到：

\[
z_O
=
\operatorname{CrossAttn}
(e_O^z,V_c),
\]

\[
z_A
=
\operatorname{CrossAttn}
(e_A^z,V_c),
\]

\[
z_P
=
\operatorname{CrossAttn}
(e_P^z,V_c).
\]

然后：

\[
S_O
=
\operatorname{sim}(q_O,z_O),
\]

\[
S_A
=
\operatorname{sim}(q_A,z_A),
\]

\[
S_P
=
\operatorname{sim}(q_P,z_P).
\]

最终：

\[
\boxed{
S_{\mathrm{Attention}}
=
S_g
+
\frac{\beta}{3}
(
S_O+S_A+S_P
)
}
\]

---

# 11. Predicate Branch 的特殊性

Predicate 与 Object / Attribute 不完全相同。

例如：

```text
dog left of chair
person holding umbrella
cup on table
man standing behind car
```

Predicate 需要建模对象之间的关系，因此可能需要更强的全局 interaction。

第一版仍建议三个 semantic token 使用完全相同结构。

如果后续发现 Predicate 明显较弱，再单独将 Predicate branch 改成：

```text
2-layer attention block
```

或者：

```text
Predicate Query
    ↓
Cross Attention to all patches
    ↓
Self Attention / FFN
```

第一版不要单独复杂化 Predicate。

---

# 12. 方案二的 Counterfactual Patching

Attention Token 方案使用与方案一相同的 semantic patching。

例如 Object negative 得到：

\[
q_O^-.
\]

构造：

\[
\widetilde Q_O
=
[
q_g^+,
q_O^-,
q_A^+,
q_P^+
].
\]

要求：

\[
S(Q^+,z^+)
>
S(\widetilde Q_O,z^+).
\]

Attribute 与 Predicate 同理。

因此：

- Attention 负责发现语义信息；
- Counterfactual Patching 负责 semantic specialization；
- Retrieval score 负责最终任务监督。

---

# 13. 为什么 Attention 方案可能超过 Adapter

Semantic Adapter：

\[
\text{先压缩，再语义分解}.
\]

Semantic Attention Token：

\[
\text{先语义读取，再分别压缩}.
\]

如果 O/A/P 细粒度信息在 global embedding 中损失较多，Attention Token 应该明显优于 Adapter。

它尤其可能更适合：

- attribute localization；
- pronoun / deictic reference；
- object binding；
- spatial relation；
- action / predicate reasoning。

---

# 14. 两个方案统一使用一个 Ranking Objective

不设计三个独立 loss。

定义所有希望满足的 score ordering：

\[
\mathcal O_i.
\]

包含：

\[
S(Q_i^+,z_i)
>
S(Q_i^+,z_j),
\]

\[
S(Q_i^+,z_i)
>
S(\widetilde Q_{O,i},z_i),
\]

\[
S(Q_i^+,z_i)
>
S(\widetilde Q_{A,i},z_i),
\]

\[
S(Q_i^+,z_i)
>
S(\widetilde Q_{P,i},z_i).
\]

统一目标：

\[
\boxed{
\mathcal L_i
=
\log
\left[
1+
\sum_{(a,b)\in\mathcal O_i}
\exp
\left(
\frac{S(b)-S(a)+m}{\tau}
\right)
\right]
}
\]

因此方法仍然只有一个核心训练逻辑：

> 正确 retrieval score 应高于错误 candidate 或 semantic corruption 对应的 score。

---

# 15. 第一轮实验设计

第一轮重点不是追求最终最强性能，而是判断：

> O/A/P semantic matching architecture 本身是否有效。

## Experiment A：Full Baseline

现有 Full baseline。

## Experiment B：Semantic Residual Adapter

初始化：

```text
Full baseline
```

冻结：

```text
image encoder
text encoder
fusion head
global projection
```

训练：

```text
query O/A/P adapters
target O/A/P adapters
```

使用普通 CIR ranking + O/A/P counterfactual patching。

## Experiment C：Semantic Attention Tokens

同样从 Full baseline 初始化。

尽可能冻结原 backbone。

新增：

```text
3 query semantic tokens
3 target semantic tokens
lightweight attention block
semantic projection
```

Score 与 Adapter 完全相同：

\[
S_g+
\frac{\beta}{3}
(S_O+S_A+S_P).
\]

这样 B / C 的主要区别只有：

> semantic representation 如何生成。

---

# 16. 推荐第一轮超参数

### Semantic dimension

\[
d_s=64.
\]

### Residual weight

测试：

\[
\beta
\in
\{
0.05,
0.1,
0.2
\}.
\]

主实验先：

\[
\beta=0.1.
\]

### Optimizer

AdamW。

新增模块学习率：

\[
10^{-4}.
\]

Weight decay：

\[
0.01.
\]

Warmup：

\[
5\%.
\]

### Epoch

第一轮：

\[
5
\]

个 epoch 先判断趋势。

---

# 17. 关键评估指标

不能只看 Recall。

需要同时报告：

## Standard Retrieval

- CIRR R@1；
- R@5；
- R@10；
- Group R@1；
- Group R@2；
- Group R@3；
- PinPoint mAP@10；
- PinPoint NegRecall@10。

## Semantic Branch Diagnostic

分别计算：

\[
S_O,
\quad
S_A,
\quad
S_P.
\]

对于 Object corruption：

\[
\Delta_O
=
S_O(q^+,z^+)
-
S_O(q_O^-,z^+).
\]

Attribute：

\[
\Delta_A
=
S_A(q^+,z^+)
-
S_A(q_A^-,z^+).
\]

Predicate：

\[
\Delta_P
=
S_P(q^+,z^+)
-
S_P(q_P^-,z^+).
\]

还应检查 cross-component response。

理想情况下，Object corruption 应主要影响 Object component，而不是同时影响三个 component。

---

# 18. 最重要的真实 Candidate 诊断

对真实 retrieval Top-K candidates：

\[
z_j
\]

记录：

\[
[
S_g,
S_O,
S_A,
S_P
].
\]

重点比较：

- positive target；
- rank-violating negatives。

其中 rank-violating negative 定义为：

\[
S_g(q,z^-)
>
S_g(q,z^+).
\]

真正希望看到的是：

\[
S_g(z^+)
<
S_g(z^-)
\]

但加入 O/A/P 后：

\[
S_{\mathrm{final}}(z^+)
>
S_{\mathrm{final}}(z^-).
\]

这才说明 semantic component 真正修复了 retrieval ranking。

这个指标的重要性高于单独的 negative-text Win Rate。

---

# 19. 两个方案的公平比较

最终建议输出：

| 模型 | CIRR R@1 | Group R@1 | PinPoint mAP | NegRecall | O/A/P specialization |
|---|---:|---:|---:|---:|---:|
| Full baseline | | | | | - |
| Residual Adapter | | | | | |
| Attention Token | | | | | |

同时记录：

- 新增参数量；
- 训练耗时；
- inference FLOPs；
- embedding 维度；
- Top-K ranking correction rate。

---

# 20. 对两个方案的预期

## Semantic Residual Adapter

更可能：

- 保留 baseline Recall；
- 小幅改善 Group Recall / NegRecall；
- 参数量很小；
- 稳定；
- 容易训练。

如果它有效，说明：

\[
\boxed{
\text{原始 CIR global representation 中已经存在可提取的 O/A/P 信息。}
}
\]

## Semantic Attention Token

更可能：

- 在 Attribute / Predicate 上产生更强 semantic specialization；
- 对指代、关系和局部属性更敏感；
- 有机会超过 Adapter；
- 训练和工程复杂度略高。

如果 Attention Token 明显优于 Adapter，说明：

\[
\boxed{
\text{O/A/P 信息需要在 token-level interaction 阶段显式提取，}
}
\]

而不是简单从最终 global embedding 中恢复。

---

# 21. 当前实验优先级

第一优先级：

\[
\boxed{
\textbf{Semantic Residual Adapter + Counterfactual Patching}
}
\]

第二优先级：

\[
\boxed{
\textbf{Semantic Attention Tokens + Counterfactual Patching}
}
\]

实验顺序建议：

```text
Full Baseline
    ↓
Residual Adapter
    ↓
Residual Adapter + Counterfactual Patching
    ↓
Attention Tokens
    ↓
Attention Tokens + Counterfactual Patching
```

---

# 22. 最终研究假设

这两个方案共同验证的核心假设是：

\[
\boxed{
\text{CIR retrieval failure can be partially decomposed into}
\\
\text{object-, attribute-, and predicate-level mismatches.}
}
\]

结构化 counterfactual text 不是最终目标。

它的作用是提供：

\[
\boxed{
\text{typed semantic corruption supervision}.
}
\]

通过这种监督训练：

\[
S_O,
\quad
S_A,
\quad
S_P,
\]

让模型除了 global semantic similarity 之外，还具有：

- object compatibility；
- attribute compatibility；
- predicate compatibility。

最终希望实现：

\[
\boxed{
\text{Global Retrieval}
+
\text{Structured Semantic Verification}
}
\]

并让 negative-text 中已经观察到的语义敏感性真正转化为真实 candidate image ranking 的提升。
