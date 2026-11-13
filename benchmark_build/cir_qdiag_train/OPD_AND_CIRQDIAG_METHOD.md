# OPD 与 CIR-QDiag 方法总结

## 1. 研究问题与整体框架

Composed Image Retrieval（CIR）输入一张或两张 Reference 图像以及一条修改文本，要求
模型从图库中检索满足修改要求的 Target 图像。设 Reference 为 $I_r$，修改文本为
$m$，CIR 模型生成归一化的 composed-query embedding：

\[
q=f(I_r,m),
\]

图库图像 $y$ 的归一化 embedding 为 $z_y$，检索分数为：

\[
s(q,y)=q^\top z_y.
\]

我们的工作包含两个彼此独立、功能互补的部分：

1. **CIR-QDiag**：把一条正确修改文本局部改成只含一个语义错误的文本，诊断模型是否
   真正理解 object、attribute 和 predicate 等细粒度条件。
2. **On-Policy Distillation（OPD）**：由 Student 当前的检索分布选出最需要学习的候选
   图像，再在这个 Student-induced 支持集上蒸馏 Teacher 的排序分布。

两部分保持严格的数据隔离：OPD 使用原始 CIRR train 训练；CIR-QDiag 来自 PinPoint，
只用于模型诊断，不进入 OPD 训练。

---

## 2. CIR-QDiag 的构造

### 2.1 从 PinPoint 构造检索等价组

CIR-QDiag 的源数据为 PinPoint licensed 数据，共 7,635 条 query。首先清理候选图像数组
中的空值和重复 ID，然后按照以下四项标注的精确组合进行分组：

\[
(I_{r,1},\ I_{r,2}\text{（可选）},\ \mathcal T^+,\ \mathcal T^-_{\mathrm{official}}),
\]

其中 $\mathcal T^+$ 是完整 Positive image ID 集合，
$\mathcal T^-_{\mathrm{official}}$ 是完整官方 Negative image ID 集合。只有四项完全一致的
query 才属于同一组；分组不依赖文本相似度或语言模型判断。因此，组内的 2–6 条
instruction 是具有相同检索标注的 operational paraphrases。

只保留含 2–6 条 instruction 的组后，得到：

| 项目 | 数量 |
|---|---:|
| 合格检索等价组 | 1,683 |
| 合格组内原始 instruction | 6,553 |
| 被排除的 singleton instruction | 987 |
| 被排除的超大组（7–17 条 instruction） | 9 组 |

当前 CIR-QDiag 使用 seed 42 随机抽取其中 500 组，共包含 1,933 条原始正向
instruction。

### 2.2 语义类型与局部编辑类型

对每条正向 instruction $m^+$，构造一个 $3\times3$ 的局部反事实空间：

\[
\text{semantic target}
\in
\{\text{object},\text{attribute},\text{predicate}\},
\]

\[
\text{edit operation}
\in
\{\text{ADD},\text{DELETE},\text{SUBSTITUTE}\}.
\]

三个语义方向分别表示：

- **Object**：实体身份、实体类别或某类实体是否存在；
- **Attribute**：已有实体的颜色、材质、大小、状态等属性；
- **Predicate**：实体的动作或实体之间的关系。

三种编辑操作分别满足严格的跨度约束：

- **ADD**：只插入一个连续语义跨度，删除该跨度即可恢复原文本；
- **DELETE**：只删除一个连续语义跨度，保留其余语义不变；
- **SUBSTITUTE**：只用一个连续跨度替换另一个连续跨度。

除必要的空格和句末标点外，不允许改写其他文本。某个语义单元无法进行可靠的单次
编辑时允许返回 `null`，因此每条 instruction 最多产生 9 条候选，但不要求九个位置全部
存在。

### 2.3 Text-only 生成

每条 instruction 分别进行 ADD、DELETE 和 SUBSTITUTE 三次结构化生成；每次同时返回
object、attribute 和 predicate 三个位置。实际运行的生成模型为 `gpt-5.6-terra`。

生成阶段只接收原始 instruction，不接收 Reference、Positive 或官方 Negative 图像。
这样，生成器只负责提出局部语言反事实，而不能根据评测图像有针对性地编写负例。

### 2.4 Group-level vision-grounded 审核

生成后，对每个检索等价组执行一次统一的多模态审核。审核模型同时接收：

- 该组的 1–2 张 Reference 图像；
- 从完整 Positive 集合中确定性选取的至多 4 张 Positive 图像；
- 该组全部原始 instruction；
- 该组全部生成候选。

PinPoint 官方 Negative 图像不进入生成或审核过程，只在最终数据中保留其 ID。审核模型
同样使用 `gpt-5.6-terra`。

ADD 和 SUBSTITUTE 的审核要求为：新增或替换后的事实必须对所有送审 Positive 图像
明确为假，并且该事实能够从 Reference 和 Positive 图像中可靠判断。DELETE 的语义不同：
它只要求准确移除一个有意义的约束，使新文本自然、完整并且严格弱于原文本，不要求
弱化后的 query 与 Positive 图像矛盾。

所有候选还必须满足：只发生一次编辑、语义类型正确、未编辑内容保持不变、语言自然，
并且声明的 `before_span` / `after_span` 确实能够重构编辑前后的文本。

### 2.5 最终数据规模

500 个组共生成 11,743 条候选，审核后接受 5,576 条局部编辑文本：

| Semantic target | ADD | DELETE | SUBSTITUTE | 合计 |
|---|---:|---:|---:|---:|
| Object | 947 | 643 | 767 | 2,357 |
| Attribute | 411 | 1,383 | 567 | 2,361 |
| Predicate | 240 | 308 | 310 | 858 |
| **合计** | **1,598** | **2,334** | **1,644** | **5,576** |

平均每条原始 instruction 有 2.89 条审核通过的局部编辑文本。

CIR-QDiag 的主诊断集合定义为 ADD 与 SUBSTITUTE：

\[
\mathcal D_{\mathrm{strict}}
=
\mathcal D_{\mathrm{ADD}}
\cup
\mathcal D_{\mathrm{SUBSTITUTE}}.
\]

它包含 3,242 条严格反事实文本，覆盖 471 个组和 1,580 条原始 instruction。按语义方向
划分为：

| Strict semantic subset | ADD | SUBSTITUTE | 合计 |
|---|---:|---:|---:|
| Object | 947 | 767 | 1,714 |
| Attribute | 411 | 567 | 978 |
| Predicate | 240 | 310 | 550 |
| **合计** | **1,598** | **1,644** | **3,242** |

DELETE 的 2,334 条文本表示约束弱化，单独报告，不纳入 strict 主结果。

---

## 3. CIR-QDiag 的诊断协议

### 3.1 正向 query、反事实 query 与 Target

对一条原始 instruction $m_i^+$ 及其局部反事实 $m_{i,n}^-$，使用完全相同的
Reference 图像构造：

\[
q_i^+=f(I_r,m_i^+),
\qquad
q_{i,n}^-=f(I_r,m_{i,n}^-).
\]

对于该检索等价组中的每一张 Positive target $t\in\mathcal T_g^+$，分别计算：

\[
s_{i,t}^+=\operatorname{sim}(q_i^+,z_t),
\qquad
s_{i,n,t}^-=\operatorname{sim}(q_{i,n}^-,z_t).
\]

诊断分数定义为：

\[
\boxed{
\Delta_{g,i,n,t}
=
s_{i,t}^+-s_{i,n,t}^-
}
\]

若模型理解被修改的细粒度条件，原始正确 query 对 Positive target 的分数应当高于只含
一个错误条件的反事实 query，因此 $\Delta$ 应为正。该指标只比较同一模型、同一
Reference、同一 Target 下的一对局部改写，能够消除大量与样本难度和绝对相似度尺度
有关的混杂因素。

### 3.2 Mean Delta 与 Raw Win

Mean Delta 衡量局部错误条件造成的平均相似度下降：

\[
\operatorname{MeanDelta}
=
\mathbb E[\Delta].
\]

Raw Win 将每个 Target 上的比较转成二值结果：

\[
W_{\mathrm{raw}}(g,i,n,t)
=
\mathbb I[\Delta_{g,i,n,t}>0].
\]

Raw Win Rate 是这些二值结果的层次平衡平均。它回答：当正确文本被替换为只含一个局部
语义错误的文本时，模型有多大比例会降低该组 Positive target 的相似度。

### 3.3 用组内同义改写校准 Stable Win

不同 embedding 模型的相似度尺度和正常文本波动不同，直接使用 $\Delta>0$ 可能把轻微
措辞变化造成的数值抖动当作语义辨别能力。因此，CIR-QDiag 为每个模型独立估计正常
波动阈值。

同一检索等价组中的原始 instruction 共享完全相同的检索标注。对任意两条组内正向
instruction $m_i^+,m_j^+$ 以及任意组内 Positive target $t$，计算：

\[
\epsilon_{g,i,j,t}
=
\left|
s(f(I_r,m_i^+),t)
-
s(f(I_r,m_j^+),t)
\right|.
\]

每个组总权重相同，对全部 $\epsilon$ 取 group-balanced 第 95 百分位：

\[
\tau_{95}
=
Q_{0.95}^{\mathrm{group\text{-}balanced}}(\epsilon).
\]

Stable Win 定义为：

\[
W_{\mathrm{stable}}(g,i,n,t)
=
\mathbb I[\Delta_{g,i,n,t}>\tau_{95}].
\]

因此 Stable Win 不只要求方向正确，还要求正负 query 的分数差超过该模型由正常同义
改写产生的 95% 波动范围。由于 $\tau_{95}$ 按模型独立校准，它也避免把某个模型较大的
原始相似度尺度误认为更强的细粒度辨别能力。

### 3.4 Random-10 separation

Delta 只回答 $q^+$ 与 $q^-$ 在自己的 Positive target 上是否分离。为了确认模型仍然
保持正常的 query–image 配对结构，还为每条反事实 query 使用 seed 42 从其他组中确定性、
不放回地抽取 10 个无关样本。

主要控制量包括：

\[
\operatorname{Gap}^{+}_{\mathrm{image}}
=
\mathbb E_{t\in\mathcal T_g^+}s(q^+,t)
-
\mathbb E_{r\in\mathrm{Random10}(g)}s(q^+,r),
\]

\[
\operatorname{Gap}^{-}_{\mathrm{image}}
=
\mathbb E_{t\in\mathcal T_g^+}s(q^-,t)
-
\mathbb E_{r\in\mathrm{Random10}(g)}s(q^-,r).
\]

同时抽取其他组的 10 条正向 query 和 10 条反事实 query，与当前组的 Positive target
比较，得到 own-query 与 random-query separation。随机 ID 对所有待比较模型保持一致，
从而保证跨模型结果可直接配对比较。

### 3.5 分层统计与置信区间

QDiag 中不同组含有不同数量的 instruction、反事实文本和 Positive target。为避免候选
数量较多的组主导总体结果，所有汇总采用层次平衡：

1. 对同一反事实文本的全部 Positive target 求平均；
2. 对同一原始 instruction 下的反事实文本求平均；
3. 对同一检索等价组中的 instruction 求平均；
4. 最后对组求平均，每个组的总权重相同。

对某个子集 $S$，可以写成：

\[
M_S
=
\frac{1}{|\mathcal G_S|}
\sum_{g\in\mathcal G_S}
\frac{1}{|\mathcal I_g^S|}
\sum_{i\in\mathcal I_g^S}
\frac{1}{|\mathcal N_{g,i}^S|}
\sum_{n\in\mathcal N_{g,i}^S}
\frac{1}{|\mathcal T_g^+|}
\sum_{t\in\mathcal T_g^+}
h_{g,i,n,t},
\]

其中 $h$ 可以是 $\Delta$、Raw Win 指示函数或 Stable Win 指示函数。Raw Win 和
Stable Win 的 95% 置信区间通过 group-level bootstrap 得到；
$\tau_{95}$ 也同时报告 group-level bootstrap 置信区间。

### 3.6 报告维度

主结果报告 strict（ADD + SUBSTITUTE）总体指标，并进一步拆分为：

- `strict|object`；
- `strict|attribute`；
- `strict|predicate`；
- ADD；
- SUBSTITUTE。

DELETE 单独报告。每个模型最终得到 Mean Delta、Raw Win、Stable Win、模型专属
$\tau_{95}$、Random-10 separation 及相应覆盖率和置信区间。

需要注意，CIR-QDiag 是一个成对相似度响应诊断：它不需要在完整 PinPoint 图库上检索，
也不把 Delta 或 Win Rate 解释为 Recall、mAP 或 NegRecall。它直接测量模型对一个局部
语义条件发生错误时，Target 相似度是否产生方向正确且足够稳定的响应。

---

## 4. Retrieval OPD

### 4.1 训练样本与检索分布

对 CIRR train 样本：

\[
x=(I_r,m,y^+),
\]

$I_r$ 是 Reference 图像，$m$ 是修改文本，$y^+$ 是黄金 Target。Student 参数为
$\theta$，冻结的 Teacher 参数为 $\phi$。二者对任意图库图像 $y$ 的相似度分别为：

\[
s_\theta(x,y),
\qquad
s_\phi(x,y).
\]

使用温度 $\tau$ 可以在完整图库上定义两个检索策略：

\[
\pi_\theta^\tau(y\mid x)
=
\frac{\exp(s_\theta(x,y)/\tau)}
{\sum_{y'\in\mathcal G}\exp(s_\theta(x,y')/\tau)},
\]

\[
\pi_\phi^\tau(y\mid x)
=
\frac{\exp(s_\phi(x,y)/\tau)}
{\sum_{y'\in\mathcal G}\exp(s_\phi(x,y')/\tau)}.
\]

直接在完整图库上计算两个分布的散度开销较大，而且绝大多数图像是没有训练信息的
easy candidates。因此 OPD 只在 Student 当前最可能访问的局部检索支持集上训练。

### 4.2 Student on-policy Top-K 候选

在第 $e$ 个 epoch 开始时，使用当前 Student $\theta_e$ 重新编码 CIRR train 的完整
gallery 与全部 composed query。排除 Reference 和 Target 后，选择 Student 得分最高的
$K-1$ 张非 Target 图像，并强制加入黄金 Target：

\[
\boxed{
\mathcal C_e^K(x)
=
\{y^+\}
\cup
\operatorname{TopK}_{K-1}
\left{
s_{\theta_e}(x,y):
y\in\mathcal G\setminus\{I_r,y^+\}
\right}
}
\]

因此每条 query 都有且仅有 $K$ 个候选：一个黄金 Target 和 $K-1$ 个当前 Student
最容易混淆的 hard images。候选由 Student 当前策略产生，并在每个 epoch 重新刷新，
所以这是 epoch-level on-policy 的检索蒸馏。

候选选择只确定图像 ID。进入 mini-batch 后，Reference、文本和所有候选图像都由更新后
的当前 Student 重新前向计算 embedding，不使用挖掘阶段保存的旧 Student 分数参与损失。

### 4.3 同一支持集上的 Teacher 与 Student 分布

Teacher 和当前 Student 在同一个 $\mathcal C_e^K(x)$ 上打分，并分别重新归一化：

\[
p_\theta(y\mid x,\mathcal C_e^K)
=
\frac{\exp(s_\theta(x,y)/\tau)}
{\sum_{c\in\mathcal C_e^K(x)}\exp(s_\theta(x,c)/\tau)},
\]

\[
p_\phi(y\mid x,\mathcal C_e^K)
=
\frac{\exp(s_\phi(x,y)/\tau)}
{\sum_{c\in\mathcal C_e^K(x)}\exp(s_\phi(x,c)/\tau)}.
\]

这等价于把完整图库检索分布条件化到 Student 当前的 Top-$K$ 支持集。Teacher 不负责
替 Student 选择候选，而是在 Student 暴露出的决策区域中提供完整的相对排序监督。

### 4.4 Teacher 有效样本条件

强制插入 Target 可以保证每条样本都能计算排序分布，但 Teacher 对某些候选集合本身也
可能不够确定。我们只保留 Teacher 将黄金 Target 排在候选 Top-$R$ 内的样本：

\[
v_\phi(x)
=
\mathbb I
\left[
\operatorname{rank}_{s_\phi}
\left(y^+\mid x,\mathcal C_e^K(x)\right)
\le R
\right].
\]

主设置使用 $R=10$。这个条件只决定该 query 是否进入本轮梯度计算；对于通过条件的
样本，Student 学习 Teacher 在全部 $K$ 个候选上的完整概率分布，而不只学习 Target
是否排第一。

### 4.5 OPD 训练目标

在 Student 产生的候选支持集上，最小化 Student 检索分布到 Teacher 检索分布的
KL 散度：

\[
\boxed{
\mathcal L_{\mathrm{OPD}}(\theta)
=
\frac{1}{\sum_i v_\phi(x_i)}
\sum_i
v_\phi(x_i)
D_{\mathrm{KL}}
\left(
p_\theta(\cdot\mid x_i,\mathcal C_e^K)
\;\middle\|\;
p_\phi(\cdot\mid x_i,\mathcal C_e^K)
\right)
}
\]

展开为：

\[
D_{\mathrm{KL}}(p_\theta\|p_\phi)
=
\sum_{y\in\mathcal C_e^K(x)}
p_\theta(y)
\log\frac{p_\theta(y)}{p_\phi(y)}.
\]

Teacher 参数和 Teacher embedding cache 全程固定；候选 ID 的 Top-$K$ 选择不参与反向
传播；梯度只更新 Student。因为损失定义在 Student 自己的高概率候选区域，Student 会
持续把新的决策边界和新的 hard images 暴露给 Teacher 监督。

### 4.6 完整训练算法

```text
Input:
  CIRR train queries and gallery
  trainable Student
  frozen Teacher
  candidate size K, temperature tau, Teacher target rank R

Precompute:
  Teacher gallery embeddings
  Teacher composed-query embeddings

For epoch e = 1 ... E:
  1. 用当前 Student 编码完整 train gallery 与全部 train queries
  2. 对每条 query 检索 Student Top-(K-1) non-target images
  3. 强制插入黄金 Target，形成固定宽度候选集 C_e^K(x)
  4. 从 Teacher cache 读取同一候选集上的 Teacher scores
  5. 计算 Teacher target rank，并保留 rank <= R 的样本
  6. 对每个 mini-batch：
       a. 当前 Student 重新编码 Reference、文本和候选图像
       b. Teacher 与 Student scores 分别除以 tau 后做 candidate softmax
       c. 计算 D_KL(p_student || p_teacher)
       d. 对有效样本求平均并更新 Student
  7. 下一 epoch 使用更新后的 Student 重新挖掘候选
```

### 4.7 当前主设置

| 项目 | 设置 |
|---|---|
| 训练数据 | CIRR train，28,225 条 query |
| Train gallery | 16,939 张图像 |
| Student | 原始 CLIP ViT-L/14 |
| Teacher | 冻结的 BGE-VL-MLLM-S2 |
| Candidate policy | Student 每个 epoch 动态检索 |
| Candidate size | $K=32$：Target + 31 个 Student hard images |
| Distillation temperature | $\tau=0.05$ |
| Teacher validity | Target rank $\le 10$ |
| Objective | $D_{\mathrm{KL}}(p_\theta\|p_\phi)$ |
| Student optimization | Full finetuning |
| Epochs | 4 |
| Batch size | 32 |
| Head learning rate | $1\times10^{-5}$ |
| Backbone learning rate | $5\times10^{-7}$ |
| Precision | BF16 |
| Scheduler | 5% warmup + cosine decay |

CLIP Student 首先分别编码 Reference 与修改文本。其 composed query 为：

\[
q_\theta
=
\operatorname{norm}
\left[
z_r+z_m+
h_\theta(z_r,z_m,z_r\odot z_m)
\right],
\]

其中 $h_\theta$ 是残差融合头，输入为图像 embedding、文本 embedding 及二者逐维交互；
残差输出层采用零初始化，使训练从标准的
$\operatorname{norm}(z_r+z_m)$ CLIP composition 开始。训练时视觉编码器、文本编码器和
融合头共同更新。

---

## 5. 两部分如何形成闭环

CIR-QDiag 与 OPD 分别回答两个不同问题：

| 部分 | 核心问题 | 数据 | 输出 |
|---|---|---|---|
| CIR-QDiag | 模型是否对一个 object/attribute/predicate 局部错误产生正确且稳定的相似度响应？ | PinPoint 派生诊断集 | Delta、Raw Win、Stable Win、Random-10 separation |
| OPD | 如何在 Student 当前最容易混淆的真实图像候选上学习更强 Teacher 的排序知识？ | 原始 CIRR train | 经过蒸馏的 CIR Student |

OPD 不读取 CIR-QDiag 的反事实文本；CIR-QDiag 也不参与候选挖掘或梯度更新。因此，训练后
在 QDiag 上观察到的变化反映的是排序知识向细粒度 composed-query 表示的迁移，而不是对
诊断文本的直接拟合。
