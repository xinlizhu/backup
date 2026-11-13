# Rank / OPD 全部模型结果

本文记录 Rank / OPD 主方法、普通 InfoNCE 对照、候选策略消融，以及不同 Student / Teacher
组合的模型矩阵。前半部分的候选策略消融统一从原始 `CLIP ViT-L/14` 初始化，使用冻结的
BGE-VL-MLLM-S2 Teacher；新增模型矩阵覆盖 CLIP-B、CLIP-L、Qwen3-VL-Embedding-2B
Student 以及 S2、Qwen3-VL-Embedding-8B Teacher。各矩阵实验均训练 4 epochs、每轮保存。
另补充 `CLIP-L ← Qwen3-VL-Reranker-2B` 的实验；该实验目前只完成并评估
epoch 1，因此所有表格均显式标为 E1，不与四轮最终 checkpoint 混淆。
本文也记录两项后续消融：其一是在候选支持集中直接插入 composed q− embedding 的
K=16/32/64 sweep；其二是不做 Student 检索和 Teacher 过滤、仅用随机图片候选拟合
Teacher 原始相似度几何的 Random Centered MSE baseline。两项均从原始 CLIP-L 独立
初始化、full finetune 4 epochs。

## CIR-QDiag 的构造

CIR-QDiag 基于 PinPoint 的 composed image retrieval 数据构造。原始 PinPoint 含 7,635
条 query；我们首先按照以下四项完全一致的检索标注进行精确分组：Reference image 1、
可选的 Reference image 2、完整 Positive image ID 集合，以及完整 official Negative
image ID 集合。只保留含 2–6 条 instruction 的组，得到 1,683 个合格组、6,553 条
instruction；随后以 seed 42 随机抽取 500 组作为当前 QDiag，共包含 1,933 条原始正向
instruction。组内文本不是通过文本相似度聚类得到，而是共享完全相同检索标注的
operational paraphrases。

对每条正向 instruction，`gpt-5.6-terra` 在完全不查看图像的情况下，按照

```text
semantic target ∈ {object, attribute, predicate}
edit operation  ∈ {ADD, DELETE, SUBSTITUTE}
```

构造最多 $3\times3=9$ 条局部编辑候选。每条候选只允许改变一个连续语义跨度：ADD
增加一个约束，DELETE 删除一个约束，SUBSTITUTE 用新约束替换原约束；除必要的空格和
标点外，其余文本必须保持不变。无法可靠构造某一单元格时允许返回 `null`，所以并非
每条 instruction 都有九条候选。

生成之后使用同一个 `gpt-5.6-terra` 进行一次 group-level vision-grounded 审核。
审核模型同时看到该组的 1–2 张 Reference 图像、确定性选取的至多 4 张 Positive 图像、
全部原 instruction 和全部候选文本，但不查看 PinPoint 官方 Negative 图像。
ADD/SUBSTITUTE 只有在新增事实对所有
送审 Positive 图像均明确为假时才会通过；DELETE 只要求准确删除一个有意义的约束，
形成自然、严格更弱的 query。所有候选还必须满足单次编辑、语义类型正确和语言流畅；
ADD/SUBSTITUTE 的新事实还必须能够从 Positive 图像中可靠判断。该过程避免使用官方
Negative 图像构造或筛选 q−，防止将 PinPoint 的 retrieval 标注泄漏到诊断文本中。

最终 `final.jsonl` 共 500 行，每行对应一个 group；其中共生成 11,743 条候选，审核
通过 5,576 条（平均每条原 instruction 2.89 条）：

| Semantic target | ADD | DELETE | SUBSTITUTE | 合计 |
|---|---:|---:|---:|---:|
| Object | 947 | 643 | 767 | 2,357 |
| Attribute | 411 | 1,383 | 567 | 2,361 |
| Predicate | 240 | 308 | 310 | 858 |
| **合计** | **1,598** | **2,334** | **1,644** | **5,576** |

QDiag 的严格反事实集合定义为 ADD + SUBSTITUTE，共 3,242 条；它们在保留 Reference
图像的情况下，将原正向文本 $m^+$ 替换为仅有一个错误语义约束的 $m^-$。DELETE 的
2,334 条样本是“约束被移除后的更弱 query”，单独用于弱化查询诊断，不作为严格
negative。因而主要 QDiag 判断检验的是同一 target 是否满足
$s(q^+,t)>s(q^-,t)$，而不是重新定义 PinPoint 的官方图像检索标签。

# On-Policy Rank Distillation（OPD）

对于训练样本 $x=(I_{\mathrm{ref}},t,y^+)$，Student 使用当前参数从图库中检索候选，并强制加入正确的 target：

$$
\mathcal C_\theta^{K}(x)
=
\{y^+\}
\cup
\operatorname{TopK}_{K-1}
\left\{
s_\theta(x,y):y\in\mathcal G\setminus\{y^+\}
\right\}.
$$

在同一个候选集合上，Teacher 和 Student 分别产生经过温度归一化的排序分布：

$$
p_\phi^\tau(y\mid x,\mathcal C_\theta^K)
=
\frac{
\exp\left(s_\phi(x,y)/\tau\right)
}{
\sum_{y'\in\mathcal C_\theta^K}
\exp\left(s_\phi(x,y')/\tau\right)
},
$$

$$
p_\theta^\tau(y\mid x,\mathcal C_\theta^K)
=
\frac{
\exp\left(s_\theta(x,y)/\tau\right)
}{
\sum_{y'\in\mathcal C_\theta^K}
\exp\left(s_\theta(x,y')/\tau\right)
}.
$$

只保留 Teacher 将正确 target 排在候选 Top-$R$ 内的样本：

$$
m_\phi(x)
=
\mathbb I
\left[
\operatorname{rank}_{s_\phi}
\left(y^+\mid x,\mathcal C_\theta^K(x)\right)
\le R
\right].
$$

最终的 Top-$K$ On-Policy Distillation 损失为：

$$
\boxed{
\mathcal L_{\mathrm{OPD}}^{\mathrm{TopK}}(\theta)
=
\frac{
\mathbb E_{x\sim\mathcal D}
\left[
m_\phi(x)\,
D_{\mathrm{KL}}
\left(
p_\phi^\tau(\cdot\mid x,\mathcal C_\theta^K(x))
\;\middle\|\;
p_\theta^\tau(\cdot\mid x,\mathcal C_\theta^K(x))
\right)
\right]
}{
\mathbb E_{x\sim\mathcal D}[m_\phi(x)]
}
}
$$

展开后为：

$$
\mathcal L_{\mathrm{OPD}}^{\mathrm{TopK}}(\theta)
=
\frac{1}{\sum_i m_\phi(x_i)}
\sum_i m_\phi(x_i)
\sum_{y\in\mathcal C_\theta^K(x_i)}
p_\phi^\tau(y\mid x_i)
\log
\frac{
p_\phi^\tau(y\mid x_i)
}{
p_\theta^\tau(y\mid x_i)
}.
$$

如果在实验中未明确说明，我们的具体设置是：

$$
K=32,\qquad R=10,\qquad \tau=0.05.
$$

候选集合由当前 Student 动态检索，因此监督集中在 Student 当前最容易混淆的区域。Teacher 不负责选择候选，只负责在 Student 产生的候选集合内提供完整的软排序分布。

这里优化的是 forward KL：

$$
D_{\mathrm{KL}}
\left(
p_{\mathrm{Teacher}}
\middle\|
p_{\mathrm{Student}}
\right).
$$


这里还有个reverse KL方式，暂时未进行实验

$$
D_{\mathrm{KL}}
\left(
p_{\mathrm{Student}}
\middle\|
p_{\mathrm{Teacher}}
\right).
$$

## 指标口径

- CIRR R@K、Group R@K：百分数，越高越好；共 4,181 条 val query。
- PinPoint：共 7,635 条 query；mAP@10 越高越好，NegRecall@10 越低越好。
- FashionIQ：dress、shirt、toptee 三类宏平均；标准指标为 R@10/R@50，越高越好。
- QDiag strict：只统计 ADD/SUBSTITUTE negatives，共 3,242 条。
- `Delta = sim(q+, target) - sim(q-, target)`；Raw Win 为 `Delta > 0`。
- Stable Win 为 `Delta > tau95`；tau95 由每个模型的同义改写波动单独校准。

## 模型与训练目标

| Model | 候选支持集 | 刷新 | 训练目标 | 有效样本 |
|---|---|---|---|---|
| CLIP-L original | — | — | 无后训练 | — |
| CLIP-L Full InfoNCE | batch negatives | 每 step | target one-hot CE | 全部 query |
| Dynamic Student Rank E8 | target + 当前 Student Top-31 | 每 epoch | Teacher→Student KL | Teacher target rank ≤ 10 |
| CLIP-L ← Qwen3-VL-Reranker-2B (E1) | target + 当前 Student Top-31 | 每 epoch | Reranker→Student KL | Reranker target rank ≤ 10 |
| Dynamic Student Rank E4 | target + 当前 Student Top-31 | 每 epoch | Teacher→Student KL | Teacher target rank ≤ 10 |
| Teacher Top-K Rank E4 | target + Teacher Top-31 | 固定 | Teacher→Student KL | Teacher target rank ≤ 10 |
| Random Rank E4 | target + 31 张随机图 | 每 epoch | Teacher→Student KL | Teacher target rank ≤ 10 |
| Random Centered MSE E4 | target + 31 张随机图 | 每 epoch | `1/K‖(s-t)-mean(s-t)‖²` | 全部 query |
| Dynamic Student InfoNCE E4 | target + 当前 Student Top-31 | 每 epoch | target one-hot CE | 全部 query |
| Dynamic Student InfoNCE + Top-10 E4 | target + 当前 Student Top-31 | 每 epoch | target one-hot CE | Teacher target rank ≤ 10 |
| Mixed q− Rank K=16/32/64 E4 | target + owned composed q− + Student hard images | 每 epoch | Teacher→Student KL | Teacher target 在混合候选中 rank ≤ 10 |
| BGE-VL-MLLM-S2 Teacher | — | — | 冻结 Teacher | — |

`Dynamic Student Rank E8` 是此前较长训练计划得到的主 checkpoint；`Dynamic Student Rank E4`
是为候选策略消融重新训练的四轮学习率计划对照。二者不能仅按 epoch 编号直接比较。

## CIRR val

| Model | R@1 | R@5 | R@10 | R@50 | Group R@1 | Group R@2 | Group R@3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP-L original | 12.7960 | 37.7661 | 51.1600 | 78.4023 | 34.5133 | 58.8854 | 76.6324 |
| Dynamic Student Rank E8 | 44.0804 | 78.0674 | 88.4238 | 97.8235 | 70.1507 | 87.4193 | 95.0012 |
| CLIP-L ← Qwen3-VL-Reranker-2B (E1) | 30.9735 | 65.6302 | 78.9524 | 95.4317 | 56.5654 | 79.1677 | 90.5286 |
| Dynamic Student Rank E4 | **42.4540** | **76.4889** | **87.1562** | **97.4408** | **67.7828** | **86.9648** | **94.4989** |
| Teacher Top-K Rank E4 | 41.7843 | 74.9821 | 85.9364 | 97.1777 | 67.3284 | 85.9603 | 94.2597 |
| Random Rank E4 | 36.3071 | 71.4183 | 83.3533 | 96.9385 | 61.3011 | 82.9467 | 92.8008 |
| Random Centered MSE E4 | 35.9005 | 70.3420 | 82.1813 | 96.3406 | 61.1337 | 82.0617 | 92.2746 |
| Dynamic Student InfoNCE E4 | 33.9153 | 69.4810 | 81.6073 | 95.2643 | 59.1485 | 81.0093 | 91.5331 |
| Dynamic Student InfoNCE + Top-10 E4 | 37.0486 | 73.1404 | 84.1665 | 96.6754 | 62.2339 | 84.0947 | 92.9682 |
| Mixed q− Rank K=16 E4 | 32.8630 | 66.1564 | 78.9524 | 94.5946 | 58.9333 | 80.2200 | 90.9352 |
| Mixed q− Rank K=32 E4 | 31.6431 | 65.3193 | 77.9718 | 94.7381 | 57.4743 | 79.4786 | 90.7438 |
| Mixed q− Rank K=64 E4 | 31.0691 | 64.8170 | 78.3066 | 94.7142 | 56.4697 | 79.0242 | 90.3372 |
| BGE-VL-MLLM-S2 Teacher | 49.4140 | 80.3157 | 89.2849 | 97.5604 | 75.1017 | 90.0981 | 96.1014 |

粗体只在四轮候选策略消融内部比较。Teacher 的 R@1/R@5/R@10 与 Group Recall 仍然
最高；OPD Rank 的 R@50 略高于 Teacher。

## 四轮候选策略的 CIRR 训练曲线

| Run | Epoch | R@1 | R@10 | Group R@1 |
|---|---:|---:|---:|---:|
| Dynamic Student Rank | 1 | 39.9904 | 85.6254 | 66.2999 |
| Dynamic Student Rank | 2 | 41.9039 | 86.7735 | 66.7065 |
| Dynamic Student Rank | 3 | **42.5257** | 87.0605 | **67.9503** |
| Dynamic Student Rank | 4 | 42.4540 | **87.1562** | 67.7828 |
| Teacher Top-K Rank | 1 | 38.3879 | 84.0947 | 65.4150 |
| Teacher Top-K Rank | 2 | 41.3777 | **85.9364** | 67.2806 |
| Teacher Top-K Rank | 3 | **41.7843** | 85.8885 | **67.5197** |
| Teacher Top-K Rank | 4 | **41.7843** | **85.9364** | 67.3284 |
| Random Rank | 1 | 30.9495 | 79.2155 | 55.5130 |
| Random Rank | 2 | 35.3026 | 82.6118 | 60.4162 |
| Random Rank | 3 | **36.3310** | 83.2337 | **61.3011** |
| Random Rank | 4 | 36.3071 | **83.3533** | **61.3011** |
| Dynamic Student InfoNCE | 1 | 26.9314 | 69.5767 | **60.0574** |
| Dynamic Student InfoNCE | 2 | 24.8983 | 68.6917 | 51.4709 |
| Dynamic Student InfoNCE | 3 | 17.1968 | 53.6714 | 50.4664 |
| Dynamic Student InfoNCE | 4 | **33.9153** | **81.6073** | 59.1485 |
| Dynamic Student InfoNCE + Top-10 | 1 | 29.3231 | 74.6711 | 60.7271 |
| Dynamic Student InfoNCE + Top-10 | 2 | 31.3801 | 77.9957 | 56.5893 |
| Dynamic Student InfoNCE + Top-10 | 3 | 35.0634 | 81.6551 | **62.8797** |
| Dynamic Student InfoNCE + Top-10 | 4 | **37.0486** | **84.1665** | 62.2339 |

Dynamic Student Rank 的 R@1/Group R@1 在 epoch 3 最好，R@10 在 epoch 4 最好，说明
三至四轮已进入平台期。未筛选 InfoNCE 在 epoch 2–3 明显退化，epoch 4 又恢复；加入
Teacher Top-10 筛选后，R@1/R@10 四轮单调提升，训练震荡基本消失。

## Reranker as Teacher：epoch 1(事实上我觉得这个可以不看，在学生模型召回的top 32个上面排序，reranker的能力不如其余老师，而且训练用时非常夸张)

该实验从原始 `CLIP ViT-L/14` 独立初始化并进行 full finetune。Student 在 CIRR train
上选择当前 Top-32（target 强制置于候选第 0 位），Qwen3-VL-Reranker-2B 通过
`/v1/score` 对每个 query 的 32 张候选图输出原始 0–1 relevance score；不传显式
instruction。Teacher 与 Student 分数均以固定 `T=0.05` softmax，只保留 Reranker 将
target 排在候选 Top-10 内的 query，优化
`D_KL(p_reranker || p_student)`。本轮共有 28,225 条 train query，训练 883 optimizer
steps；Reranker 候选打分约 7 小时 40 分，模型训练约 36 分钟。

第一轮的候选由同一个原始 CLIP-L 产生，因此 Reranker、S2、Qwen-8B 三种 Teacher
面对的 `28,225 × 32` 候选矩阵完全相同，可直接比较：

| Teacher（相同 E1 candidates） | Target Top-1 | Target Top-10 | Target mean rank | Entropy (`T=0.05`) |
|---|---:|---:|---:|---:|
| BGE-VL-MLLM-S2 | 0.4503 | 0.8728 | 4.6074 | 2.9499 |
| Qwen3-VL-Embedding-8B | 0.4235 | 0.8329 | 5.4780 | 2.3546 |
| Qwen3-VL-Reranker-2B | 0.3929 | 0.8252 | 5.7134 | 1.7774 |

Reranker KL 在第一轮内部确实发生了有效学习。下表使用开头 5 个与结尾 5 个
日志窗口的平均值，而不是单个 minibatch：

| Training proxy | E1 开头 | E1 结尾 | 变化 |
|---|---:|---:|---:|
| KL loss | 1.6925 | 0.9833 | -0.7092 |
| Student target Top-1 | 0.0822 | 0.2294 | +0.1472 |
| Student–Teacher Top-1 agreement | 0.1884 | 0.3616 | +0.1731 |
| Student target margin | -0.0691 | -0.0432 | +0.0259 |

正式 CIRR val 的 epoch-1 对比如下：

| Model（均为 CLIP-L） | R@1 | R@5 | R@10 | R@50 | Group R@1 | Group R@2 | Group R@3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original（E0） | 12.7960 | 37.7661 | 51.1600 | 78.4023 | 34.5133 | 58.8854 | 76.6324 |
| Reranker KL（E1） | 30.9735 | 65.6302 | 78.9524 | 95.4317 | 56.5654 | 79.1677 | 90.5286 |
| S2 Dynamic Student Rank（E1） | 39.9904 | 74.0014 | 85.6254 | 96.7950 | 66.2999 | 85.6254 | 93.9488 |
| S2 Random Rank（E1） | 30.9495 | 65.7737 | 79.2155 | 95.7905 | 55.5130 | 78.5697 | 90.3851 |
| S2-filtered InfoNCE（E1） | 29.3231 | 61.0859 | 74.6711 | 92.4898 | 60.7271 | 81.7029 | 92.4659 |

Reranker E1 相对原始 CLIP-L 提升 `+18.1775` R@1、`+27.7924` R@10 和
`+22.0521` Group R@1，证明其分数可以作为有效的 listwise 蒸馏监督。但它比相同
Student-candidate 协议的 S2 Rank E1 低 `9.0169` R@1、`6.6730` R@10 和 `9.7345`
Group R@1；其标准 Recall 与 S2 Random Rank E1 基本相同。结合上面的 Teacher 诊断，
当前差距与 Reranker 在这些候选上较低的 target Top-1/Top-10 以及更尖锐的软分布一致。
这里只能得出 epoch-1 结论；PinPoint、QDiag、FashionIQ 及后续 epoch 尚未评估。

## PinPoint 官方指标

| Model | mAP@10 ↑ | NegRecall@10 ↓ |
|---|---:|---:|
| CLIP-L original | 0.0462 | 0.2056 |
| Dynamic Student Rank E8 | 0.0890 | 0.1719 |
| Dynamic Student Rank E4 | **0.0886** | 0.1740 |
| Teacher Top-K Rank E4 | 0.0869 | 0.1693 |
| Random Rank E4 | 0.0800 | 0.1927 |
| Random Centered MSE E4 | 0.0808 | 0.1880 |
| Dynamic Student InfoNCE E4 | 0.0651 | 0.1760 |
| Dynamic Student InfoNCE + Top-10 E4 | 0.0723 | 0.1797 |
| Mixed q− Rank K=16 E4 | 0.0772 | 0.1871 |
| Mixed q− Rank K=32 E4 | 0.0758 | 0.1913 |
| Mixed q− Rank K=64 E4 | 0.0739 | 0.1938 |
| BGE-VL-MLLM-S2 Teacher | 0.1262 | 0.1219 |

在保留的四轮消融中，Dynamic Student Rank 的 mAP@10 最高；Teacher Top-K Rank 的
NegRecall@10 最低。Teacher 本身同时保持最高 mAP 和最低 NegRecall。

## QDiag strict ADD/SUBSTITUTE

| Model | tau95 | Mean Delta | Raw Win | Stable Win |
|---|---:|---:|---:|---:|
| CLIP-L original | 0.042547 | 0.006177 | 0.661461 | 0.020157 |
| Dynamic Student Rank E8 | 0.082127 | **0.027603** | 0.756761 | 0.095163 |
| Dynamic Student Rank E4 | 0.071450 | 0.025387 | **0.780768** | 0.098707 |
| Teacher Top-K Rank E4 | 0.047538 | 0.012076 | 0.718977 | 0.053413 |
| Random Rank E4 | 0.059378 | 0.016002 | 0.740030 | 0.058667 |
| Random Centered MSE E4 | 0.053417 | 0.016597 | 0.778631 | 0.069275 |
| Dynamic Student InfoNCE E4 | 0.080798 | 0.023152 | 0.755062 | 0.064538 |
| Dynamic Student InfoNCE + Top-10 E4 | 0.085784 | 0.026884 | 0.764896 | 0.076642 |
| Mixed q− Rank K=16 E4 | 0.046627 | 0.010251 | 0.716884 | 0.040088 |
| Mixed q− Rank K=32 E4 | 0.050394 | 0.011400 | 0.718197 | 0.043166 |
| Mixed q− Rank K=64 E4 | 0.052837 | 0.012440 | 0.721490 | 0.047323 |
| BGE-VL-MLLM-S2 Teacher | 0.071220 | 0.021749 | 0.742554 | 0.091780 |

Dynamic Student Rank 的 strict Raw Win 与 Stable Win 均超过 Teacher。tau95 是各
embedding 空间内部阈值，不能仅按数值大小跨模型判断稳定性。

### 细粒度 Stable Win

| Model | strict Object | strict Attribute | strict Predicate | DELETE |
|---|---:|---:|---:|---:|
| CLIP-L original | 0.0254 | 0.0089 | 0.0005 | 0.0199 |
| Dynamic Student Rank E8 | 0.1090 | 0.0712 | **0.0519** | 0.0115 |
| Dynamic Student Rank E4 | 0.1112 | 0.0798 | 0.0385 | 0.0092 |
| Teacher Top-K Rank E4 | 0.0496 | 0.0549 | 0.0188 | 0.0172 |
| Random Rank E4 | 0.0501 | 0.0640 | 0.0133 | 0.0206 |
| Random Centered MSE E4 | 0.0586 | 0.0829 | 0.0175 | 0.0186 |
| Dynamic Student InfoNCE E4 | 0.0510 | 0.0825 | 0.0161 | 0.0259 |
| Dynamic Student InfoNCE + Top-10 E4 | 0.0640 | 0.1011 | 0.0178 | 0.0232 |
| Mixed q− Rank K=16 E4 | 0.0267 | 0.0608 | 0.0047 | 0.0224 |
| Mixed q− Rank K=32 E4 | 0.0293 | 0.0677 | 0.0045 | 0.0216 |
| Mixed q− Rank K=64 E4 | 0.0336 | 0.0744 | 0.0051 | 0.0196 |
| BGE-VL-MLLM-S2 Teacher | 0.0762 | **0.1112** | 0.0310 | **0.0634** |

Student Rank 对 Object 的提升最明显；Attribute 仍由 Teacher 最强。DELETE 不是当前
Rank 目标直接优化的严格反事实方向，因此仍是主要短板。

## 五种消融的四轮 QDiag 轨迹

下表补充保留的 `5 methods × 4 checkpoints` 的 strict 结果。`Own-random` 是正问句对
自身 target 的相似度减去对随机 target 的平均相似度，用于区分“细粒度反事实能力”与
“一般正负样本分离能力”。

| Model | Epoch | tau95 | Mean Delta | Raw Win | Stable Win | Own-random |
|---|---:|---:|---:|---:|---:|---:|
| Dynamic Student Rank | 1 | 0.064419 | 0.020851 | 0.774187 | 0.078588 | 0.128644 |
| Dynamic Student Rank | 2 | 0.068422 | 0.023923 | 0.781005 | 0.094454 | 0.151873 |
| Dynamic Student Rank | **3** | 0.071231 | 0.025380 | 0.779928 | **0.099628** | 0.150433 |
| Dynamic Student Rank | 4 | 0.071450 | 0.025387 | 0.780768 | 0.098707 | 0.151091 |
| Teacher Top-K Rank | 1 | 0.040829 | 0.007178 | 0.667235 | 0.032797 | 0.081752 |
| Teacher Top-K Rank | 2 | 0.044629 | 0.010148 | 0.702389 | 0.044343 | 0.089835 |
| Teacher Top-K Rank | 3 | 0.046818 | 0.011716 | 0.717794 | 0.052673 | 0.091831 |
| Teacher Top-K Rank | **4** | 0.047538 | 0.012076 | 0.718977 | **0.053413** | 0.092813 |
| Random Rank | 1 | 0.055873 | 0.013132 | 0.724457 | 0.044239 | 0.147865 |
| Random Rank | 2 | 0.058370 | 0.015358 | 0.739165 | 0.054727 | 0.147708 |
| Random Rank | 3 | 0.059461 | 0.016024 | 0.740883 | 0.058259 | 0.148461 |
| Random Rank | **4** | 0.059378 | 0.016002 | 0.740030 | **0.058667** | 0.148011 |
| Dynamic Student InfoNCE | 1 | 0.073720 | 0.018540 | 0.731339 | 0.053774 | 0.125703 |
| Dynamic Student InfoNCE | 2 | 0.071467 | 0.016493 | 0.728173 | 0.041690 | 0.171644 |
| Dynamic Student InfoNCE | 3 | 0.076884 | 0.020098 | 0.741500 | 0.057190 | 0.154687 |
| Dynamic Student InfoNCE | **4** | 0.080798 | 0.023152 | 0.755062 | **0.064538** | 0.210445 |
| Dynamic Student InfoNCE + Top-10 | 1 | 0.078098 | 0.021426 | 0.750030 | 0.061073 | 0.137488 |
| Dynamic Student InfoNCE + Top-10 | 2 | 0.075610 | 0.021677 | 0.758214 | 0.061954 | 0.200561 |
| Dynamic Student InfoNCE + Top-10 | 3 | 0.085102 | 0.026391 | 0.763909 | 0.075940 | 0.190002 |
| Dynamic Student InfoNCE + Top-10 | **4** | 0.085784 | 0.026884 | 0.764896 | **0.076642** | 0.211082 |

### 四轮诊断结论

1. **Rank 在 epoch 3 已经饱和。** Dynamic Rank 的 Stable Win 在 epoch 3 达到
   `0.099628`，epoch 4 为 `0.098707`，只下降 `0.000921`；同时 CIRR R@10 在 epoch 4
   略高。因此没有证据支持继续增加训练轮数，统一报告 epoch 4 是合理的。
2. **Student policy candidates 是 Rank KD 的关键。** Dynamic Rank 第一轮的 Stable
   Win `0.078588` 已高于 Teacher Top-K Rank 第四轮的 `0.053413` 和 Random Rank
   第四轮的 `0.058667`。Teacher-centric 候选或随机候选即使继续训练，也没有追上
   Student 决策边界附近的 hard candidates。
3. **Teacher Top-10 筛选稳定了 InfoNCE，但不能替代软排序。** 筛选版 InfoNCE 在
   四轮的 Stable Win 均高于未筛选版，epoch 4 的增益为 `+0.012104`；Dynamic Rank
   在相同 Teacher Top-10 有效样本规则下又比筛选版 InfoNCE 高 `+0.022065`。这与
   CIRR、PinPoint 和 FashionIQ 上的排序一致。
4. **Rank 学到的是局部语义排序，不是更强的全局分离。** epoch 4 的筛选版 InfoNCE
   具有更高 Mean Delta（`0.026884` 对 `0.025387`）和显著更高 Own-random
   （`0.211082` 对 `0.151091`），但 Dynamic Rank 的 Raw Win/Stable Win 反而更高
   （`0.780768/0.098707` 对 `0.764896/0.076642`），且 tau95 更低。这说明 OPD 的
   收益来自在局部候选集上复制 Teacher 的相对排序并降低语义扰动敏感性，而不是简单
   增大正样本与无关图片之间的距离。
5. **能力提升具有语义方向性。** 保留的五组中最佳 Object 和 Predicate Stable Win
   都来自 Dynamic Rank epoch 3（`0.111564`、`0.041444`）；最佳 Attribute 来自筛选版
   InfoNCE epoch 4（`0.101125`）。DELETE 的最高值只有 `0.027820`，且来自 Teacher
   Top-K Rank epoch 1，进一步确认当前 Rank 目标主要修复 ADD/SUBSTITUTE，不会自动
   修复 DELETE。

完整逐轮 ADD/SUBSTITUTE/DELETE、Object/Attribute/Predicate 表以及机器可读结果见：

- `outputs/rank_candidate_policy_e4/QDIAG_ALL_EPOCHS.md`
- `outputs/rank_candidate_policy_e4/qdiag_all_epochs_summary.csv`

## FashionIQ val 跨数据集迁移

| Model | Average R@10 | Average R@50 |
|---|---:|---:|
| CLIP-L original | 20.2010 | 36.4484 |
| CLIP-L Full InfoNCE | 26.1036 | 46.5442 |
| Dynamic Student Rank E8 | 31.3227 | 52.3614 |
| Dynamic Student Rank E4 | **31.6529** | **52.4326** |
| Teacher Top-K Rank E4 | 31.6225 | 52.3925 |
| Random Rank E4 | 30.8637 | 51.6963 |
| Random Centered MSE E4 | 29.6161 | 50.3958 |
| Dynamic Student InfoNCE E4 | 29.4679 | 49.7322 |
| Dynamic Student InfoNCE + Top-10 E4 | 29.5640 | 50.2682 |
| Mixed q− Rank K=16 E4 | 28.9936 | 49.6810 |
| Mixed q− Rank K=32 E4 | 28.7785 | 49.8109 |
| Mixed q− Rank K=64 E4 | 29.2779 | 50.0896 |
| BGE-VL-MLLM-S2 Teacher | 34.1685 | 55.1285 |

普通 Full InfoNCE 已带来跨数据集增益，但 Student Rank 仍在其上提高约 5.2 点 R@10
和 5.8 点 R@50。保留的四轮消融中 Dynamic Student Rank 的跨数据集迁移最好。

## Candidate mining 诊断

`source target R@32` 是强制注入 target 前，候选来源自身能否召回 target。Teacher
候选固定，因此表中 final 与 epoch 1 相同；这不代表最终 Student 的真实训练集 R@32。

| Run | Epoch-1 source target R@32 | Final mining R@32 | Epoch-1 Teacher target Top-10 | Final Teacher target Top-10 |
|---|---:|---:|---:|---:|
| Dynamic Student Rank | 50.09% | 86.79% | 87.28% | 77.17% |
| Teacher Top-K Rank | 87.59% | 87.59% | 73.30% | 73.30% |
| Random Rank | 0.00% | 0.00% | 99.99% | 99.99% |
| Random Centered MSE | 0.00% | 0.00% | 99.99% | 99.99%（仅记录，不过滤） |
| Dynamic Student InfoNCE | 50.09% | 51.73% | — | — |
| Dynamic Student InfoNCE + Top-10 | 50.09% | 85.27% | 87.28% | 81.56% |

Random candidates 几乎总能让 Teacher 把 target 排进 Top-10，却产生最弱的 Rank 结果，
说明高 Teacher 置信度不等于有效监督；候选必须位于真实、模型相关的混淆区域。

## Random Centered MSE 与 mixed q− candidates

本节记录两项在候选策略消融之后完成的实验。所有 Student 都从原始
`CLIP ViT-L/14` 独立初始化并 full finetune 4 epochs。

### 统一最终结果

| Model | CIRR R@1 | CIRR R@10 | Group R@1 | FashionIQ R@10 | FashionIQ R@50 | PinPoint mAP@10 | PinPoint NegRecall@10 | QDiag Delta | QDiag Raw Win | QDiag Stable Win | QDiag tau95 | QDiag Own-Random |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP-L original | 12.7960 | 51.1600 | 34.5133 | 20.2010 | 36.4484 | 0.0462 | 0.2056 | 0.006177 | 0.661461 | 0.020157 | 0.042547 | 0.126864 |
| Random Rank E4 | 36.3071 | 83.3533 | 61.3011 | 30.8637 | 51.6963 | 0.0800 | 0.1927 | 0.016002 | 0.740030 | 0.058667 | 0.059378 | 0.148011 |
| **Random Centered MSE E4** | **35.9005** | **82.1813** | **61.1337** | **29.6161** | **50.3958** | **0.0808** | **0.1880** | **0.016597** | **0.778631** | **0.069275** | **0.053417** | **0.133379** |
| Dynamic Student Rank E4 | 42.4540 | 87.1562 | 67.7828 | 31.6529 | 52.4326 | 0.0886 | 0.1740 | 0.025387 | 0.780768 | 0.098707 | 0.071450 | 0.151091 |
| Mixed q− Rank K=16 E4 | 32.8630 | 78.9524 | 58.9333 | 28.9936 | 49.6810 | 0.0772 | 0.1871 | 0.010251 | 0.716884 | 0.040088 | 0.046627 | 0.098660 |
| Mixed q− Rank K=32 E4 | 31.6431 | 77.9718 | 57.4743 | 28.7785 | 49.8109 | 0.0758 | 0.1913 | 0.011400 | 0.718197 | 0.043166 | 0.050394 | 0.107970 |
| Mixed q− Rank K=64 E4 | 31.0691 | 78.3066 | 56.4697 | 29.2779 | 50.0896 | 0.0739 | 0.1938 | 0.012440 | 0.721490 | 0.047323 | 0.052837 | 0.115664 |
| BGE-VL-MLLM-S2 Teacher | 49.4140 | 89.2849 | 75.1017 | 34.1685 | 55.1285 | 0.1262 | 0.1219 | 0.021749 | 0.742554 | 0.091780 | 0.071220 | 0.161308 |

粗体的 Random Centered MSE 行是本轮新增 baseline，不表示各列全局最优。PinPoint
NegRecall@10 越低越好；其余表中性能指标越高越好。QDiag tau95 是各 embedding 空间
内部的同义改写噪声阈值，不应单独按大小跨模型排序。

### Random Centered MSE baseline（这是另外一个蒸馏embedding的方法，今年7月刚发布在arxiv上，可以算是个baseline）

该实验不做 Student 全图库召回：每条 query 的支持集为 target 加 31 张随机且不重复的
非 target、非 reference 图片，每个 epoch 重新采样。它不做任何 Teacher Top-1/Top-10
过滤，全部 28,225 条 query 参与训练。Teacher 和 Student 的原始候选相似度分别为
`t`、`s`，优化：

$$
\mathcal{L}_{\mathrm{CMSE}}
=\frac{1}{B}\sum_{b=1}^{B}\frac{1}{K}
\left\|(s_b-t_b)-\text{mean}_k(s_b-t_b)\right\|_2^2.
$$


不使用 softmax、KL 或 temperature scaling。它只移除每条 query 上 Student–Teacher
残差的整体加性偏置，直接匹配候选之间的原始相似度几何。

| Epoch | Centered MSE（epoch 末） | Centered residual MAE | CIRR R@1 | R@10 | Group R@1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.000989 | 0.024467 | 31.1887 | 78.3545 | 56.0631 |
| 2 | 0.000842 | 0.022541 | 34.8242 | 81.1289 | 60.2487 |
| 3 | 0.000789 | 0.022022 | 35.4461 | **82.4683** | 61.0380 |
| 4 | 0.000776 | 0.021642 | **35.9005** | 82.1813 | **61.1337** |

与相同初始化、相同 seed、相同随机候选和相同四轮 schedule 的 Random Rank 相比，
Centered MSE 的 CIRR R@1/R@10 分别低 `0.4066/1.1720`，FashionIQ R@10 低
`1.2476`；但 PinPoint mAP 从 `0.0800` 增至 `0.0808`、NegRecall 从 `0.1927`
降至 `0.1880`，QDiag Raw/Stable Win 从 `0.740030/0.058667` 增至
`0.778631/0.069275`。其 strict Raw Win 95% CI 为 `[0.760891, 0.795803]`，与
Random Rank 的 `[0.721282, 0.757293]` 基本不重叠；Stable Win CI 仍有重叠，不能据此
声称稳定胜率显著提高。随机候选下 S2 将 target 排为 Top-1 的比例约 97%，说明这条
baseline 主要匹配易负例上的全局相似度几何，缺少 Student 决策边界附近的 hard images。

### Mixed q− candidate K sweep

每条训练样本把 composed q− embedding 直接插入候选支持集：

`target image + owned q− embeddings + Student 动态 hard images`

固定总宽度分别为 K=16/32/64，平均每条样本插入 9.11 个 q−；K=16 最多保留 14 个
q− 以确保至少有一张 hard image，整个训练集只截断 35 个 q−。Teacher 和 Student 都
计算 q+ 对该混合支持集的相似度，以 `T=0.05` 做 forward KL，并要求 Teacher target
在混合候选 Top-10 内。

| K | E1 raw image target R@K | E4 raw image target R@K | E1 Teacher target Top-10 | E4 Teacher target Top-10 | E4 Teacher target mean rank | 4-epoch train time |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.3727 | 0.6594 | 0.3039 | 0.2498 | 11.80 | 1.05 h |
| 32 | 0.5009 | 0.7802 | 0.2531 | 0.2099 | 14.31 | 2.14 h |
| 64 | 0.6421 | 0.8760 | 0.2261 | 0.1958 | 17.23 | 4.26 h |

相同总宽度 K=32 的 image-only Dynamic Student Rank 在 epoch 4 的 Teacher target
Top-10 为 `0.7717`，而插入 q− 后降至 `0.2099`；Teacher target Top-1 也从
`0.3452` 降至 `0.0157`。q+ 与 q− 共享 reference image，且通常只改变一个细粒度
语义，因此 query–query 相似度很高，混合排序中 q− 经常压过 target image。由此约
75%–80% 样本被 Top-10 规则过滤；对剩余样本，KL 仍然是在模仿 Teacher 给 q− 的高分，
而不是显式把 q− 推开。

三组 mixed-q− 实验内部，K=16 的 CIRR、PinPoint 与大部分常规检索指标最好；K=64 的
QDiag Delta、Raw/Stable Win、Own-Random 最高，但三组均低于 image-only Dynamic
Student Rank。这个结果表明 composed q− 是 query-side counterfactual，不适合直接与
image candidates 放进同一个 listwise softmax；它更适合作为独立 query 测量或监督其
对纯图像候选排序的响应。

## Teacher Top-10 筛选与软排序的贡献分解

三组 Dynamic Student Top-K 使用相同初始化、数据、K、优化器和训练轮数：

| Model | CIRR R@1 | R@10 | Group R@1 | PinPoint mAP@10 | QDiag Stable Win | FashionIQ R@10 |
|---|---:|---:|---:|---:|---:|---:|
| InfoNCE，全部 query | 33.9153 | 81.6073 | 59.1485 | 0.0651 | 0.064538 | 29.4679 |
| InfoNCE，Teacher Top-10 query | 37.0486 | 84.1665 | 62.2339 | 0.0723 | 0.076642 | 29.5640 |
| Rank KD，Teacher Top-10 query | 42.4540 | 87.1562 | 67.7828 | 0.0886 | 0.098707 | 31.6529 |

先加入相同的 Teacher Top-10 筛选，InfoNCE 相对未筛选版本获得：

- CIRR R@1 `+3.1333`、R@10 `+2.5592`、Group R@1 `+3.0854`；
- PinPoint mAP@10 `+0.0072`；
- QDiag Mean Delta `+0.003732`、Raw Win `+0.009834`、Stable Win `+0.012104`；
- FashionIQ R@10 `+0.0961`、R@50 `+0.5360`。

这项筛选解释了未筛选 InfoNCE 到 Rank KD 总差距中的约 36.7% R@1、46.1% R@10、
35.7% Group R@1、30.6% PinPoint mAP 和 35.4% Stable Win，但只解释 4.4% 的
FashionIQ R@10 差距。其余优势来自 Teacher 的完整软排序分布，跨数据集迁移尤其依赖
这一信息。NegRecall@10 从 0.1760 变为 0.1797，说明筛选提升正样本检索，但没有改善
negative-image 排除。

epoch 1 是最严格的单变量比较：Rank KD 与 filtered InfoNCE 的初始候选、Teacher Top-10
mask、query 顺序和超参数完全相同。仅把 one-hot CE 换成 Teacher KL 后，Rank 的 R@1
从 29.3231 提升到 39.9904，R@10 从 74.6711 提升到 85.6254，Group R@1 从
60.7271 提升到 66.2999。这直接证明收益不只是有效样本筛选造成的。

## 主要结论

1. Teacher Top-10 筛选是必要的稳定器：它消除了 Dynamic InfoNCE 在 epoch 2–3 的明显
   崩塌，并显著改善 CIRR、PinPoint mAP 和 QDiag。
2. 在使用完全相同筛选后，Rank KD 仍比 InfoNCE 的 CIRR R@1 高 5.41 点、Group R@1
   高 5.55 点、PinPoint mAP 高 0.0163、FashionIQ R@10 高 2.09 点，证明完整 Teacher
   排序分布是独立且主要的收益来源。
3. Student 相关 hard candidates 明显优于 random candidates；Teacher 自己选择的 Top-K
   也没有超过 Student policy 选择，说明训练应聚焦 Student 当前决策边界。
4. Dynamic Student Rank 的 QDiag 在 epoch 3 已基本饱和：Stable Win 为 0.099628；
   epoch 4 为 0.098707，仅下降 0.000921，同时 CIRR R@10 略有提高。因此三至四轮是
   当前结果支持的训练区间，不应仅按单个诊断点选择 checkpoint。
5. Teacher 仍保持最强的常规检索与 negative-image 排除能力；Student Rank 在 QDiag
   strict 反事实区分上可以超过 Teacher，表明蒸馏并非简单复制 Teacher 排名。
6. 当前四轮结果为单 seed；若将不同候选策略之间的小幅差异写成论文强结论，仍需
   多 seed 或 paired significance test。
7. 在完全相同的随机候选上，Centered MSE 相比 softmax-KL 提高 QDiag Raw/Stable Win
   和 PinPoint negative 排除，但 CIRR/FashionIQ 略低，说明原始相似度几何是有效监督，
   同时 random candidates 仍限制了真实 retrieval 能力。
8. 把 composed q− 直接混入 image candidate softmax 会使 Teacher target Top-10 有效率
   从 image-only K=32 的 77.17% 降至 20.99%，并全面弱于 Dynamic Rank。q− 与 q+
   相似不等于它应作为低分 image candidate；后续应在 query-side 使用 counterfactual
   response，而不是混合 query/image 支持集。

## Student–Teacher 模型矩阵

本节只比较 Rank-OPD。CLIP Student 使用全量微调；Qwen-2B Student 使用 query-side
LoRA（image encoder 与 gallery 固定）。候选由 Student 动态检索 Top-32，并强制加入
target；Teacher 将 target 排在候选 Top-10 内的 query 才参与蒸馏。目标函数均为
forward KL `D_KL(p_teacher || p_student)`。除显式标注 E1 的 Reranker 行外，其余蒸馏
Student 均报告 epoch 4。

| Model | Student | Teacher | CIRR R@1 | CIRR R@10 | Group R@1 | FashionIQ R@10 | FashionIQ R@50 | PinPoint mAP@10 | PinPoint NegRecall@10 | QDiag Delta | QDiag Raw Win | QDiag Stable Win | QDiag tau95 | QDiag Own-Random |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BGE-VL-MLLM-S2 | — | — | 49.4140 | 89.2849 | 75.1017 | 34.1685 | 55.1285 | 0.1262 | 0.1219 | 0.0217 | 0.7426 | 0.0918 | 0.0712 | 0.1613 |
| Qwen3-VL-Embedding-8B | — | — | 47.6680 | 86.7257 | 70.9400 | 47.4424 | 68.2903 | 0.1539 | 0.1465 | 0.0479 | 0.8166 | 0.1321 | 0.1129 | 0.3040 |
| CLIP-B original | CLIP-B | — | 12.6046 | 50.0120 | 32.6955 | 16.9389 | 31.5377 | 0.0399 | 0.2051 | 0.0051 | 0.6636 | 0.0232 | 0.0360 | 0.1246 |
| CLIP-L original | CLIP-L | — | 12.7960 | 51.1600 | 34.5133 | 20.2010 | 36.4484 | 0.0462 | 0.2056 | 0.0062 | 0.6615 | 0.0202 | 0.0425 | 0.1269 |
| Qwen3-VL-Embedding-2B original | Qwen-2B | — | 37.2877 | 80.8180 | 63.4776 | 31.0111 | 51.9094 | 0.1152 | 0.1604 | 0.0361 | 0.7678 | 0.1048 | 0.1038 | 0.2882 |
| CLIP-L ← S2 | CLIP-L | BGE-S2 | 42.4540 | 87.1562 | 67.7828 | 31.6529 | 52.4326 | 0.0886 | 0.1740 | 0.0254 | 0.7808 | 0.0987 | 0.0714 | 0.1511 |
| CLIP-L ← Qwen3-VL-Reranker-2B (E1) | CLIP-L | Qwen3-VL-Reranker-2B | 30.9735 | 78.9524 | 56.5654 | — | — | — | — | — | — | — | — | — |
| CLIP-B ← S2 | CLIP-B | BGE-S2 | 40.4927 | 86.2951 | 65.7498 | 26.8361 | 48.4028 | 0.0743 | 0.1694 | 0.0261 | 0.7889 | 0.1040 | 0.0717 | 0.1563 |
| CLIP-B ← Qwen-8B | CLIP-B | Qwen-8B | 33.9632 | 82.8988 | 59.5790 | 25.4648 | 46.1603 | 0.0683 | 0.1856 | 0.0283 | 0.7689 | 0.1004 | 0.0816 | 0.2210 |
| CLIP-L ← Qwen-8B | CLIP-L | Qwen-8B | 36.7376 | 83.5924 | 61.6599 | 31.6753 | 52.5181 | 0.0835 | 0.1913 | 0.0284 | 0.7765 | 0.0988 | 0.0815 | 0.2112 |
| Qwen-2B query-LoRA ← Qwen-8B | Qwen-2B query-LoRA | Qwen-8B | 46.8787 | 86.6539 | 70.7486 | 44.3374 | 65.2264 | 0.1367 | 0.1497 | 0.0467 | 0.8211 | **0.1821** | 0.0943 | 0.2847 |

### QDiag 完成情况

这轮 QDiag 已全部跑完。上表中的 `Delta / Raw Win / Stable Win / tau95 / Own-Random`
均取自 `strict`（ADD + SUBSTITUTE）子集，而不是占位值。新增五个实际诊断条目为：

| 诊断模型 | groups | instructions | negatives | strict Object Stable Win | strict Attribute Stable Win | strict Predicate Stable Win | DELETE Stable Win |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP-B original | 471 | 1,580 | 3,242 | 0.0265 | 0.0133 | 0.0029 | 0.0275 |
| CLIP-B ← S2 | 471 | 1,580 | 3,242 | 0.1272 | 0.0650 | 0.0337 | 0.0083 |
| CLIP-B ← Qwen-8B | 471 | 1,580 | 3,242 | 0.1019 | 0.0810 | 0.0294 | 0.0193 |
| CLIP-L ← Qwen-8B | 471 | 1,580 | 3,242 | 0.0965 | 0.0940 | 0.0247 | 0.0180 |
| Qwen-2B query-LoRA ← Qwen-8B | 471 | 1,580 | 3,242 | **0.2083** | **0.1591** | **0.0469** | **0.0620** |

其中前三个细粒度列使用 `strict|object`、`strict|attribute`、`strict|predicate`；DELETE
使用单独的 `delete` 子集，因此 DELETE 不包含在 strict 总分中。

### 模型矩阵结论

1. 所有 Student 在四轮蒸馏后都相对各自原版大幅提高 CIRR、FashionIQ、PinPoint mAP
   和 QDiag，说明 Rank-OPD 不依赖某一个 Student 架构。
2. 对 CLIP-B/L，S2 Teacher 的 CIRR 与 PinPoint 结果均好于 Qwen-8B Teacher；Qwen-8B
   则带来更高的 QDiag Delta、tau95 和 Own-Random。固定 temperature 下 Qwen-8B 的
   Teacher 分布更尖锐，因此这里不仅改变了 Teacher，也改变了监督分布的熵。
3. 同架构族的 Qwen-2B ← Qwen-8B 最强：CIRR R@1 从 37.2877 提升至 46.8787，几乎
   追平 8B Teacher 的 47.6680；FashionIQ R@10 提升 13.3263 点；QDiag Stable Win
   达到 0.1821，超过 8B Teacher 的 0.1321。
4. Qwen-2B Student 的 QDiag strict Raw Win 也略高于 8B Teacher（0.8211 vs 0.8166）。
   Stable Win 超过 Teacher 并不表示所有检索能力都超过 Teacher：它同时受到各模型
   tau95 校准阈值影响，且 2B 在 FashionIQ、PinPoint 上仍低于 8B。
5. CLIP 是 full finetune，Qwen-2B 是 query-LoRA，不能把两类结果直接解释成单纯的
   Student 尺度效应；可靠比较应优先在各自原始模型内看蒸馏增益。
