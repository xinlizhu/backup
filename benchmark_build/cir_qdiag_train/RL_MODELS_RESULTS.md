# CLIP-L Retrieval RL 与 OPD 超参数消融

本文单独记录已经完成的 CLIP-L retrieval RL 实验。RL 阶段使用 CIRR train 的黄金
target 作为奖励信号，不调用 Teacher；除 `Original CLIP-L → Full-sequence RL` 外，所有
RL 实验都从同一个 `Dynamic Student Rank E4`（OPD E4）checkpoint 开始。所有模型均为
CLIP ViT-L/14 full finetune，训练 4 epochs，每轮保存 checkpoint。

本文只汇报已经完成评估的旧 RL 实验。为完整呈现 RL 初始化模型的选择依据，本文同时
收录已经完成的 OPD candidate $K$、temperature 和 KL direction 消融。

## RL 方法

### 候选集合与排序策略

对训练样本 $x=(I_{\mathrm{ref}},t,y^+)$，在每个 epoch 开始时由 Student 从 CIRR train
gallery 中检索候选，并强制将黄金 target 放入候选集合：

$$
\mathcal C_\theta^{32}(x)
=
\{y^+\}
\cup
\operatorname{TopK}_{31}
\left\{
s_\theta(x,y):y\in\mathcal G\setminus\{y^+\}
\right\}.
$$

因此每条 query 的训练支持集包含 1 张 target 和 31 张 Student hard images。训练时重新
前向计算当前 Student 在这 32 张图片上的相似度。给定尚未选出的候选集合
$\mathcal A_t$，Plackett--Luce 排序策略为：

$$
\pi_\theta(y_t=i\mid x,y_{<t})
=
\frac{
\exp\left(s_\theta(x,i)/\tau\right)
}{
\sum_{j\in\mathcal A_t}
\exp\left(s_\theta(x,j)/\tau\right)
},
\qquad \tau=0.05.
$$

每条 query 独立采样 $G=16$ 个 rollout，每个 rollout 内不放回采样，最大排序长度
$K=10$。同一个 rollout 内图片不会重复，不同 rollout 之间允许自然重复。

### Target NDCG@10 reward

如果 target 在某条 rollout 中首次出现在第 $r_g$ 位，则该 rollout 的单正样本
NDCG@10 reward 为：

$$
R_g=
\begin{cases}
\displaystyle \frac{1}{\log_2(r_g+1)}, & r_g\le 10,\\[6pt]
0, & r_g>10.
\end{cases}
$$

对同一 query 的 16 个 reward 使用组内均值和总体标准差进行标准化：

$$
\mu_R=\frac{1}{G}\sum_{g=1}^{G}R_g,
\qquad
\sigma_R=
\sqrt{
\frac{1}{G}\sum_{g=1}^{G}(R_g-\mu_R)^2
},
$$

$$
A_g=
\begin{cases}
\displaystyle\frac{R_g-\mu_R}{\sigma_R+\epsilon}, & \sigma_R>0,\\[6pt]
0, & \sigma_R=0.
\end{cases}
$$

当一个 query 的 16 个 rollout 得到完全相同的 reward 时，该 query 没有组内相对信号，
其 advantage 被置为零。

### Full-sequence group-relative REINFORCE

最初版本总是采样完整的 10 个位置。设 rollout $y^{(g)}$ 的序列 log-probability 为：

$$
\log\pi_\theta(y^{(g)}\mid x)
=
\sum_{t=1}^{10}
\log\pi_\theta
\left(y_t^{(g)}\mid x,y_{<t}^{(g)}\right).
$$

训练目标为：

$$
\mathcal L_{\mathrm{FullSeq}}
=
-\frac{1}{G}
\sum_{g=1}^{G}
A_g\log\pi_\theta(y^{(g)}\mid x).
$$

该版本在 target 已经被选中后仍对后续位置施加同一个 advantage，因此后续实验改为
target-stopped causal prefix。

### Causal-prefix rollout

在 causal-prefix 版本中，rollout 一旦选择 target 就立即终止。其有效长度为：

$$
L_g=
\begin{cases}
r_g, & r_g\le 10,\\
10, & r_g>10.
\end{cases}
$$

只有 $t\le L_g$ 的 action 进入 policy loss；target 后的 suffix 被完全 mask。Causal-prefix
REINFORCE 的目标为：

$$
\mathcal L_{\mathrm{CP\text{-}REINFORCE}}
=
-\frac{1}{G}
\sum_{g=1}^{G}
A_g
\sum_{t=1}^{L_g}
\log\pi_\theta
\left(y_t^{(g)}\mid x,y_{<t}^{(g)}\right).
$$

该方法每个 rollout group 只更新一次。

### Causal-prefix GRPO

GRPO 和 GSPO 先由 batch-start policy $\pi_{\mathrm{old}}$ 采样 rollout，再复用同一组
rollout 更新两次。GRPO 在每个有效位置上计算 importance ratio：

$$
\rho_{g,t}(\theta)
=
\frac{
\pi_\theta(y_t^{(g)}\mid x,y_{<t}^{(g)})
}{
\pi_{\mathrm{old}}(y_t^{(g)}\mid x,y_{<t}^{(g)})
}.
$$

使用 $\varepsilon=0.2$ 的逐位置 clipping：

$$
\mathcal L_{\mathrm{CP\text{-}GRPO}}
=
-\frac{1}{G}
\sum_{g=1}^{G}
\frac{1}{L_g}
\sum_{t=1}^{L_g}
\min\left[
\rho_{g,t}A_g,
\operatorname{clip}(\rho_{g,t},1-\varepsilon,1+\varepsilon)A_g
\right].
$$

### Causal-prefix GSPO

GSPO 不对每个位置分别裁剪，而是先计算长度归一化的 sequence-level ratio：

$$
\rho_g^{\mathrm{seq}}(\theta)
=
\exp\left[
\frac{1}{L_g}
\sum_{t=1}^{L_g}
\log
\frac{
\pi_\theta(y_t^{(g)}\mid x,y_{<t}^{(g)})
}{
\pi_{\mathrm{old}}(y_t^{(g)}\mid x,y_{<t}^{(g)})
}
\right].
$$

带 clipping 的 GSPO 目标为：

$$
\mathcal L_{\mathrm{CP\text{-}GSPO}}
=
-\frac{1}{G}
\sum_{g=1}^{G}
\min\left[
\rho_g^{\mathrm{seq}}A_g,
\operatorname{clip}(\rho_g^{\mathrm{seq}},1-\varepsilon,1+\varepsilon)A_g
\right],
\qquad \varepsilon=0.2.
$$

`GSPO No-Clip` 使用相同的 sequence-level ratio，但不执行
$[1-\varepsilon,1+\varepsilon]$ 裁剪。

### OPD 的 K、temperature 与 KL direction

OPD 消融均从原始 CLIP-L 初始化，使用冻结的 BGE-VL-MLLM-S2 Teacher。Student 每个
epoch 动态检索并强制插入 target：

$$
\mathcal C_\theta^K(x)
=
\{y^+\}
\cup
\operatorname{TopK}_{K-1}
\left\{s_\theta(x,y):y\in\mathcal G\setminus\{y^+\}\right\}.
$$

Teacher 和 Student 在同一个候选集合上产生温度归一化分布：

$$
p_\phi^\tau(y\mid x,\mathcal C_\theta^K)
=
\frac{\exp(s_\phi(x,y)/\tau)}
{\sum_{y'\in\mathcal C_\theta^K}\exp(s_\phi(x,y')/\tau)},
$$

$$
p_\theta^\tau(y\mid x,\mathcal C_\theta^K)
=
\frac{\exp(s_\theta(x,y)/\tau)}
{\sum_{y'\in\mathcal C_\theta^K}\exp(s_\theta(x,y')/\tau)}.
$$

Forward KL 和 Reverse KL 分别为：

$$
\mathcal L_{\mathrm{Forward}}
=
D_{\mathrm{KL}}\left(p_\phi^\tau\middle\|p_\theta^\tau\right),
\qquad
\mathcal L_{\mathrm{Reverse}}
=
D_{\mathrm{KL}}\left(p_\theta^\tau\middle\|p_\phi^\tau\right).
$$

所有消融只保留 Teacher 将 target 排在候选 Top-10 内的样本，训练 4 epochs。实验覆盖
$K\in\{8,16,32\}$、$\tau\in\{0.05,1\}$ 以及 forward/reverse KL。$\tau=1$
表示直接对 cosine similarity 做 softmax，并不表示取消 softmax。K=64 运行时未产生有效
checkpoint，因此不进入结果表。

## 模型与训练目标

所有后训练实验都使用 batch size 32、BF16、head learning rate $10^{-5}$、backbone
learning rate $5\times10^{-7}$、cosine schedule、4 epochs 和 seed 42。RL 阶段不使用
Teacher 分数，也不进行 Teacher target Top-10 过滤。

| Model | 初始化 | 候选 | Rollout | 更新 | 训练目标 |
|---|---|---|---|---:|---|
| CLIP-L original | 原始 CLIP-L | — | — | — | 无后训练 |
| OPD E4 initialization | 原始 CLIP-L → OPD E4 | Student Top-32 | — | — | Teacher→Student forward KL |
| OPD E4 → Target CE E4 | OPD E4 | Student Top-32 | — | 1× | target one-hot candidate CE |
| Original CLIP-L → Full-sequence RL E4 | 原始 CLIP-L | Student Top-32 | 16×Top-10 | 1× | group-normalized full-sequence REINFORCE |
| OPD E4 → Full-sequence RL E4 | OPD E4 | Student Top-32 | 16×Top-10 | 1× | group-normalized full-sequence REINFORCE |
| OPD E4 → CP-REINFORCE E4 | OPD E4 | Student Top-32 | 16×target-stopped Top-10 | 1× | causal-prefix REINFORCE |
| OPD E4 → CP-GRPO E4 | OPD E4 | Student Top-32 | 16×target-stopped Top-10 | 2× | position-level ratio，clip 0.2 |
| OPD E4 → CP-GSPO No-Clip E4 | OPD E4 | Student Top-32 | 16×target-stopped Top-10 | 2× | sequence-level ratio，无 clipping |
| OPD E4 → CP-GSPO E4 | OPD E4 | Student Top-32 | 16×target-stopped Top-10 | 2× | sequence-level ratio，clip 0.2 |

OPD 超参数消融保持 Student、Teacher、有效样本筛选和其他训练配置不变，只改变
candidate $K$、temperature 或 KL direction：

| Model | Candidate $K$ | Temperature $\tau$ | KL direction | 训练轮数 |
|---|---:|---:|---|---:|
| OPD K=32 Forward KL, $\tau=0.05$ | 32 | 0.05 | Teacher $\rightarrow$ Student | 4 |
| OPD K=16 Forward KL, $\tau=0.05$ | 16 | 0.05 | Teacher $\rightarrow$ Student | 4 |
| OPD K=8 Forward KL, $\tau=0.05$ | 8 | 0.05 | Teacher $\rightarrow$ Student | 4 |
| OPD K=32 Reverse KL, $\tau=0.05$ | 32 | 0.05 | Student $\rightarrow$ Teacher | 4 |
| OPD K=32 Forward KL, $\tau=1$ | 32 | 1 | Teacher $\rightarrow$ Student | 4 |
| OPD K=32 Reverse KL, $\tau=1$ | 32 | 1 | Student $\rightarrow$ Teacher | 4 |

## CIRR val

CIRR val 共 4,181 条 query。表中均为 epoch-4 checkpoint；R@K 和 Group R@K 的单位为
百分数。

| Model | R@1 | R@5 | R@10 | R@50 | Group R@1 | Group R@2 | Group R@3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP-L original | 12.7960 | 37.7661 | 51.1600 | 78.4023 | 34.5133 | 58.8854 | 76.6324 |
| OPD E4 initialization | **42.4540** | **76.4889** | **87.1562** | 97.4408 | **67.7828** | **86.9648** | **94.4989** |
| OPD E4 → Target CE E4 | 41.2820 | 76.8955 | 87.0605 | **97.7517** | 65.9651 | 86.3908 | 94.1880 |
| Original CLIP-L → Full-sequence RL E4 | 23.4394 | 53.7192 | 66.4434 | 86.4387 | 48.3856 | 72.0641 | 85.4341 |
| OPD E4 → Full-sequence RL E4 | 37.2399 | 73.1643 | 84.2143 | 96.8907 | 62.6405 | 84.6927 | 93.2791 |
| OPD E4 → CP-REINFORCE E4 | 37.3595 | 73.3557 | 84.5731 | 96.8429 | 62.7601 | 84.7644 | 93.2552 |
| OPD E4 → CP-GRPO E4 | 38.4836 | 73.5709 | 83.9512 | 96.0775 | 64.4583 | 85.1232 | 93.4944 |
| OPD E4 → CP-GSPO No-Clip E4 | 38.8185 | 73.5948 | 84.2621 | 96.1732 | 64.3148 | 85.0514 | 93.3030 |
| OPD E4 → CP-GSPO E4 | 38.6271 | 73.4035 | 84.0708 | 96.2688 | 64.2430 | 85.0275 | 93.4705 |

OPD candidate $K$、temperature 与 KL direction 消融的 epoch-4 结果如下：

| Candidate $K$ | Temperature $\tau$ | KL direction | R@1 | R@5 | R@10 | R@50 | Group R@1 | Group R@2 | Group R@3 |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.05 | Forward | 42.4540 | 76.4889 | 87.1562 | 97.4408 | 67.7828 | **86.9648** | 94.4989 |
| 16 | 0.05 | Forward | 43.1476 | 77.0390 | 87.0844 | **97.4647** | 68.0938 | 86.8453 | 94.5228 |
| 8 | 0.05 | Forward | **43.7694** | **77.1586** | **87.4193** | 97.3451 | **68.8591** | 86.6539 | **94.6424** |
| 32 | 0.05 | Reverse | 42.6214 | 76.2497 | 87.1083 | 97.4169 | 67.3762 | 86.5343 | 94.4989 |
| 32 | 1 | Forward | 41.9995 | 75.8431 | 86.8213 | 97.3451 | 67.2327 | 86.2951 | 94.3554 |
| 32 | 1 | Reverse | 41.9995 | 75.7713 | 86.9409 | 97.3451 | 67.3762 | 86.1995 | 94.4033 |

在固定 Forward KL 和 $\tau=0.05$ 时，$K=8$ 取得最高的 R@1、R@5、R@10、
Group R@1 和 Group R@3。相对 $K=32$，$K=8$ 的 R@1/R@10/Group R@1 分别提高
1.3154/0.2631/1.0763 个百分点。在 $K=32$ 时，Reverse KL 没有显示稳定优势；
$\tau=1$ 的整体结果也低于 $\tau=0.05$。

四个 causal-prefix 实验的逐轮曲线如下：

| Model | Epoch | R@1 | R@10 | Group R@1 |
|---|---:|---:|---:|---:|
| CP-REINFORCE | 1 | 35.4461 | 83.2337 | 61.0380 |
| CP-REINFORCE | 2 | 37.1442 | **84.5970** | **62.9036** |
| CP-REINFORCE | 3 | **37.5269** | 84.5492 | 62.4731 |
| CP-REINFORCE | 4 | 37.3595 | 84.5731 | 62.7601 |
| CP-GRPO | 1 | 36.9529 | 83.6403 | 62.6166 |
| CP-GRPO | 2 | **38.9381** | **84.6927** | 64.1713 |
| CP-GRPO | 3 | 38.7467 | 84.0708 | **64.5539** |
| CP-GRPO | 4 | 38.4836 | 83.9512 | 64.4583 |
| CP-GSPO No-Clip | 1 | 36.9768 | 83.8316 | 62.6884 |
| CP-GSPO No-Clip | 2 | **39.0576** | **84.6448** | 64.0517 |
| CP-GSPO No-Clip | 3 | 38.7945 | 84.5252 | **64.4104** |
| CP-GSPO No-Clip | 4 | 38.8185 | 84.2621 | 64.3148 |
| CP-GSPO | 1 | 36.8811 | 83.6881 | 62.4492 |
| CP-GSPO | 2 | **39.1533** | **84.6687** | 64.1234 |
| CP-GSPO | 3 | 38.8185 | 84.1665 | **64.5779** |
| CP-GSPO | 4 | 38.6271 | 84.0708 | 64.2430 |

原始 CLIP-L 直接进行 full-sequence RL 后，CIRR R@1 从 12.7960 提升到 23.4394，说明
黄金 target reward 确实能够学习检索信号。但从 OPD E4 继续 RL 时，所有最终 checkpoint
都低于 OPD 初始化。Causal-prefix 修正了 target 后错误 credit：相对 full-sequence RL，
最佳 R@1 从 37.2399 提升到 38.8185，最佳 Group R@1 从 62.6405 提升到 64.4583，
但仍未恢复 OPD E4 的 42.4540/67.7828。Causal 系列通常在 epoch 2 达到较好的 CIRR
点位，继续训练到 epoch 4 没有带来进一步提升。

## FashionIQ

FashionIQ val 共 6,016 条 query。R@K 为 dress、shirt、toptee 三个类别的宏平均，标准
指标为 R@10 和 R@50；表中均使用 epoch-4 checkpoint。

| Model | Avg R@1 | Avg R@5 | Avg R@10 | Avg R@50 |
|---|---:|---:|---:|---:|
| CLIP-L original | 0.4682 | 13.9806 | 20.2010 | 36.4484 |
| OPD E4 initialization | **8.1267** | **24.0912** | **31.6529** | **52.4326** |
| OPD E4 → Target CE E4 | 5.9298 | 22.1104 | 30.0011 | 50.8950 |
| Original CLIP-L → Full-sequence RL E4 | 0.8640 | 16.6139 | 23.0361 | 40.7752 |
| OPD E4 → Full-sequence RL E4 | 4.1714 | 20.1444 | 28.0575 | 48.2491 |
| OPD E4 → CP-REINFORCE E4 | 3.9390 | 20.1305 | 27.8303 | 48.0359 |
| OPD E4 → CP-GRPO E4 | 3.9365 | 19.3820 | 26.5974 | 47.0235 |
| OPD E4 → CP-GSPO No-Clip E4 | 3.9011 | 19.3350 | 26.2962 | 46.7887 |
| OPD E4 → CP-GSPO E4 | 3.8179 | 19.4819 | 26.2957 | 46.9560 |

OPD candidate $K$、temperature 与 KL direction 消融的 FashionIQ epoch-4 结果如下：

| Candidate $K$ | Temperature $\tau$ | KL direction | Avg R@1 | Avg R@5 | Avg R@10 | Avg R@50 |
|---:|---:|---|---:|---:|---:|---:|
| 32 | 0.05 | Forward | 8.1267 | 24.0912 | 31.6529 | 52.4326 |
| 16 | 0.05 | Forward | 8.1448 | 23.9109 | 31.7746 | 52.8466 |
| 8 | 0.05 | Forward | **8.6079** | **24.4559** | **32.2379** | **52.9462** |
| 32 | 0.05 | Reverse | 8.2091 | 24.1056 | 31.5293 | 52.2481 |
| 32 | 1 | Forward | 8.3068 | 24.0250 | 31.2156 | 51.8531 |
| 32 | 1 | Reverse | 8.4590 | 24.0887 | 31.4182 | 51.9484 |

FashionIQ 上同样是 $K=8$、Forward KL、$\tau=0.05$ 在所有四个指标上最高。
相对 $K=32$ 的原始 OPD 设置，其 Avg R@10 和 Avg R@50 分别提高 0.5850 和
0.5136 个百分点，说明较小的 on-policy hard candidate 集合在 CIRR 和跨数据集
FashionIQ 上都有轻微优势。

RL 从原始 CLIP-L 开始时，FashionIQ R@10/R@50 分别提高 2.8351/4.3268，说明其学习到
的部分 CIRR retrieval 能力可以迁移。但从 OPD E4 开始的全部 RL 实验均损害跨数据集
泛化；其中表现最好的 CP-REINFORCE 仍比 OPD 初始化低 3.8226 R@10 和 4.3967 R@50。
这与 CIRR 结果共同表明：当前 target-only、局部 Top-32 reward 能优化训练候选中的 target
位置，却不能维持 OPD 已学习的完整语义排序结构。
