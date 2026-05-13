# IP Embedding Loss 设计决策说明

> **用途**：解释本次 IP 虚拟形象 embedding 训练升级中，为什么从普通 `infonce` 升级到 `multi_positive_infonce`，以及为什么引入 `Sub-center ArcFace` 作为辅助 loss。本文档侧重算法决策逻辑，不替代运行脚本说明。  
> **对应实现背景**：见 `projects/ips/IP_EMBEDDING_UPGRADE_PLAN.md`。  
> **更新时间**：2026-05-09

## 1. 任务本质

我们的线上链路是两阶段检索判断：

```text
query 图像
  -> embedding 模型召回 vault top-K 图片
  -> VLM / reranker 对 query-candidate 做精判
  -> 判定是否疑似 IP 侵权
```

因此 embedding 阶段的目标不是闭集分类准确率，而是 **top-K 召回**。只要召回集合里出现同 IP 图片，后续 reranker 仍有机会修正；如果 embedding 阶段漏召，后续无法补救。

这个任务不同于传统人脸识别。虚拟 IP 的同一身份可能跨越发型、服装、皮肤、画风、视角、武器、配饰和形态；不同 IP 又可能共享相似画风或元素。因此我们面对的是：

```text
类内方差可能大于类间方差。
```

这意味着训练目标不应该简单地把同一 IP 的所有图片压到一个单中心，而应该学习：

- 同 IP 多形态之间可召回；
- 不同 IP 之间保持足够边界；
- 允许同一个 IP 在 embedding 空间中形成多个合理子簇；
- 对历史标注噪声、同 IP 别名、漏合并皮肤保持鲁棒。

## 2. 证据基础

### 2.1 embedding 训练和 Qwen3 / Qwen3-VL 适配

SWIFT 官方 embedding 训练文档说明，框架支持纯文本和多模态 embedding 模型，包括 `qwen3-embedding` 和 `qwen3-vl-embedding`，并要求模型 forward 输出包含 `last_hidden_state` 形式的 embedding tensor。见 [SWIFT Embedding Training 文档](https://swift.readthedocs.io/en/v3.12/BestPractices/Embedding.html)。

Qwen3 Embedding 官方说明强调 embedding / reranker 组合适用于检索和排序任务，并且模型支持 instruction-aware 使用方式；官方 README 还提到针对下游任务定制 instruction 通常能带来收益。见 [Qwen3-Embedding 官方仓库](https://github.com/QwenLM/Qwen3-Embedding)。

Qwen3-VL-Embedding 官方仓库说明其面向图像、视频、文本和混合模态检索，并采用 dual-tower embedding 与 reranker 配套的检索架构，这与我们的线上“embedding 召回 + VLM/reranker 精判”一致。见 [Qwen3-VL-Embedding 官方仓库](https://github.com/QwenLM/Qwen3-VL-Embedding)。

### 2.2 为什么需要多正例

Supervised Contrastive Learning 论文把自监督对比学习扩展到监督场景：同类样本在 embedding 空间被拉近，不同类样本被推开，并报告其相对 cross-entropy 的稳定性和鲁棒性收益。见 [Khosla et al., Supervised Contrastive Learning, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)。

这给我们的启发是：如果一个 IP 有多个合法形态，那么一个 anchor 不应该只看一张 positive；它应该同时看到多个同 IP positive，让模型学习“同一身份可以有多个外观模式”。

不过，我们没有直接照搬完整 batch-global SupCon，而是先实现 row-local `multi_positive_infonce`。原因见第 5 节。

### 2.3 为什么需要 Sub-center

Sub-center ArcFace 的核心思想是：每个类别有 `K` 个子中心，训练样本只需要靠近该类别里最接近自己的一个子中心，而不是强迫所有样本靠近唯一中心。论文原本用于大规模 noisy face recognition，但它的关键机制正好适合我们的“同 IP 多形态”问题。见 [Deng et al., Sub-center ArcFace, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf)。

论文中还讨论了“sub-class / multi-modality”思想：子中心可以捕捉类别内部复杂分布和不同模式；这与一个游戏英雄多个皮肤、一个动漫角色多个造型高度一致。

### 2.4 为什么不能过早 hard negative

Triplet loss 和 hard negative mining 在度量学习中很常见，但 hardest negative 在训练早期会造成优化失败或坏局部最优。`Hard negative examples are hard, but useful` 论文明确分析了 hardest negatives 的问题，同时指出高类内方差数据集里这类样本又非常重要。见 [Xuan et al., ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590120.pdf)。

CurricularFace 也指出，过早强调 hard / semi-hard samples 可能带来收敛问题，因此建议训练策略从 easy 到 hard 逐步推进。见 [Huang et al., CurricularFace, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Huang_CurricularFace_Adaptive_Curriculum_Learning_Loss_for_Deep_Face_Recognition_CVPR_2020_paper.html)。

这支持我们把 hard mining 放到后续 Stage 3，而不是 Stage 1 直接上最难负例。

## 3. 当前最终采用的 loss 方案

### 3.1 Stage 1：普通显式负例 InfoNCE

Stage 1 使用当前框架已有的 `infonce`：

```text
L_stage1 = L_infonce_explicit
```

数据结构：

```text
anchor: 1 张 query 图
positive: 1 张同 IP 图
negative: N 张不同 IP 显式负例图
```

Stage 1 的作用不是解决所有跨形态召回，而是：

- 让模型适配 IP 图像域；
- 用清洗后的 merge 标注修复历史 false negative；
- 建立基本同 IP / 不同 IP 判别能力；
- 给 Stage 2 提供稳定初始化。

为什么 Stage 1 不直接上复杂 loss：

- 初始 embedding 空间还没有 IP 领域结构；
- 直接 hard mining 容易引入 false negative；
- 直接强 proxy margin 可能让模型在标注噪声和多形态上过早收紧。

### 3.2 Stage 2A：Multi-positive InfoNCE

Stage 2A 使用新实现的 `multi_positive_infonce`：

```text
L_mp = -log(
  sum_{p in P(q)} exp(sim(q, p) / tau)
  /
  [sum_{p in P(q)} exp(sim(q, p) / tau) + sum_{n in N(q)} exp(sim(q, n) / tau)]
)
```

其中：

```text
P(q) = 同 IP 的多个 positive
N(q) = 显式不同 IP negatives
```

核心变化是：一个 anchor 同时使用多张同 IP positive，而不是只使用第一张 positive。

#### Multi-positive 带来的收益

1. **更贴近线上召回目标**

线上只需要 top-K 中召回任意同 IP 形态。多正例分子 `sum exp(sim(q,p))` 与这个目标更一致：模型不只优化 anchor 与某一张固定 positive 的关系，而是优化 anchor 对同 IP positive 集合的可达性。

2. **降低单个 positive 偶然性的影响**

如果某一张 positive 与 anchor 只是同服装、同姿态或同背景，普通 InfoNCE 容易让模型学到这些表面相关性。多个 positive 同时出现时，只有跨样本稳定存在的身份线索更容易被强化。

3. **强化 hard positive**

同一 IP 的不同形态、不同皮肤、不同画风本质上是 hard positives。多正例训练让这些 hard positives 在同一训练样本里直接提供梯度，而不是依赖随机 pair 采样碰到。

4. **保持实现风险可控**

我们采用 row-local multi-positive，而不是第一版就做 batch-global SupCon。这样可以保留当前 JSONL 数据范式和显式负例采样逻辑，避免一次性引入 class-aware batch、DDP gather、same-IP mask 和 false negative 过滤等工程复杂度。

### 3.3 Stage 2B：Multi-positive InfoNCE + Sub-center ArcFace

Stage 2B 使用组合 loss：

```text
L_stage2_ip = L_multi_positive_infonce + lambda_sub * L_subcenter_arcface
```

其中 multi-positive InfoNCE 仍然是主 loss，Sub-center ArcFace 是辅助几何约束。

默认实现保持未归一化组合：

```text
stage2_loss_norm = none
L_total = L_mp + lambda_sub * L_subcenter
```

为了做 loss 尺度 ablation，新增静态参考归一化开关：

```text
stage2_loss_norm = subcenter_ref
L_total = L_mp + lambda_sub * (L_subcenter / log(num_classes))
```

这里使用 `log(num_classes)`，因为 Sub-center ArcFace 最终是 `num_classes` 分类交叉熵；在接近随机分类时，CE 的参考尺度约为 `log(num_classes)`。这个归一化不会动态平衡梯度，也不会改变 multi-positive InfoNCE，只是把 class-count-dependent 的辅助 CE 项压到更可比较的尺度，方便判断 `lambda_sub` 是否过强。

Ablation 时建议固定数据、checkpoint、学习率、`SUBCENTER_LAMBDA`、`SUBCENTER_K`，只比较：

```text
--stage2_loss_norm none
--stage2_loss_norm subcenter_ref
```

重点看 W&B 中的：

```text
train/stage2_mp_loss
train/stage2_subcenter_loss              # raw CE
train/stage2_subcenter_loss_effective    # 进入 total loss 的 CE 项
train/stage2_weighted_subcenter_loss
train/stage2_total_loss
train/grad_norm
```

如果 `subcenter_ref` 下 `stage2_weighted_subcenter_loss` 明显更稳定、`grad_norm` 尖峰减少，同时检索评估不下降，说明原始 subcenter 项对训练过强；如果检索指标变差，说明当前 `lambda_sub` 在未归一化尺度下可能刚好合适，或者需要重新调大归一化后的 `SUBCENTER_LAMBDA`。

Sub-center ArcFace 的 proxy 形状为：

```text
proxies: [num_ip_classes, K, embedding_dim]
```

每个 IP 有 `K` 个子中心。训练时，对每个样本计算它与每个 IP 的所有子中心相似度，然后在 `K` 维度取 max，得到类别级 logit。对 target IP 应用 additive angular margin。

#### Sub-center 带来的收益

1. **允许同一 IP 多子簇**

同一 IP 的机甲皮肤、默认皮肤、节日皮肤、特殊形态可能在视觉上不连续。单中心 ArcFace 会强迫这些形态挤到同一中心附近，容易破坏自然结构。Sub-center 允许它们分别靠近不同子中心。

2. **对标注噪声更鲁棒**

Sub-center ArcFace 论文的动机就是在 noisy web faces 上放松类内约束：样本只需靠近任一正类子中心。对我们的历史 IP 标注场景，这能缓解漏合并、错合并、极端变体带来的训练冲突。

3. **提供全局 class-level 几何边界**

Multi-positive InfoNCE 主要是 row-local pair/set 约束；Sub-center ArcFace 让每个 IP 拥有可训练 proxy，提供全局 class-level 边界，有利于不同 IP 之间形成更稳定的角度间隔。

4. **不替代 retrieval loss**

Sub-center 是辅助项，不是主 loss。原因是我们的线上指标是检索 top-K，而不是训练集闭集分类。只用 proxy/classification loss 可能让模型优化类别中心，却不一定优化 query-gallery pair 排序。Multi-positive InfoNCE 仍然直接对应检索关系。

#### 为什么 `K` 不宜太大

`K` 越大，每个 IP 可表达的子形态越多，但负类也有更多机会被某个子中心偶然匹配，可能增加误召回。因此初始建议：

```text
SUBCENTER_K = 3
SUBCENTER_LAMBDA = 0.05
SUBCENTER_MARGIN = 0.2
SUBCENTER_SCALE = 64
STAGE2_LOSS_NORM = none
```

如果某些 IP 实际只有单一形态，max-over-K 会让一个 dominant sub-center 获得主要梯度，其余子中心较少被激活，通常不会显著破坏训练。但 `K` 过大仍然会增加不必要自由度，所以不建议第一版使用 `K=10`。

## 4. 决策过程

### 4.1 我们先解决数据语义，再解决 loss

历史数据中存在“同一 IP 多个文件夹被当作不同 IP”的问题。如果不先 merge，同 IP 文件夹可能互相作为 negative，这会直接教模型错误边界。

因此升级顺序是：

```text
1. 合并 IP 标注，修正 false negative
2. 构建 merged infonce / merged multipos 数据
3. Stage 1 使用普通 InfoNCE warmup
4. Stage 2A 使用 multi-positive InfoNCE
5. Stage 2B 加入 Sub-center ArcFace 辅助约束
6. Stage 3 再引入 hard mining / reranker-aware 数据闭环
```

### 4.2 为什么 Stage 2 先 multi-positive，再 subcenter

Multi-positive 是最直接的召回目标升级：同一个 anchor 同时看到多个同 IP 正例，能够立即改善跨形态召回训练信号。

Sub-center 需要额外的 `ip_id`、proxy 参数、优化器接入、checkpoint 保存和 DDP/ZeRO 行为验证，工程风险更高。因此实现上先做 `multi_positive_infonce`，确认稳定后再做 `stage2_ip_embedding` 组合 loss。

### 4.3 为什么 Sub-center 属于 Stage 2，不等到 Stage 3

Stage 2 的目标就是建立多形态 IP 结构。Sub-center 是对多形态结构的几何建模，因此应该在 Stage 2 介入。如果等到 Stage 3，模型可能已经被普通 contrastive loss 拉向较单一的类结构，再引入子中心会和已有几何结构冲突。

Stage 3 更适合处理：

- ANN mined hard negatives；
- false negative / false positive 过滤；
- reranker-aware distillation；
- curriculum negative mix。

## 5. 为什么没有采用其他方案作为第一版

### 5.1 不采用纯 Cross-Entropy / 单中心分类

纯 CE 或普通分类 head 更适合闭集分类，不直接优化 top-K retrieval。它会倾向学习“每个训练类一个决策区域”，但我们的线上需求是开放检索式召回。

此外，同一 IP 多形态跨度很大时，单中心分类会鼓励类内压缩，和“保留多形态结构”的目标冲突。

### 5.2 不采用普通 ArcFace 作为主 loss

ArcFace 在人脸识别中很强，但普通 ArcFace 默认每类一个中心，并施加强 angular margin。对于虚拟 IP 这种同类多模态分布，单中心 margin 可能过强。

Sub-center ArcFace 是更合适的变体，因为它允许一个类别有多个子中心，同时保留 angular margin 的类别边界能力。

### 5.3 不采用纯 Triplet Loss

Triplet loss 的问题在于：

- 训练效果高度依赖 triplet mining；
- hardest negative 在早期容易导致坏局部最优；
- batch 内可用 triplet 数量和质量受 batch size 影响；
- 在高类内方差任务中，hard positive 和 hard negative 都容易被噪声污染。

`Hard negative examples are hard, but useful` 对这些问题有系统分析，并指出 hardest negatives 虽重要但优化困难，尤其在高类内方差数据中更突出。见 [Xuan et al., ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590120.pdf)。

因此我们没有把 triplet 作为 Stage 1/2 主 loss，而是把 hard mining 放到 Stage 3 的数据闭环里处理。

### 5.4 不直接采用完整 batch-global SupCon

完整 SupCon 通常依赖 batch 内同类样本作为 positives，并把其他类作为 negatives。它的理论方向对，但第一版直接接入有几个风险：

- 需要 PK sampler 保证每个 batch 有多个同 IP；
- 需要 DDP 下跨卡 gather，才能获得足够正负样本；
- batch 内可能出现同 IP 漏标，形成 false negative；
- 长尾 IP 样本少，batch-global positives 分布不稳定。

Decoupled Contrastive Learning 指出，SCL 会同时拉近两类 positives：同图增强视图和同类其他图片；在长尾场景中，等价处理这些 positives 会带来类内距离优化偏差。见 [Xuan & Zhang, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28459)。

因此我们先实现 row-local multi-positive InfoNCE：positive 集合由数据构建脚本显式指定，语义更可控，工程风险更低。

### 5.5 不直接采用 Multi-Similarity Loss

Multi-Similarity Loss 是强有力的 pair weighting 方法，论文提出 general pair weighting 框架，并在多个图像检索 benchmark 上取得优秀结果。见 [Wang et al., Multi-Similarity Loss, CVPR 2019](https://www.deepnlp.org/content/articles/multi-similarity-loss-with-general-pair-weighting-for-deep-metric-learning)。

我们没有第一版采用它，原因是：

- 它依赖 pair mining 和 weighting，最好配合高质量 batch / memory bank；
- 当前框架首先缺的是“多个 positive 被真正使用”，应先补这个主缺口；
- 它不直接解决“一个 IP 多个子中心”的几何建模问题；
- 后续 Stage 3 可以在 hard mining 数据闭环稳定后再评估 MS Loss。

### 5.6 不直接采用 Circle Loss

Circle Loss 通过对相似度分数动态重加权，提升优化灵活性，适用于 recognition / retrieval 任务。见 [Sun et al., Circle Loss, CVPR 2020](https://huggingface.co/papers/2002.10857)。

我们没有第一版采用它，原因类似 MS Loss：

- 它是强 pair similarity optimizer，但不是显式多中心类别建模；
- 需要额外调 margin、scale、pair mining 规则；
- 当前更高 ROI 的改动是 multi-positive 数据利用和 sub-center proxy。

### 5.7 不直接采用 Proxy Anchor / Proxy-NCA

Proxy-based loss 通常收敛快、比纯 pair-based loss 更稳定。Proxy Anchor 论文也指出 proxy 方法能提升收敛速度并增强对噪声和 outlier 的鲁棒性。见 [Kim et al., Proxy Anchor Loss, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.html)。

我们没有第一版采用 Proxy Anchor，是因为它仍然主要是 class proxy 方向，而我们的直接需求是：

- 每条样本多 positive 检索关系；
- 同 IP 多形态子中心；
- 与现有 SWIFT embedding 数据格式低侵入兼容。

Sub-center ArcFace 更贴近“多中心 IP 几何结构”这个关键矛盾。

### 5.8 不在 Stage 2 直接接入 XBM / Memory Bank

Cross-Batch Memory 论文指出，mini-batch 限制了 hard negative mining 能力，XBM 可以记忆过去 batch 的 embedding，在图像检索任务中显著提升 R@1。见 [Wang et al., Cross-Batch Memory, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html)。

这对我们的 Stage 3 很有价值，但不适合作为 Stage 2 第一版：

- IP 数据存在漏标，同 IP 可能被误作 hard negative；
- memory bank 中的 embedding 需要处理 stale feature；
- DDP 多机下实现、同步和调试成本更高；
- 需要先有一个稳定 Stage 2 模型，才能挖更可靠的 hard samples。

因此我们先实现离线 hard-mining 脚本闭环，而不是把 XBM 直接塞进训练 loop。

## 6. 最终训练策略

### 6.1 Stage 1：稳定领域适配

```text
dataset: train_merged_infonce.jsonl
loss: infonce
positive per row: 1
negative per row: explicit negatives
epoch: 1
temperature: 0.05
```

Stage 1 只做 warmup，不追求训练很久。它的产物是 Stage 2 的初始化 checkpoint。

### 6.2 Stage 2A：多正例主实验

```text
dataset: train_merged_multipos_p3.jsonl
loss: multi_positive_infonce
positive per row: 3
negative per row: explicit negatives
epoch: 1
temperature: 0.03
```

这是最重要的 A/B 实验：它直接验证多正例是否改善跨形态召回。

### 6.3 Stage 2B：Sub-center 辅助实验

```text
dataset: train_merged_multipos_p3.jsonl
loss: stage2_ip_embedding
formula: multi_positive_infonce + lambda_sub * subcenter_arcface
SUBCENTER_K: 3
SUBCENTER_LAMBDA: 0.05
SUBCENTER_MARGIN: 0.2
SUBCENTER_SCALE: 64
epoch: 1
```

Stage 2B 应与 Stage 2A 从同一个 Stage 1 checkpoint 启动，作为干净 A/B：

```text
Stage 1 checkpoint
  ├── Stage 2A: multi_positive_infonce
  └── Stage 2B: multi_positive_infonce + subcenter
```

这样才能判断 Sub-center 是否真的带来增益。

## 7. 指标设计

不能只看训练 loss。必须使用 retrieval protocol 评估：

- `Recall@1`
- `Recall@5`
- `Recall@10`
- `mAP`
- cross-form Recall@K
- 每个 IP 的 per-class recall
- head / tail IP 分组 recall

特别是 cross-form Recall@K：如果 query 是某 IP 的 A 形态，gallery 中命中的应当包含该 IP 的 B/C/D 形态。这个指标直接衡量模型是否学到 IP identity，而不是只学到服装、颜色、画风或背景。

## 8. 风险和后续方向

### 8.1 风险

- `multi_positive_infonce` 如果 positive 集合中有错标，会把 false positive 拉近。
- `Sub-center_K` 过大时，每个类有更多子中心，负类偶然匹配概率上升。
- `SUBCENTER_LAMBDA` 过大时，proxy 分类目标可能压过 retrieval 目标。
- 长尾 IP 仍然可能因为样本少而学不到稳定 identity。
- 未经过滤的 hard negative 会把漏标同 IP 当作负例，伤害训练。

### 8.2 后续方向

1. **Hard mining 数据闭环**

使用 Stage 2 模型离线编码 vault，挖 hard positives / hard negatives，再通过规则、reranker 或人工过滤。

2. **False negative 过滤**

高相似 negative 不应直接进入训练，应先用 reranker 或人工校验。

3. **Class-balanced sampling / queue**

缓解热门 IP 样本过多、长尾 IP 训练不足的问题。

4. **XBM / memory bank**

在标注质量和过滤策略稳定后，可考虑接入 memory bank 提升 hard negative 覆盖。

5. **Reranker-aware distillation**

后期可以让 embedding 排序与 VLM/reranker 判断更一致，减少召回和精排目标不一致。

## 9. 结论

本次 loss 设计的核心判断是：

```text
不要把同一 IP 的所有形态强压成单一中心；
要先让一个 anchor 同时看到多个同 IP 正例；
再用多中心 proxy 给同 IP 多形态提供几何结构；
hard negative 留到模型已有稳定结构之后再引入。
```

因此我们选择：

```text
Stage 1: infonce
Stage 2A: multi_positive_infonce
Stage 2B: multi_positive_infonce + Sub-center ArcFace
Stage 3: hard-mining / reranker-aware 数据闭环
```

这个方案的优点是：

- 与线上 top-K retrieval 目标一致；
- 直接修复当前框架只使用第一个 positive 的缺口；
- 支持同一 IP 多形态、多子簇；
- 控制工程复杂度，便于逐阶段 A/B；
- 为后续 hard mining 和 reranker-aware 训练保留扩展空间。

## 参考链接

- [SWIFT Embedding Training 文档](https://swift.readthedocs.io/en/v3.12/BestPractices/Embedding.html)
- [Qwen3-Embedding 官方仓库](https://github.com/QwenLM/Qwen3-Embedding)
- [Qwen3-VL-Embedding 官方仓库](https://github.com/QwenLM/Qwen3-VL-Embedding)
- [Khosla et al., Supervised Contrastive Learning, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)
- [Deng et al., Sub-center ArcFace, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf)
- [Xuan et al., Hard negative examples are hard, but useful, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590120.pdf)
- [Huang et al., CurricularFace, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Huang_CurricularFace_Adaptive_Curriculum_Learning_Loss_for_Deep_Face_Recognition_CVPR_2020_paper.html)
- [Wang et al., Cross-Batch Memory for Embedding Learning, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html)
- [Wang et al., Multi-Similarity Loss, CVPR 2019](https://www.deepnlp.org/content/articles/multi-similarity-loss-with-general-pair-weighting-for-deep-metric-learning)
- [Sun et al., Circle Loss, CVPR 2020](https://huggingface.co/papers/2002.10857)
- [Kim et al., Proxy Anchor Loss, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.html)
