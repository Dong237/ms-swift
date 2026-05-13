# IP 虚拟形象 Embedding 训练升级方案

本文档用于说明 `projects/ips` 后续训练升级的目标、背景、阶段设计和代码改造边界，作为 Codex 与 Claude 后续协作实现的共同上下文。

> **版本**: v2.0 (Final)
> **上次更新**: 2026-05-08
> **状态**: 已通过 Claude + Codex 双向 review，准备进入实现阶段
 
## 1. 背景与业务目标

当前任务是训练一个面向虚拟形象 / IP 识别的图像表征模型。线上链路不是直接用 embedding 给出最终判定，而是：

```text
输入图像
  -> embedding 模型编码
  -> 从 IP vault 中召回 top-K 相似图片
  -> VLM / reranker 对 query 与候选图进行二次判断
  -> 输出是否疑似 IP 侵权 / 是否需要拦截或人审
```

因此 embedding 模型的核心目标是高召回，尤其是同一 IP 在不同形态、皮肤、妆造、发型、衣着、视角下的召回。只要 top-K 中至少有一个正确同 IP 候选，后续 VLM/reranker 仍有机会判断；如果 embedding 阶段漏召，后续阶段无法补救。

该任务与传统人脸识别不同。人脸识别通常假设同一身份的视觉结构相对稳定，但虚拟 IP 可能出现：

- 同一动漫角色在不同作品、不同画风、不同服装和发型下跨度极大。
- 同一游戏英雄有默认皮肤、机甲皮肤、节日皮肤、联名皮肤等多个不连续视觉形态。
- 不同 IP 可能共享相似画风、服装元素、发色、武器或构图，导致类间距离很小。
- 训练数据可能存在漏标、混标、同 IP 别名、冷门皮肤未归并等噪声。

因此这里的核心难点是：

```text
类内方差可能大于类间方差。
```

训练目标不应是强行把同一 IP 的所有图片压成一个单中心簇，而应是在保留多形态结构的同时，让同一 IP 在检索空间中足够可召回。

## 2. 当前仓库能力与限制

当前 `ms-swift` embedding 训练链路已经可以支持 Stage 1 的基础训练：

- `--task_type embedding`
- `--loss_type infonce`
- `INFONCE_USE_BATCH=False`
- 数据格式为 anchor + positive + explicit negatives

当前相关代码位置：

- `swift/loss/embedding.py`
- `swift/loss/mapping.py`
- `swift/template/base.py`
- `projects/ips/build_ip_training_data.py`
- `projects/ips/train_ip_embedding*.sh`
- `projects/ips/eval_ip_retrieval.py`

当前已支持的 embedding loss 包括：

- `cosine_similarity`
- `contrastive`
- `online_contrastive`
- `infonce`

当前不支持本升级方案所需的 Stage 2 能力：

- 多正例 supervised contrastive / multi-positive InfoNCE。
- 同一 anchor 对多个 positive 同时计算正例分子。
- batch 内同 IP 样本作为 positive，而不是误作 negative。
- PK sampler 或 class-balanced batch。
- Sub-center ArcFace / multi-proxy auxiliary head。
- 训练时组合 loss，例如 `L_mp_infonce + lambda_sub * L_subcenter`。

特别注意：当前 `swift/template/base.py` 的 embedding encode 逻辑会将多个 positive 折叠成第一个 positive：

```python
positive = inputs.positive
if isinstance(positive, list):
    positive = positive[0]
```

因此，即使 JSONL 中写入多个 `positive_messages` / `positive_images`，当前训练也不会真正使用多个正例。

## 3. 总体训练原则

本升级方案按训练阶段组织，而不是按“逐个添加 loss”组织。每个阶段的重点是训练目标、样本难度和特征空间几何结构。

核心原则如下：

1. embedding 阶段优先优化 top-K 召回，而不是闭集分类准确率。
2. 同一 IP 的多形态不应被强行压成单一中心。
3. hard negative 必须逐步引入，且需要过滤 false hard negative。
4. hard positive 与 hard negative 同等重要，跨形态正例是该任务的关键训练信号。
5. Sub-center ArcFace 是多模态类结构约束，应在 Stage 2 开始介入，而不是等到 Stage 3。
6. VLM/reranker 是线上最终判断器，后期训练应尽量让 embedding 排序与 reranker 判断对齐。

## 4. 推荐训练阶段

### Stage 1: 显式负例 InfoNCE Warmup

Stage 1 基本沿用当前训练方式，作为领域适配和稳定 warmup。

训练目标：

- 让 Qwen3.5 / Qwen3-VL embedding 模型适应 IP 图像域。
- 建立初步 query-positive 相似度。
- 避免一开始就被复杂 hard negative 或跨形态 hard positive 拉崩。

Loss：

```text
L_stage1 = L_infonce_explicit
```

配置建议：

```text
INFONCE_USE_BATCH=False
INFONCE_TEMPERATURE=0.05 ~ 0.1
```

> **Review note**: 原方案建议 τ=0.02~0.05，但在 `USE_BATCH=False` 下有效负样本池仅有显式负例（通常 3~5 个），极低温度 + 小负样本池会导致梯度方差过大。Stage 1 应使用较温和的温度，等 Stage 2 负样本更充足后再降低。

数据组织：

```text
anchor: 当前图像
positive: 同 IP 图像 1 张
negative: 显式采样的不同 IP 图像若干张
```

负例采样：

- 以 random negative 和 medium negative 为主。
- 暂不使用 ANN mined hardest negative。
- 尽量避免同 IP 漏标图片被放进 negative。

训练长度：

- 约 0.5 到 1 epoch，或总训练步数的 10% 到 15%。

验收标准：

- loss 稳定下降。
- 基础 CMC@K / Recall@K 有提升。
- 不要求此阶段解决跨形态召回。

### Stage 2: 多正例跨形态结构学习

Stage 2 是本次升级的核心阶段。它不是简单地“添加 SupCon loss”，而是要改变训练样本组织和特征空间约束。

训练目标：

- 让同一 IP 的多个形态都成为 anchor 的正例。
- 显式学习跨皮肤、跨妆造、跨发型、跨画风的 identity 不变性。
- 允许同一 IP 在 embedding 空间中形成多个合理子簇，而不是单中心坍缩。

推荐 loss：

```text
L_stage2 = L_multi_positive_infonce + lambda_sub * L_subcenter_arcface
```

Stage 2 推荐温度：

```text
INFONCE_TEMPERATURE=0.03 ~ 0.07
```

其中 `L_multi_positive_infonce` 是主 loss，`L_subcenter_arcface` 是辅助几何约束。

#### 4.2.1 Multi-positive InfoNCE / SupCon

对一个 anchor，所有同 IP positive 同时进入正例集合：

```text
anchor: q
positives: p1, p2, ..., pM
negatives: n1, n2, ..., nN
```

主 loss 形式：

```text
L = -log(
  sum_{p in P(q)} exp(sim(q, p) / tau)
  /
  [sum_{p in P(q)} exp(sim(q, p) / tau) + sum_{n in N(q)} exp(sim(q, n) / tau)]
)
```

关键要求：

- 多个 positive 不能只取第一个。
- 同 IP 样本不能进入 negative denominator。
- 如果使用 batch 内样本，需要 class id / ip id 来 mask same-IP negatives。
- 跨形态 positive 应优先采样或加权。

第一版实现建议：

- 先实现 row-local multi-positive InfoNCE。
- 每条 JSONL 内包含 anchor + 多个 positives + 显式 negatives。
- 不急着启用 batch-global SupCon。
- 不改变现有 `infonce` 行为，新增 loss type，例如 `multi_positive_infonce`。

这样实现风险最低，且能先吃到多正例训练收益。

#### 4.2.1.1 Label Encoding 方案（关键实现细节）

当前 `infonce` 使用 `labels` tensor 中的 `1.0` 作为 group boundary marker（见 `_parse_multi_negative_sentences`）。Multi-positive 场景下，如果一个 anchor 有多个 positive，多个 `1.0` 会被误判为多个 group 的起点。

**`multi_positive_infonce` 采用新的 label 编码**（仅限该 loss type，不影响现有 `infonce`）：

```text
2.0 = group boundary（每组第一个 positive，同时标识新 group 的开始）
1.0 = additional positive（同组的其他 positive）
0.0 = negative
```

示例（2 个样本，第一个有 2 个 positive + 3 个 negative，第二个有 3 个 positive + 1 个 negative）：

```text
labels = [2, 1, 0, 0, 0, 2, 1, 1, 0]
          ^ p1   ^ n1,n2,n3  ^ p1      ^ n1
            ^ p2               ^ p2
                                 ^ p3
```

解析逻辑：`torch.nonzero(labels == 2)` 得到 group boundary，`labels >= 1` 为 positive mask。

#### 4.2.2 Sub-center ArcFace 辅助约束

Sub-center ArcFace 的作用是允许每个 IP 有多个 proxy/sub-center。例如一个 IP 可以自然分成默认形态、机甲形态、节日皮肤形态等子簇。

它不应替代 multi-positive retrieval loss，而应作为辅助项：

```text
L_stage2 = L_multi_positive_infonce + lambda_sub * L_subcenter_arcface
```

建议：

```text
K = 2 ~ 4
lambda_sub 从 0 warm up 到 0.1 ~ 0.3
margin m 从 0.1 warm up 到 0.25
scale s 可设为 32 或 64
```

为什么 Sub-center 属于 Stage 2：

- Stage 2 的目标是建立同 IP 多形态结构。
- Sub-center 正是多形态结构约束。
- 如果 Stage 2 只用普通 SupCon，模型可能已经开始把同 IP 样本往单区域压缩。
- 等到 Stage 3 再引入 sub-center，会与前面形成的几何结构产生冲突。

第一版实现建议：

- 不要在 multi-positive MVP 中立即加入 Sub-center。
- 先实现 multi-positive InfoNCE，确认训练稳定后再实现 combined loss。
- Sub-center 需要 trainable proxy 参数、class id 映射、优化器接入和 checkpoint 保存，属于第二个 patch。

Sub-center proxy 管理策略：

- **第一版要求固定 `num_classes`**。训练前扫描全量数据确定 IP 总数，构建稳定的 `ip_id -> class_index` 映射。运行时不支持动态扩展。
- proxy head 仅在训练时使用（推理时不需要 proxy），因此线上新增 IP 不需要扩展 proxy。
- 每次 from scratch 训练时重新初始化 proxy 矩阵。不需要跨训练轮次兼容旧 proxy checkpoint。

### Stage 3: Hard Positive / Hard Negative Curriculum

Stage 3 的重点不是换一套新 loss，而是改变样本难度分布。

训练目标：

- 拉近当前模型仍然召不回的跨形态同 IP 样本。
- 推开当前模型容易混淆的不同 IP 样本。
- 通过 curriculum 避免早期 hard sample 噪声破坏训练。

推荐 loss：

```text
L_stage3 = L_multi_positive_infonce + lambda_sub * L_subcenter_arcface
```

可选加入：

```text
+ lambda_ms * L_multi_similarity_or_circle
```

`Multi-Similarity Loss` 或 `Circle Loss` 可用于 hard pair 加权，但不建议第一版就上。它们适合在 hard sample 池质量较高后作为后期增强。

#### 4.3.1 Hard Positive Mining

Hard positive 是该任务的核心，不应只关注 hard negative。

挖掘方式：

1. 用当前 checkpoint 编码全量 IP vault。
2. 对每个 IP 内部计算 pairwise similarity。
3. 找出同 IP 中相似度较低但标注可信的 pair。
4. 优先采样跨形态、跨皮肤、跨妆造、跨画风 positive。

防噪策略：

- 对极端低相似度 positive 降权或送人工复核。
- 如果 VLM/reranker 也无法确认同 IP，不要强行作为 hard positive。
- 可设定 per-IP 相似度分位阈值，过滤明显异常样本。

#### 4.3.2 Hard Negative Mining

Hard negative 应从当前 embedding 的 top-K 近邻中挖：

```text
encode 全 vault
  -> 建 ANN index
  -> 对每张 query 取 top100/top200
  -> 去掉同 IP
  -> 剩余候选作为 hard negative candidates
  -> 经过 VLM/reranker 或人工规则过滤
  -> 写回下一轮训练数据
```

注意：不能直接使用 hardest negative。虚拟 IP 数据中漏标、别名、冷门皮肤未归并很常见，hardest negative 里会有大量 false negative。

推荐 negative curriculum：

```text
Stage 3 early: random : medium : hard = 4 : 4 : 2
Stage 3 late:  random : medium : hard = 2 : 3 : 5
```

其中：

- random negative: 随机不同 IP。
- medium negative: 同画风、同题材、同角色类型但不同 IP。
- hard negative: 当前模型 top-K 召回中高相似但确认不同 IP 的样本。

推荐温度：

```text
Stage 3 tau: 0.02 ~ 0.05（此时负样本数量和质量足够支撑低温度）
```

### Stage 4: Reranker-aware Fine-tuning

Stage 4 是可选但很有价值的线上对齐阶段。

训练目标：

- 让 embedding top-K 排序更接近最终 VLM/reranker 判断。
- 降低“embedding 召回相似外观但 reranker 判不同 IP”的排序浪费。

数据构造：

```text
query image
  -> 当前 embedding 召回 top-K candidates
  -> VLM/reranker 给每个 candidate 打分或给同 IP / 非同 IP 判断
  -> 形成 listwise 或 pairwise distillation 数据
```

训练方式：

- 可以先不直接改主训练 loss。
- 先离线分析 reranker 分数与 embedding 排序差异。
- 后续可加入 listwise distillation 或 soft-label pairwise loss。

该阶段优先级低于 Stage 2 和 Stage 3，但最终会更贴近线上目标。

## 4.5 数据增强策略

训练时数据增强的目标是让模型学会忽略"可变特征"（服装纹理、背景、构图），聚焦"不变特征"（面部结构、标志性配饰、体型轮廓）。

推荐增强（按优先级）：

```text
1. 中等强度 Color Jitter（不要激进）
   - 动漫/游戏 IP 的发色、瞳色、标志性配色是 identity signal
   - 过度 jitter 会破坏这些信号
   - 建议只做亮度/对比度/饱和度的温和扰动

2. 背景/纹理扰动
   - 高斯模糊、背景替换、纹理随机化
   - 保留角色主体，干扰背景依赖

3. Random Erasing（偏离面部区域）
   - 随机遮挡服装区域，迫使模型用面部/配饰识别角色
   - 避开面部、眼睛、标志性配饰区域

4. Multi-Crop（可选）
   - 2 张全局裁剪 + 4 张局部裁剪
   - 全身视角和面部特写应产生一致 embedding
```

> **Note**: 增强策略在数据管线侧实现，不影响 loss 侧的 Patch 1-3。可独立于代码改造并行推进。

## 5. 数据格式升级方向

### 5.1 Stage 1 当前格式

当前格式大致为：

```json
{
  "messages": [{"role": "user", "content": "<image>"}],
  "images": ["anchor.jpg"],
  "positive_messages": [[{"role": "user", "content": "<image>"}]],
  "positive_images": [["positive.jpg"]],
  "negative_messages": [[{"role": "user", "content": "<image>"}]],
  "negative_images": [["negative.jpg"]]
}
```

### 5.2 Stage 2 目标格式

建议扩展为：

```json
{
  "ip_id": 12345,
  "form_id": "skin_a",
  "messages": [{"role": "user", "content": "<image>"}],
  "images": ["anchor.jpg"],
  "positive_messages": [
    [{"role": "user", "content": "<image>"}],
    [{"role": "user", "content": "<image>"}]
  ],
  "positive_images": [
    ["positive_form_b.jpg"],
    ["positive_form_c.jpg"]
  ],
  "negative_messages": [
    [{"role": "user", "content": "<image>"}],
    [{"role": "user", "content": "<image>"}]
  ],
  "negative_images": [
    ["negative_medium.jpg"],
    ["negative_hard.jpg"]
  ],
  "negative_types": ["medium", "hard"]
}
```

第一版最重要的是保留多个 positives，并能在 loss 中知道每条样本的 positive 数量和 negative 数量。

> **Note**: `negative_types` 字段用于数据分析和后续 curriculum 采样脚本，不参与 loss 计算。loss 不会对不同类型的 negative 做差异化加权（第一版）。

## 6. 代码改造计划

建议分 patch 实现，不要一次性做完整研究框架。

### Patch 1: Multi-positive InfoNCE MVP

目标：

- 新增 `multi_positive_infonce` loss。
- 支持 row-local anchor + multiple positives + explicit negatives。
- 保持现有 `infonce` 行为完全不变。

实现规约（已通过 review 确认）：

```text
1. 不修改现有 infonce，新增独立的 loss type。
2. 修改 _embedding_encode 保留多个 positive（去掉 positive[0] 截断）。
3. 修改 _embedding_data_collator 支持 positive0_, positive1_, ... 展开。
4. 新增 label encoding: 2.0=group boundary, 1.0=additional positive, 0.0=negative。
5. 新增 MultiPositiveInfonceLoss，仅支持 row-local（不启用 batch-global）。
6. 新增 multi_positive_infonce eval metric。
7. 添加合成 embedding 单元测试。
8. 新增 stage2 训练脚本。
```

需要修改：

- `swift/template/base.py`（`_embedding_encode` + `_embedding_data_collator`）
- `swift/loss/embedding.py`（新增 `MultiPositiveInfonceLoss`）
- `swift/loss/mapping.py`（注册新 loss type）
- `swift/metrics/embedding.py`（新增 `MultiPositiveInfonceMetrics`）
- `projects/ips/train_ip_embedding*.sh` 或新增 stage2 脚本

验收点：

- 多个 positive 不再被折叠成第一个。
- loss 能处理每条样本不同数量的 positive / negative。
- 旧的 `--loss_type infonce` 测试仍然通过（**第一 QA 门禁**）。
- 合成 embedding 测试中，anchor 与多个 positives 更接近时 loss 更低。
- 两个 positive 都实际参与了 loss 计算（不是只用了第一个）。

### Patch 2: IP id / Class-aware Batch Metadata

目标：

- 数据中显式包含稳定 `ip_id`。
- collator 能把 `ip_id` 传给 loss。
- 为 batch-global SupCon 和 same-IP masking 做准备。

注意：

- 不要把 `ip_id` 混进旧的 float labels 逻辑。
- 需要清晰区分 group boundary、positive mask 和 class id。

### Patch 3: Sub-center ArcFace Auxiliary

目标：

- 在 Stage 2 combined loss 中加入可选 Sub-center ArcFace 辅助项。
- 每个 IP 有 K 个 trainable proxy。
- 样本只需靠近本 IP 的某一个 sub-center。

需要解决：

- proxy 参数如何注册到 model/trainer，确保 optimizer 更新。
- class id 到 proxy row 的稳定映射。
- checkpoint 保存和恢复。
- DDP 下 proxy 参数同步。
- lambda / margin / scale 的配置入口。
- `stage2_loss_norm` 默认必须为 `none`，保证旧脚本和普通 `infonce`/`multi_positive_infonce` 向后兼容。

建议新增 loss type：

```text
stage2_ip_embedding
```

内部组合：

```text
L = L_multi_positive_infonce + lambda_sub * L_subcenter_arcface
```

静态参考归一化 ablation：

```text
--stage2_loss_norm none          # 默认，保持当前行为
--stage2_loss_norm subcenter_ref # L_subcenter / log(num_classes)
```

`subcenter_ref` 只归一化 Sub-center ArcFace 辅助项，不改变 `multi_positive_infonce`。它用于判断 `SUBCENTER_LAMBDA` 是否因为 class-level CE 尺度过大而压过 retrieval loss。

### Patch 4: Hard Mining 数据闭环

目标：

- 训练后离线编码全量 vault。
- 构建 ANN index。
- 生成 hard positive / hard negative candidate JSONL。
- 通过 VLM/reranker 或人工规则过滤。
- 产出下一轮训练数据。

该 patch 主要在 `projects/ips` 下新增脚本，不应侵入 `swift` 主框架。

## 7. 评估要求

主评估指标（按优先级排列）：

```text
CMC@1 / Recall@1          -- top-1 命中率，直接反映 reranker 负担
CMC@5
CMC@10
cross-form CMC@1          -- 跨形态 top-1，核心能力指标
cross-form CMC@K
mAP@K
R-Precision
```

> **Recall@1 重要性说明**: 在线上链路中，top-1 命中同 IP 意味着 reranker 几乎不需要额外判断。Recall@1 的提升比 Recall@10 有更大的业务价值。

必须新增或单独维护 cross-form 评估：

```text
query: IP 的形态 A
gallery: 同 IP 只保留形态 B/C/D，不包含形态 A 近重复图
```

如果总 Recall@K 提升但 cross-form Recall@K 不提升，说明模型可能仍在依赖表层外观，而没有学到真正 IP identity。

线上相关指标：

- top-K 中是否至少出现一个同 IP 候选。
- reranker 前后的最终拦截 precision / recall。
- false positive 样本类型。
- false negative 样本类型。
- hard negative 中被发现为漏标同 IP 的比例。

## 8. 协作分工建议

详细的协作流程见 `projects/ips/COLLABORATION_GUIDE.md`。

核心原则：

- **非对称分工**：一个 agent 实现，另一个 QA + review。不要让两个 agent 同时自由修改同一模块。
- **串行交接**：Patch 1 的 label encoding 方案影响后续所有 patch，必须先完成并验收。
- **Claude 负责 Patch 1 和 Patch 3**（需要深度理解现有数据流：encode → collate → loss → metrics 四处联动）。
- **Codex 负责 Patch 2 和 QA**（定义清晰的 plumbing 任务 + 回归测试）。
- **Patch 4 可以和 Patch 1 并行**（完全在 `projects/ips/` 下，不碰框架代码）。

## 9. 不建议第一版做的事情

第一版不要做：

- 不要重写 backbone。
- 不要直接替换为 DINO/SigLIP 主训练栈。
- 不要无标签实现 identity/style 双分支正交解耦。
- 不要一开始就启用 hardest negative。
- 不要把 Sub-center ArcFace 做成唯一主 loss。
- 不要改坏现有 `infonce` 路径。

这些方向可以后续探索，但不应阻塞最关键的 Stage 2 MVP。

## 10. 推荐近期执行顺序

近期最小可行升级路径：

```text
1. 保留当前 Stage 1 infonce warmup。
2. 实现 row-local multi-positive InfoNCE。
3. 修改数据构造，让每个 anchor 有多个跨形态 positives。
4. 跑小规模 smoke train，确认 loss 和 retrieval 指标正常。
5. 加入 ip_id metadata，为 batch-global SupCon 和 sub-center 做准备。
6. 实现 Sub-center ArcFace auxiliary，并以小 lambda 开始 ramp。
7. 建立 ANN hard mining 脚本，逐轮引入 hard positives / hard negatives。
8. 最后考虑 reranker-aware distillation。
```

本次升级的核心判断是：

```text
Stage 2 改变 embedding 空间几何结构；
Stage 3 改变训练样本难度分布。
```

因此 Sub-center ArcFace 属于 Stage 2 的结构约束，应与 multi-positive contrastive training 同阶段引入，并通过 `lambda_sub`、margin 和 K 控制强度。Stage 3 则继续使用同一核心 loss 家族，但通过 hard positive / hard negative curriculum 让模型学习更难的边界。

## 11. 难度评估与风险

| Patch | 难度 (1-10) | 主要风险 |
|---|---|---|
| Patch 1: Multi-positive InfoNCE | 5/10 | label encoding 变更需同步改 encode/collator/loss/metrics 四处 |
| Patch 2: ip_id metadata | 3/10 | 纯数据管道 plumbing，风险低 |
| Patch 3: Sub-center ArcFace | 5/10 | proxy 参数注入 optimizer + DDP 同步（已简化：不需要动态扩展和 checkpoint 兼容） |
| Patch 4: Hard mining scripts | 3/10 | 独立脚本，不碰框架 |

Stage 1 + Stage 2 整体难度：**5/10**（如果 Stage 2 先只做 Patch 1 MVP，降到 4/10）。

最大风险：Patch 1 的 label encoding 变更可能破坏现有 `infonce` 行为。因此必须新增独立 loss type，不修改现有路径。
