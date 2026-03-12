# 从文本生成模型到嵌入模型：论文分析与方法解读

## 1. 背景：为什么 Decoder-Only 模型可以做嵌入？

传统上，文本嵌入模型主要基于 Encoder-Only 架构（如 BERT、RoBERTa），因为这类模型天然具备双向注意力机制，可以全面捕获上下文语义。然而，随着大语言模型（LLM）的快速发展，研究人员发现 Decoder-Only 模型同样可以产生高质量的文本嵌入，甚至在某些任务上超越传统 Encoder 模型。

关键洞察：
- Decoder-Only LLM 在大规模预训练过程中已经学习了丰富的语义表示
- 通过适当的池化策略和对比学习训练，可以将这些表示转化为高质量的嵌入向量
- LLM 的指令跟随能力可以用于生成任务感知的嵌入

---

## 2. 论文一：Qwen3-Embedding（arXiv: 2506.05176）

### 2.1 论文信息

- **标题**：Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models
- **作者**：通义千问团队
- **发布日期**：2025 年 6 月
- **模型规模**：0.6B、4B、8B

### 2.2 核心方法：从 Qwen3 基座模型到嵌入模型

#### 2.2.1 模型初始化

Qwen3-Embedding 直接基于 **Qwen3 Dense 基座模型**（而非指令微调版本）进行初始化，利用基座模型在大规模预训练中获得的语义理解能力。

#### 2.2.2 嵌入向量提取位置

**关键设计决策：EOS Token 池化（Last Token Pooling）**

```
输入格式: "{Instruction} {Query}<|endoftext|>"

嵌入提取: embedding = last_hidden_state[EOS_token_position]
```

具体来说：
- 在输入序列末尾附加 `[EOS]` token（即 `<|endoftext|>`）
- 从模型**最后一层（last layer）的隐藏状态**中，取 **EOS token 对应位置**的向量作为最终嵌入
- 这种方法专门为因果语言模型（Causal LM）设计：由于因果注意力掩码的存在，EOS token 是唯一一个能"看到"所有前面 token 的位置，因此其隐藏状态包含了完整的序列语义信息

#### 2.2.3 指令感知嵌入

输入格式设计支持任务感知的嵌入生成：

```
{Instruction} {Query}<|endoftext|>
```

其中 Instruction 可以根据不同任务定制，例如：
- 检索任务："Given a web search query, retrieve relevant passages that answer the query"
- 语义相似度任务："Retrieve semantically similar sentences"
- 分类任务："Classify the given text into categories"

这使得同一个模型可以通过不同指令生成适用于不同任务的嵌入。

### 2.3 多阶段训练流程

#### 阶段一：弱监督预训练（Weakly Supervised Pre-training）

- **数据规模**：约 1.5 亿对（150M pairs）
- **数据来源**：使用 Qwen3-32B 模型合成生成
- **损失函数**：基于 InfoNCE 的改进对比损失
- **数据合成策略**：
  - 覆盖四种任务类型：检索（Retrieval）、双文本挖掘（Bitext Mining）、语义相似度（STS）、分类（Classification）
  - 使用 Persona Hub 为合成查询分配角色视角，提高数据多样性
  - 控制查询的长度、难度、语言等维度
  - 支持 100+ 种语言的多语言数据合成

#### 阶段二：有监督微调（Supervised Fine-Tuning）

- **数据规模**：约 700 万高质量标注对（7M pairs）+ 约 1200 万筛选后的合成数据（12M pairs）
- **损失函数**：与阶段一相同的改进 InfoNCE 损失
- **数据质量**：使用更严格的筛选标准，确保训练数据的高质量

#### 阶段三：模型合并（Model Merging）

- **方法**：球面线性插值（Spherical Linear Interpolation, SLERP）
- **操作**：合并微调过程中保存的多个检查点
- **目的**：提升模型在不同数据分布上的鲁棒性和泛化性能

### 2.4 损失函数详解

#### InfoNCE 改进版对比损失

标准 InfoNCE 损失公式：

```
L = -(1/N) Σ_i log [exp(s(q_i, d_i^+) / τ) / Z_i]
```

Qwen3-Embedding 的改进之处在于分母 Z_i 的组成：

```
Z_i = exp(s(q_i, d_i^+) / τ)                    // 正样本相似度
    + Σ_j exp(s(q_i, d_i,j^-) / τ)              // 硬负例相似度
    + Σ_{k≠i} m_ik · exp(s(q_i, d_k^+) / τ)     // batch 内其他文档
    + Σ_{k≠i} exp(s(q_i, q_k) / τ)              // batch 内其他查询 (QQ)
    + Σ_{k≠i} exp(s(d_i^+, d_k^+) / τ)          // batch 内其他文档对 (DD)
```

**关键增强**：
- **QQ 相似度**（`INFONCE_INCLUDE_QQ`）：将 batch 内其他查询也作为负例
- **DD 相似度**（`INFONCE_INCLUDE_DD`）：将 batch 内其他文档也作为负例
- **假负例掩码**（`INFONCE_MASK_FAKE_NEGATIVE`）：当某个"负例"与正例的相似度差值超过正例得分 0.1 以上时，将其掩码为零，避免将实际上相关的样本作为负例训练

**温度参数**：τ 默认为 0.1

### 2.5 Matryoshka 表示学习（MRL）

支持灵活的嵌入维度：模型可以输出任意维度的嵌入向量，而不仅限于固定维度。这通过 Matryoshka 表示学习实现，使得低维嵌入仍然保持良好的性能。

### 2.6 结果

- Qwen3-Embedding-8B 在 MTEB 多语言排行榜上排名第一（截至 2025.06.05，得分 70.58）
- 模型采用 Apache 2.0 许可证开源

---

## 3. 论文二：Qwen3-VL-Embedding（arXiv: 2601.04720）

### 3.1 论文信息

- **标题**：Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking
- **作者**：通义千问团队
- **发布日期**：2025 年 1 月
- **模型规模**：2B、8B

### 3.2 核心方法：多模态生成模型到嵌入模型

#### 3.2.1 模型基座

基于 **Qwen3-VL**（视觉语言模型）作为基座模型，继承其多模态理解能力。

#### 3.2.2 嵌入向量提取

与 Qwen3-Embedding 一致，采用 **EOS Token 池化**：

```
embedding = last_hidden_state[EOS_token_position]
```

从基座模型最后一层提取 EOS token 对应的隐藏状态向量作为最终语义表示。

### 3.3 多阶段训练范式

#### 阶段一：大规模对比预训练（Contrastive Pre-training）

- 使用大规模多模态数据进行对比学习
- 训练数据覆盖文本、图像、文档图像、视频等多种模态

#### 阶段二：重排序模型蒸馏（Reranker Distillation）

- 利用训练好的 Qwen3-VL-Reranker 模型的知识，对嵌入模型进行蒸馏
- 通过蒸馏提升嵌入质量

### 3.4 Matryoshka 表示学习

与 Qwen3-Embedding 类似，支持灵活的嵌入维度：
- 可以在不同维度下使用嵌入向量
- 低维嵌入仍保持良好性能
- 最大支持 32K token 的输入长度

### 3.5 重排序模型（Reranker）

论文同时提出了 Qwen3-VL-Reranker：

- **架构**：交叉编码器（Cross-Encoder），使用交叉注意力机制
- **相关性评分方法**：通过预测特殊 token "yes" 和 "no" 的生成概率来表示相关性分数
- **优势**：进行更深层次、更细粒度的跨模态交互和信息融合

#### 两阶段检索流程

```
查询 → Embedding 模型（粗排/召回）→ Top-K 候选 → Reranker（精排/重排）→ 最终结果
```

### 3.6 结果

- Qwen3-VL-Embedding-8B 在 MMEB-V2 上总分 77.8，排名第一（截至 2025.01.08）
- 支持 30+ 种语言
- 无缝处理文本、图像、截图、视频等多种输入

---

## 4. 两篇论文的对比分析

| 维度 | Qwen3-Embedding | Qwen3-VL-Embedding |
|------|-----------------|-------------------|
| **基座模型** | Qwen3 Dense（纯文本 LLM） | Qwen3-VL（视觉语言模型） |
| **模态支持** | 纯文本 | 文本 + 图像 + 视频 + 文档 |
| **嵌入提取** | EOS Token（最后一层隐藏状态） | EOS Token（最后一层隐藏状态） |
| **训练阶段** | 弱监督预训练 → 有监督微调 → 模型合并 | 对比预训练 → 重排序蒸馏 |
| **数据合成** | Qwen3-32B 合成 | 多模态数据收集 |
| **损失函数** | 改进 InfoNCE | 对比损失 |
| **MRL 支持** | 支持 | 支持 |
| **模型规模** | 0.6B, 4B, 8B | 2B, 8B |
| **评测基准** | MTEB | MMEB-V2 |

### 4.1 共同的核心设计选择

1. **EOS Token 池化**：两篇论文都选择从最后一层的 EOS token 位置提取嵌入向量。这是 Decoder-Only 模型的最佳实践：
   - EOS token 位于序列末尾，在因果注意力中可以关注所有前面的 token
   - 相当于对整个序列信息的自然汇聚点

2. **保持因果注意力**：两者都没有将 Decoder-Only 模型改造为双向注意力，而是保持原有的因果注意力掩码，仅通过 EOS token 位置的表示来获取嵌入

3. **多阶段训练**：都采用了多阶段训练策略，从大规模弱监督数据逐步过渡到高质量数据

4. **指令感知**：都支持通过输入指令来定制嵌入行为

---

## 5. 从 Decoder-Only LLM 到嵌入模型的一般范式

基于这两篇论文，可以总结出将文本生成模型转化为嵌入模型的通用范式：

### 5.1 步骤总结

```
1. 选择基座模型
   └── 使用高质量预训练的 Decoder-Only LLM 作为初始化

2. 确定嵌入提取策略
   └── EOS Token Pooling: 在输入末尾添加 EOS token，
       从最后一层隐藏状态中提取该位置的向量

3. 设计训练数据
   ├── 大规模合成数据（弱监督）
   │   └── 使用强大的 LLM（如 Qwen3-32B）合成查询-文档对
   └── 高质量标注数据（有监督）
       └── 人工标注或精心筛选的数据

4. 多阶段训练
   ├── 阶段 1: 大规模弱监督对比学习
   ├── 阶段 2: 小规模高质量有监督微调
   └── 阶段 3: 模型合并（可选，如 SLERP）

5. 损失函数
   └── InfoNCE 对比损失（支持 batch 内负例、硬负例、QQ/DD 增强）

6. 高级特性
   ├── Matryoshka 表示学习（灵活维度）
   ├── 指令感知嵌入
   └── 多语言/多模态支持
```

### 5.2 为什么 EOS Token 是最佳选择？

对于因果语言模型（Causal LM），选择 EOS token 作为嵌入提取位置有以下原因：

1. **信息完整性**：EOS token 是序列中最后一个 token，由于因果注意力掩码的存在，它是唯一可以"看到"所有前面 token 的位置
2. **自然汇聚**：在自回归训练过程中，模型学习在 EOS 位置汇总前文信息，这个位置的表示天然具有序列级语义
3. **实现简单**：无需修改模型架构，只需在输入末尾添加 EOS token 即可
4. **与预训练一致**：保持了模型预训练时的因果注意力模式，不会引入分布偏移

### 5.3 替代方案的劣势

| 策略 | 缺点 |
|------|------|
| Mean Pooling | 因果注意力下，前面的 token 看不到后面的信息，平均池化会引入不完整的表示 |
| CLS Token | Decoder-Only 模型预训练时没有 CLS token，需要额外训练 |
| 双向注意力改造 | 修改注意力掩码会偏离预训练分布，需要大量额外训练来适应 |

---

## 参考资料

- Qwen3 Embedding 论文: https://arxiv.org/abs/2506.05176
- Qwen3-VL-Embedding 论文: https://arxiv.org/abs/2601.04720
- Qwen3-Embedding GitHub: https://github.com/QwenLM/Qwen3-Embedding
- Qwen3-VL-Embedding GitHub: https://github.com/QwenLM/Qwen3-VL-Embedding
- HuggingFace 模型: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- Alibaba Cloud 教程: https://www.alibabacloud.com/blog/mastering-text-embedding-and-reranker-with-qwen3_602308
