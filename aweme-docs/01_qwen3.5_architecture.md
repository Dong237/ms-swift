# Qwen3.5 架构深度分析：Gated DeltaNet 与 Gated Attention 混合架构

## 1. 概述

Qwen3.5 是通义千问团队于 2026 年 3 月发布的新一代原生多模态大模型系列，其核心创新在于采用了 **混合注意力架构（Hybrid Attention Architecture）**，将传统的全注意力（Full Softmax Attention）与线性注意力（Linear Attention，即 Gated DeltaNet）相结合。这一架构继承自 Qwen3-Next（HuggingFace model_type: `qwen3_next`），在保持模型能力的同时显著提升了推理效率。

Qwen3.5 系列包含多个规模的模型：0.8B、2B、4B、9B、27B（Dense）以及 35B-A3B、122B-A10B、397B-A17B（MoE），所有模型均采用相同的混合注意力架构。

---

## 2. 混合注意力架构：3:1 层级结构

### 2.1 层级分配

Qwen3.5 的核心设计理念是将 Transformer 的 token-mixing 层分为两种类型：

- **线性注意力层（Linear Attention）**：使用 Gated DeltaNet 机制
- **全注意力层（Full Attention）**：使用 Gated Attention 机制

两者按照 **3:1** 的比例交替排列，由配置参数 `full_attention_interval` 控制：

```
[线性, 线性, 线性, 全注意力, 线性, 线性, 线性, 全注意力, ...]
```

这意味着约 **75%** 的层使用线性注意力，**25%** 的层使用全注意力。

### 2.2 设计动机

- **全注意力层**：提供全局上下文建模和强大的信息检索能力
- **线性注意力层**：提供高效的 O(1) 逐 token 推理能力，实现与序列长度线性相关的计算复杂度

---

## 3. Gated DeltaNet（线性注意力层）

### 3.1 背景

Gated DeltaNet 源自 ICLR 2025 论文 《Gated Delta Networks: Improving Mamba2 with Delta Rule》（arXiv: 2412.06464），由 Songlin Yang 等人提出。该方法结合了 Mamba2 的门控衰减机制（Gating）和 DeltaNet 的 Delta 规则（Delta Rule），实现了对两者的超越。

### 3.2 核心问题

传统的线性注意力（Vanilla Linear Attention）存在严重的 **记忆过载（Memory Overload）** 问题：在键值关联记忆系统中，只能不断添加新的键值对关联，无法擦除已有信息。随着序列长度增加，这导致累积的"检索错误"不断增大，严重影响模型性能。

### 3.3 数学公式

Gated DeltaNet 的递归形式如下：

**门控计算：**
```
g_t = exp(α_t)        // 衰减门（decay gate），控制记忆衰减速率
β_t = σ(b_t)          // 更新门（update gate），控制新输入的更新强度
```

**状态更新（核心 Delta 规则）：**
```
S_t = g_t · S_{t-1} + k̃_t ⊗ [β_t · (v_t − (g_t · S_{t-1})^T k̃_t)]
```

**输出计算：**
```
o_t = q̃_t^T · S_t / √d_k
```

其中：
- `S_t` 是 (d_k × d_v) 的状态矩阵，大小固定，不随序列长度增长
- `g_t` 是指数衰减门，控制旧记忆的保留程度
- `β_t` 是 sigmoid 更新门，控制新信息写入的强度
- `v_t − (g_t · S_{t-1})^T k̃_t` 是 Delta 规则的核心：计算当前期望值与预测值的误差，实现纠错式更新
- `k̃_t` 和 `q̃_t` 经过 L2 归一化处理

### 3.4 关键组件

| 组件 | 作用 | 替代的传统组件 |
|------|------|--------------|
| Delta 规则 | 纠错式记忆更新，提升检索精度 | Softmax 注意力的全局关注 |
| 指数门控（Exponential Gating） | 自适应记忆衰减，防止饱和 | 无（传统线性注意力无衰减机制） |
| 因果卷积（Causal Conv1D） | 提供局部上下文信息，kernel size=4 | RoPE 位置编码 |
| L2 归一化（Q/K） | 替代 Softmax 归一化 | Softmax 归一化 |

### 3.5 计算复杂度

| 方法 | 复杂度 | 状态大小 |
|------|--------|---------|
| 标准注意力 (SDPA) | O(n²d) | KV Cache 随序列长度线性增长 |
| Gated DeltaNet (递归) | O(nd²) | 固定大小矩阵 S (d_k × d_v) |
| Gated DeltaNet (分块并行) | O(n · C · d²) | C 为 chunk size，默认 64 |

### 3.6 投影结构

Gated DeltaNet 层的输入经过两个线性投影：
- `in_proj_qkvz`：将 hidden_size 投影到 Q、K、V、Z（门控向量）
- `in_proj_ba`：将 hidden_size 投影到 β 和 α 参数

输出处理流程：RMSNorm → SiLU 门控 → out_proj 线性变换

### 3.7 与 Mamba2 / Kimi Linear 的对比

- **Qwen3.5**：使用 **标量门控**（scalar gate），每个注意力头一个衰减值
- **Kimi Linear**：使用 **通道级门控**（channel-wise gate），每个特征维度一个衰减值
- **Mamba2**：仅有门控机制，无 Delta 规则
- **DeltaNet**：仅有 Delta 规则，无门控机制

---

## 4. Gated Attention（全注意力层）

### 4.1 机制

Gated Attention 本质上是在标准全注意力（Grouped-Query Attention + RoPE）基础上增加了一个 **sigmoid 输出门**：

```
1. 计算标准注意力输出: attn_output = Attention(Q, K, V)  // 使用 GQA + RoPE
2. 计算门控信号: gate = σ(W_gate · x)                     // sigmoid 门控
3. 门控输出: output = gate ⊙ attn_output                  // 逐元素相乘
```

### 4.2 设计目的

- **消除注意力汇聚点（Attention Sinks）**：通过门控抑制无意义的高注意力值
- **消除极端激活值（Massive Activations）**：通过门控平滑异常大的激活
- **提升训练稳定性**：特别是在大规模训练时

### 4.3 与 Gated DeltaNet 的区别

- Gated DeltaNet 使用 **SiLU** 激活函数作为输出门
- Gated Attention 使用 **Sigmoid** 激活函数作为输出门

---

## 5. Qwen3.5-0.8B 模型配置

| 参数 | 值 |
|------|-----|
| 模型类型 | `qwen3_5`（HuggingFace model_type） |
| 参数量 | 0.8B |
| 隐藏层维度 | 1024 |
| 层数 | 24 |
| 层级布局 | 6 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN)) |
| 词表大小 | 248,320（Padded） |
| 上下文长度 | 262,144（原生支持） |
| **Gated DeltaNet 配置** | |
| - 线性注意力头数 (Q/K) | 16 |
| - 线性注意力头数 (V) | 16 |
| - 头维度 | 128 |
| **Gated Attention 配置** | |
| - Q 头数 | 8 |
| - KV 头数 | 2 (GQA) |
| - 头维度 | 256 |
| - RoPE 维度 | 64 |
| **FFN 配置** | |
| - 中间维度 | 3584 |
| - 激活函数 | SiLU |
| LM Head | 248,320（与 token embedding 权重绑定） |

### 5.1 状态缓存（Hybrid Cache）

推理时的状态管理：
- **全注意力层**：标准 KV Cache，大小 (batch, num_kv_heads, seq_len, head_dim)，随序列长度线性增长
- **线性注意力层**：固定大小递归状态 (batch, num_v_heads, key_head_dim, value_head_dim) = (B, 16, 128, 128)

0.8B 模型的线性注意力层总固定内存开销约为 **18 × (B × 16 × 128 × 128) = 18 × 256KB ≈ 4.5MB/batch**（FP16），与序列长度无关。

---

## 6. Qwen3.5 vs Qwen3 对比

| 维度 | Qwen3 | Qwen3.5 |
|------|-------|---------|
| **注意力架构** | 标准 Transformer（纯 Softmax Attention） | 混合架构（75% Gated DeltaNet + 25% Gated Attention） |
| **计算复杂度** | O(n²d) | 混合：大部分层 O(nd²)，少部分层 O(n²d) |
| **多模态支持** | 独立模型（Qwen3-VL 为单独模型线） | 原生多模态（从预训练阶段开始融合） |
| **MoE 支持** | 独立 MoE 模型（Qwen3-30B-A3B） | 较大模型均采用超稀疏 MoE |
| **语言支持** | 119 种语言 | 201 种语言/方言 |
| **长上下文** | 128K（部分模型） | 262K（原生） |
| **推理效率** | 标准 KV Cache，推理成本随序列长度增长 | 大部分层使用固定大小状态，长上下文推理效率显著提升 |
| **训练稳定性** | 标准方法 | 零中心 LayerNorm、权重衰减 LayerNorm 等稳定性优化 |
| **位置编码** | RoPE（所有层） | 全注意力层用 RoPE，线性注意力层用 Causal Conv1D |
| **模型规模** | Dense: 0.6B-32B; MoE: 30B-A3B, 235B-A22B | Dense: 0.8B-27B; MoE: 35B-A3B, 122B-A10B, 397B-A17B |
| **激活参数比（最大MoE）** | ~9.4%（22B/235B） | ~4.3%（17B/397B），更极致的稀疏性 |

---

## 7. 其他架构创新

### 7.1 超稀疏 MoE（Ultra-Sparse MoE）

Qwen3.5 在较大规模模型中采用超稀疏 MoE 策略：
- 从 512 个专家中仅激活 10+1 个
- 实现极低的每 token FLOPs，同时保持模型容量
- 397B-A17B 仅激活约 4.3% 的参数

### 7.2 多 Token 预测（Multi-Token Prediction, MTP）

- 提升预训练模型性能
- 加速推理速度（推测解码）

### 7.3 原生多模态

- **早期融合**：在多模态 token 上进行早期融合训练
- **无需独立视觉适配器**：不同于 Qwen3 需要独立的 Qwen3-VL 模型
- 在视觉任务上超越 Qwen3-VL（MMMU: 85.0 vs 80.6）

### 7.4 训练稳定性优化

- 零中心 LayerNorm（Zero-centered LayerNorm）
- 带权重衰减的 LayerNorm（Weight-decayed LayerNorm）
- 其他稳定性增强措施

---

## 8. 行业背景与意义

Qwen3.5 的发布标志着多个重要趋势：

1. **稀疏 MoE 成为默认扩展策略**：不再是可选方案
2. **线性注意力进入生产环境**：从学术研究走向实际部署
3. **注意力机制成为新的竞争焦点**：Qwen 选择 Gated DeltaNet，DeepSeek 选择 MLA（Multi-head Latent Attention），代表了两种不同的技术路线

---

## 参考资料

- Qwen3.5 官方博客: https://qwen.ai/blog?id=qwen3.5
- Gated Delta Networks 论文 (ICLR 2025): https://arxiv.org/abs/2412.06464
- DeltaNet 解析: https://sustcsonglin.github.io/blog/2024/deltanet-1/
- Flash Linear Attention 实现: https://github.com/fla-org/flash-linear-attention
- Qwen3.5 HuggingFace 模型页: https://huggingface.co/Qwen/Qwen3.5-0.8B
- HuggingFace Transformers 文档: https://huggingface.co/docs/transformers/model_doc/qwen3_5
- Qwen3.5 架构分析: https://huggingface.co/blog/mlabonne/qwen35
