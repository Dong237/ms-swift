# ms-swift qwen3point5 分支对 Qwen3.5 嵌入训练的支持分析

## 1. ms-swift 框架概述

ms-swift（Scalable lightWeight Infrastructure for Fine-Tuning）是 ModelScope 开发的大模型微调和部署框架，支持 600+ 纯文本 LLM 和 400+ 多模态模型的训练、推理、评估、量化和部署全流程。

---

## 2. 已有的嵌入训练支持

### 2.1 任务类型（Task Type）

ms-swift 已原生支持嵌入训练任务类型：

```python
# swift/arguments/base_args/model_args.py (line 71)
task_type: Literal['causal_lm', 'seq_cls', 'embedding', 'reranker', 'generative_reranker']
```

通过 `--task_type embedding` 即可启用嵌入训练模式。

### 2.2 损失函数

在 `swift/loss/embedding.py` 中已实现完整的嵌入训练损失函数：

| 损失函数 | 类名 | 说明 |
|---------|------|------|
| 余弦相似度损失 | `CosineSimilarityLoss` | MSE 损失在余弦相似度上 |
| 对比损失 | `ContrastiveLoss` | 孪生网络式对比学习，带 margin |
| 在线对比损失 | `OnlineContrastiveLoss` | 硬正/负例在线挖掘 |
| **InfoNCE 损失** | `InfonceLoss` | **支持 Qwen3-Embedding 风格的增强** |

**InfoNCE 损失的高级特性**（`swift/loss/embedding.py`, lines 113-322）：
- 温度缩放（`INFONCE_TEMPERATURE`，默认 0.1）
- Batch 内负例（`INFONCE_USE_BATCH`）
- 硬负例数量控制（`INFONCE_HARD_NEGATIVES`）
- QQ 相似度增强（`INFONCE_INCLUDE_QQ`）
- DD 相似度增强（`INFONCE_INCLUDE_DD`）
- 假负例掩码（`INFONCE_MASK_FAKE_NEGATIVE`）
- 分布式训练支持（DDP/Megatron）

损失函数注册表（`swift/loss/mapping.py`）：
```python
loss_map = {
    'cosine_similarity': CosineSimilarityLoss,
    'contrastive': ContrastiveLoss,
    'online_contrastive': OnlineContrastiveLoss,
    'infonce': InfonceLoss,
}
```

### 2.3 训练器（Trainer）

**标准嵌入训练器**（`swift/trainers/embedding_trainer.py`）：
```python
class EmbeddingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gather_function = gather_for_unpadded_tensors
```

**Megatron 嵌入训练器**（`swift/megatron/trainers/embedding_trainer.py`）：
- 支持 padding-free 训练
- 集成损失计算与分布式 gathering
- 支持 infonce 和 paired 评估指标

### 2.4 模板系统

**已有的 Qwen3 嵌入模板**（`swift/template/templates/qwen.py`, lines 95-114）：

```python
class Qwen3EmbTemplate(Template):
    def _preprocess_inputs(self, inputs: StdTemplateInputs) -> None:
        super()._preprocess_inputs(inputs)
        if inputs.system is not None:
            inputs.messages[0]['content'] = inputs.system + ' ' + inputs.messages[0]['content']
            inputs.system = None
        if len(inputs.messages) % 2 == 1 and inputs.messages[-1]['role'] != 'assistant':
            inputs.messages.append({'role': 'assistant', 'content': ''})
        return inputs

register_template(
    TemplateMeta(
        LLMTemplateType.qwen3_emb,
        template_cls=Qwen3EmbTemplate,
        suffix=['<|endoftext|>'],
        prefix=[],
        chat_sep=[],
        prompt=['{{QUERY}}']))
```

模板特点：
- 将 system 指令与第一条消息合并
- 在末尾添加空的 assistant 消息
- 使用 `<|endoftext|>` 作为后缀（EOS token）

### 2.5 已注册的 Qwen3 嵌入模型

在 `swift/model/constant.py` 中：
```python
class LLMModelType:
    qwen3_emb = 'qwen3_emb'       # Qwen3-Embedding 系列
    qwen3_reranker = 'qwen3_reranker'  # Qwen3-Reranker 系列

class MLLMModelType:
    qwen3_vl_emb = 'qwen3_vl_emb'         # Qwen3-VL-Embedding 系列
    qwen3_vl_reranker = 'qwen3_vl_reranker' # Qwen3-VL-Reranker 系列
```

已注册的嵌入模型：
- `Qwen/Qwen3-Embedding-0.6B`
- `Qwen/Qwen3-Embedding-4B`
- `Qwen/Qwen3-Embedding-8B`
- `Qwen/Qwen3-VL-Embedding-2B`（多模态）
- `Qwen/Qwen3-VL-Embedding-8B`（多模态）

### 2.6 现有训练示例

**嵌入训练脚本**（`examples/train/embedding/qwen3/qwen3_emb.sh`）：
```bash
swift sft \
    --model Qwen/Qwen3-Embedding-4B \
    --task_type embedding \
    --loss_type infonce \
    --dataset sentence-transformers/stsb:positive \
    --padding_free true \
    --deepspeed zero2
```

**嵌入推理示例**（`examples/train/embedding/qwen3/infer.py`、`examples/infer/demo_embedding.py`）

### 2.7 分布式训练支持

- DDP（Data Distributed Parallel）
- DeepSpeed（ZeRO-2, ZeRO-3）
- FSDP/FSDP2
- Megatron 并行（TP, PP, CP, EP, SP）

### 2.8 微调方法支持

- LoRA / QLoRA / DoRA
- Adapter / LISA / ReFT
- 全参数微调
- Flash Attention 2/3

---

## 3. Qwen3.5 模型在 ms-swift 中的注册状态

### 3.1 模型注册

在 `swift/model/models/qwen.py`（lines 1161-1186）中，Qwen3.5 已注册为 **MLLM 模型类型**：

```python
register_model(
    ModelMeta(
        MLLMModelType.qwen3_5,  # 注册为多模态模型类型
        [
            ModelGroup([
                Model('Qwen/Qwen3.5-0.8B', 'Qwen/Qwen3.5-0.8B'),
                Model('Qwen/Qwen3.5-2B', 'Qwen/Qwen3.5-2B'),
                Model('Qwen/Qwen3.5-4B', 'Qwen/Qwen3.5-4B'),
                Model('Qwen/Qwen3.5-9B', 'Qwen/Qwen3.5-9B'),
                Model('Qwen/Qwen3.5-27B', 'Qwen/Qwen3.5-27B'),
                # ... base variants
            ], TemplateType.qwen3_5),
        ],
        Qwen3_5Loader,
        model_arch=ModelArch.qwen2_vl,
        architectures=['Qwen3_5ForConditionalGeneration'],
        requires=['transformers>=5.0.0.dev', 'qwen_vl_utils>=0.0.14', 'decord'],
        tags=['vision', 'video']))
```

### 3.2 模型加载器

```python
class Qwen3_5Loader(Qwen3VLLoader):
    def get_model(self, model_dir, config, processor, model_kwargs):
        from transformers import Qwen3_5ForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or Qwen3_5ForConditionalGeneration
        return Qwen2VLLoader.get_model(self, model_dir, config, processor, model_kwargs)
```

### 3.3 模型常量

```python
# swift/model/constant.py
class MLLMModelType:
    qwen3_5 = 'qwen3_5'
    qwen3_5_moe = 'qwen3_5_moe'
```

---

## 4. 差距分析：将 Qwen3.5 用于嵌入训练缺少什么

### 4.1 缺少专用嵌入模型类型

**现状**：没有 `qwen3_5_emb` 模型类型

```python
# swift/model/constant.py 中缺少：
class LLMModelType:
    # qwen3_5_emb = 'qwen3_5_emb'  ← 不存在
class MLLMModelType:
    # qwen3_5_emb = 'qwen3_5_emb'  ← 不存在
```

**影响**：无法直接使用 `--model Qwen/Qwen3.5-0.8B --task_type embedding` 命令来启动嵌入训练

### 4.2 缺少专用嵌入模板

**现状**：没有 Qwen3.5 的嵌入模板

已有的 Qwen3 嵌入模板（`qwen3_emb`）是为 Qwen3 设计的，使用 `<|endoftext|>` 作为 EOS token。Qwen3.5 的 tokenizer 可能使用不同的特殊 token。

**需要创建**：类似 `Qwen3EmbTemplate` 的 `Qwen3_5EmbTemplate`，需确认：
- EOS token 是否相同
- 输入格式是否需要调整（Qwen3.5 是多模态模型，输入处理可能不同）

### 4.3 模型架构兼容性问题

**核心问题**：Qwen3.5 使用 `Qwen3_5ForConditionalGeneration`（条件生成模型），而嵌入训练需要的是获取隐藏状态的前向传播。

现有的嵌入训练流程（`swift/template/base.py`, line 540）：
```python
elif self.task_type == 'embedding':
    encoded = self._embedding_encode(inputs)
```

嵌入训练在 forward 过程中需要获取 `last_hidden_state`：
```python
# swift/loss/embedding.py, line 135
sentences = outputs['last_hidden_state']
```

**关键问题**：`Qwen3_5ForConditionalGeneration` 的 forward 方法是否输出 `last_hidden_state`？作为一个 VL 条件生成模型，其输出格式可能与纯文本 LLM 不同。

### 4.4 LoRA Task Type 映射

在 `swift/pipelines/train/tuner.py`（lines 168-169）中：
```python
if task_type == 'EMBEDDING':
    task_type = None  # peft LoRA 不支持 embedding task type
```

这意味着使用 LoRA 微调嵌入模型时，peft 的 task_type 设为 None。这应该对 Qwen3.5 也适用，但需要验证。

### 4.5 缺少训练示例和文档

**现状**：
- `examples/train/embedding/` 只有 `qwen3/` 目录，没有 `qwen3_5/`
- 没有 Qwen3.5 嵌入训练的文档说明
- 没有 Qwen3.5 嵌入推理的示例代码

---

## 5. 已支持 vs 需要开发的功能

### 5.1 已支持（可直接复用）

| 功能 | 文件路径 | 说明 |
|------|---------|------|
| `task_type='embedding'` | `swift/arguments/base_args/model_args.py` | 嵌入任务类型声明 |
| InfoNCE 损失 | `swift/loss/embedding.py` | 完整实现，支持 QQ/DD/假负例 |
| 对比损失 / 余弦相似度损失 | `swift/loss/embedding.py` | 多种损失函数 |
| EmbeddingTrainer | `swift/trainers/embedding_trainer.py` | 嵌入训练器 |
| MegatronEmbeddingTrainer | `swift/megatron/trainers/embedding_trainer.py` | Megatron 嵌入训练器 |
| 嵌入编码逻辑 | `swift/template/base.py` (line 540) | `_embedding_encode` 方法 |
| 分布式训练 | 多个文件 | DDP/DeepSpeed/FSDP/Megatron |
| LoRA/全参数微调 | `swift/pipelines/train/tuner.py` | 微调方法 |
| Qwen3.5 模型加载 | `swift/model/models/qwen.py` | Qwen3_5Loader |
| 损失函数注册 | `swift/loss/mapping.py` | loss_map |

### 5.2 需要开发或适配

| 功能 | 优先级 | 工作量 | 说明 |
|------|--------|--------|------|
| Qwen3.5 嵌入模型类型注册 | **P0** | 小 | 在 constant.py 和 qwen.py 中注册 |
| Qwen3.5 嵌入模板 | **P0** | 小 | 类似 Qwen3EmbTemplate，确认 EOS token |
| 验证 forward 输出 | **P0** | 中 | 确认 Qwen3_5ForConditionalGeneration 能输出 last_hidden_state |
| 可能的模型包装器 | P1 | 中 | 如果 forward 输出不兼容，需要包装器提取隐藏状态 |
| 训练示例脚本 | P1 | 小 | `examples/train/embedding/qwen3_5/` |
| 推理示例代码 | P1 | 小 | 嵌入推理示例 |
| 文档更新 | P2 | 小 | 更新 BestPractices/Embedding.md |
| Megatron 嵌入支持 | P2 | 中 | `swift/megatron/model/mm_gpts/qwen3_5.py` 中适配 |

---

## 6. 实现方案建议

### 6.1 方案 A：最小改动方案（推荐先验证）

直接尝试使用现有的 Qwen3.5 模型 + `--task_type embedding`，观察是否能正常工作：

```bash
swift sft \
    --model Qwen/Qwen3.5-0.8B \
    --task_type embedding \
    --loss_type infonce \
    --dataset sentence-transformers/stsb:positive \
    --template_type qwen3_emb \
    --deepspeed zero2
```

**可能遇到的问题**：
1. 模板不匹配 — 需要确认 Qwen3.5 的 tokenizer 是否与 `qwen3_emb` 模板兼容
2. 模型 forward 输出不包含 `last_hidden_state` — 需要适配
3. 多模态组件干扰 — 可能需要冻结视觉编码器

### 6.2 方案 B：完整适配方案

#### 步骤 1：注册嵌入模型类型

```python
# swift/model/constant.py
class MLLMModelType:
    qwen3_5_emb = 'qwen3_5_emb'  # 新增
```

#### 步骤 2：创建嵌入模板

```python
# swift/template/templates/qwen.py 新增
class Qwen3_5EmbTemplate(Template):
    def _preprocess_inputs(self, inputs: StdTemplateInputs) -> None:
        super()._preprocess_inputs(inputs)
        # 复用 Qwen3EmbTemplate 的逻辑
        if inputs.system is not None:
            inputs.messages[0]['content'] = inputs.system + ' ' + inputs.messages[0]['content']
            inputs.system = None
        if len(inputs.messages) % 2 == 1 and inputs.messages[-1]['role'] != 'assistant':
            inputs.messages.append({'role': 'assistant', 'content': ''})
        return inputs

register_template(
    TemplateMeta(
        'qwen3_5_emb',  # 新模板类型
        template_cls=Qwen3_5EmbTemplate,
        suffix=['<|endoftext|>'],  # 确认 Qwen3.5 的 EOS token
        prefix=[],
        chat_sep=[],
        prompt=['{{QUERY}}']))
```

#### 步骤 3：注册模型

```python
# swift/model/models/qwen.py 新增
register_model(
    ModelMeta(
        MLLMModelType.qwen3_5_emb,
        [
            ModelGroup([
                Model('Qwen/Qwen3.5-0.8B', 'Qwen/Qwen3.5-0.8B'),
                Model('Qwen/Qwen3.5-0.8B-Base', 'Qwen/Qwen3.5-0.8B-Base'),
                # ... 其他规模
            ], TemplateType.qwen3_5_emb),
        ],
        Qwen3_5Loader,  # 或自定义 Loader
        model_arch=ModelArch.qwen2_vl,
        architectures=['Qwen3_5ForConditionalGeneration'],
        requires=['transformers>=5.0.0.dev'],
    ))
```

#### 步骤 4：验证 forward 输出

需要确认 `Qwen3_5ForConditionalGeneration.forward()` 是否返回 `last_hidden_state`。如果不返回，需要创建包装器：

```python
class Qwen3_5ForEmbedding(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model.model  # 获取内部的语言模型部分

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.model(input_ids, attention_mask=attention_mask, **kwargs)
        # 提取 EOS token 位置的隐藏状态
        last_hidden_state = outputs.last_hidden_state
        # 获取每个样本最后一个非 padding token 的位置
        eos_positions = attention_mask.sum(dim=1) - 1
        embeddings = last_hidden_state[range(len(eos_positions)), eos_positions]
        return {'last_hidden_state': embeddings}
```

#### 步骤 5：创建训练示例

```bash
# examples/train/embedding/qwen3_5/qwen3_5_emb.sh
swift sft \
    --model Qwen/Qwen3.5-0.8B \
    --task_type embedding \
    --loss_type infonce \
    --dataset sentence-transformers/stsb:positive \
    --padding_free true \
    --deepspeed zero2
```

---

## 7. 风险与注意事项

### 7.1 Transformers 版本依赖

Qwen3.5 需要 `transformers>=5.0.0.dev`，这是一个开发版本。需要确保：
- 该版本稳定可用
- 不会与其他依赖冲突
- 嵌入训练相关的 API 在该版本中正常工作

### 7.2 多模态组件处理

Qwen3.5 是原生多模态模型，包含视觉编码器。在纯文本嵌入训练场景下：
- 视觉编码器参数会占用显存但不参与训练
- 需要考虑是否冻结视觉部分
- 模型加载时可能需要额外的处理

### 7.3 混合注意力的 padding-free 训练

ms-swift 的嵌入训练支持 `--padding_free true`，但 Gated DeltaNet 层的 padding-free 实现需要验证：
- 标准注意力的 padding-free 是通过合并 batch 并使用 position_ids 区分样本
- Gated DeltaNet 使用递归状态，padding-free 实现可能不同

### 7.4 Megatron 支持

`swift/megatron/model/mm_gpts/qwen3_5.py` 中已有 Qwen3.5 的 Megatron 支持（`Qwen3_5MoeGatedDeltaNet`），但嵌入训练的 Megatron 适配可能需要额外工作。

---

## 8. 总结

### 已支持的关键能力

- 嵌入训练基础设施（task_type, loss, trainer）完整
- Qwen3.5 模型加载和基础训练支持
- InfoNCE 损失函数（含 Qwen3-Embedding 增强特性）
- 分布式训练和多种微调方法

### 需要补充的关键缺失

1. **P0**：Qwen3.5 嵌入模型类型注册和专用模板
2. **P0**：验证 `Qwen3_5ForConditionalGeneration` 的 forward 输出是否包含 `last_hidden_state`
3. **P1**：训练/推理示例脚本
4. **P1**：如需要，创建模型包装器提取嵌入
5. **P2**：文档和 Megatron 适配

### 预计工作量

- **方案 A（最小改动）**：约 1-2 天的验证和调试
- **方案 B（完整适配）**：约 3-5 天的开发和测试

### 建议路径

```
1. 先用方案 A 快速验证可行性
   ├── 成功 → 记录配置，补充文档和示例
   └── 失败 → 分析问题原因

2. 根据方案 A 的结果决定方案 B 的具体范围
   ├── 模板不兼容 → 创建新模板
   ├── forward 输出不兼容 → 创建模型包装器
   └── 多模态组件干扰 → 冻结策略

3. 完善文档和示例
```

---

## 参考文件路径

| 文件 | 路径 |
|------|------|
| 模型常量定义 | `swift/model/constant.py` |
| Qwen 模型注册 | `swift/model/models/qwen.py` |
| 嵌入损失函数 | `swift/loss/embedding.py` |
| 损失函数映射 | `swift/loss/mapping.py` |
| 嵌入训练器 | `swift/trainers/embedding_trainer.py` |
| Megatron 嵌入训练器 | `swift/megatron/trainers/embedding_trainer.py` |
| 模板基类 | `swift/template/base.py` |
| Qwen 模板 | `swift/template/templates/qwen.py` |
| 基础参数定义 | `swift/arguments/base_args/model_args.py` |
| 微调器 | `swift/pipelines/train/tuner.py` |
| Qwen3 嵌入训练示例 | `examples/train/embedding/qwen3/qwen3_emb.sh` |
| Qwen3 嵌入推理示例 | `examples/train/embedding/qwen3/infer.py` |
| 嵌入推理 Demo | `examples/infer/demo_embedding.py` |
| 嵌入训练测试 | `tests/train/test_embedding.py` |
| Megatron 嵌入测试 | `tests/megatron/test_embedding.py` |
| Megatron Qwen3.5 模型 | `swift/megatron/model/mm_gpts/qwen3_5.py` |
| 嵌入文档 | `docs/source/BestPractices/Embedding.md` |
