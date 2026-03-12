# Qwen3.5 嵌入训练支持实现报告

## 1. 变更概述

本次更新在 ms-swift 框架的 `qwen3point5-embedding` 分支中添加了 Qwen3.5 模型的嵌入训练支持，使用户能够通过 `--model_type qwen3_5_emb --task_type embedding` 将 Qwen3.5 系列模型（0.8B-9B Dense）训练为文本嵌入模型。

**核心方法**：EOS Token 池化 + InfoNCE 对比学习，与 Qwen3-Embedding 技术路线一致。

---

## 2. 修改的文件

### 2.1 `swift/trainers/mixin.py` — 修复 transformers v5.x 兼容性

**问题**：`_disable_group_by_length` 上下文管理器在 transformers v5.x 上崩溃，因为 `TrainingArguments` 不再包含 `group_by_length` 属性。

**错误信息**：
```
AttributeError: 'TrainingArguments' object has no attribute 'group_by_length'
```

**修复方式**：在 `_disable_group_by_length` 方法中添加 `hasattr` 防御性检查：

```python
# 修改前
@contextmanager
def _disable_group_by_length(self):
    group_by_length = getattr(self.args, 'group_by_length', False)
    self.args.group_by_length = False  # ← 这里直接设置会报错
    try:
        yield
    finally:
        self.args.group_by_length = group_by_length

# 修改后
@contextmanager
def _disable_group_by_length(self):
    group_by_length = getattr(self.args, 'group_by_length', False)
    if hasattr(self.args, 'group_by_length'):
        self.args.group_by_length = False
    try:
        yield
    finally:
        if hasattr(self.args, 'group_by_length'):
            self.args.group_by_length = group_by_length
```

**影响范围**：仅在 `get_eval_dataloader` 方法中使用此上下文管理器，确保评估数据加载器不使用按长度分组策略。修复后向后兼容，transformers v4.x 上行为不变。

---

### 2.2 `swift/model/constant.py` — 添加模型类型常量

在 `MLLMModelType` 类中新增 `qwen3_5_emb` 常量：

```python
class MLLMModelType:
    ...
    qwen3_5 = 'qwen3_5'
    qwen3_5_moe = 'qwen3_5_moe'
    qwen3_5_emb = 'qwen3_5_emb'  # 新增
    ...
```

**设计决策**：使用 `MLLMModelType`（多模态模型类型）而非 `LLMModelType`，因为 Qwen3.5 是原生多模态模型，不存在纯文本变体。这与 `qwen3_vl_emb` 的设计模式一致。

---

### 2.3 `swift/template/constant.py` — 添加模板类型常量

在 `MLLMTemplateType` 类中新增 `qwen3_5_emb` 常量：

```python
class MLLMTemplateType:
    ...
    qwen3_5 = 'qwen3_5'
    qwen3_5_emb = 'qwen3_5_emb'  # 新增
    ...
```

---

### 2.4 `swift/template/templates/qwen.py` — 创建嵌入模板

新增 `Qwen3_5EmbTemplate` 类及其注册，位于 `Qwen3_5Template` 注册之后：

```python
class Qwen3_5EmbTemplate(Qwen3_5Template):
    def _preprocess_inputs(self, inputs: StdTemplateInputs) -> None:
        super()._preprocess_inputs(inputs)
        if len(inputs.messages) % 2 == 1 and inputs.messages[-1]['role'] != 'assistant':
            inputs.messages.append({'role': 'assistant', 'content': ''})

register_template(
    QwenTemplateMeta(
        MLLMTemplateType.qwen3_5_emb,
        default_system=None,
        suffix=['<|endoftext|>'],
        template_cls=Qwen3_5EmbTemplate,
    ))
```

**设计说明**：
- 继承 `Qwen3_5Template`，保留 Qwen3.5 的 token ID 定义（`image_token_id=248056` 等）和编码逻辑
- 在预处理时确保消息数量为偶数（添加空的 assistant 消息），这是嵌入编码所需的格式
- 使用 `<|endoftext|>` 作为后缀（EOS token），与 Qwen3-Embedding 一致
- `default_system=None`：不设默认系统提示，用户可通过数据中的 system 字段自定义指令

---

### 2.5 `swift/model/models/qwen.py` — 注册嵌入模型

新增 `Qwen3_5EmbLoader` 类和模型注册，位于 `qwen3_5` 模型注册之后：

```python
class Qwen3_5EmbLoader(Qwen3_5Loader):
    pass

register_model(
    ModelMeta(
        MLLMModelType.qwen3_5_emb,
        [ModelGroup([
            Model('Qwen/Qwen3.5-0.8B', ...),
            Model('Qwen/Qwen3.5-0.8B-Base', ...),
            Model('Qwen/Qwen3.5-2B', ...),
            # ... 包含 0.8B 到 9B 的 Dense 模型及其 Base 变体
        ], TemplateType.qwen3_5_emb)],
        Qwen3_5EmbLoader,
        model_arch=ModelArch.qwen2_vl,
        architectures=['Qwen3_5ForConditionalGeneration'],
        requires=['transformers>=5.0.0.dev', 'qwen_vl_utils>=0.0.14', 'decord'],
        tags=['vision', 'video']))
```

**设计说明**：
- `Qwen3_5EmbLoader` 继承 `Qwen3_5Loader`，完全复用加载逻辑
- 架构使用 `Qwen3_5ForConditionalGeneration`，与标准 Qwen3.5 一致
- 仅注册 Dense 模型（0.8B-9B），不包含 MoE 变体
- 同一个物理模型（如 `Qwen/Qwen3.5-0.8B`）同时注册在 `qwen3_5` 和 `qwen3_5_emb` 模型类型下。用户通过 `--model_type` 参数选择使用生成模式还是嵌入模式

---

## 3. 新增的文件

### 3.1 `examples/train/embedding/qwen3_5/qwen3_5_emb.sh`

Qwen3.5 嵌入训练示例脚本，包含：
- 阶段一（弱监督预训练）的完整命令
- 阶段二（有监督微调）的注释模板
- 关键超参数说明

### 3.2 `examples/train/embedding/qwen3_5/infer.py`

Qwen3.5 嵌入推理示例代码，支持：
- 全参数训练后的推理
- LoRA 微调后的推理
- 余弦相似度计算

---

## 4. 未修改的关键组件（已有支持）

以下组件已在 ms-swift 中实现，无需修改即可用于 Qwen3.5 嵌入训练：

| 组件 | 文件路径 | 说明 |
|------|---------|------|
| InfoNCE 损失 | `swift/loss/embedding.py` | 完整实现，含 QQ/DD/假负例增强 |
| EmbeddingTrainer | `swift/trainers/embedding_trainer.py` | 嵌入训练器，支持分布式 |
| 嵌入编码 | `swift/template/base.py` `_embedding_encode()` | anchor/positive/negative 编码 |
| 数据整理 | `swift/template/base.py` `_embedding_data_collator()` | batch 构建 |
| 训练器选择 | `swift/trainers/trainer_factory.py` | task_type='embedding' → EmbeddingTrainer |
| LoRA 适配 | `swift/pipelines/train/tuner.py` | embedding task_type 的 LoRA 映射 |

---

## 5. 技术原理

### 为什么这个方案可行？

1. **模型输出兼容**：`Qwen3_5ForConditionalGeneration.forward()` 返回包含 `last_hidden_state` 的输出，InfoNCE 损失函数通过 `outputs['last_hidden_state']` 获取嵌入向量

2. **EOS Token 池化与混合注意力兼容**：
   - Gated DeltaNet 层的递归状态 `S_t` 在处理到 EOS token 时已累积全部序列信息
   - Gated Attention 层通过因果注意力使 EOS token 可以关注所有前面的 token
   - 因此 EOS 位置的隐藏状态天然包含完整的序列语义

3. **设计模式已验证**：`qwen3_vl_emb`（Qwen3-VL-Embedding）已证明 `ConditionalGeneration` 架构的模型可以成功用于嵌入训练

---

## 6. 注意事项

1. **模型为多模态架构**：Qwen3.5 包含视觉编码器。在纯文本嵌入训练中，视觉编码器不会被使用但仍占用显存。建议在显存紧张时冻结视觉编码器参数。

2. **transformers 版本要求**：需要 `transformers>=5.0.0.dev`（Qwen3.5 模型的 HuggingFace 支持所需）。

3. **`--dataloader_drop_last true`**：嵌入训练必须设置此参数，否则评估时的 gather 操作可能因不同 rank 的 batch 大小不一致而报错。

4. **`--attn_impl sdpa`**：Qwen3.5 的混合注意力架构可能不完全兼容 `flash_attn`，建议使用 `sdpa`（Scaled Dot Product Attention）。
