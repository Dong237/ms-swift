# Qwen3.5-0.8B 嵌入模型训练完整指南

本指南详细说明如何使用更新后的 ms-swift 框架将 Qwen3.5-0.8B 训练为文本嵌入模型。

---

## 1. 环境安装

### 1.1 基础环境要求

- Python >= 3.9
- CUDA >= 11.8
- PyTorch >= 2.0

### 1.2 安装 ms-swift

```bash
# 克隆仓库并切换到嵌入训练分支
git clone https://github.com/your-repo/ms-swift.git
cd ms-swift
git checkout qwen3point5-embedding

# 安装依赖
pip install -e .

# 安装 Qwen3.5 所需的额外依赖
pip install transformers>=5.0.0.dev
pip install qwen_vl_utils>=0.0.14
pip install decord

# 安装 DeepSpeed（推荐用于分布式训练）
pip install deepspeed
```

### 1.3 验证安装

```bash
python -c "
from swift.model.constant import MLLMModelType
print('qwen3_5_emb model type:', MLLMModelType.qwen3_5_emb)

from swift.template.constant import MLLMTemplateType
print('qwen3_5_emb template type:', MLLMTemplateType.qwen3_5_emb)

print('Installation verified successfully!')
"
```

---

## 2. 数据准备

ms-swift 支持多种数据格式用于嵌入训练。

### 2.1 格式一：Messages 格式（推荐）

这是 ms-swift 的原生格式，使用 JSONL 文件，每行一个 JSON 对象：

```jsonl
{"messages": [{"role": "user", "content": "查询文本"}], "positive_messages": [[{"role": "user", "content": "相关文档1"}]], "negative_messages": [[{"role": "user", "content": "不相关文档1"}], [{"role": "user", "content": "不相关文档2"}]]}
{"messages": [{"role": "user", "content": "另一个查询"}], "positive_messages": [[{"role": "user", "content": "相关文档"}]]}
```

**字段说明**：
| 字段 | 必须 | 说明 |
|------|------|------|
| `messages` | 是 | 锚点/查询文本（anchor） |
| `positive_messages` | 是 | 正例列表（与查询相关的文档） |
| `negative_messages` | 否 | 负例列表（与查询不相关的文档，即硬负例） |

**示例：弱监督预训练数据**（无硬负例，使用 batch 内负例）：
```jsonl
{"messages": [{"role": "user", "content": "什么是机器学习？"}], "positive_messages": [[{"role": "user", "content": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习而无需被明确编程。"}]]}
{"messages": [{"role": "user", "content": "北京的天气如何？"}], "positive_messages": [[{"role": "user", "content": "北京属于温带季风气候，四季分明，夏季高温多雨，冬季寒冷干燥。"}]]}
```

**示例：有监督微调数据**（带硬负例）：
```jsonl
{"messages": [{"role": "user", "content": "什么是深度学习？"}], "positive_messages": [[{"role": "user", "content": "深度学习是机器学习的子集，使用多层神经网络来学习数据中的复杂模式。"}]], "negative_messages": [[{"role": "user", "content": "浅层学习方法通常使用简单的线性模型。"}], [{"role": "user", "content": "强化学习是通过试错来学习最优策略的方法。"}]]}
```

### 2.2 格式二：简单 Query-Document 格式

更简洁的 JSONL 格式，适合快速实验：

```jsonl
{"query": "查询文本", "positive": "相关文档", "negative": "不相关文档"}
{"query": "另一个查询", "positive": "相关文档"}
```

**注意**：需要配合对应的数据预处理器使用，请参考 `docs/source_en/BestPractices/Embedding.md`。

### 2.3 格式三：使用 HuggingFace 数据集

ms-swift 内置支持多种 HuggingFace 数据集：

```bash
# STS-B（语义文本相似度）
--dataset sentence-transformers/stsb:positive

# 更多数据集请参考 ms-swift 文档
```

### 2.4 带指令的嵌入数据

如果需要指令感知嵌入（提高特定任务性能），在数据中添加 system 字段：

```jsonl
{"messages": [{"role": "system", "content": "Given a web search query, retrieve relevant passages"}, {"role": "user", "content": "什么是机器学习？"}], "positive_messages": [[{"role": "user", "content": "机器学习是..."}]]}
```

系统提示会自动与查询合并为嵌入输入。

---

## 3. 训练

### 3.1 阶段一：弱监督对比预训练

**目标**：使用大规模数据（可以是合成数据）进行初步的嵌入表示学习。

```bash
# 环境变量配置
export CUDA_VISIBLE_DEVICES=0,1
export INFONCE_TEMPERATURE=0.1    # InfoNCE 温度参数
export NPROC_PER_NODE=2           # GPU 数量

# 训练命令
swift sft \
    --model Qwen/Qwen3.5-0.8B \
    --model_type qwen3_5_emb \
    --task_type embedding \
    --loss_type infonce \
    --tuner_type full \
    --learning_rate 1e-4 \
    --dataset /path/to/your/pretrain_data.jsonl \
    --attn_impl sdpa \
    --torch_dtype bfloat16 \
    --load_from_cache_file true \
    --split_dataset_ratio 0.02 \
    --eval_strategy steps \
    --output_dir output/qwen3_5_emb_phase1 \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --num_train_epochs 3 \
    --max_length 4096 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --warmup_ratio 0.05 \
    --dataloader_drop_last true \
    --deepspeed zero2
```

**关键参数说明**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `--model` | 模型路径 | HuggingFace 模型 ID 或本地路径 |
| `--model_type` | `qwen3_5_emb` | 指定使用 Qwen3.5 嵌入模型类型 |
| `--task_type` | `embedding` | 嵌入训练任务 |
| `--loss_type` | `infonce` | InfoNCE 对比损失 |
| `--tuner_type` | `full` | 全参数微调（也可用 `lora`） |
| `--learning_rate` | `1e-4` | 阶段一使用较大学习率 |
| `--attn_impl` | `sdpa` | Qwen3.5 混合注意力建议使用 sdpa |
| `--dataloader_drop_last` | `true` | 必须设置，防止分布式评估报错 |
| `--deepspeed` | `zero2` | 推荐使用 DeepSpeed ZeRO-2 |

**InfoNCE 环境变量**：

| 环境变量 | 默认值 | 说明 |
|---------|-------|------|
| `INFONCE_TEMPERATURE` | `0.1` | 温度参数，越小越严格 |
| `INFONCE_USE_BATCH` | `True` | 使用 batch 内负例 |
| `INFONCE_HARD_NEGATIVES` | `None` | 每个样本的硬负例数量 |
| `INFONCE_INCLUDE_QQ` | `False` | 阶段一建议 False |
| `INFONCE_INCLUDE_DD` | `False` | 阶段一建议 False |
| `INFONCE_MASK_FAKE_NEGATIVE` | `False` | 阶段一建议 False |

**使用 LoRA 微调**（显存不足时）：

```bash
swift sft \
    --model Qwen/Qwen3.5-0.8B \
    --model_type qwen3_5_emb \
    --task_type embedding \
    --loss_type infonce \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --learning_rate 5e-5 \
    ... # 其他参数同上
```

### 3.2 阶段二：有监督微调

**目标**：使用高质量标注数据进行精细化训练，启用高级 InfoNCE 增强。

```bash
export CUDA_VISIBLE_DEVICES=0,1
export INFONCE_TEMPERATURE=0.1
export INFONCE_INCLUDE_QQ=True          # 启用 QQ 相似度增强
export INFONCE_INCLUDE_DD=True          # 启用 DD 相似度增强
export INFONCE_MASK_FAKE_NEGATIVE=True  # 启用假负例掩码
export NPROC_PER_NODE=2

swift sft \
    --model output/qwen3_5_emb_phase1/checkpoint-best \
    --model_type qwen3_5_emb \
    --task_type embedding \
    --loss_type infonce \
    --tuner_type full \
    --learning_rate 1e-5 \
    --dataset /path/to/your/supervised_data.jsonl \
    --attn_impl sdpa \
    --torch_dtype bfloat16 \
    --load_from_cache_file true \
    --split_dataset_ratio 0.02 \
    --eval_strategy steps \
    --output_dir output/qwen3_5_emb_phase2 \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --num_train_epochs 2 \
    --max_length 4096 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --warmup_ratio 0.05 \
    --dataloader_drop_last true \
    --deepspeed zero2
```

**与阶段一的关键差异**：

| 参数 | 阶段一 | 阶段二 |
|------|--------|--------|
| `--model` | 原始 Qwen3.5-0.8B | 阶段一的最佳检查点 |
| `--learning_rate` | `1e-4` | `1e-5`（更小） |
| `--num_train_epochs` | 3 | 2（更少） |
| `INFONCE_INCLUDE_QQ` | False | **True** |
| `INFONCE_INCLUDE_DD` | False | **True** |
| `INFONCE_MASK_FAKE_NEGATIVE` | False | **True** |
| 数据 | 大规模弱监督数据 | 小规模高质量标注数据 |

---

## 4. 推理

### 4.1 使用训练好的模型进行推理

```python
import torch
from swift.infer_engine import InferRequest, TransformersEngine

# 加载全参数训练的模型
engine = TransformersEngine(
    'output/qwen3_5_emb_phase2/checkpoint-best',  # 阶段二最佳检查点
    model_type='qwen3_5_emb',
    task_type='embedding',
    torch_dtype=torch.bfloat16)

# 构建推理请求
infer_requests = [
    InferRequest(messages=[{'role': 'user', 'content': '什么是机器学习？'}]),
    InferRequest(messages=[{'role': 'user', 'content': '机器学习是一种让计算机从数据中学习的方法。'}]),
    InferRequest(messages=[{'role': 'user', 'content': '今天北京天气晴朗。'}]),
]

# 获取嵌入向量
resp_list = engine.infer(infer_requests)
embeddings = [torch.tensor(resp.data[0].embedding) for resp in resp_list]
embedding_matrix = torch.stack(embeddings)

# 计算余弦相似度
norms = embedding_matrix.norm(dim=1, keepdim=True)
similarity = (embedding_matrix @ embedding_matrix.T) / (norms @ norms.T)
print(f'余弦相似度矩阵:\n{similarity}')

# 预期输出：查询0与文档1相似度高，与文档2相似度低
```

### 4.2 使用 LoRA 适配器推理

```python
engine = TransformersEngine(
    'Qwen/Qwen3.5-0.8B',
    model_type='qwen3_5_emb',
    task_type='embedding',
    torch_dtype=torch.bfloat16,
    adapters=['output/qwen3_5_emb_phase2/checkpoint-best'])  # LoRA 权重路径

# 后续推理代码与全参数训练相同
```

### 4.3 批量编码文本

```python
import torch
from swift.infer_engine import InferRequest, TransformersEngine

engine = TransformersEngine(
    'output/qwen3_5_emb_phase2/checkpoint-best',
    model_type='qwen3_5_emb',
    task_type='embedding',
    torch_dtype=torch.bfloat16)

# 批量编码文档
documents = [
    "机器学习是人工智能的一个分支...",
    "深度学习使用多层神经网络...",
    "自然语言处理是AI的重要方向...",
    # ... 更多文档
]

infer_requests = [
    InferRequest(messages=[{'role': 'user', 'content': doc}])
    for doc in documents
]

resp_list = engine.infer(infer_requests)
doc_embeddings = torch.stack([
    torch.tensor(resp.data[0].embedding) for resp in resp_list
])

# 编码查询
query = "什么是深度学习？"
query_resp = engine.infer([
    InferRequest(messages=[{'role': 'user', 'content': query}])
])
query_embedding = torch.tensor(query_resp[0].data[0].embedding)

# 计算查询与所有文档的相似度
scores = torch.cosine_similarity(
    query_embedding.unsqueeze(0), doc_embeddings, dim=1
)
print(f'相似度分数: {scores}')
print(f'最相关文档索引: {scores.argmax().item()}')
```

---

## 5. 训练技巧

### 5.1 Batch Size 建议

嵌入训练使用 batch 内负例，因此 **batch size 越大越好**。建议：
- 单卡显存充足：`per_device_train_batch_size=8` 或更大
- 多卡训练：使用 `gradient_accumulation_steps` 增大有效 batch size
- 使用 DeepSpeed ZeRO-2 节省显存

### 5.2 序列长度

- 嵌入训练通常不需要太长的上下文
- 建议 `max_length=512-2048` 对于大多数任务已足够
- 更长的 `max_length` 会降低训练速度但可能提升长文本嵌入质量

### 5.3 温度参数

- `INFONCE_TEMPERATURE=0.1` 是常用默认值
- 较小的温度（如 0.05）使分布更尖锐，正负例区分更严格
- 较大的温度（如 0.2）使分布更平滑，训练更稳定

### 5.4 数据质量

- 阶段一数据量 > 质量，可使用大模型（如 Qwen3-32B）合成
- 阶段二质量 > 数据量，使用精心标注的高质量数据
- 硬负例的质量对最终效果影响很大

---

## 6. 常见问题

### Q1: 报错 `AttributeError: 'TrainingArguments' object has no attribute 'group_by_length'`

**原因**：transformers v5.x 移除了 `group_by_length` 属性。
**解决**：确保使用 `qwen3point5-embedding` 分支，该分支已修复此问题。

### Q2: 显存不足

**解决方案**：
- 使用 LoRA 微调：`--tuner_type lora --lora_rank 8`
- 降低 batch size：`--per_device_train_batch_size 1`
- 使用 DeepSpeed ZeRO-3：`--deepspeed zero3`
- 降低 max_length：`--max_length 512`

### Q3: 训练 loss 不下降

**排查方向**：
- 检查数据格式是否正确（正例是否确实与查询相关）
- 降低学习率
- 增大 batch size（更多 batch 内负例有助于对比学习）
- 检查温度参数是否合理

### Q4: 能否在 Qwen3.5-0.8B 上同时训练文本和图像嵌入？

**理论上可以**，但需要额外工作：
- 当前模板设计仅处理文本输入
- 如需多模态嵌入，需要参考 `qwen3_vl_emb` 的模板设计
- 建议先在纯文本上验证效果

---

## 参考资料

- ms-swift 嵌入训练文档：`docs/source_en/BestPractices/Embedding.md`
- Qwen3-Embedding 论文：https://arxiv.org/abs/2506.05176
- Qwen3.5 模型：https://huggingface.co/Qwen/Qwen3.5-0.8B
- 训练示例脚本：`examples/train/embedding/qwen3_5/qwen3_5_emb.sh`
- 推理示例代码：`examples/train/embedding/qwen3_5/infer.py`
