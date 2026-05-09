# IP 图片 Embedding 模型：训练数据构建与评估方案

## 目录

- [1. 任务背景](#1-任务背景)
- [2. 训练数据构建](#2-训练数据构建)
- [3. 评估方案](#3-评估方案)
- [4. 使用方式](#4-使用方式)

---

## 1. 任务背景

**目标**：训练一个图片 Embedding 模型，使同一 IP 角色的不同图片在向量空间中距离更近，不同 IP 的图片距离更远。

**生产链路**：
1. 所有 IP 图片预先 embedding，存入向量库
2. 新图片到达 → embedding → 检索向量库中最相似的 top-K 张图片
3. 将 top-K 候选图片交给 VLM（视觉语言模型）判断是否为同一 IP

**数据结构**：每个 IP 对应一个文件夹，文件夹内有该 IP 的多张图片（约 1-10 张）。

```
IP_image/                    # 训练集
├── ip_001/ (img1.jpg, img2.jpg, ..., img5.jpg)
├── ip_002/ (img1.jpg, img2.jpg, img3.jpg)
└── ...     (约 22,000 个文件夹)

IP_image_val/                # 测试集
├── ip_001/ (img1.jpg, ..., img4.jpg)
└── ...     (约 1,000 个文件夹)
```

---

## 2. 训练数据构建

> 脚本：`build_ip_training_data.py`

### 2.1 核心思路

使用 **InfoNCE 对比学习**：每个训练样本由三部分组成：

| 组成部分 | 来源 | 作用 |
|---------|------|------|
| **Anchor（锚点）** | 某个 IP 文件夹中的一张图 | 查询图片 |
| **Positive（正样本）** | 同一 IP 文件夹中的另一张图 | 模型应判断为"相似" |
| **Negatives（负样本）** | 其他 IP 文件夹中的图片，每个样本 N 张 | 模型应判断为"不相似" |

### 2.2 正样本对的生成

对每个 IP 文件夹，枚举所有可能的图片配对 C(n, 2)：

| 文件夹图片数 | C(n,2) 配对数 | 说明 |
|------------|-------------|------|
| 1 张 | 0（跳过） | 无法构成正样本对 |
| 2 张 | 1 | |
| 5 张 | 10 | |
| 10 张 | 45 → 上限截断为 25 | `--max_pairs_per_ip` 控制 |

每个配对中，两张图互为 anchor 和 positive（实际只取一个方向）。

### 2.3 负样本的选取

对每个 (anchor, positive) 样本：
- 从**其他 IP 文件夹**中随机选取 N 个不同的 IP
- 每个 IP 随机取 1 张图作为负样本
- N 由 `--num_negatives` 控制（默认 20）

**为什么使用显式负样本而非 batch 内负样本？**

batch 内负样本（`INFONCE_USE_BATCH=True`）有一个风险：同一 batch 中如果出现同一 IP 的不同图片，它们会被错误地当作负样本，导致模型学到"把同 IP 图片推远"的错误信号（即 **假负样本问题**）。

使用显式负样本（`INFONCE_USE_BATCH=False`）可以完全避免这个问题——每个样本的负样本是我们明确指定的、确定来自不同 IP 的图片。

### 2.4 输出格式

每行一个 JSONL 样本，符合 ms-swift embedding 训练格式：

```json
{
  "messages": [{"role": "user", "content": "<image>"}],
  "images": ["IP_image/ip_001/img1.jpg"],
  "positive_messages": [[{"role": "user", "content": "<image>"}]],
  "positive_images": [["IP_image/ip_001/img3.jpg"]],
  "negative_messages": [
    [{"role": "user", "content": "<image>"}],
    [{"role": "user", "content": "<image>"}]
  ],
  "negative_images": [
    ["IP_image/ip_042/img2.jpg"],
    ["IP_image/ip_099/img1.jpg"]
  ]
}
```

**注意**：系统指令（system prompt）不写入数据，而是通过训练命令的 `--system` 参数统一注入，确保训练和推理时使用完全相同的指令。

### 2.5 数据量估算

以实际数据为例（22,613 个 IP，平均 5 张/IP）：

| 参数 | 值 |
|-----|-----|
| 有效 IP 数（≥2 张图） | ~20,331 |
| 每 IP 最多正样本对 | 25 |
| 每样本显式负样本数 | 20 |
| 总训练样本数 | ~300,000 |
| InfoNCE 每步 forward 次数 | batch_size × (1 anchor + 1 positive + 20 negatives) = batch_size × 22 |

---

## 3. 评估方案

> 脚本：`eval_ip_retrieval.py`

### 3.1 评估协议：Leave-one-out

**不是**随机挑 1 张做 query，而是**每张图片都轮流做一次 query**。

具体步骤：

1. 将测试集所有图片放入同一个池子（约 4,900 张）
2. 计算所有图片之间的余弦相似度矩阵（N × N）
3. 对角线设为 -∞（不能检索到自己）
4. 对每张图片（来自 ≥2 张图的 IP）：
   - 该图片作为 query
   - 所有其他图片作为 gallery
   - 该 query 的相关图片数 n_rel = 同 IP 图片数 - 1
   - 按相似度排序 gallery，计算各项指标

```
示例：IP_042 有 5 张图（img1-img5）

当 img1 做 query 时：
  n_rel = 4（img2, img3, img4, img5 是正确答案）
  gallery = 所有其他 ~4,899 张图
  看 top-K 中有多少是 img2/img3/img4/img5

当 img2 做 query 时：
  n_rel = 4（img1, img3, img4, img5 是正确答案）
  ...依此类推，5 张图各做一次 query
```

**为什么用 leave-one-out 而不是每个 IP 只选 1 个 query？**

| | 每 IP 1 个 query | Leave-one-out |
|--|-----------------|---------------|
| query 数 | ~884 | ~3,900 |
| 是否确定性 | 否（依赖随机种子） | 是（每次结果相同） |
| 统计误差 | 较大（±0.02） | 较小（±0.01） |
| 测试覆盖 | 只测了部分图片 | 每张图片都测到 |

### 3.2 评估指标

我们计算 4 类指标，各有侧重：

#### CMC@K（Cumulative Matching Characteristic）

> **top-K 中是否至少找到 1 个正确结果？**（二元判断：是/否）

```
CMC@K = 所有 query 中「top-K 包含至少 1 个同 IP 图片」的比例
```

**为什么重要**：这是最贴近生产链路的指标。生产中检索 top-K 给 VLM，VLM 只需要看到**至少 1 张**正确图片就能确认身份。如果 CMC@K 很低，说明 VLM 根本没机会看到正确候选。

**举例**：
- CMC@10 = 0.95 → 95% 的查询在 top-10 中至少有 1 个正确结果
- CMC@1 = 0.85 → 85% 的查询最近邻就是正确的

#### Recall@K（召回率）

> **top-K 中找到了多少比例的正确结果？**（覆盖度）

```
Recall@K = top-K 中正确结果数 / 该 query 的总正确结果数(n_rel)
```

**与 CMC 的区别**：CMC 只关心"有没有找到"，Recall 关心"找到了多少"。

**举例**（某 IP 有 5 张图，query 的 n_rel = 4）：
- top-10 中找到 3 个正确 → Recall@10 = 3/4 = 0.75，CMC@10 = 1.0
- top-10 中找到 0 个正确 → Recall@10 = 0，CMC@10 = 0

**注意**：Recall@1 在 n_rel > 1 时上限不是 1.0（例如 n_rel=4 时 Recall@1 最多 = 1/4 = 0.25），所以小 K 值下 CMC 比 Recall 更有意义。

#### R-Precision（自适应精确率）

> **在 top-K_q 结果中，有多少是正确的？K_q = 该 query 的实际相关数**

```
R-Precision = top-n_rel 中正确结果数 / n_rel
```

**为什么用 R-Precision 而不是固定 K 的 Precision@K？**

固定 K 的 Precision@K 会因 IP 图片数不同而产生误导。例如：
- IP 只有 2 张图（n_rel=1），Precision@10 最高只能是 0.1
- IP 有 10 张图（n_rel=9），Precision@10 最高可以是 0.9

R-Precision 用每个 query 自己的 n_rel 作为截断点，公平对待不同大小的 IP。

#### mAP@K（平均精度均值）

> **正确结果在排序中的位置有多好？**（排序质量）

```
AP@K = (1/min(n_rel, K)) × Σ_{j=1}^{K} Precision@j × rel(j)
mAP@K = 所有 query 的 AP@K 的平均
```

**直觉解释**：如果正确结果排在第 1、2、3 位，AP 接近 1.0；如果排在第 8、9、10 位，AP 很低。mAP 奖励"正确结果排得越靠前越好"。

### 3.3 指标选择总结

| 指标 | 回答的问题 | 生产意义 |
|-----|-----------|---------|
| **CMC@K** | 能不能找到？ | VLM 能否看到至少 1 个正确候选 |
| **Recall@K** | 找到了多少？ | 检索的覆盖度 |
| **R-Precision** | 检索精度如何？ | 自适应的精确率 |
| **mAP@K** | 排序好不好？ | 正确结果是否排在前面 |

### 3.4 输出示例

```
============================================================
IP EMBEDDING RETRIEVAL EVALUATION
============================================================
  Protocol:          Leave-one-out
  Queries:           3918
  Total images:      4934

  ── CMC (hit rate: at least 1 correct in top K) ──
  CMC@1              0.8523
  CMC@3              0.9412
  CMC@5              0.9687
  CMC@10             0.9891  <-- production cutoff

  ── Recall (coverage) ──
  Recall@1           0.2341
  Recall@3           0.5123
  Recall@5           0.7234
  Recall@10          0.9234  <-- primary

  ── R-Precision (Precision@K_q) ──
  R-Precision:       0.8156

  ── Ranking quality ──
  mAP@10             0.7891  <-- primary

  ── Recall@10 by IP size ──
    n_rel= 1: Recall@10=0.9812  (n=164 queries)
    n_rel= 2: Recall@10=0.9534  (n=168 queries)
    ...

  ── Failures: 42/3918 queries with Recall@10 < 1.0 ──
  Top 15 worst:
    ...
```

### 3.5 失败分析

脚本自动输出 Recall@K < 1.0 的最差 query 列表，包括：
- 该 query 所属的 IP
- n_rel（有多少正确答案）
- 各项指标值
- top-5 检索到的 IP 及相似度

通过失败分析可以定位：
- 哪些 IP 的图片难以区分（视觉相似的不同 IP）
- 哪些特定图片是离群点（同 IP 中差异太大的图片）
- 是否需要补充训练数据或挖掘困难负样本

---

## 4. 使用方式

### 4.1 构建训练数据

```bash
python build_ip_training_data.py \
    --input_dir /mnt/bn/youxiang-lf/data/facial_ip/IP_image \
    --output /mnt/bn/youxiang-lf/data/facial_ip/train_test_data/open_training_data.jsonl \
    --num_negatives 20 \
    --max_pairs_per_ip 25 \
    --num_workers 64
```

### 4.2 训练模型

```bash
# Qwen3.5-0.8B
bash train_ip_embedding_0.8B.sh

# Qwen3.5-9B
bash train_ip_embedding_9B.sh

# Qwen3-VL-Embedding-8B
bash train_ip_embedding_qwen3vl8B.sh
```

关键训练参数：
- `--system "提取该IP角色的身份特征，关注角色本身而非背景或姿态"` — 系统指令
- `INFONCE_USE_BATCH=False` — 只使用显式负样本
- `INFONCE_TEMPERATURE=0.02` — 温度参数

### 4.3 评估模型

评估脚本同时支持 **NVIDIA GPU** 和**华为昇腾 910B NPU**，自动检测可用硬件，无需额外配置。

```bash
# 完整流程：embedding + 评估（GPU 或 NPU 均可）
python eval_ip_retrieval.py \
    --model /path/to/checkpoint \
    --model_type qwen3_5_emb \
    --instruction "提取该IP角色的身份特征，关注角色本身而非背景或姿态" \
    --test_dir /mnt/bn/youxiang-lf/data/facial_ip/IP_image_val \
    --output /path/to/eval_embeddings.jsonl \
    --K 10

# 从已有 embedding 重新评估（跳过推理，秒级完成，纯 CPU）
python eval_ip_retrieval.py \
    --embeddings /path/to/eval_embeddings.jsonl \
    --K 5
```

**硬件兼容说明**：
- 脚本通过 ms-swift 的 `get_device_count()` 自动检测 GPU/NPU
- 多卡并行时同时设置 `CUDA_VISIBLE_DEVICES` 和 `ASCEND_RT_VISIBLE_DEVICES`，各运行时只读取自己的环境变量
- NPU 环境需预装 `torch_npu` 和 CANN 工具包
- 指标计算阶段为纯 numpy 运算，不依赖任何加速硬件

### 4.4 文件清单

| 文件 | 说明 |
|-----|------|
| `build_ip_training_data.py` | 训练数据构建 |
| `count_ip_distribution.py` | 数据分布统计 |
| `eval_ip_retrieval.py` | 端到端评估（embedding + 指标计算） |
| `train_ip_embedding_0.8B.sh` | Qwen3.5-0.8B 训练脚本 |
| `train_ip_embedding_9B.sh` | Qwen3.5-9B 训练脚本 |
| `train_ip_embedding_qwen3vl8B.sh` | Qwen3-VL-Embedding-8B 训练脚本 |
