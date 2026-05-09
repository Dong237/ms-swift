#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protocol A: IP Embedding 检索评估
================================

本脚本是 IP embedding 评估闭环的第 3 步：读取 encode_ip_eval_manifest.py 生成的
embedding JSONL，按 Protocol A 计算检索指标。

Protocol A 定义：
  - query：所有 is_query=true 的 test 图片。
  - gallery：所有 is_gallery=true 的 train + test 图片。
  - 每个 query 评估时，gallery 中所有 image_path 与 query image_path 完全相同的
    记录都会被 mask 为不可检索，避免自检索命中。
  - relevant：gallery 中 ip_name 与 query ip_name 相同，且未被 self-mask 的记录。
  - 若某个 query 在 self-mask 后没有任何 relevant gallery，则该 query 被跳过，
    计入 skipped_no_relevant_queries。

为什么使用 train+test gallery：
  这模拟一个更大的候选池：既包含已有 vault/train 图片，也包含测试集中其他同 IP
  变体。它能衡量 embedding 是否把同 IP 的跨形态样本召回到 top-K，而不是只在训练
  图库中找最近邻。

指标：
  - Hit@K / CMC@K：top-K 中是否至少有一个同 IP 图片，最贴近“召回后交给 reranker”
    的业务目标。
  - Recall@K：top-K 覆盖了多少同 IP 相关图片。
  - Precision@K：top-K 里同 IP 图片占比。
  - mAP@K：top-K 排序质量。
  - MRR：第一个正确召回的位置倒数。
  - R-Precision：取 K_q = 该 query 的 relevant 数量时的 precision。
  - train_hit@K / test_hit@K：正确 top-K 命中来自 train 还是其他 test 图片。

输入 embeddings JSONL：
  {
    "record_id": "test:000000123",
    "image_path": "/abs/path/img.jpg",
    "ip_name": "蜡笔小新_樱田妮妮",
    "split": "test",
    "is_query": true,
    "is_gallery": true,
    "embedding": [0.01, ...]
  }

输出：
  - eval_metrics.json：整体指标与数据分布。
  - per_query_results.jsonl：每个 query 的指标、top-K 结果、相关样本数量。
  - failure_queries.jsonl：Hit@primary_K=0 的失败 query，便于人工诊断。

示例：
  python projects/ips/evaluation/evaluate_ip_retrieval_protocol_a.py \\
      --embeddings /mnt/bn/.../eval_embeddings.jsonl \\
      --output_dir /mnt/bn/.../eval_results \\
      --K 10 \\
      --extra_k 30
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_json(path: str, obj: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    rank = (len(values) - 1) * p / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(values[lower])
    return values[lower] * (upper - rank) + values[upper] * (rank - lower)


def load_embedding_table(path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray]:
    rows = read_jsonl(path)
    if not rows:
        raise RuntimeError(f'Embeddings JSONL is empty: {path}')
    missing = [row.get('record_id') or row.get('image_path') for row in rows if 'embedding' not in row]
    if missing:
        raise ValueError(f'{len(missing)} rows are missing embedding. First: {missing[0]}')

    dims = sorted(set(len(row['embedding']) for row in rows))
    if len(dims) != 1:
        raise ValueError(f'Inconsistent embedding dimensions: {dims}')

    queries = [row for row in rows if bool(row.get('is_query'))]
    gallery = [row for row in rows if bool(row.get('is_gallery'))]
    if not queries:
        raise RuntimeError('No query rows found. Expected test rows with is_query=true.')
    if not gallery:
        raise RuntimeError('No gallery rows found. Expected is_gallery=true rows.')

    q_emb = np.array([row['embedding'] for row in queries], dtype=np.float32)
    g_emb = np.array([row['embedding'] for row in gallery], dtype=np.float32)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-8)
    g_emb = g_emb / (np.linalg.norm(g_emb, axis=1, keepdims=True) + 1e-8)
    return queries, gallery, q_emb, g_emb


def compute_query_metrics(query: dict[str, Any],
                          sims: np.ndarray,
                          gallery: list[dict[str, Any]],
                          k_values: list[int],
                          top_detail_k: int) -> tuple[dict[str, Any] | None, dict[str, list[float]]]:
    q_ip = query['ip_name']
    q_path = query['image_path']
    gallery_ips = np.array([row['ip_name'] for row in gallery])
    gallery_paths = np.array([row['image_path'] for row in gallery])
    gallery_splits = np.array([row.get('split', 'unknown') for row in gallery])

    self_mask = gallery_paths == q_path
    sims = sims.copy()
    sims[self_mask] = -np.inf
    relevant = (gallery_ips == q_ip) & (~self_mask)
    n_rel = int(relevant.sum())
    if n_rel == 0:
        return None, {}

    max_eval_k = max(max(k_values), n_rel, top_detail_k)
    fetch_k = min(max_eval_k, len(gallery))
    if fetch_k < len(gallery):
        top_idx = np.argpartition(-sims, fetch_k - 1)[:fetch_k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
    else:
        top_idx = np.argsort(-sims)

    top_relevant = relevant[top_idx].astype(int)
    top_splits = gallery_splits[top_idx]

    result: dict[str, Any] = {
        'query_record_id': query.get('record_id'),
        'query_ip': q_ip,
        'query_path': q_path,
        'query_split': query.get('split'),
        'n_relevant': n_rel,
        'n_relevant_train': int((relevant & (gallery_splits == 'train')).sum()),
        'n_relevant_test': int((relevant & (gallery_splits == 'test')).sum()),
    }
    aggregates = {
        'r_precision': [],
        'mrr': [],
    }

    first_hit_positions = np.where(top_relevant == 1)[0]
    mrr = 1.0 / float(first_hit_positions[0] + 1) if len(first_hit_positions) else 0.0
    result['mrr'] = round(mrr, 6)
    aggregates['mrr'].append(mrr)

    r_rel = top_relevant[:n_rel]
    r_precision = float(r_rel.sum()) / n_rel
    result['r_precision'] = round(r_precision, 6)
    aggregates['r_precision'].append(r_precision)

    for k in k_values:
        rel_k = top_relevant[:k]
        n_hit = int(rel_k.sum())
        hit = 1.0 if n_hit > 0 else 0.0
        recall = n_hit / n_rel
        precision = n_hit / k

        ap = 0.0
        seen_rel = 0
        for idx, is_rel in enumerate(rel_k):
            if is_rel:
                seen_rel += 1
                ap += seen_rel / (idx + 1)
        ap = ap / min(n_rel, k)

        train_hit = 1.0 if np.any((top_relevant[:k] == 1) & (top_splits[:k] == 'train')) else 0.0
        test_hit = 1.0 if np.any((top_relevant[:k] == 1) & (top_splits[:k] == 'test')) else 0.0

        result[f'hit@{k}'] = round(hit, 6)
        result[f'CMC@{k}'] = round(hit, 6)
        result[f'recall@{k}'] = round(recall, 6)
        result[f'precision@{k}'] = round(precision, 6)
        result[f'mAP@{k}'] = round(ap, 6)
        result[f'train_hit@{k}'] = round(train_hit, 6)
        result[f'test_hit@{k}'] = round(test_hit, 6)

        aggregates.setdefault(f'hit@{k}', []).append(hit)
        aggregates.setdefault(f'recall@{k}', []).append(recall)
        aggregates.setdefault(f'precision@{k}', []).append(precision)
        aggregates.setdefault(f'mAP@{k}', []).append(ap)
        aggregates.setdefault(f'train_hit@{k}', []).append(train_hit)
        aggregates.setdefault(f'test_hit@{k}', []).append(test_hit)

    detail_n = min(top_detail_k, len(top_idx))
    result['top_results'] = []
    for rank, gi in enumerate(top_idx[:detail_n], start=1):
        result['top_results'].append({
            'rank': rank,
            'ip_name': gallery[gi]['ip_name'],
            'image_path': gallery[gi]['image_path'],
            'split': gallery[gi].get('split'),
            'similarity': round(float(sims[gi]), 6),
            'correct': bool(relevant[gi]),
        })
    return result, aggregates


def evaluate(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    queries, gallery, q_emb, g_emb = load_embedding_table(args.embeddings)
    k_values = sorted(set([1, 3, 5, args.K] + args.extra_k))
    max_k = max(k_values)
    if max_k <= 0:
        raise ValueError('K values must be positive.')

    gallery_split_counts = Counter(row.get('split', 'unknown') for row in gallery)
    query_split_counts = Counter(row.get('split', 'unknown') for row in queries)
    gallery_ip_counts = Counter(row['ip_name'] for row in gallery)
    query_ip_counts = Counter(row['ip_name'] for row in queries)

    print('\nEvaluation input distribution:')
    print(f'  Embeddings:         {args.embeddings}')
    print(f'  Queries:            {len(queries):,} rows, {len(query_ip_counts):,} IPs, splits={dict(query_split_counts)}')
    print(f'  Gallery pool:       {len(gallery):,} rows, {len(gallery_ip_counts):,} IPs, splits={dict(gallery_split_counts)}')
    print(f'  Protocol:           test queries vs train+test gallery, same image_path masked per query')
    print(f'  K values:           {k_values}')

    all_metrics: dict[str, list[float]] = defaultdict(list)
    per_query = []
    skipped_no_relevant = 0
    t0 = time.time()

    for start in range(0, len(queries), args.query_chunk_size):
        end = min(start + args.query_chunk_size, len(queries))
        sims_block = q_emb[start:end] @ g_emb.T
        for local_idx, sims in enumerate(sims_block):
            query = queries[start + local_idx]
            result, aggregates = compute_query_metrics(query, sims, gallery, k_values, args.top_detail_k)
            if result is None:
                skipped_no_relevant += 1
                continue
            per_query.append(result)
            for key, vals in aggregates.items():
                all_metrics[key].extend(vals)
        print(f'  Evaluated {end}/{len(queries)} query rows', end='\r')
    print()

    if not per_query:
        raise RuntimeError('No evaluable queries after self-mask; every query has zero relevant gallery rows.')

    metrics: dict[str, Any] = {
        'protocol': 'A_test_query_gallery_train_plus_test_self_path_masked',
        'embeddings': args.embeddings,
        'query_rows': len(queries),
        'gallery_rows': len(gallery),
        'evaluated_queries': len(per_query),
        'skipped_no_relevant_queries': skipped_no_relevant,
        'query_ip_count': len(query_ip_counts),
        'gallery_ip_count': len(gallery_ip_counts),
        'query_split_counts': dict(query_split_counts),
        'gallery_split_counts': dict(gallery_split_counts),
        'K': args.K,
        'k_values': k_values,
        'elapsed_seconds': round(time.time() - t0, 3),
    }
    for key, vals in sorted(all_metrics.items()):
        metrics[key] = round(float(np.mean(vals)), 6)
    n_rel_values = [row['n_relevant'] for row in per_query]
    metrics['n_relevant_min'] = min(n_rel_values)
    metrics['n_relevant_avg'] = round(float(np.mean(n_rel_values)), 3)
    metrics['n_relevant_p50'] = round(percentile(n_rel_values, 50), 3)
    metrics['n_relevant_p90'] = round(percentile(n_rel_values, 90), 3)
    metrics['n_relevant_p95'] = round(percentile(n_rel_values, 95), 3)
    metrics['n_relevant_max'] = max(n_rel_values)

    return metrics, per_query


def print_and_save(args: argparse.Namespace, metrics: dict[str, Any], per_query: list[dict[str, Any]]) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, 'eval_metrics.json')
    per_query_path = os.path.join(args.output_dir, 'per_query_results.jsonl')
    failures_path = os.path.join(args.output_dir, 'failure_queries.jsonl')

    primary_k = args.K
    failures = [row for row in per_query if row.get(f'hit@{primary_k}', 0.0) == 0.0]
    failures.sort(key=lambda row: (row.get(f'recall@{primary_k}', 0.0), row.get('mrr', 0.0)))

    write_json(metrics_path, metrics)
    write_jsonl(per_query_path, per_query)
    write_jsonl(failures_path, failures)

    print('\nProtocol A retrieval metrics:')
    print(f'  Output dir:             {args.output_dir}')
    print(f'  Evaluated queries:      {metrics["evaluated_queries"]:,}')
    print(f'  Skipped no-relevant:    {metrics["skipped_no_relevant_queries"]:,}')
    print(f'  Gallery rows:           {metrics["gallery_rows"]:,}')
    print('  Relevant/query:         '
          f'min={metrics["n_relevant_min"]:,}, '
          f'avg={metrics["n_relevant_avg"]:.1f}, '
          f'p50={metrics["n_relevant_p50"]:.1f}, '
          f'p90={metrics["n_relevant_p90"]:.1f}, '
          f'p95={metrics["n_relevant_p95"]:.1f}, '
          f'max={metrics["n_relevant_max"]:,}')
    for k in metrics['k_values']:
        print(f'  Hit@{k:<3} / CMC@{k:<3}:    {metrics[f"hit@{k}"]:.6f}')
        print(f'  Recall@{k:<3}:          {metrics[f"recall@{k}"]:.6f}')
        print(f'  Precision@{k:<3}:       {metrics[f"precision@{k}"]:.6f}')
        print(f'  mAP@{k:<3}:             {metrics[f"mAP@{k}"]:.6f}')
        print(f'  train_hit@{k:<3}:       {metrics[f"train_hit@{k}"]:.6f}')
        print(f'  test_hit@{k:<3}:        {metrics[f"test_hit@{k}"]:.6f}')
    print(f'  MRR:                  {metrics["mrr"]:.6f}')
    print(f'  R-Precision:          {metrics["r_precision"]:.6f}')
    print(f'  Failures Hit@{primary_k}=0: {len(failures):,}')
    print(f'  Metrics:              {metrics_path}')
    print(f'  Per-query:            {per_query_path}')
    print(f'  Failures:             {failures_path}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate IP embedding retrieval with Protocol A.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--embeddings', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--extra_k', type=int, nargs='*', default=[30])
    parser.add_argument('--query_chunk_size', type=int, default=256)
    parser.add_argument('--top_detail_k', type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.K <= 0:
        raise ValueError('--K must be positive.')
    if args.query_chunk_size <= 0:
        raise ValueError('--query_chunk_size must be positive.')
    if args.top_detail_k <= 0:
        raise ValueError('--top_detail_k must be positive.')
    metrics, per_query = evaluate(args)
    print_and_save(args, metrics, per_query)


if __name__ == '__main__':
    main()
