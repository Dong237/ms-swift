"""
IP hard sample 挖掘脚本
======================

本脚本实现 Patch 4 的第一步：用训练好的 embedding checkpoint 在全量 IP vault
上挖掘 hard positive 与 hard negative 候选样本。

它做什么：
  1. 读取 IP vault。可以传入目录树 `--vault_dir`，也可以传入已经准备好的
     JSONL manifest `--vault_manifest`。
  2. 如果已经有 embedding JSONL，可用 `--embeddings` 直接跳过编码。
     否则脚本会调用 `aweme-docs/sh/infer_embedding_multimodal_swift.py`
     对 manifest 编码，并读取生成的 `*_with_embedding.jsonl`。
  3. 对全量图片 embedding 做 L2 normalize，优先用 faiss 建内积 ANN index；
     如果环境没有 faiss，则自动回退到 chunked NumPy 矩阵乘法检索。
  4. 对每张 query 检索 top-K，提取高相似但不同 IP 的 hard negative。
  5. 对每个 IP 内部按 sibling 相似度升序挖低相似同 IP 图片作为 hard positive，
     避免只从 top-K 中拿到 easy positive。

输入格式：
  - vault_dir:
      root/
        ip_a/*.jpg
        ip_b/*.png
  - vault_manifest JSONL:
      {"ip_name": "ip_a", "image_path": "/path/a.jpg", "images": ["/path/a.jpg"]}
  - embeddings JSONL 支持两种格式：
      扁平格式: {"ip_name": "ip_a", "image_path": "...", "embedding": [...]}
      聚合格式: {"ip_name": "ip_a", "embeddings": {"/path/a.jpg": [...]}}

输出：
  - hard_positives.jsonl
      query_ip, query_image, positive_ip, positive_image, similarity, rank, pair_id
  - hard_negatives.jsonl
      query_ip, query_image, negative_ip, negative_image, similarity, rank, pair_id
  - mining_report.json

示例：
  python projects/ips/hard-mining/mine_hard_samples.py \\
      --vault_dir /mnt/bn/data/IP_image_train \\
      --model /mnt/bn/models/qwen-ip/checkpoint-1000 \\
      --output_dir /mnt/bn/data/ip_hard_mining/round1

  python projects/ips/hard-mining/mine_hard_samples.py \\
      --embeddings /mnt/bn/data/eval_embeddings.jsonl \\
      --output_dir /mnt/bn/data/ip_hard_mining/round1

注意事项：
  - 第一版只负责挖候选，不负责调用 VLM/reranker 做最终真伪判断。
  - hardest negative 中可能包含漏标同 IP，必须继续经过 filter_hard_samples.py。
  - 单图 IP 不会产生 hard positive，但仍会作为 hard negative gallery。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
USER_IMAGE_MSG = {'role': 'user', 'content': '<image>'}
DEFAULT_INSTRUCTION = '提取图片中IP角色的视觉特征表示'


def read_jsonl(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def write_json(obj: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def get_image_path(record: dict[str, Any]) -> str | None:
    if record.get('image_path'):
        return record['image_path']
    images = record.get('images')
    if images:
        return images[0]
    return None


def get_ip_name(record: dict[str, Any], image_path: str | None = None) -> str:
    for key in ('ip_name', 'ip_id', 'label'):
        value = record.get(key)
        if value is not None and value != '':
            return str(value)
    if image_path:
        parent = os.path.basename(os.path.dirname(image_path))
        if parent:
            return parent
    return 'unknown'


def make_pair_id(kind: str, query_image: str, candidate_image: str) -> str:
    raw = f'{kind}\t{query_image}\t{candidate_image}'.encode('utf-8')
    return hashlib.sha1(raw).hexdigest()[:16]


def scan_vault_dir(vault_dir: str) -> list[dict[str, Any]]:
    records = []
    ip_dirs = []
    with os.scandir(vault_dir) as entries:
        for entry in entries:
            if entry.is_dir():
                ip_dirs.append(entry.path)
    ip_dirs.sort()

    for ip_dir in ip_dirs:
        ip_name = os.path.basename(ip_dir)
        for root, _, files in os.walk(ip_dir):
            for filename in sorted(files):
                ext = os.path.splitext(filename)[1].lower()
                if ext not in IMAGE_EXTS:
                    continue
                image_path = os.path.abspath(os.path.join(root, filename))
                records.append({
                    'ip_name': ip_name,
                    'image_path': image_path,
                    'messages': [USER_IMAGE_MSG],
                    'images': [image_path],
                })
    return records


def prepare_manifest(args: argparse.Namespace) -> str | None:
    if args.embeddings:
        return None

    if args.vault_manifest:
        return args.vault_manifest

    if not args.vault_dir:
        raise ValueError('Provide --embeddings, or provide --vault_dir/--vault_manifest with --model.')

    print(f'[1/4] Scanning vault_dir: {args.vault_dir}')
    records = scan_vault_dir(args.vault_dir)
    if not records:
        raise ValueError(f'No images found in vault_dir: {args.vault_dir}')

    manifest_path = args.manifest_output or os.path.join(args.output_dir, 'vault_manifest.jsonl')
    write_jsonl(records, manifest_path)
    print(f'  Manifest: {manifest_path} ({len(records):,} images)')
    return manifest_path


def run_embedding_inference(args: argparse.Namespace, manifest_path: str) -> str:
    if not args.model:
        raise ValueError('Missing --model when --embeddings is not provided.')

    embeddings_dir = args.embedding_output_dir or os.path.join(args.output_dir, 'embeddings')
    os.makedirs(embeddings_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(manifest_path))[0]
    expected = os.path.join(embeddings_dir, f'{base}_with_embedding.jsonl')
    if os.path.exists(expected) and os.path.getsize(expected) > 0 and not args.force_reembed:
        print(f'[2/4] Reusing existing embeddings: {expected}')
        return expected

    infer_script = args.infer_script
    cmd = [
        sys.executable,
        infer_script,
        '--model',
        args.model,
        '--input',
        manifest_path,
        '--output_dir',
        embeddings_dir,
        '--model_type',
        args.model_type,
        '--batch_size',
        str(args.batch_size),
        '--instruction',
        args.instruction,
    ]
    if args.num_gpus is not None and args.num_gpus > 0:
        cmd.extend(['--num_gpus', str(args.num_gpus)])

    print('[2/4] Running embedding inference:')
    print('  ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)
    if not os.path.exists(expected):
        raise FileNotFoundError(f'Embedding output not found: {expected}')
    return expected


def load_embeddings(path: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    items: list[dict[str, Any]] = []
    skipped = 0

    for row in rows:
        if isinstance(row.get('embeddings'), dict):
            ip_name = get_ip_name(row)
            for image_path, embedding in row['embeddings'].items():
                items.append({'ip': ip_name, 'image': image_path, 'embedding': embedding})
            continue

        embedding = row.get('embedding')
        image_path = get_image_path(row)
        if embedding is None or image_path is None:
            skipped += 1
            continue
        items.append({
            'ip': get_ip_name(row, image_path),
            'image': image_path,
            'embedding': embedding,
        })

    if not items:
        raise ValueError(f'No usable embeddings found in {path}')

    dims = {len(item['embedding']) for item in items}
    if len(dims) != 1:
        raise ValueError(f'Inconsistent embedding dimensions: {sorted(dims)}')

    print(f'  Loaded embeddings: {len(items):,} images, dim={next(iter(dims))}, skipped={skipped}')
    return items


def normalize_embeddings(items: list[dict[str, Any]]) -> np.ndarray:
    matrix = np.array([item['embedding'] for item in items], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-8)
    if not np.isfinite(matrix).all():
        raise ValueError('Embedding matrix contains NaN or inf after normalization.')
    return matrix


def search_topk_faiss(matrix: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import faiss  # type: ignore
    except ImportError:
        return None

    fetch_k = min(matrix.shape[0], top_k + 1)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    sims, indices = index.search(matrix, fetch_k)
    return indices, sims


def search_topk_numpy(matrix: np.ndarray, top_k: int, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    n = matrix.shape[0]
    fetch_k = min(n, top_k + 1)
    all_indices = np.empty((n, fetch_k), dtype=np.int64)
    all_sims = np.empty((n, fetch_k), dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sims = matrix[start:end] @ matrix.T
        row_ids = np.arange(start, end)
        sims[np.arange(end - start), row_ids] = -np.inf

        if fetch_k < n:
            part = np.argpartition(-sims, fetch_k - 1, axis=1)[:, :fetch_k]
        else:
            part = np.argsort(-sims, axis=1)[:, :fetch_k]
        part_sims = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_sims, axis=1)
        top_indices = np.take_along_axis(part, order, axis=1)
        top_sims = np.take_along_axis(part_sims, order, axis=1)

        all_indices[start:end] = top_indices
        all_sims[start:end] = top_sims
        print(f'    NumPy search: {end:,}/{n:,}', end='\r')
    print()
    return all_indices, all_sims


def search_topk(matrix: np.ndarray, top_k: int, backend: str, chunk_size: int) -> tuple[np.ndarray, np.ndarray, str]:
    if backend in {'auto', 'faiss'}:
        result = search_topk_faiss(matrix, top_k)
        if result is not None:
            return result[0], result[1], 'faiss'
        if backend == 'faiss':
            raise RuntimeError('Requested --ann_backend faiss, but faiss is not installed.')
    indices, sims = search_topk_numpy(matrix, top_k, chunk_size)
    return indices, sims, 'numpy'


def mine_hard_negatives(items: list[dict[str, Any]], indices: np.ndarray, sims: np.ndarray,
                        per_query: int) -> list[dict[str, Any]]:
    records = []
    for query_idx, item in enumerate(items):
        q_ip = item['ip']
        q_image = item['image']
        found = 0
        rank = 0
        for cand_idx, sim in zip(indices[query_idx], sims[query_idx]):
            if cand_idx < 0 or cand_idx == query_idx:
                continue
            cand = items[int(cand_idx)]
            if cand['image'] == q_image:
                continue
            rank += 1
            if cand['ip'] == q_ip:
                continue
            records.append({
                'query_ip': q_ip,
                'query_image': q_image,
                'negative_ip': cand['ip'],
                'negative_image': cand['image'],
                'similarity': round(float(sim), 6),
                'rank': rank,
                'pair_id': make_pair_id('neg', q_image, cand['image']),
                'source': 'ann_topk',
            })
            found += 1
            if found >= per_query:
                break
    return records


def mine_hard_positives(items: list[dict[str, Any]], matrix: np.ndarray,
                        per_query: int) -> list[dict[str, Any]]:
    by_ip: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(items):
        by_ip[item['ip']].append(idx)

    records = []
    for ip_name, ip_indices in sorted(by_ip.items()):
        if len(ip_indices) < 2:
            continue
        ip_arr = np.array(ip_indices, dtype=np.int64)
        ip_matrix = matrix[ip_arr]
        for local_pos, query_idx in enumerate(ip_indices):
            q_image = items[query_idx]['image']
            sims = ip_matrix @ matrix[query_idx]
            sims[local_pos] = np.inf
            order = np.argsort(sims)
            rank = 0
            for local_cand in order:
                cand_idx = ip_indices[int(local_cand)]
                if cand_idx == query_idx:
                    continue
                cand_image = items[cand_idx]['image']
                rank += 1
                records.append({
                    'query_ip': ip_name,
                    'query_image': q_image,
                    'positive_ip': ip_name,
                    'positive_image': cand_image,
                    'similarity': round(float(sims[local_cand]), 6),
                    'rank': rank,
                    'pair_id': make_pair_id('pos', q_image, cand_image),
                    'source': 'intra_ip_low_similarity',
                })
                if rank >= per_query:
                    break
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Mine hard positive/negative IP image candidates from embeddings.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--vault_dir', type=str, default=None,
                        help='IP vault root directory; each child folder is one IP.')
    parser.add_argument('--vault_manifest', type=str, default=None,
                        help='Optional JSONL manifest for vault images.')
    parser.add_argument('--embeddings', type=str, default=None,
                        help='Precomputed embeddings JSONL; skips model inference.')
    parser.add_argument('--model', type=str, default=None,
                        help='Trained embedding checkpoint path; required without --embeddings.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for mined outputs.')
    parser.add_argument('--manifest_output', type=str, default=None,
                        help='Where to save scanned manifest when --vault_dir is used.')
    parser.add_argument('--embedding_output_dir', type=str, default=None,
                        help='Directory for embedding inference outputs.')
    parser.add_argument('--infer_script', type=str, default='aweme-docs/sh/infer_embedding_multimodal_swift.py',
                        help='Embedding inference script to call.')
    parser.add_argument('--model_type', type=str, default='qwen3_5_emb',
                        help='Model type for inference.')
    parser.add_argument('--instruction', type=str, default=DEFAULT_INSTRUCTION,
                        help='System instruction for embedding inference.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Per-GPU inference batch size.')
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='Number of GPUs for inference; omitted means infer script default.')
    parser.add_argument('--top_k', type=int, default=100,
                        help='ANN top-K neighbors to inspect per query.')
    parser.add_argument('--hard_positive_per_query', type=int, default=3,
                        help='Lowest-similarity same-IP positives to output per query.')
    parser.add_argument('--hard_negative_per_query', type=int, default=10,
                        help='Highest-similarity different-IP negatives to output per query.')
    parser.add_argument('--ann_backend', choices=['auto', 'faiss', 'numpy'], default='auto',
                        help='ANN backend.')
    parser.add_argument('--chunk_size', type=int, default=4096,
                        help='Query chunk size for NumPy fallback search.')
    parser.add_argument('--force_reembed', action='store_true',
                        help='Re-run embedding inference even if output exists.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    total_t0 = time.time()
    manifest_path = prepare_manifest(args)
    embeddings_path = args.embeddings or run_embedding_inference(args, manifest_path)

    print(f'[3/4] Loading embeddings: {embeddings_path}')
    items = load_embeddings(embeddings_path)
    matrix = normalize_embeddings(items)

    print(f'[4/4] Mining hard samples (N={len(items):,}, top_k={args.top_k})')
    indices, sims, backend = search_topk(matrix, args.top_k, args.ann_backend, args.chunk_size)
    print(f'  ANN backend: {backend}')

    hard_negatives = mine_hard_negatives(items, indices, sims, args.hard_negative_per_query)
    hard_positives = mine_hard_positives(items, matrix, args.hard_positive_per_query)

    pos_path = os.path.join(args.output_dir, 'hard_positives.jsonl')
    neg_path = os.path.join(args.output_dir, 'hard_negatives.jsonl')
    report_path = os.path.join(args.output_dir, 'mining_report.json')
    write_jsonl(hard_positives, pos_path)
    write_jsonl(hard_negatives, neg_path)

    ip_counts = defaultdict(int)
    for item in items:
        ip_counts[item['ip']] += 1
    report = {
        'embeddings_path': embeddings_path,
        'manifest_path': manifest_path,
        'num_images': len(items),
        'num_ips': len(ip_counts),
        'single_image_ips': sum(1 for count in ip_counts.values() if count == 1),
        'ann_backend': backend,
        'top_k': args.top_k,
        'hard_positive_per_query': args.hard_positive_per_query,
        'hard_negative_per_query': args.hard_negative_per_query,
        'hard_positives': len(hard_positives),
        'hard_negatives': len(hard_negatives),
        'elapsed_sec': round(time.time() - total_t0, 2),
    }
    write_json(report, report_path)

    print('Done.')
    print(f'  hard positives: {pos_path} ({len(hard_positives):,})')
    print(f'  hard negatives: {neg_path} ({len(hard_negatives):,})')
    print(f'  report:         {report_path}')


if __name__ == '__main__':
    main()
