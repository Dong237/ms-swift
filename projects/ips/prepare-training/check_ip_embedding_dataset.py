#!/usr/bin/env python3
"""
IP Embedding 训练 JSONL 一体化校验脚本
=====================================

用途：
  在启动 ms-swift embedding 训练前，检查由
  build_ip_training_data_from_merge_json.py / run_build_ip_training_data_from_merge_json.sh
  生成的 JSONL 是否满足训练要求，尤其是 Sub-center ArcFace 所需的 ip_id 元数据。

检查内容：
  1. JSONL 基础结构：
     - 每行必须是合法 JSON object；
     - 必须包含 messages/images/positive_images/negative_images 等核心字段；
     - anchor 至少 1 张图，positive 至少 1 张图；
     - positive_images/negative_images 必须是 List[List[str]] 结构。

  2. ip_id / ip_id_map：
     - 默认要求每行包含整数 ip_id 和字符串 ip_name；
     - 加载 <dataset>.ip_id_map.json 或 --ip_id_map 指定文件；
     - 检查 ip_id 是否在 [0, num_classes) 范围内；
     - 如果 id_to_ip 存在，检查 row["ip_name"] 是否与 id_to_ip[ip_id] 一致。

  3. 正负例冲突：
     - anchor 不能出现在 positive 或 negative 中；
     - positive 和 negative 不能出现同一图片；
     - 同一行内 negative 重复会计入 warning。

  4. 图片文件：
     - 默认检查所有图片路径是否存在且 PIL 可读；
     - 可通过 --skip_image_check 跳过；
     - 可通过 --max_image_checks 限制检查图片引用数量，用于快速抽查。

  5. 数据分布：
     - 输出行数、class 数、positive/negative 数量分布；
     - 输出 unique ip_id 数、samples/ip 分布；
     - 输出图片引用数、缺失/损坏数量。

示例：
  python3 projects/ips/prepare-training/check_ip_embedding_dataset.py \\
      --dataset /mnt/bn/youxiang-lf/data/facial_ip/output_train_test_set/training/train_merged_multipos_p3.jsonl

  # 快速检查结构和 ip_id，不检查图片可读性：
  python3 projects/ips/prepare-training/check_ip_embedding_dataset.py \\
      --dataset /mnt/bn/.../train_merged_multipos_p3.jsonl \\
      --skip_image_check

  # 只抽查前 10000 个图片引用：
  python3 projects/ips/prepare-training/check_ip_embedding_dataset.py \\
      --dataset /mnt/bn/.../train_merged_multipos_p3.jsonl \\
      --max_image_checks 10000

退出码：
  - 所有 error 级检查通过：退出 0；
  - 存在缺失 ip_id、ip_id 越界、JSON 解析失败、图片缺失/损坏等 error：退出 1。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from PIL import Image
except ImportError:
    Image = None


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    rank = (len(values) - 1) * percentile / 100
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    if lower == upper:
        return float(values[lower])
    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _iter_group_images(value: Any) -> Iterable[str]:
    if not isinstance(value, list):
        return
    for group in value:
        if isinstance(group, list):
            for item in group:
                if isinstance(item, str):
                    yield item


def _load_ip_id_map(path: str) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict):
        raise ValueError(f'ip_id_map root must be object, got {type(mapping).__name__}')
    if 'num_classes' not in mapping:
        raise ValueError('ip_id_map missing required key: num_classes')
    return mapping


def _check_image(path: str) -> tuple[bool, str | None]:
    if not os.path.isfile(path):
        return False, 'missing'
    if Image is None:
        return True, None
    try:
        with Image.open(path) as image:
            image.verify()
        return True, None
    except Exception as exc:
        return False, f'{type(exc).__name__}: {exc}'


def _count_lines(path: Path) -> int:
    count = 0
    with path.open('rb') as f:
        for _ in f:
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Check IP embedding training JSONL before infonce/multi-positive/sub-center training.')
    parser.add_argument('--dataset', default="/mnt/bn/youxiang-lf/data/facial_ip/output_train_test_set/training/train_merged_multipos_p3.jsonl", help='训练 JSONL 路径。')
    parser.add_argument('--ip_id_map', default="/mnt/bn/youxiang-lf/data/facial_ip/output_train_test_set/training/train_merged_multipos_p3.jsonl.ip_id_map.json",
                        help='ip_id_map JSON 路径；默认使用 <dataset>.ip_id_map.json。')
    parser.add_argument('--allow_missing_ip_id', action='store_true',
                        help='允许缺失 ip_id。仅适合普通 infonce/multi_positive_infonce，不适合 stage2_ip_embedding。')
    parser.add_argument('--skip_image_check', action='store_true',
                        help='跳过图片存在性和 PIL 可读性检查。')
    parser.add_argument('--max_rows', type=int, default=0,
                        help='最多检查多少行；0 表示全量检查。')
    parser.add_argument('--max_image_checks', type=int, default=0,
                        help='最多检查多少个图片引用；0 表示全量检查。')
    parser.add_argument('--max_examples', type=int, default=10,
                        help='每类错误最多打印多少个例子。')
    parser.add_argument('--no_progress', action='store_true',
                        help='关闭 tqdm 进度条。默认开启；若环境未安装 tqdm 会自动退回普通输出。')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    ip_id_map_path = Path(args.ip_id_map or f'{args.dataset}.ip_id_map.json')

    if not dataset_path.is_file():
        print(f'ERROR: dataset not found: {dataset_path}')
        sys.exit(1)

    mapping = None
    num_classes = None
    id_to_ip = {}
    if ip_id_map_path.is_file():
        mapping = _load_ip_id_map(str(ip_id_map_path))
        num_classes = int(mapping['num_classes'])
        id_to_ip = {int(k): v for k, v in mapping.get('id_to_ip', {}).items()}
    elif not args.allow_missing_ip_id:
        print(f'ERROR: ip_id_map not found: {ip_id_map_path}')
        sys.exit(1)

    errors: Counter[str] = Counter()
    warnings: Counter[str] = Counter()
    examples: dict[str, list[Any]] = {}

    def add_example(key: str, value: Any) -> None:
        bucket = examples.setdefault(key, [])
        if len(bucket) < args.max_examples:
            bucket.append(value)

    rows = 0
    image_refs = 0
    image_checks = 0
    unique_images: set[str] = set()
    ip_counts: Counter[int] = Counter()
    pos_count_dist: Counter[int] = Counter()
    neg_count_dist: Counter[int] = Counter()

    total_rows = None
    if not args.no_progress and tqdm is not None:
        total_rows = args.max_rows if args.max_rows else _count_lines(dataset_path)

    with dataset_path.open('r', encoding='utf-8') as f:
        iterator = enumerate(f, 1)
        if not args.no_progress and tqdm is not None:
            iterator = tqdm(iterator, total=total_rows, desc='Checking JSONL', unit='row')

        for line_no, line in iterator:
            if args.max_rows and rows >= args.max_rows:
                break
            line = line.strip()
            if not line:
                warnings['empty_line'] += 1
                add_example('empty_line', {'line': line_no})
                continue

            try:
                row = json.loads(line)
            except Exception as exc:
                errors['json_parse_error'] += 1
                add_example('json_parse_error', {'line': line_no, 'error': repr(exc)})
                continue
            rows += 1

            if not isinstance(row, dict):
                errors['row_not_object'] += 1
                add_example('row_not_object', {'line': line_no, 'type': type(row).__name__})
                continue

            for key in ['messages', 'images', 'positive_images', 'negative_images']:
                if key not in row:
                    errors[f'missing_{key}'] += 1
                    add_example(f'missing_{key}', {'line': line_no})

            anchors = row.get('images', [])
            if not isinstance(anchors, list) or not all(isinstance(x, str) for x in anchors):
                errors['bad_images_field'] += 1
                add_example('bad_images_field', {'line': line_no, 'value': anchors})
                anchors = []
            if not anchors:
                errors['empty_anchor_images'] += 1
                add_example('empty_anchor_images', {'line': line_no})

            positives = list(_iter_group_images(row.get('positive_images', [])))
            negatives = list(_iter_group_images(row.get('negative_images', [])))

            if not positives:
                errors['empty_positive_images'] += 1
                add_example('empty_positive_images', {'line': line_no})

            pos_count_dist[len(row.get('positive_images', [])) if isinstance(row.get('positive_images'), list) else -1] += 1
            neg_count_dist[len(row.get('negative_images', [])) if isinstance(row.get('negative_images'), list) else -1] += 1

            if 'ip_id' not in row:
                if not args.allow_missing_ip_id:
                    errors['missing_ip_id'] += 1
                    add_example('missing_ip_id', {'line': line_no, 'keys': sorted(row.keys())})
            else:
                ip_id = row['ip_id']
                if not isinstance(ip_id, int):
                    errors['ip_id_not_int'] += 1
                    add_example('ip_id_not_int', {'line': line_no, 'ip_id': ip_id})
                elif num_classes is not None and not (0 <= ip_id < num_classes):
                    errors['ip_id_out_of_range'] += 1
                    add_example('ip_id_out_of_range', {'line': line_no, 'ip_id': ip_id, 'num_classes': num_classes})
                else:
                    ip_counts[ip_id] += 1
                    if id_to_ip and 'ip_name' in row and row['ip_name'] != id_to_ip.get(ip_id):
                        errors['ip_name_mismatch'] += 1
                        add_example('ip_name_mismatch', {
                            'line': line_no,
                            'ip_id': ip_id,
                            'row_ip_name': row.get('ip_name'),
                            'map_ip_name': id_to_ip.get(ip_id),
                        })

            anchor_set = set(anchors)
            pos_set = set(positives)
            neg_set = set(negatives)
            if anchor_set & pos_set:
                errors['anchor_positive_collision'] += 1
                add_example('anchor_positive_collision', {'line': line_no, 'images': sorted(anchor_set & pos_set)[:5]})
            if anchor_set & neg_set:
                errors['anchor_negative_collision'] += 1
                add_example('anchor_negative_collision', {'line': line_no, 'images': sorted(anchor_set & neg_set)[:5]})
            if pos_set & neg_set:
                errors['positive_negative_collision'] += 1
                add_example('positive_negative_collision', {'line': line_no, 'images': sorted(pos_set & neg_set)[:5]})
            if len(negatives) != len(neg_set):
                warnings['duplicate_negative_in_row'] += 1
                add_example('duplicate_negative_in_row', {'line': line_no})

            for image in list(anchors) + positives + negatives:
                image_refs += 1
                unique_images.add(image)
                if args.skip_image_check:
                    continue
                if args.max_image_checks and image_checks >= args.max_image_checks:
                    continue
                image_checks += 1
                ok, reason = _check_image(image)
                if not ok:
                    key = 'missing_image' if reason == 'missing' else 'bad_image'
                    errors[key] += 1
                    add_example(key, {'line': line_no, 'image': image, 'reason': reason})

    sample_counts = list(ip_counts.values())
    print('\nDataset check summary')
    print('=' * 60)
    print(f'Dataset:                 {dataset_path}')
    print(f'IP ID map:               {ip_id_map_path if ip_id_map_path.exists() else "N/A"}')
    print(f'Rows checked:            {rows:,}')
    print(f'Num classes:             {num_classes if num_classes is not None else "N/A"}')
    print(f'Unique ip_ids in rows:   {len(ip_counts):,}')
    if sample_counts:
        print('Samples/IP:              '
              f'min={min(sample_counts):,}, '
              f'avg={sum(sample_counts) / len(sample_counts):.1f}, '
              f'p50={_percentile(sample_counts, 50):.1f}, '
              f'p90={_percentile(sample_counts, 90):.1f}, '
              f'p95={_percentile(sample_counts, 95):.1f}, '
              f'max={max(sample_counts):,}')
    print(f'Positive groups/row:     {dict(sorted(pos_count_dist.items()))}')
    print(f'Negative groups/row:     {dict(sorted(neg_count_dist.items()))}')
    print(f'Image refs seen:         {image_refs:,}')
    print(f'Unique image paths:      {len(unique_images):,}')
    print(f'Image refs checked:      {image_checks:,}' if not args.skip_image_check else 'Image refs checked:      skipped')

    if warnings:
        print('\nWarnings')
        for key, value in sorted(warnings.items()):
            print(f'  {key}: {value:,}')

    if errors:
        print('\nErrors')
        for key, value in sorted(errors.items()):
            print(f'  {key}: {value:,}')

    if examples:
        print('\nExamples')
        for key, values in sorted(examples.items()):
            print(f'  {key}:')
            for value in values:
                print(f'    {value}')

    if errors:
        print('\nFAILED: dataset has blocking errors.')
        sys.exit(1)

    print('\nPASSED: dataset is structurally valid for the requested checks.')


if __name__ == '__main__':
    main()
