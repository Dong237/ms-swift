#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建 IP Embedding 检索评估 Manifest
==================================

本脚本是 IP embedding 评估闭环的第 1 步：把训练集与测试集的“标准 IP 合并标注
JSON”转换成一个统一的图片 manifest。后续编码脚本只读取这个 manifest 做 embedding
推理，评估脚本再基于 manifest 中的 split 信息执行检索指标计算。

评估协议采用 Protocol A：
  - query：所有 test split 图片。
  - gallery：train split 图片 + test split 图片组成的全池。
  - 对每一个 test query，评估时会排除与当前 query 完全相同 image_path 的 gallery
    记录，避免自检索命中。
  - 因此 manifest 中 test 图片会同时标记为 is_query=true 与 is_gallery=true；这表示
    它属于全局 gallery pool，但在它自己作为 query 时会被评估脚本按 image_path mask。

输入 JSON 格式：
  {
    "蜡笔小新_樱田妮妮": {
      "标准名称": "蜡笔小新_樱田妮妮",
      "合并Keys": ["蜡笔小新外传:玩具大战_妮妮", "蜡笔小新_樱田妮妮"],
      "文件夹列表": [
        "/mnt/bn/youxiang-hl/data/facial_ip/IP_image/12608",
        "/mnt/bn/youxiang-hl/data/facial_ip/IP_image/1579"
      ],
      "总图片": 15
    }
  }

duplicate-folder 冲突处理：
  与训练数据构建脚本保持一致。若同一个历史图片文件夹出现在多个不同标准 IP 下，
  这些标准 IP 被视为互相冲突的类别标注。脚本会在扫描图片前解决冲突：
    1. 先合并相同“标准名称”的 JSON entry；
    2. 建立 folder -> standard IP 的归属关系；
    3. 共享 folder 连接到的多个 standard IP 构成一个冲突组件；
    4. 每个组件只保留一个 standard IP，丢弃其余 standard IP；
    5. 保留优先级为：唯一文件夹数量最多、JSON 中最早出现、标准 IP 名字典序最小。
  被丢弃的 IP 不会进入 manifest，也不会作为 query/gallery/label 参与评估。

输出 JSONL 格式：
  {
    "record_id": "test:000000123",
    "image_path": "/abs/path/img.jpg",
    "ip_name": "蜡笔小新_樱田妮妮",
    "split": "test",
    "source_folder": "/abs/path/folder",
    "is_query": true,
    "is_gallery": true
  }

同时输出 <manifest>.report.json，记录 train/test 的 IP 数、图片数、duplicate-folder
冲突解决结果、跨 split image_path 重叠等诊断信息。

示例：
  python projects/ips/evaluation/build_ip_eval_manifest.py \\
      --train_json /mnt/bn/youxiang-hl/data/facial_ip/output_train_test_set/train_set.json \\
      --test_json /mnt/bn/youxiang-hl/data/facial_ip/output_train_test_set/test_set.json \\
      --output /mnt/bn/youxiang-hl/data/facial_ip/output_train_test_set/eval/eval_manifest.jsonl \\
      --num_workers 64
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
DEFAULT_TRAIN_JSON = '/mnt/bn/youxiang-hl/data/facial_ip/output_train_test_set/train_set.json'
DEFAULT_TEST_JSON = '/mnt/bn/youxiang-hl/data/facial_ip/output_train_test_set/test_set.json'


try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, total=None, desc='', unit='', **kw):
            self.iterable = iterable
            self.total = total or (len(iterable) if iterable and hasattr(iterable, '__len__') else 0)
            self.desc = desc
            self.n = 0
            self._last_print = 0

        def __iter__(self):
            for item in self.iterable:
                yield item
                self.update(1)
            print()

        def update(self, n=1):
            self.n += n
            step = max(1, self.total // 100) if self.total else 1
            if self.n - self._last_print >= step:
                self._last_print = self.n
                pct = self.n / self.total * 100 if self.total else 0
                print(f'\r  {self.desc}: {self.n}/{self.total} ({pct:.0f}%)', end='', flush=True)

        def close(self):
            print()


def _read_json(path: str) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f'Expected JSON root dict, got {type(data).__name__}: {path}')
    return data


def _write_json(path: str, obj: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def _extract_folder_list(ip_key: str, value: Any) -> tuple[str, list[str], list[str]]:
    if isinstance(value, list):
        return ip_key, [str(x) for x in value], [ip_key]
    if not isinstance(value, dict):
        raise ValueError(f'IP entry {ip_key!r} must be dict or list, got {type(value).__name__}')

    standard_name = str(value.get('标准名称') or value.get('standard_name') or ip_key)
    folders = value.get('文件夹列表')
    if folders is None:
        folders = value.get('folder_list') or value.get('folders')
    if folders is None:
        raise ValueError(f'IP entry {ip_key!r} has no 文件夹列表/folder_list/folders field.')
    if not isinstance(folders, list):
        raise ValueError(f'IP entry {ip_key!r} folder list must be a list.')

    merged_keys = value.get('合并Keys') or value.get('merged_keys') or [ip_key]
    if not isinstance(merged_keys, list):
        merged_keys = [str(merged_keys)]
    return standard_name, [str(x) for x in folders], [str(x) for x in merged_keys]


def _normalize_folder(folder: str, image_root: Optional[str]) -> str:
    folder_path = Path(folder).expanduser()
    if not folder_path.is_absolute() and image_root:
        folder_path = Path(image_root).expanduser() / folder_path
    return str(folder_path)


def _resolve_duplicate_folder_conflicts(ip_to_folders: dict[str, list[str]],
                                        ip_to_merged_keys: dict[str, list[str]],
                                        ip_order: dict[str, int],
                                        strict_duplicates: bool) -> dict[str, Any]:
    folder_to_ips: dict[str, set[str]] = {}
    for ip_name, folders in ip_to_folders.items():
        for folder in set(folders):
            folder_to_ips.setdefault(folder, set()).add(ip_name)

    duplicate_folders = []
    for folder in sorted(folder_to_ips):
        owners = sorted(folder_to_ips[folder], key=lambda ip: (ip_order.get(ip, sys.maxsize), ip))
        if len(owners) <= 1:
            continue
        first_ip = owners[0]
        for owner in owners[1:]:
            duplicate_folders.append({'folder': folder, 'first_ip': first_ip, 'second_ip': owner})

    if not duplicate_folders:
        return {
            'duplicate_folders': [],
            'duplicate_folder_count': 0,
            'duplicate_conflict_component_count': 0,
            'duplicate_conflict_dropped_ip_count': 0,
            'duplicate_conflict_kept_ip_count': 0,
            'duplicate_conflict_resolution_examples': [],
        }

    msg = f'Found {len(duplicate_folders)} folders assigned to multiple standard IPs.'
    if strict_duplicates:
        raise ValueError(msg + f' First duplicate: {duplicate_folders[0]}')

    conflict_ips = set()
    adjacency: dict[str, set[str]] = {}
    for owners in folder_to_ips.values():
        if len(owners) <= 1:
            continue
        for owner in owners:
            conflict_ips.add(owner)
            adjacency.setdefault(owner, set()).update(owners - {owner})

    components = []
    visited = set()
    for ip_name in sorted(conflict_ips, key=lambda ip: (ip_order.get(ip, sys.maxsize), ip)):
        if ip_name in visited:
            continue
        stack = [ip_name]
        component = set()
        visited.add(ip_name)
        while stack:
            current = stack.pop()
            component.add(current)
            for nxt in sorted(adjacency.get(current, set()), key=lambda ip: (ip_order.get(ip, sys.maxsize), ip)):
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        if len(component) > 1:
            components.append(component)

    dropped_ips = set()
    resolution_examples = []
    for component in components:
        winner = min(component, key=lambda ip: (-len(set(ip_to_folders[ip])), ip_order.get(ip, sys.maxsize), ip))
        losers = sorted(component - {winner}, key=lambda ip: (ip_order.get(ip, sys.maxsize), ip))
        dropped_ips.update(losers)
        shared_folders = sorted(
            folder for folder, owners in folder_to_ips.items()
            if len(owners & component) > 1
        )
        resolution_examples.append({
            'winner': winner,
            'winner_folder_count': len(set(ip_to_folders[winner])),
            'dropped_ips': losers,
            'dropped_folder_counts': {ip: len(set(ip_to_folders[ip])) for ip in losers},
            'shared_folder_count': len(shared_folders),
            'shared_folders': shared_folders[:20],
        })

    for ip_name in dropped_ips:
        ip_to_folders.pop(ip_name, None)
        ip_to_merged_keys.pop(ip_name, None)

    print(f'  WARNING: {msg}')
    print('  Duplicate-folder conflicts auto-resolved before image scanning:')
    print(f'    conflict components: {len(components):,}')
    print(f'    kept standard IPs:   {len(components):,}')
    print(f'    dropped standard IPs:{len(dropped_ips):,}')
    if resolution_examples:
        first = resolution_examples[0]
        print(f'    first resolution: keep {first["winner"]!r}, drop {first["dropped_ips"][:5]}')

    return {
        'duplicate_folders': duplicate_folders[:100],
        'duplicate_folder_count': len(duplicate_folders),
        'duplicate_conflict_component_count': len(components),
        'duplicate_conflict_dropped_ip_count': len(dropped_ips),
        'duplicate_conflict_kept_ip_count': len(components),
        'duplicate_conflict_resolution_examples': resolution_examples[:100],
    }


def _scan_one_folder(folder_path: str, recursive: bool) -> tuple[str, list[str], Optional[str]]:
    folder = Path(folder_path).expanduser()
    images: list[str] = []
    try:
        if recursive:
            iterator = folder.rglob('*')
            for path in iterator:
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                    images.append(str(path))
        else:
            with os.scandir(folder) as entries:
                for entry in entries:
                    if entry.is_file() and os.path.splitext(entry.name)[1].lower() in IMAGE_EXTS:
                        images.append(entry.path)
    except (FileNotFoundError, PermissionError, OSError) as exc:
        return str(folder), [], f'{type(exc).__name__}: {exc}'
    return str(folder), sorted(set(images)), None


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    rank = (len(values) - 1) * percentile / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(values[lower])
    return values[lower] * (upper - rank) + values[upper] * (rank - lower)


def load_split(split: str, json_path: str, image_root: Optional[str], recursive: bool,
               num_workers: int, strict_duplicates: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    print(f'\n[{split}] Reading merge JSON: {json_path}')
    raw = _read_json(json_path)
    ip_to_folders: dict[str, list[str]] = {}
    ip_to_merged_keys: dict[str, list[str]] = {}
    ip_order: dict[str, int] = {}

    for raw_idx, (ip_key, value) in enumerate(raw.items()):
        standard_name, folders, merged_keys = _extract_folder_list(ip_key, value)
        normalized = [_normalize_folder(folder, image_root) for folder in folders]
        if standard_name not in ip_order:
            ip_order[standard_name] = raw_idx
        ip_to_folders.setdefault(standard_name, []).extend(normalized)
        ip_to_merged_keys.setdefault(standard_name, []).extend(merged_keys)

    standard_ips_before = len(ip_to_folders)
    duplicate_report = _resolve_duplicate_folder_conflicts(
        ip_to_folders, ip_to_merged_keys, ip_order, strict_duplicates)

    all_folders = sorted(set(folder for folders in ip_to_folders.values() for folder in folders))
    print(f'  Standard IPs before duplicate resolution: {standard_ips_before:,}')
    print(f'  Standard IPs after duplicate resolution:  {len(ip_to_folders):,}')
    print(f'  Unique folders to scan: {len(all_folders):,}')

    folder_to_images: dict[str, list[str]] = {}
    scan_errors = []
    if num_workers <= 1:
        iterator = (_scan_one_folder(folder, recursive) for folder in all_folders)
        for folder, images, error in tqdm(iterator, total=len(all_folders), desc=f'{split} scan', unit='dir'):
            folder_to_images[folder] = images
            if error:
                scan_errors.append({'folder': folder, 'error': error})
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_scan_one_folder, folder, recursive): folder for folder in all_folders}
            pbar = tqdm(total=len(futures), desc=f'{split} scan', unit='dir')
            for future in as_completed(futures):
                folder, images, error = future.result()
                folder_to_images[folder] = images
                if error:
                    scan_errors.append({'folder': folder, 'error': error})
                pbar.update(1)
            pbar.close()

    rows = []
    ip_image_counts = Counter()
    folder_image_counts = Counter()
    seen_split_paths = set()
    duplicate_image_paths_inside_split = 0
    record_idx = 0
    for ip_name in sorted(ip_to_folders.keys()):
        for folder in sorted(set(ip_to_folders[ip_name])):
            for image_path in folder_to_images.get(folder, []):
                if image_path in seen_split_paths:
                    duplicate_image_paths_inside_split += 1
                    continue
                seen_split_paths.add(image_path)
                rows.append({
                    'record_id': f'{split}:{record_idx:012d}',
                    'image_path': image_path,
                    'ip_name': ip_name,
                    'split': split,
                    'source_folder': folder,
                    'is_query': split == 'test',
                    'is_gallery': True,
                })
                record_idx += 1
                ip_image_counts[ip_name] += 1
                folder_image_counts[folder] += 1

    ip_counts = list(ip_image_counts.values())
    report = {
        'split': split,
        'json_path': json_path,
        'standard_ips_in_json': standard_ips_before,
        'standard_ips_after_duplicate_resolution': len(ip_to_folders),
        'ips_with_images': len(ip_image_counts),
        'image_count': len(rows),
        'unique_folders': len(all_folders),
        'folders_with_images': len(folder_image_counts),
        'duplicate_image_paths_inside_split': duplicate_image_paths_inside_split,
        'scan_errors': scan_errors[:100],
        'scan_error_count': len(scan_errors),
        'images_per_ip_min': min(ip_counts) if ip_counts else 0,
        'images_per_ip_avg': sum(ip_counts) / len(ip_counts) if ip_counts else 0.0,
        'images_per_ip_p50': _percentile(ip_counts, 50),
        'images_per_ip_p90': _percentile(ip_counts, 90),
        'images_per_ip_p95': _percentile(ip_counts, 95),
        'images_per_ip_max': max(ip_counts) if ip_counts else 0,
        **duplicate_report,
    }
    print_split_distribution(report)
    return rows, report


def print_split_distribution(report: dict[str, Any]) -> None:
    print(f'\n{report["split"]} distribution:')
    print(f'  Standard IPs before duplicate resolution: {report["standard_ips_in_json"]:,}')
    print(f'  Standard IPs after duplicate resolution:  {report["standard_ips_after_duplicate_resolution"]:,}')
    print(f'  Duplicate folders:                        {report["duplicate_folder_count"]:,}')
    print(f'  Duplicate conflict components:            {report["duplicate_conflict_component_count"]:,}')
    print(f'  Dropped standard IPs from conflicts:       {report["duplicate_conflict_dropped_ip_count"]:,}')
    print(f'  IPs with images:                           {report["ips_with_images"]:,}')
    print(f'  Images:                                    {report["image_count"]:,}')
    print('  Images/IP:                                '
          f'min={report["images_per_ip_min"]:,}, '
          f'avg={report["images_per_ip_avg"]:.1f}, '
          f'p50={report["images_per_ip_p50"]:.1f}, '
          f'p90={report["images_per_ip_p90"]:.1f}, '
          f'p95={report["images_per_ip_p95"]:.1f}, '
          f'max={report["images_per_ip_max"]:,}')


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build Protocol-A IP retrieval evaluation manifest.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--train_json', type=str, default=DEFAULT_TRAIN_JSON)
    parser.add_argument('--test_json', type=str, default=DEFAULT_TEST_JSON)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--image_root', type=str, default=None)
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--num_workers', type=int, default=64)
    parser.add_argument('--strict_duplicates', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    train_rows, train_report = load_split(
        'train', args.train_json, args.image_root, args.recursive, args.num_workers, args.strict_duplicates)
    test_rows, test_report = load_split(
        'test', args.test_json, args.image_root, args.recursive, args.num_workers, args.strict_duplicates)

    all_rows = train_rows + test_rows
    if not test_rows:
        raise RuntimeError('No test images found; Protocol A requires test queries.')
    if not all_rows:
        raise RuntimeError('No images found in train/test manifests.')

    write_jsonl(args.output, all_rows)

    train_paths = {row['image_path'] for row in train_rows}
    test_paths = {row['image_path'] for row in test_rows}
    train_ips = {row['ip_name'] for row in train_rows}
    test_ips = {row['ip_name'] for row in test_rows}
    common_ips = train_ips & test_ips
    cross_split_path_overlap = sorted(train_paths & test_paths)

    report = {
        'protocol': 'A_test_query_gallery_train_plus_test_self_path_masked',
        'manifest_path': args.output,
        'train': train_report,
        'test': test_report,
        'total_rows': len(all_rows),
        'query_rows': len(test_rows),
        'gallery_rows': len(all_rows),
        'train_image_count': len(train_rows),
        'test_image_count': len(test_rows),
        'train_ip_count': len(train_ips),
        'test_ip_count': len(test_ips),
        'common_ip_count': len(common_ips),
        'test_only_ip_count': len(test_ips - train_ips),
        'train_only_ip_count': len(train_ips - test_ips),
        'cross_split_duplicate_image_path_count': len(cross_split_path_overlap),
        'cross_split_duplicate_image_path_examples': cross_split_path_overlap[:100],
        'elapsed_seconds': round(time.time() - t0, 3),
    }
    report_path = args.output + '.report.json'
    _write_json(report_path, report)

    print('\nFinal manifest distribution:')
    print(f'  Output manifest:                    {args.output}')
    print(f'  Report:                             {report_path}')
    print(f'  Total rows/images:                  {len(all_rows):,}')
    print(f'  Query rows (test):                  {len(test_rows):,}')
    print(f'  Gallery rows before self masking:   {len(all_rows):,}')
    print(f'  Train images:                       {len(train_rows):,}')
    print(f'  Test images:                        {len(test_rows):,}')
    print(f'  Train IPs:                          {len(train_ips):,}')
    print(f'  Test IPs:                           {len(test_ips):,}')
    print(f'  Common train/test IPs:              {len(common_ips):,}')
    print(f'  Test-only IPs:                      {len(test_ips - train_ips):,}')
    print(f'  Cross-split duplicate image paths:  {len(cross_split_path_overlap):,}')
    print(f'  Done in {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
