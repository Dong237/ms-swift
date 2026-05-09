"""
基于 IP 合并标注 JSON 构建 Embedding 训练数据
=============================================

这个脚本用于新的 IP 标注 policy：多个历史 IP 文件夹现在可能被统一合并为同一个
标准 IP。脚本先读取合并 JSON，再把同一个标准 IP 下的所有文件夹图片聚合到一起，
随后按训练 loss 的需要生成 anchor / positive / negative JSONL。

输入 JSON 格式示例：

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

输出 JSONL 兼容 ms-swift embedding 训练格式：

  {
    "messages": [{"role": "user", "content": "<image>"}],
    "images": ["anchor.jpg"],
    "positive_messages": [
      [{"role": "user", "content": "<image>"}],
      [{"role": "user", "content": "<image>"}]
    ],
    "positive_images": [["pos1.jpg"], ["pos2.jpg"]],
    "negative_messages": [[{"role": "user", "content": "<image>"}]],
    "negative_images": [["neg1.jpg"]]
  }

两种训练数据模式：

  1. --loss_type infonce
     每条样本只写入 1 个 positive，适配旧的普通 InfoNCE 训练。

  2. --loss_type multi_positive_infonce 或 --loss_type supcon
     每条样本写入 --num_positives 个 positive（若该 IP 图片不足则使用可用数量），
     适配 Patch 1 新增的 multi_positive_infonce / SupCon-style 训练。

注意事项：

  - 本脚本只构建数据，不修改 swift 主框架。
  - negative 只会从不同“标准 IP”里采样；被 merge 到同一标准 IP 的历史文件夹
    不会再互相作为负例。
  - 重要：如果同一个历史文件夹出现在多个不同“标准 IP”的 文件夹列表 中，这不是
    普通重复，而是互相冲突的类别标注。脚本会在扫描图片前先解析这类冲突：
      1. 先把具有相同“标准名称”的 JSON entry 合并成一个标准 IP group；
      2. 再建立 folder -> standard IP 的归属关系，并把共享文件夹连接到的多个
         standard IP 视为一个冲突组件；
      3. 每个冲突组件只保留一个 standard IP group，丢弃其余 group；
      4. 保留规则依次为：文件夹数量最多者优先、JSON 中更早出现者优先、名称字典序
         更小者优先；
      5. 被丢弃的 standard IP 不会参与正例、负例、ip_id 或 ip_id_map 构建。
    这样可以避免同一批图片同时被当成两个类别训练，尤其避免 Sub-center ArcFace
    把同一图片推向两个 proxy。
  - --num_positives 控制每条样本里的 positive 数量；--max_samples_per_ip 控制每个
    标准 IP 最多输出多少条训练样本。
  - 对于普通 infonce，即使传入 --num_positives > 1，脚本也只会写入 1 个 positive，
    因为旧 loss 路径只消费第一个 positive。
  - 默认只做路径级去重；开启 --md5_dedup 后，会在每个标准 IP 内按文件内容 MD5
    做物理级去重，适合处理 merge 后多个历史文件夹里出现完全重复图片的情况。
    注意：--md5_dedup 只在已保留的单个 standard IP 内去重，不能替代上面的
    duplicate-folder 冲突解析。
  - 开启 --include_metadata 或 --include_ip_id 后，每条样本会写入稳定的整数 ip_id
    和字符串 ip_name。ip_id 按 sorted(standard_ip_name) 分配，并额外输出
    <output>.ip_id_map.json，供后续 Sub-center ArcFace proxy 矩阵索引使用。

示例：

  # 构建旧 InfoNCE 数据：每条样本 1 个正例
  python build_ip_training_data_from_merge_json.py \\
      --merge_json /mnt/bn/data/ip_merge.json \\
      --output /mnt/bn/data/train_infonce_merged.jsonl \\
      --loss_type infonce \\
      --num_negatives 20 \\
      --max_samples_per_ip 25 \\
      --num_workers 64

  # 构建 Stage 2 multi-positive / SupCon-style 数据：每条样本 3 个正例
  python build_ip_training_data_from_merge_json.py \\
      --merge_json /mnt/bn/data/ip_merge.json \\
      --output /mnt/bn/data/train_multipos_merged.jsonl \\
      --loss_type multi_positive_infonce \\
      --num_positives 3 \\
      --num_negatives 20 \\
      --max_samples_per_ip 25 \\
      --num_workers 64
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any, Optional


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
USER_IMAGE_MSG = {'role': 'user', 'content': '<image>'}


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


def _scan_one_folder(folder_path: str, recursive: bool) -> tuple[str, list[str], Optional[str]]:
    """Scan one folder and return image paths plus an optional error string."""
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

    images = sorted(set(images))
    return str(folder), images, None


def _file_md5(file_path: str) -> tuple[Optional[str], Optional[str]]:
    """Return file MD5 and optional error string."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                hash_md5.update(chunk)
        return hash_md5.hexdigest(), None
    except (FileNotFoundError, PermissionError, OSError) as exc:
        return None, f'{type(exc).__name__}: {exc}'


def _dedupe_one_ip_by_md5(item: tuple[str, list[str]]) -> tuple[str, list[str], dict[str, Any]]:
    """Deduplicate images within one standard IP by physical MD5."""
    ip_name, images = item
    unique_images = []
    seen_md5: dict[str, str] = {}
    duplicates = []
    errors = []

    for image in images:
        md5, error = _file_md5(image)
        if error is not None:
            errors.append({'image': image, 'error': error})
            continue
        if md5 is None:
            errors.append({'image': image, 'error': 'empty md5'})
            continue
        if md5 in seen_md5:
            duplicates.append({
                'kept_image': seen_md5[md5],
                'duplicate_image': image,
                'md5': md5,
            })
            continue
        seen_md5[md5] = image
        unique_images.append(image)

    stats = {
        'ip_name': ip_name,
        'input_images': len(images),
        'unique_images': len(unique_images),
        'duplicate_images': len(duplicates),
        'error_images': len(errors),
        'duplicate_examples': duplicates[:5],
        'error_examples': errors[:5],
    }
    return ip_name, unique_images, stats


def md5_dedupe_ip_images(ip_to_images: dict[str, list[str]], num_workers: int) -> tuple[dict[str, list[str]],
                                                                                       dict[str, Any]]:
    """Apply MD5 deduplication inside each standard IP."""
    print(f'  MD5 dedup enabled: deduplicating images inside each standard IP with {num_workers} workers ...')
    deduped: dict[str, list[str]] = {}
    per_ip_stats = []

    items = sorted(ip_to_images.items())
    if num_workers <= 1:
        iterator = (_dedupe_one_ip_by_md5(item) for item in items)
        for ip_name, unique_images, stats in tqdm(iterator, total=len(items), desc='MD5 dedup', unit='IP'):
            deduped[ip_name] = unique_images
            per_ip_stats.append(stats)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_dedupe_one_ip_by_md5, item): item[0] for item in items}
            pbar = tqdm(total=len(futures), desc='MD5 dedup', unit='IP')
            for future in as_completed(futures):
                ip_name, unique_images, stats = future.result()
                deduped[ip_name] = unique_images
                per_ip_stats.append(stats)
                pbar.update(1)
            pbar.close()

    input_images = sum(stat['input_images'] for stat in per_ip_stats)
    unique_images = sum(stat['unique_images'] for stat in per_ip_stats)
    duplicate_images = sum(stat['duplicate_images'] for stat in per_ip_stats)
    error_images = sum(stat['error_images'] for stat in per_ip_stats)
    duplicate_rate = duplicate_images / input_images if input_images else 0.0
    top_duplicate_ips = sorted(
        (stat for stat in per_ip_stats if stat['duplicate_images'] > 0),
        key=lambda x: x['duplicate_images'],
        reverse=True,
    )[:20]

    print('  MD5 dedup summary:')
    print(f'    input images:     {input_images:,}')
    print(f'    unique images:    {unique_images:,}')
    print(f'    duplicates:       {duplicate_images:,} ({duplicate_rate:.2%})')
    print(f'    read errors:       {error_images:,}')

    report = {
        'md5_dedup_enabled': True,
        'md5_input_images': input_images,
        'md5_unique_images': unique_images,
        'md5_duplicate_images': duplicate_images,
        'md5_duplicate_rate': duplicate_rate,
        'md5_error_images': error_images,
        'md5_top_duplicate_ips': top_duplicate_ips,
    }
    return deduped, report


def _read_merge_json(path: str) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f'Expected merge JSON root to be a dict, got {type(data).__name__}')
    return data


def _extract_folder_list(ip_key: str, value: Any) -> tuple[str, list[str], list[str]]:
    """Return standard_name, folder_list, merged_keys from one JSON entry."""
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


def _resolve_duplicate_folder_conflicts(ip_to_folders: dict[str, list[str]],
                                        ip_to_merged_keys: dict[str, list[str]],
                                        ip_order: dict[str, int],
                                        strict_duplicates: bool) -> dict[str, Any]:
    """Drop conflicting standard IP groups that share historical folders."""
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
            duplicate_folders.append({
                'folder': folder,
                'first_ip': first_ip,
                'second_ip': owner,
            })

    empty_summary = {
        'duplicate_folders': [],
        'duplicate_folder_count': 0,
        'duplicate_conflict_component_count': 0,
        'duplicate_conflict_dropped_ip_count': 0,
        'duplicate_conflict_kept_ip_count': 0,
        'duplicate_conflict_resolution_examples': [],
    }
    if not duplicate_folders:
        return empty_summary

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
        winner = min(
            component,
            key=lambda ip: (-len(set(ip_to_folders[ip])), ip_order.get(ip, sys.maxsize), ip),
        )
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
            'dropped_folder_counts': {
                ip: len(set(ip_to_folders[ip]))
                for ip in losers
            },
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


def load_ip_groups_from_merge_json(args: argparse.Namespace) -> tuple[dict[str, list[str]], dict[str, Any]]:
    print(f'  Reading merge JSON: {args.merge_json}')
    raw = _read_merge_json(args.merge_json)

    ip_to_folders: dict[str, list[str]] = {}
    ip_to_merged_keys: dict[str, list[str]] = {}
    ip_order: dict[str, int] = {}

    for raw_idx, (ip_key, value) in enumerate(raw.items()):
        standard_name, folders, merged_keys = _extract_folder_list(ip_key, value)
        normalized_folders = []
        for folder in folders:
            folder_path = Path(folder).expanduser()
            if not folder_path.is_absolute() and args.image_root:
                folder_path = Path(args.image_root).expanduser() / folder_path
            folder_str = str(folder_path)
            normalized_folders.append(folder_str)

        if standard_name not in ip_order:
            ip_order[standard_name] = raw_idx
        ip_to_folders.setdefault(standard_name, [])
        ip_to_folders[standard_name].extend(normalized_folders)
        ip_to_merged_keys.setdefault(standard_name, [])
        ip_to_merged_keys[standard_name].extend(merged_keys)

    standard_ips_before_resolution = len(ip_to_folders)
    duplicate_report = _resolve_duplicate_folder_conflicts(
        ip_to_folders,
        ip_to_merged_keys,
        ip_order,
        args.strict_duplicates,
    )

    all_folders = sorted(set(folder for folders in ip_to_folders.values() for folder in folders))
    print(f'  Standard IPs before duplicate resolution: {standard_ips_before_resolution:,}')
    print(f'  Standard IPs after duplicate resolution:  {len(ip_to_folders):,}')
    print(f'  Unique folders to scan: {len(all_folders):,}')

    folder_to_images: dict[str, list[str]] = {}
    scan_errors = []
    print(f'  Scanning images with {args.num_workers} workers ...')
    if args.num_workers <= 1:
        iterator = (_scan_one_folder(folder, args.recursive) for folder in all_folders)
        for folder, images, error in tqdm(iterator, total=len(all_folders), desc='Scanning folders', unit='dir'):
            folder_to_images[folder] = images
            if error is not None:
                scan_errors.append({'folder': folder, 'error': error})
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(_scan_one_folder, folder, args.recursive): folder
                for folder in all_folders
            }
            pbar = tqdm(total=len(futures), desc='Scanning folders', unit='dir')
            for future in as_completed(futures):
                folder, images, error = future.result()
                folder_to_images[folder] = images
                if error is not None:
                    scan_errors.append({'folder': folder, 'error': error})
                pbar.update(1)
            pbar.close()

    ip_to_images_before_filter: dict[str, list[str]] = {}
    for ip_name, folders in ip_to_folders.items():
        images = []
        for folder in sorted(set(folders)):
            images.extend(folder_to_images.get(folder, []))
        images = sorted(set(images))
        ip_to_images_before_filter[ip_name] = images

    md5_report = {'md5_dedup_enabled': False}
    if args.md5_dedup:
        ip_to_images_before_filter, md5_report = md5_dedupe_ip_images(ip_to_images_before_filter, args.num_workers)

    ip_to_images: dict[str, list[str]] = {}
    skipped_small = 0
    for ip_name, images in ip_to_images_before_filter.items():
        if len(images) >= args.min_images:
            ip_to_images[ip_name] = images
        else:
            skipped_small += 1

    merged_ip_count = sum(1 for folders in ip_to_folders.values() if len(set(folders)) > 1)
    report = {
        'merge_json': args.merge_json,
        'standard_ips_in_json': standard_ips_before_resolution,
        'standard_ips_after_duplicate_resolution': len(ip_to_folders),
        'valid_ips': len(ip_to_images),
        'skipped_ips_lt_min_images': skipped_small,
        'unique_folders': len(all_folders),
        'merged_ip_count': merged_ip_count,
        **duplicate_report,
        'scan_errors': scan_errors[:100],
        'scan_error_count': len(scan_errors),
        **md5_report,
        'merged_keys_per_ip': {
            ip: sorted(set(keys))
            for ip, keys in ip_to_merged_keys.items()
            if ip in ip_to_images
        },
    }
    return ip_to_images, report


def _comb_count(n: int, k: int) -> int:
    if k < 0 or n < k:
        return 0
    return math.comb(n, k)


def _sample_infonce_items(images: list[str], max_samples: int, rng: random.Random) -> list[tuple[str, list[str]]]:
    """Generate anchor + one positive samples, preserving old unordered-pair behavior when small."""
    possible = _comb_count(len(images), 2)
    if possible <= max_samples:
        return [(a, [b]) for a, b in combinations(images, 2)]

    pair_set = set()
    while len(pair_set) < max_samples:
        a, b = rng.sample(images, 2)
        key = (a, b) if a < b else (b, a)
        pair_set.add(key)
    return [(a, [b]) for a, b in sorted(pair_set)]


def _sample_multipos_items(images: list[str], num_positives: int, max_samples: int,
                           rng: random.Random) -> list[tuple[str, list[str]]]:
    """Generate anchor + multiple positive samples with bounded enumeration."""
    effective_pos = min(num_positives, len(images) - 1)
    if effective_pos <= 0:
        return []

    possible_per_anchor = _comb_count(len(images) - 1, effective_pos)
    possible_total = len(images) * possible_per_anchor

    if possible_total <= max_samples:
        items = []
        for anchor in images:
            candidates = [img for img in images if img != anchor]
            for positives in combinations(candidates, effective_pos):
                items.append((anchor, list(positives)))
        return items

    item_set = set()
    attempts = 0
    max_attempts = max(1000, max_samples * 50)
    while len(item_set) < max_samples and attempts < max_attempts:
        attempts += 1
        anchor = rng.choice(images)
        candidates = [img for img in images if img != anchor]
        positives = tuple(sorted(rng.sample(candidates, effective_pos)))
        item_set.add((anchor, positives))

    if len(item_set) < max_samples:
        print(f'  WARNING: only sampled {len(item_set)} unique multi-positive items '
              f'from {len(images)} images after {attempts} attempts.')

    return [(anchor, list(positives)) for anchor, positives in sorted(item_set)]


def _sample_negatives(ip_idx: int, ip_names: list[str], ip_to_images: dict[str, list[str]],
                      num_negatives: int, rng: random.Random) -> list[str]:
    if len(ip_names) < 2:
        raise ValueError('Need at least 2 valid standard IPs to sample negatives.')

    negatives = []
    used_images = set()
    attempts = 0
    max_attempts = max(1000, num_negatives * 50)

    while len(negatives) < num_negatives and attempts < max_attempts:
        attempts += 1
        neg_idx = rng.randint(0, len(ip_names) - 1)
        if neg_idx == ip_idx:
            continue
        neg_ip = ip_names[neg_idx]
        neg_img = rng.choice(ip_to_images[neg_ip])
        if neg_img in used_images and len(used_images) < sum(len(ip_to_images[name]) for name in ip_names) - len(
                ip_to_images[ip_names[ip_idx]]):
            continue
        used_images.add(neg_img)
        negatives.append(neg_img)

    while len(negatives) < num_negatives:
        neg_idx = rng.randint(0, len(ip_names) - 1)
        if neg_idx == ip_idx:
            continue
        negatives.append(rng.choice(ip_to_images[ip_names[neg_idx]]))

    return negatives


def _make_sample(anchor: str, positives: list[str], negatives: list[str], include_metadata: bool,
                 include_ip_id: bool, ip_name: Optional[str] = None, ip_id: Optional[int] = None) -> dict[str, Any]:
    sample = {
        'messages': [USER_IMAGE_MSG],
        'images': [anchor],
        'positive_messages': [[USER_IMAGE_MSG] for _ in positives],
        'positive_images': [[image] for image in positives],
        'negative_messages': [[USER_IMAGE_MSG] for _ in negatives],
        'negative_images': [[image] for image in negatives],
    }
    if include_metadata:
        sample['ip_name'] = ip_name
        sample['anchor_image'] = anchor
        sample['positive_image_paths'] = positives
        sample['negative_image_paths'] = negatives
    if include_ip_id:
        sample['ip_id'] = ip_id
        sample['ip_name'] = ip_name
    return sample


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * percentile / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_values[lower])
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def build_training_jsonl(ip_to_images: dict[str, list[str]], args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    ip_names = sorted(ip_to_images.keys())
    ip_to_id = {name: idx for idx, name in enumerate(ip_names)}
    id_to_ip = {str(idx): name for name, idx in ip_to_id.items()}
    loss_type = 'multi_positive_infonce' if args.loss_type == 'supcon' else args.loss_type

    if loss_type == 'infonce' and args.num_positives != 1:
        print('  WARNING: --loss_type infonce only consumes one positive; '
              f'writing 1 positive per sample instead of --num_positives={args.num_positives}.')
    effective_num_positives = 1 if loss_type == 'infonce' else max(1, args.num_positives)

    print('  Generating anchor-positive samples ...')
    all_items: list[tuple[int, str, list[str]]] = []
    samples_per_ip: Counter[str] = Counter()
    positives_per_sample: Counter[int] = Counter()

    for ip_idx, ip_name in enumerate(tqdm(ip_names, desc='Sample generation', unit='IP')):
        images = ip_to_images[ip_name]
        if loss_type == 'infonce':
            items = _sample_infonce_items(images, args.max_samples_per_ip, rng)
        else:
            items = _sample_multipos_items(images, effective_num_positives, args.max_samples_per_ip, rng)

        for anchor, positives in items:
            all_items.append((ip_idx, anchor, positives))
            samples_per_ip[ip_name] += 1
            positives_per_sample[len(positives)] += 1

    print(f'  Total training samples before shuffle: {len(all_items):,}')
    rng.shuffle(all_items)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'  Writing {output_path} with {args.num_negatives} negatives per sample ...')
    count = 0
    with output_path.open('w', encoding='utf-8') as f:
        for ip_idx, anchor, positives in tqdm(all_items, desc='Writing', unit='sample'):
            negatives = _sample_negatives(ip_idx, ip_names, ip_to_images, args.num_negatives, rng)
            sample = _make_sample(
                anchor,
                positives,
                negatives,
                include_metadata=args.include_metadata,
                include_ip_id=args.include_ip_id or args.include_metadata,
                ip_name=ip_names[ip_idx],
                ip_id=ip_to_id[ip_names[ip_idx]],
            )
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            count += 1

    ip_id_map_path = str(output_path) + '.ip_id_map.json'
    with open(ip_id_map_path, 'w', encoding='utf-8') as f:
        json.dump({
            'num_classes': len(ip_names),
            'ip_to_id': ip_to_id,
            'id_to_ip': id_to_ip,
        }, f, ensure_ascii=False, indent=2)

    sample_counts = list(samples_per_ip.values())
    return {
        'output': str(output_path),
        'ip_id_map': ip_id_map_path,
        'num_classes': len(ip_names),
        'loss_type': loss_type,
        'samples': count,
        'num_negatives': args.num_negatives,
        'requested_num_positives': args.num_positives,
        'effective_num_positives': effective_num_positives,
        'max_samples_per_ip': args.max_samples_per_ip,
        'samples_per_ip_min': min(samples_per_ip.values()) if samples_per_ip else 0,
        'samples_per_ip_max': max(samples_per_ip.values()) if samples_per_ip else 0,
        'samples_per_ip_avg': sum(samples_per_ip.values()) / len(samples_per_ip) if samples_per_ip else 0.0,
        'samples_per_ip_p50': _percentile(sample_counts, 50),
        'samples_per_ip_p90': _percentile(sample_counts, 90),
        'samples_per_ip_p95': _percentile(sample_counts, 95),
        'positives_per_sample': dict(sorted(positives_per_sample.items())),
    }


def write_report(output: str, report: dict[str, Any]) -> str:
    report_path = output + '.report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report_path


def _print_distribution(report: dict[str, Any]) -> None:
    print('\nData distribution:')
    print(f'  Standard IPs before duplicate resolution: {report["standard_ips_in_json"]:,}')
    print(f'  Standard IPs after duplicate resolution:  {report["standard_ips_after_duplicate_resolution"]:,}')
    print(f'  Duplicate folders:                        {report["duplicate_folder_count"]:,}')
    print(f'  Duplicate conflict components:            {report["duplicate_conflict_component_count"]:,}')
    print(f'  Dropped standard IPs from conflicts:       {report["duplicate_conflict_dropped_ip_count"]:,}')
    print(f'  Valid IPs after min_images filter:         {report["valid_ips"]:,}')
    print(f'  Total images:                             {report["image_total"]:,}')
    print('  Images/IP:                                '
          f'min={report["images_per_ip_min"]:,}, '
          f'avg={report["images_per_ip_avg"]:.1f}, '
          f'p50={report["images_per_ip_p50"]:.1f}, '
          f'p90={report["images_per_ip_p90"]:.1f}, '
          f'p95={report["images_per_ip_p95"]:.1f}, '
          f'max={report["images_per_ip_max"]:,}')
    print(f'  Training samples:                         {report["samples"]:,}')
    print('  Samples/IP:                               '
          f'min={report["samples_per_ip_min"]:,}, '
          f'avg={report["samples_per_ip_avg"]:.1f}, '
          f'p50={report["samples_per_ip_p50"]:.1f}, '
          f'p90={report["samples_per_ip_p90"]:.1f}, '
          f'p95={report["samples_per_ip_p95"]:.1f}, '
          f'max={report["samples_per_ip_max"]:,}')
    print(f'  Positives/sample distribution:            {report["positives_per_sample"]}')
    print(f'  Num classes:                              {report["num_classes"]:,}')
    print(f'  Output JSONL:                             {report["output"]}')
    print(f'  IP ID map:                                {report["ip_id_map"]}')
    print(f'  Report:                                   {report["report_path"]}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='从 IP 合并标注 JSON 构建 embedding 训练 JSONL。',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--merge_json', type=str, required=True,
                        help='合并标注 JSON，key 为标准 IP，value 内含 文件夹列表。')
    parser.add_argument('--output', type=str, required=True,
                        help='输出训练 JSONL 文件路径。')
    parser.add_argument('--image_root', type=str, default=None,
                        help='当 文件夹列表 中包含相对路径时，用这个根目录拼接。绝对路径不受影响。')
    parser.add_argument('--loss_type', type=str, default='infonce',
                        choices=['infonce', 'multi_positive_infonce', 'supcon'],
                        help='输出数据面向的训练 loss。supcon 是 multi_positive_infonce 的别名。')
    parser.add_argument('--num_positives', type=int, default=1,
                        help='每条样本写入多少个正例。infonce 模式会强制写 1 个。')
    parser.add_argument('--num_negatives', type=int, default=20,
                        help='每条样本采样多少个 explicit negative。')
    parser.add_argument('--max_samples_per_ip', '--max_pairs_per_ip', dest='max_samples_per_ip',
                        type=int, default=25,
                        help='每个标准 IP 最多输出多少条样本。--max_pairs_per_ip 为旧参数兼容别名。')
    parser.add_argument('--min_images', type=int, default=2,
                        help='每个标准 IP 至少需要多少图片。')
    parser.add_argument('--num_workers', type=int, default=64,
                        help='图片文件夹扫描并行进程数。')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子。')
    parser.add_argument('--recursive', action='store_true',
                        help='递归扫描每个文件夹；默认只扫描文件夹一级图片。')
    parser.add_argument('--md5_dedup', action='store_true',
                        help='在每个标准 IP 内按文件内容 MD5 去重；默认只做路径级去重。')
    parser.add_argument('--include_metadata', action='store_true',
                        help='在 JSONL 中额外写入 ip_name/anchor_image 等调试字段。训练通常不需要。')
    parser.add_argument('--include_ip_id', action='store_true',
                        help='在 JSONL 中写入稳定整数 ip_id 和 ip_name，用于后续 Sub-center ArcFace。')
    parser.add_argument('--strict_duplicates', action='store_true',
                        help='如果同一文件夹出现在多个标准 IP 下则报错；默认仅告警。')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_positives < 1:
        raise ValueError('--num_positives must be >= 1')
    if args.num_negatives < 0:
        raise ValueError('--num_negatives must be >= 0')
    if args.max_samples_per_ip < 1:
        raise ValueError('--max_samples_per_ip must be >= 1')
    if args.min_images < 2:
        raise ValueError('--min_images must be >= 2 for positive pair construction')

    t0 = time.time()
    print(f'\n[1/2] Loading merged IP groups')
    ip_to_images, scan_report = load_ip_groups_from_merge_json(args)

    if len(ip_to_images) < 2:
        print('ERROR: Need at least 2 valid standard IPs to build negatives.')
        sys.exit(1)

    counts = [len(images) for images in ip_to_images.values()]
    print(f'  Valid standard IPs: {len(ip_to_images):,}')
    print(f'  Images total:       {sum(counts):,}')
    print(f'  Images/IP:          min={min(counts)}, max={max(counts)}, avg={sum(counts)/len(counts):.1f}')

    print(f'\n[2/2] Building JSONL '
          f'(loss_type={args.loss_type}, positives={args.num_positives}, negatives={args.num_negatives})')
    build_report = build_training_jsonl(ip_to_images, args)

    elapsed = time.time() - t0
    report = {
        **scan_report,
        **build_report,
        'image_total': sum(counts),
        'images_per_ip_min': min(counts),
        'images_per_ip_max': max(counts),
        'images_per_ip_avg': sum(counts) / len(counts),
        'images_per_ip_p50': _percentile(counts, 50),
        'images_per_ip_p90': _percentile(counts, 90),
        'images_per_ip_p95': _percentile(counts, 95),
        'seed': args.seed,
        'elapsed_seconds': round(elapsed, 3),
    }
    report_path = write_report(args.output, report)
    report['report_path'] = report_path

    print(f'\n{"=" * 60}')
    print(f'Done in {elapsed:.1f}s ({elapsed/60:.1f}min)')
    _print_distribution(report)
    print(f'  File size:          {os.path.getsize(args.output) / 1024 / 1024:.1f} MB')
    print('\nTo train:')
    if build_report['loss_type'] == 'infonce':
        print('  INFONCE_USE_BATCH=False INFONCE_TEMPERATURE=0.05 \\')
        print('  swift sft --task_type embedding --loss_type infonce \\')
    else:
        print('  INFONCE_TEMPERATURE=0.05 \\')
        print('  swift sft --task_type embedding --loss_type multi_positive_infonce \\')
    print(f'      --dataset {args.output}')


if __name__ == '__main__':
    main()
