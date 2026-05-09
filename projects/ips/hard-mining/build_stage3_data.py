"""
Stage 3 hard-mining 训练数据构建脚本
==================================

本脚本实现 Patch 4 的第三步：把原始训练 JSONL 与过滤后的 hard positive /
hard negative 候选混合，生成 Stage 3 使用的训练数据。

它做什么：
  1. 读取原始训练 JSONL，保持原有 anchor、positive 和消息结构。
  2. 读取 filtered_hard_positives.jsonl，按当前 anchor image 精确匹配，
     最多追加 --max_extra_positives 个同 IP hard positive。
  3. 读取 filtered_hard_negatives.jsonl，按当前 anchor image 优先匹配 hard
     negative；不足时从全局 hard pool 补齐。
  4. 对 negative 按 curriculum 比例 random:medium:hard 混合，默认 2:3:5。
     - random 来自原始样本已有 negative。
     - medium 若原始样本存在 negative_types=medium 则使用，否则 medium 额度
       自动回退到 random。
     - hard 来自 filtered hard negative。
  5. 输出 multi-positive 兼容 JSONL：positive_images 可以包含多张正例；
     negative_types 标记每个负例来源。

输入：
  - 原始训练数据 JSONL，格式沿用当前 ms-swift embedding 数据：
      messages/images/positive_messages/positive_images/negative_messages/negative_images
  - filtered_hard_positives.jsonl
  - filtered_hard_negatives.jsonl

输出：
  - stage3_train.jsonl
  - build_stage3_report.json

示例：
  python projects/ips/hard-mining/build_stage3_data.py \\
      --train_data /mnt/bn/data/ip/train.jsonl \\
      --hard_positives /mnt/bn/data/ip_hard/filtered_hard_positives.jsonl \\
      --hard_negatives /mnt/bn/data/ip_hard/filtered_hard_negatives.jsonl \\
      --output /mnt/bn/data/ip/stage3_train.jsonl

注意事项：
  - 本脚本不修改 swift 主框架，只生成下一阶段训练数据。
  - hard/medium/random 池不足时不会报错，会按 fallback 自动补齐，并在报告中记录。
  - 输出面向后续 multi_positive_infonce，因此允许多个 positive。
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any


USER_IMAGE_MSG = {'role': 'user', 'content': '<image>'}


def read_jsonl(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(obj: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def get_anchor_image(record: dict[str, Any]) -> str | None:
    images = record.get('images')
    if images:
        return images[0]
    return record.get('image_path')


def get_ip_from_path(path: str | None) -> str | None:
    if not path:
        return None
    parent = os.path.basename(os.path.dirname(path))
    return parent or None


def path_keys(path: str | None) -> list[str]:
    if not path:
        return []
    keys = [path]
    try:
        abs_path = os.path.abspath(path)
        if abs_path != path:
            keys.append(abs_path)
    except OSError:
        pass
    return keys


def index_by_path(records: list[dict[str, Any]], image_key: str) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        image = record.get(image_key)
        for key in path_keys(image):
            indexed[key].append(record)
    return indexed


def extract_positive_items(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = record.get('positive_messages') or []
    images = record.get('positive_images') or []
    items = []
    for idx, image_group in enumerate(images):
        if not image_group:
            continue
        msg = messages[idx] if idx < len(messages) else [USER_IMAGE_MSG]
        items.append({'image': image_group[0], 'message': msg})
    return items


def extract_negative_items(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = record.get('negative_messages') or []
    images = record.get('negative_images') or []
    types = record.get('negative_types') or []
    items = []
    for idx, image_group in enumerate(images):
        if not image_group:
            continue
        msg = messages[idx] if idx < len(messages) else [USER_IMAGE_MSG]
        neg_type = types[idx] if idx < len(types) else 'random'
        items.append({'image': image_group[0], 'message': msg, 'type': neg_type})
    return items


def make_item(image: str, source_type: str) -> dict[str, Any]:
    return {'image': image, 'message': [USER_IMAGE_MSG], 'type': source_type}


def parse_mix(mix: str) -> tuple[int, int, int]:
    parts = mix.split(':')
    if len(parts) != 3:
        raise ValueError('--negative_mix must be formatted as random:medium:hard, e.g. 2:3:5')
    values = tuple(int(p) for p in parts)
    if sum(values) <= 0 or any(v < 0 for v in values):
        raise ValueError('--negative_mix values must be non-negative and sum to > 0')
    return values


def allocate_counts(total: int, mix: tuple[int, int, int]) -> dict[str, int]:
    names = ['random', 'medium', 'hard']
    weights = dict(zip(names, mix))
    weight_sum = sum(mix)
    raw = {name: total * weights[name] / weight_sum for name in names}
    counts = {name: int(raw[name]) for name in names}
    remain = total - sum(counts.values())
    priority = {'hard': 2, 'medium': 1, 'random': 0}
    order = sorted(names, key=lambda n: (raw[n] - counts[n], priority[n]), reverse=True)
    for name in order[:remain]:
        counts[name] += 1
    return counts


def item_key(item: dict[str, Any]) -> str:
    return item['image']


def sample_items(pool: list[dict[str, Any]], count: int, rng: random.Random,
                 selected: set[str], anchor_image: str | None) -> list[dict[str, Any]]:
    candidates = [
        item for item in pool
        if item.get('image') and item['image'] != anchor_image and item_key(item) not in selected
    ]
    if not candidates or count <= 0:
        return []
    if len(candidates) <= count:
        chosen = candidates[:]
    else:
        chosen = rng.sample(candidates, count)
    for item in chosen:
        selected.add(item_key(item))
    return chosen


def hard_negative_items_for_anchor(anchor_image: str | None,
                                   hard_by_anchor: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if not anchor_image:
        return []
    merged = []
    seen = set()
    for key in path_keys(anchor_image):
        for record in hard_by_anchor.get(key, []):
            image = record.get('negative_image')
            if image and image not in seen:
                merged.append(record)
                seen.add(image)
    merged.sort(key=lambda r: float(r.get('similarity', 0.0)), reverse=True)
    return [make_item(record['negative_image'], 'hard') for record in merged]


def hard_positive_items_for_anchor(anchor_image: str | None,
                                   hard_by_anchor: dict[str, list[dict[str, Any]]],
                                   max_extra: int,
                                   existing: set[str]) -> list[dict[str, Any]]:
    if not anchor_image or max_extra <= 0:
        return []
    merged = []
    seen = set()
    for key in path_keys(anchor_image):
        for record in hard_by_anchor.get(key, []):
            image = record.get('positive_image')
            if image and image not in seen and image not in existing and image != anchor_image:
                merged.append(record)
                seen.add(image)
    merged.sort(key=lambda r: float(r.get('similarity', 1.0)))
    return [{'image': record['positive_image'], 'message': [USER_IMAGE_MSG]} for record in merged[:max_extra]]


def build_stage3(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    train_records = read_jsonl(args.train_data)
    hard_pos = read_jsonl(args.hard_positives)
    hard_neg = read_jsonl(args.hard_negatives)
    hard_pos_by_anchor = index_by_path(hard_pos, 'query_image')
    hard_neg_by_anchor = index_by_path(hard_neg, 'query_image')
    global_hard_pool = [make_item(r['negative_image'], 'hard') for r in hard_neg if r.get('negative_image')]

    all_original_negatives = []
    for record in train_records:
        all_original_negatives.extend(extract_negative_items(record))
    global_random_pool = [dict(item, type='random') for item in all_original_negatives]
    global_medium_pool = [dict(item, type='medium') for item in all_original_negatives if item.get('type') == 'medium']

    mix = parse_mix(args.negative_mix)
    target_counts = allocate_counts(args.num_negatives, mix)
    report: Counter[str] = Counter()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for record in train_records:
            out = deepcopy(record)
            anchor_image = get_anchor_image(record)

            positives = extract_positive_items(record)
            existing_positive_images = {item['image'] for item in positives}
            extra_pos = hard_positive_items_for_anchor(
                anchor_image, hard_pos_by_anchor, args.max_extra_positives, existing_positive_images)
            positives.extend(extra_pos)
            report['extra_positives_added'] += len(extra_pos)

            original_negs = extract_negative_items(record)
            sample_random = [dict(item, type='random') for item in original_negs if item.get('type') != 'medium']
            sample_medium = [dict(item, type='medium') for item in original_negs if item.get('type') == 'medium']
            if not sample_random:
                sample_random = [dict(item, type='random') for item in original_negs]

            selected: set[str] = set()
            chosen = []

            hard_pool = hard_negative_items_for_anchor(anchor_image, hard_neg_by_anchor)
            hard_count = target_counts['hard']
            hard_chosen = sample_items(hard_pool, hard_count, rng, selected, anchor_image)
            if len(hard_chosen) < hard_count:
                report['hard_exact_shortage'] += hard_count - len(hard_chosen)
                hard_chosen.extend(sample_items(global_hard_pool, hard_count - len(hard_chosen), rng, selected,
                                                anchor_image))
            chosen.extend(hard_chosen)

            medium_count = target_counts['medium']
            random_count = target_counts['random']
            if sample_medium:
                medium_chosen = sample_items(sample_medium, medium_count, rng, selected, anchor_image)
                if len(medium_chosen) < medium_count:
                    report['medium_shortage'] += medium_count - len(medium_chosen)
                    medium_chosen.extend(sample_items(global_medium_pool, medium_count - len(medium_chosen), rng,
                                                      selected, anchor_image))
                chosen.extend(medium_chosen)
            else:
                report['medium_fallback_to_random'] += medium_count
                random_count += medium_count

            random_chosen = sample_items(sample_random, random_count, rng, selected, anchor_image)
            if len(random_chosen) < random_count:
                report['random_shortage'] += random_count - len(random_chosen)
                random_chosen.extend(sample_items(global_random_pool, random_count - len(random_chosen), rng,
                                                  selected, anchor_image))
            chosen.extend(random_chosen)

            if len(chosen) < args.num_negatives:
                report['final_negative_shortage'] += args.num_negatives - len(chosen)
                chosen.extend(sample_items(global_random_pool + global_hard_pool, args.num_negatives - len(chosen),
                                           rng, selected, anchor_image))
            chosen = chosen[:args.num_negatives]

            out['positive_messages'] = [item['message'] for item in positives]
            out['positive_images'] = [[item['image']] for item in positives]
            out['negative_messages'] = [item['message'] for item in chosen]
            out['negative_images'] = [[item['image']] for item in chosen]
            out['negative_types'] = [item.get('type', 'random') for item in chosen]

            report['samples_written'] += 1
            report['total_positives'] += len(positives)
            report['total_negatives'] += len(chosen)
            for neg_type in out['negative_types']:
                report[f'negative_type_{neg_type}'] += 1

            f.write(json.dumps(out, ensure_ascii=False) + '\n')

    report_dict = dict(report)
    report_dict.update({
        'train_data': args.train_data,
        'hard_positives': args.hard_positives,
        'hard_negatives': args.hard_negatives,
        'output': args.output,
        'num_input_samples': len(train_records),
        'num_hard_positive_records': len(hard_pos),
        'num_hard_negative_records': len(hard_neg),
        'num_negatives_target': args.num_negatives,
        'negative_mix': args.negative_mix,
        'allocated_negative_counts': target_counts,
        'max_extra_positives': args.max_extra_positives,
        'seed': args.seed,
    })
    return report_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build Stage 3 IP embedding training JSONL with filtered hard samples.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--train_data', type=str, required=True,
                        help='Original training JSONL.')
    parser.add_argument('--hard_positives', type=str, required=True,
                        help='Filtered hard positives JSONL.')
    parser.add_argument('--hard_negatives', type=str, required=True,
                        help='Filtered hard negatives JSONL.')
    parser.add_argument('--output', type=str, required=True,
                        help='Output stage3_train.jsonl.')
    parser.add_argument('--num_negatives', type=int, default=5,
                        help='Number of negatives per output sample.')
    parser.add_argument('--negative_mix', type=str, default='2:3:5',
                        help='Curriculum mix random:medium:hard.')
    parser.add_argument('--max_extra_positives', type=int, default=1,
                        help='Max hard positives to append per anchor.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--report', type=str, default=None,
                        help='Optional report path; default is build_stage3_report.json next to output.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_stage3(args)
    report_path = args.report or os.path.join(os.path.dirname(os.path.abspath(args.output)),
                                              'build_stage3_report.json')
    write_json(report, report_path)
    print('Done.')
    print(f'  output: {args.output}')
    print(f'  report: {report_path}')
    print(f'  samples: {report["samples_written"]:,}')
    print(f'  negatives: {report["total_negatives"]:,}')
    print(f'  extra positives added: {report["extra_positives_added"]:,}')


if __name__ == '__main__':
    main()
