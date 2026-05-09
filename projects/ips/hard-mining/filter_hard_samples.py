"""
IP hard sample 规则过滤脚本
==========================

本脚本实现 Patch 4 的第二步：对 mine_hard_samples.py 输出的 hard positive /
hard negative 候选做本地规则过滤，并兼容外部 VLM/reranker 预打分结果。

它做什么：
  1. 读取 hard_positives.jsonl 与 hard_negatives.jsonl。
  2. 对 positive 候选做规则过滤：
     - query_ip 必须等于 positive_ip
     - query_image 与 positive_image 不能相同
     - similarity 不低于 --min_positive_sim
  3. 对 negative 候选做规则过滤：
     - query_ip 必须不同于 negative_ip
     - query_image 与 negative_image 不能相同
     - similarity 不低于 --min_negative_sim
  4. 如果候选记录包含外部判定字段，则继续过滤：
     - verdict: positive 只接受 same_ip/positive/match/yes 等；negative
       只接受 different_ip/negative/no_match/no 等。
     - vlm_score 或 reranker_score: 视为“同 IP 概率/置信度”。
       positive 要求分数 >= --positive_score_threshold；
       negative 要求分数 <= --negative_same_ip_score_max。
  5. 按 pair_id 去重，输出过滤后的 JSONL 和统计报告。

输入：
  - hard_positives.jsonl
  - hard_negatives.jsonl

输出：
  - filtered_hard_positives.jsonl
  - filtered_hard_negatives.jsonl
  - filter_report.json

示例：
  python projects/ips/hard-mining/filter_hard_samples.py \\
      --hard_positives /path/hard_positives.jsonl \\
      --hard_negatives /path/hard_negatives.jsonl \\
      --output_dir /path/filtered

注意事项：
  - 第一版不会主动调用 VLM/reranker，只消费记录里已经存在的 verdict /
    vlm_score / reranker_score 字段。
  - 对 hard negative，外部分数越高越像“同 IP”，因此要求低于阈值。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from typing import Any


POSITIVE_VERDICTS = {'same_ip', 'positive', 'match', 'matched', 'same', 'yes', 'true', '1'}
NEGATIVE_VERDICTS = {
    'different_ip', 'different', 'diff_ip', 'negative', 'not_match', 'no_match', 'mismatch', 'no', 'false', '0'
}


def read_jsonl(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
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


def pair_id(kind: str, query_image: str, candidate_image: str) -> str:
    raw = f'{kind}\t{query_image}\t{candidate_image}'.encode('utf-8')
    return hashlib.sha1(raw).hexdigest()[:16]


def get_similarity(record: dict[str, Any]) -> float | None:
    value = record.get('similarity')
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_same_ip_score(record: dict[str, Any]) -> float | None:
    scores = []
    for key in ('vlm_score', 'reranker_score'):
        if key not in record or record[key] is None:
            continue
        try:
            scores.append(float(record[key]))
        except (TypeError, ValueError):
            continue
    if not scores:
        return None
    return max(scores)


def normalize_verdict(record: dict[str, Any]) -> str | None:
    verdict = record.get('verdict')
    if verdict is None:
        return None
    return str(verdict).strip().lower().replace('-', '_').replace(' ', '_')


def reject(reason: str, counter: Counter[str]) -> tuple[bool, str]:
    counter[reason] += 1
    return False, reason


def keep_positive(record: dict[str, Any], args: argparse.Namespace,
                  counter: Counter[str], seen: set[str]) -> tuple[bool, str]:
    q_ip = str(record.get('query_ip', ''))
    p_ip = str(record.get('positive_ip', ''))
    q_image = record.get('query_image')
    p_image = record.get('positive_image')
    if not q_ip or not p_ip or q_ip != p_ip:
        return reject('positive_ip_mismatch', counter)
    if not q_image or not p_image or q_image == p_image:
        return reject('positive_same_or_missing_image', counter)

    sim = get_similarity(record)
    if sim is None or sim < args.min_positive_sim:
        return reject('positive_similarity_too_low', counter)

    pid = record.get('pair_id') or pair_id('pos', q_image, p_image)
    if pid in seen:
        return reject('positive_duplicate_pair_id', counter)

    verdict = normalize_verdict(record)
    if verdict is not None and verdict not in POSITIVE_VERDICTS:
        return reject('positive_verdict_rejected', counter)

    score = get_same_ip_score(record)
    if score is not None and score < args.positive_score_threshold:
        return reject('positive_external_score_too_low', counter)

    record['pair_id'] = pid
    seen.add(pid)
    return True, 'kept'


def keep_negative(record: dict[str, Any], args: argparse.Namespace,
                  counter: Counter[str], seen: set[str]) -> tuple[bool, str]:
    q_ip = str(record.get('query_ip', ''))
    n_ip = str(record.get('negative_ip', ''))
    q_image = record.get('query_image')
    n_image = record.get('negative_image')
    if not q_ip or not n_ip or q_ip == n_ip:
        return reject('negative_ip_not_different', counter)
    if not q_image or not n_image or q_image == n_image:
        return reject('negative_same_or_missing_image', counter)

    sim = get_similarity(record)
    if sim is None or sim < args.min_negative_sim:
        return reject('negative_similarity_too_low', counter)

    pid = record.get('pair_id') or pair_id('neg', q_image, n_image)
    if pid in seen:
        return reject('negative_duplicate_pair_id', counter)

    verdict = normalize_verdict(record)
    if verdict is not None and verdict not in NEGATIVE_VERDICTS:
        return reject('negative_verdict_rejected', counter)

    score = get_same_ip_score(record)
    if score is not None and score > args.negative_same_ip_score_max:
        return reject('negative_external_score_too_high', counter)

    record['pair_id'] = pid
    seen.add(pid)
    return True, 'kept'


def filter_records(records: list[dict[str, Any]], kind: str,
                   args: argparse.Namespace) -> tuple[list[dict[str, Any]], Counter[str]]:
    kept = []
    counter: Counter[str] = Counter()
    seen: set[str] = set()
    for record in records:
        if kind == 'positive':
            ok, _ = keep_positive(record, args, counter, seen)
        else:
            ok, _ = keep_negative(record, args, counter, seen)
        if ok:
            kept.append(record)
            counter['kept'] += 1
    counter['input'] = len(records)
    return kept, counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Filter mined hard IP samples with rules and optional external scores.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--hard_positives', type=str, required=True,
                        help='Input hard_positives.jsonl.')
    parser.add_argument('--hard_negatives', type=str, required=True,
                        help='Input hard_negatives.jsonl.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory.')
    parser.add_argument('--min_positive_sim', type=float, default=0.05,
                        help='Minimum embedding similarity for hard positives.')
    parser.add_argument('--min_negative_sim', type=float, default=0.30,
                        help='Minimum embedding similarity for hard negatives.')
    parser.add_argument('--positive_score_threshold', type=float, default=0.70,
                        help='Minimum external same-IP score for positives.')
    parser.add_argument('--negative_same_ip_score_max', type=float, default=0.30,
                        help='Maximum external same-IP score allowed for negatives.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    positives = read_jsonl(args.hard_positives)
    negatives = read_jsonl(args.hard_negatives)

    filtered_pos, pos_counter = filter_records(positives, 'positive', args)
    filtered_neg, neg_counter = filter_records(negatives, 'negative', args)

    pos_path = os.path.join(args.output_dir, 'filtered_hard_positives.jsonl')
    neg_path = os.path.join(args.output_dir, 'filtered_hard_negatives.jsonl')
    report_path = os.path.join(args.output_dir, 'filter_report.json')
    write_jsonl(filtered_pos, pos_path)
    write_jsonl(filtered_neg, neg_path)

    report = {
        'inputs': {
            'hard_positives': args.hard_positives,
            'hard_negatives': args.hard_negatives,
        },
        'outputs': {
            'filtered_hard_positives': pos_path,
            'filtered_hard_negatives': neg_path,
        },
        'thresholds': {
            'min_positive_sim': args.min_positive_sim,
            'min_negative_sim': args.min_negative_sim,
            'positive_score_threshold': args.positive_score_threshold,
            'negative_same_ip_score_max': args.negative_same_ip_score_max,
        },
        'positive_counts': dict(pos_counter),
        'negative_counts': dict(neg_counter),
    }
    write_json(report, report_path)

    print('Done.')
    print(f'  positives: {len(filtered_pos):,}/{len(positives):,} -> {pos_path}')
    print(f'  negatives: {len(filtered_neg):,}/{len(negatives):,} -> {neg_path}')
    print(f'  report:    {report_path}')


if __name__ == '__main__':
    main()
