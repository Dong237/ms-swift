#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
编码 IP 评估 Manifest 为 Embedding JSONL
=======================================

本脚本是 IP embedding 评估闭环的第 2 步：读取 build_ip_eval_manifest.py 生成的
manifest，对其中每张图片调用 embedding 模型推理，并输出带 embedding 的扁平 JSONL。

为什么拆成独立脚本：
  - embedding 推理最耗时，单独缓存后可以用同一份 embedding 反复计算不同 K、
    不同过滤规则或不同错误分析报告。
  - manifest 的 split/is_query/is_gallery 元信息会原样保留，评估脚本不需要再读取
    原始 train/test JSON。

输入 manifest JSONL：
  {
    "record_id": "test:000000123",
    "image_path": "/abs/path/img.jpg",
    "ip_name": "蜡笔小新_樱田妮妮",
    "split": "test",
    "source_folder": "/abs/path/folder",
    "is_query": true,
    "is_gallery": true
  }

输出 embeddings JSONL：
  {
    "record_id": "test:000000123",
    "image_path": "/abs/path/img.jpg",
    "ip_name": "蜡笔小新_樱田妮妮",
    "split": "test",
    "source_folder": "/abs/path/folder",
    "is_query": true,
    "is_gallery": true,
    "embedding": [0.01, ...]
  }

推理后端：
  - backend=swift：使用 ms-swift TransformersEngine(task_type='embedding')，适合
    Qwen3.5 embedding / Qwen-VL embedding checkpoint。

多设备方式：
  主进程按 round-robin 把 manifest 切成 shard，每个 worker 子进程只看到一个设备
  CUDA_VISIBLE_DEVICES 或 ASCEND_RT_VISIBLE_DEVICES，并将自己的 shard 写为临时 JSONL。
  全部 worker 成功后主进程合并输出；若任意 worker 失败则整体失败，避免静默缺图。

示例：
  python projects/ips/evaluation/encode_ip_eval_manifest.py \\
      --manifest /mnt/bn/.../eval_manifest.jsonl \\
      --model /mnt/bn/.../checkpoint-best \\
      --model_type qwen3_5_emb \\
      --output /mnt/bn/.../eval_embeddings.jsonl \\
      --num_devices 8 \\
      --batch_size 64
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from typing import Any


DEFAULT_INSTRUCTION = '提取该IP角色的身份特征，关注角色本身而非背景或姿态'


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


def try_import_torch_npu() -> bool:
    try:
        import torch_npu  # noqa: F401
        return True
    except Exception:
        return False


def cuda_device_count() -> int:
    import torch
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def npu_device_count() -> int:
    import torch
    if not try_import_torch_npu() or not hasattr(torch, 'npu'):
        return 0
    try:
        return int(torch.npu.device_count())
    except Exception:
        return 0


def resolve_device_backend(device_backend: str) -> str:
    if device_backend in {'cuda', 'npu'}:
        return device_backend
    if device_backend != 'auto':
        raise ValueError(f'Unsupported device backend: {device_backend}')
    if cuda_device_count() > 0:
        return 'cuda'
    if npu_device_count() > 0:
        return 'npu'
    try:
        from swift.utils import get_device_count
        if int(get_device_count()) > 0:
            return 'npu'
    except Exception:
        pass
    raise RuntimeError('No CUDA GPU or Ascend NPU device detected.')


def detect_device_count(device_backend: str) -> int:
    if device_backend == 'cuda':
        return cuda_device_count()
    if device_backend == 'npu':
        count = npu_device_count()
        if count == 0:
            try:
                from swift.utils import get_device_count
                count = int(get_device_count())
            except Exception:
                pass
        return count
    raise ValueError(f'Unsupported device backend: {device_backend}')


def configure_visible_device(rank: int, device_backend: str, env: dict[str, str]) -> dict[str, str]:
    env = env.copy()
    if device_backend == 'cuda':
        env['CUDA_VISIBLE_DEVICES'] = str(rank)
        env.pop('ASCEND_RT_VISIBLE_DEVICES', None)
    elif device_backend == 'npu':
        env['ASCEND_RT_VISIBLE_DEVICES'] = str(rank)
        env['CUDA_VISIBLE_DEVICES'] = ''
    else:
        raise ValueError(f'Unsupported device backend: {device_backend}')
    return env


def get_torch_dtype(dtype_name: str):
    import torch
    if dtype_name == 'auto':
        return torch.bfloat16
    if dtype_name == 'float32':
        return torch.float32
    if dtype_name == 'float16':
        return torch.float16
    if dtype_name == 'bfloat16':
        return torch.bfloat16
    raise ValueError(f'Unsupported torch dtype: {dtype_name}')


def build_swift_infer_request(image_path: str, instruction: str):
    from swift.infer_engine import InferRequest
    content = [{'type': 'image', 'image': image_path}]
    messages = []
    if instruction:
        messages.append({'role': 'system', 'content': instruction})
    messages.append({'role': 'user', 'content': content})
    return InferRequest(messages=messages)


def run_swift_worker(worker_args: dict[str, Any]) -> None:
    import torch

    device_backend = worker_args['device_backend']
    if device_backend == 'npu':
        try_import_torch_npu()
        if hasattr(torch, 'npu'):
            try:
                torch.npu.set_device(0)
            except Exception:
                torch.npu.set_device('npu:0')
    elif device_backend == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(0)

    from swift.infer_engine import TransformersEngine

    rows = read_jsonl(worker_args['shard_input'])
    rank = worker_args['rank']
    print(f'[{device_backend.upper()} {rank}] {len(rows)} images, loading model ...')
    engine = TransformersEngine(
        worker_args['model'],
        model_type=worker_args['model_type'],
        task_type='embedding',
        torch_dtype=get_torch_dtype(worker_args['torch_dtype']),
    )
    print(f'[{device_backend.upper()} {rank}] Model loaded. Starting inference ...')

    t0 = time.time()
    done = 0
    total = len(rows)
    batch_size = worker_args['batch_size']
    with open(worker_args['shard_output'], 'w', encoding='utf-8') as f:
        for start in range(0, total, batch_size):
            batch_rows = rows[start:start + batch_size]
            requests = [
                build_swift_infer_request(row['image_path'], worker_args['instruction'])
                for row in batch_rows
            ]
            responses = engine.infer(requests)
            for row, resp in zip(batch_rows, responses):
                emb = resp.data[0].embedding
                if hasattr(emb, 'tolist'):
                    emb = emb.tolist()
                out = dict(row)
                out['embedding'] = emb
                f.write(json.dumps(out, ensure_ascii=False) + '\n')
            done += len(batch_rows)
            elapsed = time.time() - t0
            speed = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / speed if speed > 0 else 0
            print(f'[{device_backend.upper()} {rank}] {done}/{total} '
                  f'({done / total * 100:.1f}%) | {speed:.0f} img/s | ETA {eta:.0f}s   ', end='\r')
    elapsed = time.time() - t0
    print(f'\n[{device_backend.upper()} {rank}] Done: {done} images in {elapsed:.1f}s')


def encode_manifest(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(args.manifest)
    if not rows:
        raise RuntimeError(f'Manifest is empty: {args.manifest}')
    if args.reuse_existing and os.path.exists(args.output) and os.path.getsize(args.output) > 0:
        print(f'Reusing existing embeddings: {args.output}')
        return read_jsonl(args.output)

    if not args.model:
        raise ValueError('--model is required unless --reuse_existing points to an existing --output.')

    args.device_backend = resolve_device_backend(args.device_backend)
    if args.num_devices is None:
        args.num_devices = detect_device_count(args.device_backend)
    if args.num_devices <= 0:
        raise RuntimeError('No accelerator devices detected.')

    print('\nEncoding manifest:')
    print(f'  Manifest:       {args.manifest}')
    print(f'  Output:         {args.output}')
    print(f'  Backend:        {args.backend}')
    print(f'  Device backend: {args.device_backend}')
    print(f'  Devices:        {args.num_devices}')
    print(f'  Batch/device:   {args.batch_size}')
    print(f'  Model:          {args.model}')
    print(f'  Model type:     {args.model_type}')
    print(f'  Instruction:    {args.instruction!r}')

    tmp_dir = tempfile.mkdtemp(prefix='ip_eval_encode_')
    shard_inputs = []
    shard_outputs = []
    script_path = os.path.abspath(__file__)
    try:
        for rank in range(args.num_devices):
            shard = rows[rank::args.num_devices]
            shard_input = os.path.join(tmp_dir, f'shard_in_{rank}.jsonl')
            shard_output = os.path.join(tmp_dir, f'shard_out_{rank}.jsonl')
            write_jsonl(shard_input, shard)
            shard_inputs.append(shard_input)
            shard_outputs.append(shard_output)

        procs = []
        for rank in range(args.num_devices):
            worker_args = {
                'backend': args.backend,
                'device_backend': args.device_backend,
                'rank': rank,
                'model': args.model,
                'model_type': args.model_type,
                'batch_size': args.batch_size,
                'instruction': args.instruction,
                'shard_input': shard_inputs[rank],
                'shard_output': shard_outputs[rank],
                'torch_dtype': args.torch_dtype,
            }
            cmd = [sys.executable, script_path, '--__worker__', json.dumps(worker_args)]
            procs.append(subprocess.Popen(cmd, env=configure_visible_device(rank, args.device_backend, os.environ)))

        for proc in procs:
            proc.wait()
        failed = [idx for idx, proc in enumerate(procs) if proc.returncode != 0]
        if failed:
            raise RuntimeError(f'Encoding workers failed: {failed}')

        encoded = []
        for shard_output in shard_outputs:
            encoded.extend(read_jsonl(shard_output))
        if len(encoded) != len(rows):
            raise RuntimeError(f'Expected {len(rows)} encoded rows, got {len(encoded)}.')
        encoded.sort(key=lambda row: row['record_id'])
        write_jsonl(args.output, encoded)
        return encoded
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def print_distribution(rows: list[dict[str, Any]], output: str, elapsed: float) -> None:
    split_counts = Counter(row.get('split', 'unknown') for row in rows)
    ip_counts = Counter(row['ip_name'] for row in rows)
    dims = Counter(len(row['embedding']) for row in rows if 'embedding' in row)
    query_count = sum(1 for row in rows if row.get('is_query'))
    gallery_count = sum(1 for row in rows if row.get('is_gallery'))
    print('\nEncoded data distribution:')
    print(f'  Output:            {output}')
    print(f'  Rows/images:       {len(rows):,}')
    print(f'  Split counts:      {dict(split_counts)}')
    print(f'  Query rows:        {query_count:,}')
    print(f'  Gallery rows:      {gallery_count:,}')
    print(f'  Unique IPs:        {len(ip_counts):,}')
    print(f'  Embedding dims:    {dict(dims)}')
    print(f'  Done in:           {elapsed:.1f}s')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Encode Protocol-A IP eval manifest with an embedding model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--__worker__', type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--manifest', type=str, required=False)
    parser.add_argument('--output', type=str, required=False)
    parser.add_argument('--backend', choices=['swift'], default='swift')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--model_type', type=str, default='qwen3_5_emb')
    parser.add_argument('--instruction', type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument('--device_backend', choices=['auto', 'cuda', 'npu'], default='auto')
    parser.add_argument('--num_devices', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--torch_dtype', choices=['auto', 'float32', 'float16', 'bfloat16'], default='auto')
    parser.add_argument('--reuse_existing', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.__worker__:
        worker_args = json.loads(args.__worker__)
        run_swift_worker(worker_args)
        return
    if not args.manifest or not args.output:
        raise SystemExit('Main mode requires --manifest and --output.')
    if args.batch_size <= 0:
        raise ValueError('--batch_size must be positive.')
    t0 = time.time()
    rows = encode_manifest(args)
    report = {
        'manifest': args.manifest,
        'output': args.output,
        'backend': args.backend,
        'model': args.model,
        'model_type': args.model_type,
        'instruction': args.instruction,
        'device_backend': getattr(args, 'device_backend', None),
        'num_devices': getattr(args, 'num_devices', None),
        'batch_size': args.batch_size,
        'row_count': len(rows),
        'split_counts': dict(Counter(row.get('split', 'unknown') for row in rows)),
        'query_count': sum(1 for row in rows if row.get('is_query')),
        'gallery_count': sum(1 for row in rows if row.get('is_gallery')),
        'unique_ip_count': len({row['ip_name'] for row in rows}),
        'embedding_dims': dict(Counter(len(row['embedding']) for row in rows if 'embedding' in row)),
        'elapsed_seconds': round(time.time() - t0, 3),
    }
    write_json(args.output + '.report.json', report)
    print_distribution(rows, args.output, time.time() - t0)
    print(f'  Report:            {args.output}.report.json')


if __name__ == '__main__':
    main()
