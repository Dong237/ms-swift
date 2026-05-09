#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified IP Embedding Retrieval Evaluation

Supports two embedding backends with the SAME evaluation protocol, on either CUDA GPU or Ascend NPU:

1) swift
   - Your ms-swift embedding models.
   - Uses TransformersEngine(task_type='embedding').

2) danbooru_clip
   - The Danbooru CLIP model/checkpoint from the earlier V6 script.
   - Uses CLIPModel.from_pretrained(local model dir) + checkpoint .pth.

Evaluation protocol:
  - Scan IP folders under --test_dir.
  - Embed every image.
  - Put all images in one retrieval pool.
  - Mask self-match.
  - Images from IPs with >=2 images are queries.
  - Images from 1-image IPs stay in the gallery as distractors, but are not queries.
  - Report CMC@K, Recall@K, mAP@K, and R-Precision.

Embedding JSONL format:
  {"ip_name": "ip_001", "embeddings": {"/path/img1.jpg": [0.01, ...]}}

Example: evaluate your ms-swift model
  python eval_ip_retrieval_unified.py \
      --backend swift \
      --model /path/to/swift/checkpoint_or_model_dir \
      --model_type qwen3_5_emb \
      --test_dir /mnt/bn/youxiang-lf/data/facial_ip/IP_image_val \
      --output /path/to/swift_eval_embeddings.jsonl \
      --K 10

Example: evaluate Danbooru CLIP with your evaluation protocol
  python eval_ip_retrieval_unified.py \
      --backend danbooru_clip \
      --model /path/to/danbooru_clip_best_v6.pth \
      --clip_model_dir /path/to/danbooru_clip \
      --test_dir /mnt/bn/jinghan-lqa/data/IP/IP_image_val \
      --output /path/to/danbooru_eval_embeddings.jsonl \
      --batch_size 128 \
      --K 10

Example: re-evaluate from saved embeddings
  python eval_ip_retrieval_unified.py \
      --embeddings /path/to/eval_embeddings.jsonl \
      --K 10
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Iterable, List, Tuple

import numpy as np


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
DEFAULT_INSTRUCTION = '提取该IP角色的身份特征，关注角色本身而非背景或姿态'


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def read_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[dict], path: str) -> None:
    ensure_parent_dir(path)
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def normalize_state_dict_keys(state_dict: dict) -> dict:
    """Remove DataParallel 'module.' prefix if present."""
    out = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            out[k[7:]] = v
        else:
            out[k] = v
    return out


def get_torch_dtype(dtype_name: str, backend: str):
    import torch

    if dtype_name == 'auto':
        # Preserve the behavior of the original scripts:
        # swift used bfloat16; Danbooru CLIP eval used normal fp32.
        return torch.bfloat16 if backend == 'swift' else torch.float32
    if dtype_name == 'float32':
        return torch.float32
    if dtype_name == 'float16':
        return torch.float16
    if dtype_name == 'bfloat16':
        return torch.bfloat16
    raise ValueError(f'Unsupported torch dtype: {dtype_name}')


def try_import_torch_npu() -> bool:
    """Import torch_npu when available. Returns True if NPU runtime is importable."""
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
    if not try_import_torch_npu():
        return 0
    if not hasattr(torch, 'npu'):
        return 0
    try:
        return int(torch.npu.device_count())
    except Exception:
        return 0


def resolve_device_backend(device_backend: str, prefer_swift: bool = False) -> str:
    """Resolve auto/cuda/npu into a concrete device backend."""
    if device_backend in {'cuda', 'npu'}:
        return device_backend
    if device_backend != 'auto':
        raise ValueError(f'Unsupported device backend: {device_backend}')

    # For auto, prefer CUDA first when both are present.
    # Set --device_backend npu to force Ascend NPU on a mixed machine.
    if cuda_device_count() > 0:
        return 'cuda'
    if npu_device_count() > 0:
        return 'npu'

    if prefer_swift:
        try:
            from swift.utils import get_device_count
            if int(get_device_count()) > 0:
                return 'npu'
        except Exception:
            pass

    raise RuntimeError('No CUDA GPU or Ascend NPU device detected.')


def detect_device_count(device_backend: str, prefer_swift: bool = False) -> int:
    """Return visible device count for cuda/npu. `device_backend` must be concrete."""
    if device_backend == 'cuda':
        return cuda_device_count()
    if device_backend == 'npu':
        count = npu_device_count()
        if count == 0 and prefer_swift:
            try:
                from swift.utils import get_device_count
                count = int(get_device_count())
            except Exception:
                pass
        return count
    raise ValueError(f'detect_device_count expects cuda or npu, got {device_backend}')


def configure_visible_device_env(rank: int, device_backend: str) -> None:
    """Restrict a worker process to one physical accelerator before torch import."""
    rank_str = str(rank)
    if device_backend == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = rank_str
        os.environ.pop('ASCEND_RT_VISIBLE_DEVICES', None)
    elif device_backend == 'npu':
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = rank_str
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        raise ValueError(f'Unsupported device backend: {device_backend}')


def set_torch_device_0(device_backend: str):
    """Return torch.device for the single visible worker device."""
    import torch

    if device_backend == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA backend requested, but torch.cuda.is_available() is False.')
        torch.cuda.set_device(0)
        return torch.device('cuda:0')

    if device_backend == 'npu':
        if not try_import_torch_npu():
            raise RuntimeError('NPU backend requested, but torch_npu cannot be imported.')
        if not hasattr(torch, 'npu'):
            raise RuntimeError('NPU backend requested, but torch.npu is not available.')
        try:
            is_available = bool(torch.npu.is_available())
        except Exception:
            is_available = npu_device_count() > 0
        if not is_available:
            raise RuntimeError('NPU backend requested, but torch.npu is not available.')
        try:
            torch.npu.set_device(0)
        except Exception:
            torch.npu.set_device('npu:0')
        return torch.device('npu:0')

    raise ValueError(f'Unsupported device backend: {device_backend}')


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Scan IP folders
# ─────────────────────────────────────────────────────────────────────────────

def _scan_one_folder(folder_path: str) -> Tuple[str, List[str]]:
    name = os.path.basename(folder_path.rstrip('/'))
    images = []
    try:
        with os.scandir(folder_path) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                ext = os.path.splitext(entry.name)[1].lower()
                if ext in IMAGE_EXTS:
                    images.append(entry.path)
    except (PermissionError, OSError) as e:
        print(f'  Warning: cannot scan {folder_path}: {e}', file=sys.stderr)

    images.sort()
    return name, images


def scan_ip_folders(test_dir: str, num_workers: int = 8) -> Dict[str, List[str]]:
    if not test_dir:
        raise ValueError('--test_dir is required when embedding images.')
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f'Test dir not found: {test_dir}')

    print(f'[Step 1] Scanning {test_dir} ...')
    subdirs = [e.path for e in os.scandir(test_dir) if e.is_dir()]
    subdirs.sort()
    print(f'  Found {len(subdirs)} subdirectories')

    ip_folders = {}
    if not subdirs:
        raise RuntimeError(f'No subdirectories found under test_dir: {test_dir}')

    max_workers = max(1, int(num_workers))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_scan_one_folder, d): d for d in subdirs}
        done = 0
        for future in as_completed(futures):
            name, images = future.result()
            if images:
                ip_folders[name] = images
            done += 1
            if done % 200 == 0 or done == len(futures):
                print(f'  Scanned {done}/{len(futures)} folders', end='\r')
    print()

    total_imgs = sum(len(v) for v in ip_folders.values())
    print(f'  {len(ip_folders)} IPs with images, {total_imgs} total images')
    if total_imgs == 0:
        raise RuntimeError(f'No images found under test_dir: {test_dir}')
    return ip_folders


# ─────────────────────────────────────────────────────────────────────────────
# Step 2A: swift embedding backend
# ─────────────────────────────────────────────────────────────────────────────

def build_swift_infer_request(image_path: str, instruction: str):
    """Build InferRequest for a single image."""
    from swift.infer_engine import InferRequest

    content = [{'type': 'image', 'image': image_path}]
    messages = []
    if instruction:
        messages.append({'role': 'system', 'content': instruction})
    messages.append({'role': 'user', 'content': content})
    return InferRequest(messages=messages)


def run_swift_worker(rank: int,
                     model: str,
                     model_type: str,
                     batch_size: int,
                     instruction: str,
                     shard_input: str,
                     shard_output: str,
                     torch_dtype_name: str,
                     device_backend: str) -> None:
    import torch
    if device_backend == 'npu':
        try_import_torch_npu()
        if hasattr(torch, 'npu'):
            try:
                torch.npu.set_device(0)
            except Exception:
                try:
                    torch.npu.set_device('npu:0')
                except Exception:
                    pass
    elif device_backend == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(0)

    from swift.infer_engine import TransformersEngine

    records = read_jsonl(shard_input)
    print(f'[{device_backend.upper()} {rank}] swift backend: {len(records)} images, loading model ...')

    torch_dtype = get_torch_dtype(torch_dtype_name, backend='swift')
    engine = TransformersEngine(
        model,
        model_type=model_type,
        task_type='embedding',
        torch_dtype=torch_dtype,
    )
    print(f'[{device_backend.upper()} {rank}] Model loaded. Starting inference ...')

    t0 = time.time()
    n_done = 0
    total = len(records)

    ensure_parent_dir(shard_output)
    with open(shard_output, 'w', encoding='utf-8') as f_out:
        for batch_start in range(0, total, batch_size):
            batch_records = records[batch_start:batch_start + batch_size]
            infer_requests = [
                build_swift_infer_request(r['image_path'], instruction)
                for r in batch_records
            ]
            resp_list = engine.infer(infer_requests)

            for r, resp in zip(batch_records, resp_list):
                emb = resp.data[0].embedding
                if hasattr(emb, 'tolist'):
                    emb = emb.tolist()
                r['embedding'] = emb
                f_out.write(json.dumps(r, ensure_ascii=False) + '\n')

            n_done += len(batch_records)
            elapsed = time.time() - t0
            speed = n_done / elapsed if elapsed > 0 else 0
            eta = (total - n_done) / speed if speed > 0 else 0
            print(
                f'[{device_backend.upper()} {rank}] {n_done}/{total} '
                f'({n_done / total * 100:.1f}%) | {speed:.0f} img/s | ETA {eta:.0f}s   ',
                end='\r',
            )

    elapsed = time.time() - t0
    speed = n_done / elapsed if elapsed > 0 else 0
    print(f'\n[{device_backend.upper()} {rank}] Done: {n_done} images in {elapsed:.1f}s ({speed:.0f} img/s)')


# ─────────────────────────────────────────────────────────────────────────────
# Step 2B: Danbooru CLIP embedding backend
# ─────────────────────────────────────────────────────────────────────────────

def run_danbooru_clip_worker(rank: int,
                             checkpoint_path: str,
                             clip_model_dir: str,
                             batch_size: int,
                             shard_input: str,
                             shard_output: str,
                             torch_dtype_name: str,
                             trust_remote_code: bool,
                             device_backend: str) -> None:
    import torch
    import torch.nn as nn
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    class CartoonRetrievalModel(nn.Module):
        def __init__(self, local_model_path: str):
            super().__init__()
            self.clip = CLIPModel.from_pretrained(
                local_model_path,
                local_files_only=True,
                trust_remote_code=trust_remote_code,
            )

        def forward(self, pixel_values):
            """
            Return projected CLIP image embeddings as a Tensor.

            This avoids cases where get_image_features() returns
            BaseModelOutputWithPooling instead of a Tensor.
            """
            vision_outputs = self.clip.vision_model(
                pixel_values=pixel_values,
                return_dict=True,
            )
            pooled_output = vision_outputs.pooler_output

            if hasattr(self.clip, "visual_projection"):
                return self.clip.visual_projection(pooled_output)

            return pooled_output

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Danbooru checkpoint not found: {checkpoint_path}')
    if not os.path.isdir(clip_model_dir):
        raise FileNotFoundError(f'Danbooru CLIP local model dir not found: {clip_model_dir}')

    records = read_jsonl(shard_input)
    print(f'[{device_backend.upper()} {rank}] danbooru_clip backend: {len(records)} images, loading model ...')

    device = set_torch_device_0(device_backend)

    torch_dtype = get_torch_dtype(torch_dtype_name, backend='danbooru_clip')
    processor = CLIPProcessor.from_pretrained(
        clip_model_dir,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )

    model = CartoonRetrievalModel(clip_model_dir)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    state_dict = normalize_state_dict_keys(state_dict)
    model.load_state_dict(state_dict)
    model = model.to(device=device, dtype=torch_dtype)
    model.eval()

    print(f'[{device_backend.upper()} {rank}] Model loaded. Starting inference ...')

    def load_pixel_values(image_path: str):
        img = Image.open(image_path).convert('RGB')
        pixel_values = processor.image_processor(
            img,
            return_tensors='pt',
        )['pixel_values'].squeeze(0)
        return pixel_values

    t0 = time.time()
    n_done = 0
    total = len(records)

    ensure_parent_dir(shard_output)
    with open(shard_output, 'w', encoding='utf-8') as f_out:
        for batch_start in range(0, total, batch_size):
            batch_records = records[batch_start:batch_start + batch_size]

            pixel_batch = []
            kept_records = []
            for r in batch_records:
                try:
                    pixel_batch.append(load_pixel_values(r['image_path']))
                    kept_records.append(r)
                except Exception as e:
                    print(
                        f'\n[{device_backend.upper()} {rank}] Warning: failed to read image {r["image_path"]}: {e}',
                        file=sys.stderr,
                    )

            if not pixel_batch:
                n_done += len(batch_records)
                continue

            x = torch.stack(pixel_batch, dim=0).to(device=device, dtype=torch_dtype)
            with torch.no_grad():
                feats = model(x)
                feats = torch.nn.functional.normalize(feats, dim=-1)
                feats = feats.detach().float().cpu().numpy()

            for r, emb in zip(kept_records, feats):
                r['embedding'] = emb.tolist()
                f_out.write(json.dumps(r, ensure_ascii=False) + '\n')

            n_done += len(batch_records)
            elapsed = time.time() - t0
            speed = n_done / elapsed if elapsed > 0 else 0
            eta = (total - n_done) / speed if speed > 0 else 0
            print(
                f'[{device_backend.upper()} {rank}] {n_done}/{total} '
                f'({n_done / total * 100:.1f}%) | {speed:.0f} img/s | ETA {eta:.0f}s   ',
                end='\r',
            )

    elapsed = time.time() - t0
    speed = n_done / elapsed if elapsed > 0 else 0
    print(f'\n[{device_backend.upper()} {rank}] Done: {n_done} images in {elapsed:.1f}s ({speed:.0f} img/s)')


# ─────────────────────────────────────────────────────────────────────────────
# Worker dispatch and multi-GPU embedding
# ─────────────────────────────────────────────────────────────────────────────

def run_worker_from_json(worker_args: dict) -> None:
    backend = worker_args['backend']
    if backend == 'swift':
        run_swift_worker(
            rank=worker_args['rank'],
            model=worker_args['model'],
            model_type=worker_args['model_type'],
            batch_size=worker_args['batch_size'],
            instruction=worker_args['instruction'],
            shard_input=worker_args['shard_input'],
            shard_output=worker_args['shard_output'],
            torch_dtype_name=worker_args['torch_dtype'],
            device_backend=worker_args['device_backend'],
        )
    elif backend == 'danbooru_clip':
        run_danbooru_clip_worker(
            rank=worker_args['rank'],
            checkpoint_path=worker_args['model'],
            clip_model_dir=worker_args['clip_model_dir'],
            batch_size=worker_args['batch_size'],
            shard_input=worker_args['shard_input'],
            shard_output=worker_args['shard_output'],
            torch_dtype_name=worker_args['torch_dtype'],
            trust_remote_code=worker_args['trust_remote_code'],
            device_backend=worker_args['device_backend'],
        )
    else:
        raise ValueError(f'Unsupported backend: {backend}')


def embed_all_images(ip_folders: Dict[str, List[str]],
                     backend: str,
                     device_backend: str,
                     model_path: str,
                     model_type: str,
                     instruction: str,
                     clip_model_dir: str,
                     batch_size: int,
                     num_gpus: int,
                     output_path: str,
                     torch_dtype: str,
                     trust_remote_code: bool) -> Dict[str, Dict[str, List[float]]]:
    """
    Embed all images across all IPs using subprocess-per-device parallelism.
    Saves result as JSONL: one line per IP with all its embeddings.
    """
    if not model_path:
        raise ValueError('--model is required when embedding images.')
    if backend == 'danbooru_clip' and not clip_model_dir:
        raise ValueError('--clip_model_dir is required for --backend danbooru_clip.')
    if int(num_gpus) <= 0:
        raise RuntimeError('No accelerator devices detected. Set --num_gpus or check CUDA/NPU visibility.')

    print(f'\n[Step 2] Embedding all images')
    print(f'  Backend: {backend}')
    print(f'  Device backend: {device_backend}')
    print(f'  Devices: {num_gpus}')
    print(f'  Batch per device: {batch_size}')

    flat_records = []
    for ip_name in sorted(ip_folders.keys()):
        for img_path in ip_folders[ip_name]:
            flat_records.append({'ip_name': ip_name, 'image_path': img_path})

    total = len(flat_records)
    print(f'  Total images to embed: {total}')

    tmp_dir = tempfile.mkdtemp(prefix='ip_eval_shards_')
    script_path = os.path.abspath(__file__)
    shard_inputs = []
    shard_outputs = []

    try:
        # Round-robin shard across devices.
        for rank in range(num_gpus):
            shard = flat_records[rank::num_gpus]
            si = os.path.join(tmp_dir, f'shard_in_{rank}.jsonl')
            so = os.path.join(tmp_dir, f'shard_out_{rank}.jsonl')
            write_jsonl(shard, si)
            shard_inputs.append(si)
            shard_outputs.append(so)

        procs = []
        t0 = time.time()
        for rank in range(num_gpus):
            worker_args = {
                'backend': backend,
                'device_backend': device_backend,
                'rank': rank,
                'model': model_path,
                'model_type': model_type,
                'batch_size': batch_size,
                'instruction': instruction,
                'clip_model_dir': clip_model_dir,
                'shard_input': shard_inputs[rank],
                'shard_output': shard_outputs[rank],
                'torch_dtype': torch_dtype,
                'trust_remote_code': trust_remote_code,
            }

            cmd = [sys.executable, script_path, '--__worker__', json.dumps(worker_args)]
            env = os.environ.copy()
            if device_backend == 'cuda':
                env['CUDA_VISIBLE_DEVICES'] = str(rank)
                env.pop('ASCEND_RT_VISIBLE_DEVICES', None)
            elif device_backend == 'npu':
                env['ASCEND_RT_VISIBLE_DEVICES'] = str(rank)
                env['CUDA_VISIBLE_DEVICES'] = ''
            else:
                raise ValueError(f'Unsupported device backend: {device_backend}')
            procs.append(subprocess.Popen(cmd, env=env))

        for p in procs:
            p.wait()

        failed = [i for i, p in enumerate(procs) if p.returncode != 0]
        if failed:
            raise RuntimeError(
                f'Workers {failed} failed with non-zero exit codes. '
                f'Check model path, GPU memory, batch size, and backend dependencies.'
            )

        elapsed = time.time() - t0
        speed = total / elapsed if elapsed > 0 else 0
        print(f'  All devices done in {elapsed:.1f}s ({speed:.0f} img/s)')

        print('  Merging shard outputs ...')
        merged = []
        for so in shard_outputs:
            if os.path.exists(so):
                with open(so, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            merged.append(json.loads(line))

        if len(merged) != total:
            missing = total - len(merged)
            raise RuntimeError(
                f'Data loss during merge: expected {total} records, got {len(merged)} '
                f'({missing} missing). Some images may have failed or workers crashed.'
            )

        print(f'  Merged {len(merged)}/{total} records')

        ip_embeddings = defaultdict(dict)
        for r in merged:
            ip_embeddings[r['ip_name']][r['image_path']] = r['embedding']

        print(f'  Saving embeddings to {output_path} ...')
        ensure_parent_dir(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            for ip_name in sorted(ip_embeddings.keys()):
                line = {
                    'ip_name': ip_name,
                    'embeddings': ip_embeddings[ip_name],
                }
                f.write(json.dumps(line, ensure_ascii=False) + '\n')

        print(f'  Saved {len(ip_embeddings)} IPs to {output_path}')
        return dict(ip_embeddings)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def load_embeddings(embeddings_path: str) -> Dict[str, Dict[str, List[float]]]:
    print(f'[Step 2] Loading embeddings from {embeddings_path} ...')
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f'Embeddings file not found: {embeddings_path}')

    ip_embeddings = {}
    with open(embeddings_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ip_embeddings[obj['ip_name']] = obj['embeddings']

    total_imgs = sum(len(v) for v in ip_embeddings.values())
    print(f'  Loaded {len(ip_embeddings)} IPs, {total_imgs} images')
    if total_imgs == 0:
        raise RuntimeError(f'No embeddings found in {embeddings_path}')
    return ip_embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Leave-one-out evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(ip_embeddings: Dict[str, Dict[str, List[float]]],
             K: int,
             metadata: dict = None):
    """
    Leave-one-out retrieval evaluation.

    - All images go into one pool.
    - Self-match is masked.
    - Only images from IPs with >=2 images are valid queries.
    - Images from 1-image IPs remain in the gallery as distractors.
    """
    print(f'\n[Step 3] Leave-one-out evaluation (K={K}) ...')

    all_paths = []
    all_ips = []
    all_embs = []
    ip_image_count = {}

    for ip_name in sorted(ip_embeddings.keys()):
        emb_dict = ip_embeddings[ip_name]
        img_paths = sorted(emb_dict.keys())
        ip_image_count[ip_name] = len(img_paths)
        for p in img_paths:
            all_paths.append(p)
            all_ips.append(ip_name)
            all_embs.append(emb_dict[p])

    N = len(all_paths)
    if N < 2:
        raise RuntimeError(f'Need at least 2 images to evaluate retrieval; got {N}.')

    dims = set(len(e) for e in all_embs)
    if len(dims) > 1:
        raise ValueError(f'Inconsistent embedding dimensions across images: {sorted(dims)}')

    all_ips_arr = np.array(all_ips)
    query_mask = np.array([ip_image_count[ip] >= 2 for ip in all_ips])
    query_indices = np.where(query_mask)[0]
    n_queries = int(len(query_indices))
    n_distractor_ips = sum(1 for _, cnt in ip_image_count.items() if cnt < 2)
    n_query_ips = sum(1 for _, cnt in ip_image_count.items() if cnt >= 2)

    print(f'  Total images:   {N}')
    print(f'  Total IPs:      {len(ip_image_count)}')
    print(f'  Query IPs:      {n_query_ips} (IPs with >=2 images)')
    print(f'  Queries:        {n_queries} (every image from IPs with >=2 images)')
    print(f'  Distractor IPs: {n_distractor_ips} (1-image IPs, gallery only)')

    if n_queries == 0:
        raise RuntimeError('No valid queries: every IP has only one image.')

    E = np.array(all_embs, dtype=np.float32)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)

    print(f'  Computing similarity matrix ({N} x {N}) ...')
    sim_matrix = E @ E.T
    np.fill_diagonal(sim_matrix, -np.inf)

    k_values = [1, 3, 5, int(K)]
    if int(K) == 30:
        k_values.append(10)
    k_values = sorted(set(k_values))
    max_k = max(k_values)

    cmc_at = {k: [] for k in k_values}
    recall_at = {k: [] for k in k_values}
    ap_at = {k: [] for k in k_values}
    r_precision_list = []
    per_query = []

    print(f'  Evaluating {n_queries} queries ...')
    for idx, i in enumerate(query_indices):
        q_ip = all_ips[i]
        n_rel = ip_image_count[q_ip] - 1
        sims = sim_matrix[i]

        fetch_k = max(max_k, n_rel)
        if fetch_k < N - 1:
            top_indices = np.argpartition(-sims, fetch_k)[:fetch_k]
            top_indices = top_indices[np.argsort(-sims[top_indices])]
        else:
            top_indices = np.argsort(-sims)

        top_ips = all_ips_arr[top_indices]
        relevance = (top_ips == q_ip).astype(int)

        query_result = {
            'query_ip': q_ip,
            'query_path': all_paths[i],
            'n_relevant': int(n_rel),
        }

        for k in k_values:
            rel_k = relevance[:k]
            n_rel_in_k = int(rel_k.sum())

            cmc = 1.0 if n_rel_in_k > 0 else 0.0
            recall = n_rel_in_k / n_rel

            ap = 0.0
            n_rel_so_far = 0
            for j in range(min(k, len(relevance))):
                if relevance[j] == 1:
                    n_rel_so_far += 1
                    ap += n_rel_so_far / (j + 1)
            ap = ap / min(n_rel, k)

            cmc_at[k].append(cmc)
            recall_at[k].append(recall)
            ap_at[k].append(ap)

            query_result[f'cmc@{k}'] = round(float(cmc), 4)
            query_result[f'recall@{k}'] = round(float(recall), 4)
            query_result[f'ap@{k}'] = round(float(ap), 4)

        r_prec_rel = relevance[:n_rel]
        r_prec = float(r_prec_rel.sum()) / n_rel
        r_precision_list.append(r_prec)
        query_result['r_precision'] = round(float(r_prec), 4)

        n_detail = min(5, len(top_indices))
        query_result['top5_ips'] = top_ips[:n_detail].tolist()
        query_result['top5_sims'] = [
            round(float(sims[top_indices[j]]), 4)
            for j in range(n_detail)
        ]
        query_result['top5_correct'] = relevance[:n_detail].astype(int).tolist()

        per_query.append(query_result)

        if (idx + 1) % 500 == 0 or idx + 1 == n_queries:
            print(f'    {idx + 1}/{n_queries} queries done', end='\r')
    print()

    metrics = {}
    for k in k_values:
        metrics[f'CMC@{k}'] = round(float(np.mean(cmc_at[k])), 4)
        metrics[f'Recall@{k}'] = round(float(np.mean(recall_at[k])), 4)
        metrics[f'mAP@{k}'] = round(float(np.mean(ap_at[k])), 4)

    metrics['R-Precision'] = round(float(np.mean(r_precision_list)), 4)
    metrics['n_queries'] = n_queries
    metrics['n_total_images'] = N
    metrics['n_total_ips'] = len(ip_image_count)
    metrics['n_query_ips'] = n_query_ips
    metrics['n_distractor_ips'] = n_distractor_ips
    metrics['K'] = int(K)
    if metadata:
        metrics['_metadata'] = metadata

    return metrics, per_query, cmc_at, recall_at, r_precision_list, k_values


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Output results
# ─────────────────────────────────────────────────────────────────────────────

def print_results(metrics: dict,
                  per_query: List[dict],
                  cmc_at: dict,
                  recall_at: dict,
                  r_precision_list: List[float],
                  K: int,
                  k_values: List[int],
                  output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    report_lines = []
    report_lines.append('=' * 60)
    report_lines.append('IP EMBEDDING RETRIEVAL EVALUATION')
    report_lines.append('=' * 60)
    report_lines.append('  Protocol:          Leave-one-out')
    report_lines.append('                     every valid image queries all other images')
    report_lines.append('                     1-image IPs are gallery distractors only')
    report_lines.append(f'  Queries:           {metrics["n_queries"]}')
    report_lines.append(f'  Total images:      {metrics["n_total_images"]}')
    report_lines.append(f'  Total IPs:         {metrics["n_total_ips"]}')
    report_lines.append(f'  Query IPs:         {metrics["n_query_ips"]}')
    report_lines.append(f'  Distractor IPs:    {metrics["n_distractor_ips"]}')
    report_lines.append(f'  K:                 {K}')

    report_lines.append('')
    report_lines.append('  ── CMC (hit rate: at least 1 correct in top K) ──')
    for k in k_values:
        marker = '  <-- production cutoff' if k == K else ''
        report_lines.append(f'  CMC@{k:<4d}          {metrics[f"CMC@{k}"]:.4f}{marker}')

    report_lines.append('')
    report_lines.append('  ── Recall (coverage: fraction of relevant found in top K) ──')
    for k in k_values:
        marker = '  <-- primary' if k == K else ''
        report_lines.append(f'  Recall@{k:<4d}       {metrics[f"Recall@{k}"]:.4f}{marker}')

    report_lines.append('')
    report_lines.append('  ── R-Precision (Precision@K_q) ──')
    report_lines.append(f'  R-Precision:       {metrics["R-Precision"]:.4f}')

    report_lines.append('')
    report_lines.append('  ── Ranking quality ──')
    for k in k_values:
        marker = '  <-- primary' if k == K else ''
        report_lines.append(f'  mAP@{k:<4d}          {metrics[f"mAP@{k}"]:.4f}{marker}')

    buckets = defaultdict(list)
    for q in per_query:
        buckets[q['n_relevant']].append(q[f'recall@{K}'])

    report_lines.append('')
    report_lines.append(f'  ── Recall@{K} by IP size (n_relevant) ──')
    for n_rel in sorted(buckets.keys()):
        vals = buckets[n_rel]
        report_lines.append(
            f'    n_rel={n_rel:>2d}: Recall@{K}={np.mean(vals):.4f} '
            f'(n={len(vals)} queries)'
        )

    failures = [q for q in per_query if q[f'recall@{K}'] < 1.0]
    fail_rate = len(failures) / len(per_query) * 100 if per_query else 0.0
    report_lines.append('')
    report_lines.append(
        f'  ── Failures: {len(failures)}/{len(per_query)} queries with '
        f'Recall@{K} < 1.0 ({fail_rate:.1f}%) ──'
    )

    if failures:
        failures_sorted = sorted(
            failures,
            key=lambda q: (q[f'recall@{K}'], q['r_precision'], q[f'ap@{K}']),
        )
        top_n = min(15, len(failures_sorted))
        report_lines.append(f'  Top {top_n} worst:')
        report_lines.append(
            f'    {"IP":>12s}  {"n_rel":>5s}  {"R@" + str(K):>6s}  '
            f'{"R-Prec":>6s}  {"AP@" + str(K):>6s}'
        )
        report_lines.append('    ' + '-' * 50)
        for q in failures_sorted[:top_n]:
            report_lines.append(
                f'    {q["query_ip"]:>12s}  {q["n_relevant"]:>5d}  '
                f'{q[f"recall@{K}"]:>6.3f}  {q["r_precision"]:>6.3f}  '
                f'{q[f"ap@{K}"]:>6.3f}'
            )

    report = '\n'.join(report_lines)
    print(f'\n{report}')

    ts = time.strftime('%Y%m%d_%H%M%S')

    log_path = os.path.join(output_dir, f'eval_log_{ts}.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    print(f'\n  Log saved to {log_path}')

    metrics_path = os.path.join(output_dir, f'eval_metrics_{ts}.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f'  Metrics saved to {metrics_path}')

    pq_path = os.path.join(output_dir, f'per_query_results_{ts}.jsonl')
    with open(pq_path, 'w', encoding='utf-8') as f:
        for q in per_query:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')
    print(f'  Per-query details saved to {pq_path}')

    save_plots(
        metrics=metrics,
        buckets=buckets,
        recall_at=recall_at,
        r_precision_list=r_precision_list,
        K=K,
        output_dir=output_dir,
        ts=ts,
    )


def save_plots(metrics: dict,
               buckets: dict,
               recall_at: dict,
               r_precision_list: List[float],
               K: int,
               output_dir: str,
               ts: str) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  matplotlib not installed, skipping plots')
        return

    # Plot 1: R-Precision histogram.
    fig = plt.figure(figsize=(7, 5))
    plt.hist(r_precision_list, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(x=metrics['R-Precision'], linestyle='--', label=f'Mean={metrics["R-Precision"]:.3f}')
    plt.xlabel('R-Precision')
    plt.ylabel('Count')
    plt.title('R-Precision Distribution')
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, f'eval_r_precision_{ts}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  R-Precision plot saved to {path}')

    # Plot 2: Recall@K by n_relevant.
    x_vals = sorted(buckets.keys())
    y_vals = [np.mean(buckets[x]) for x in x_vals]

    fig = plt.figure(figsize=(7, 5))
    plt.bar(x_vals, y_vals, edgecolor='black', alpha=0.7)
    plt.xlabel('n_relevant')
    plt.ylabel(f'Mean Recall@{K}')
    plt.title(f'Recall@{K} by IP Size')
    plt.ylim(0, 1.05)
    for x, y in zip(x_vals, y_vals):
        plt.text(x, y + 0.02, f'{y:.2f}', ha='center', fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, f'eval_recall_by_ip_size_{ts}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Recall-by-IP-size plot saved to {path}')

    # Plot 3: Recall@K histogram.
    fig = plt.figure(figsize=(7, 5))
    plt.hist(recall_at[K], bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(x=metrics[f'Recall@{K}'], linestyle='--', label=f'Mean={metrics[f"Recall@{K}"]:.3f}')
    plt.xlabel(f'Recall@{K}')
    plt.ylabel('Count')
    plt.title(f'Recall@{K} Distribution')
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, f'eval_recall_hist_{ts}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Recall histogram saved to {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Unified IP embedding retrieval evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Internal worker mode.
    parser.add_argument('--__worker__', type=str, default=None, help=argparse.SUPPRESS)

    # Backend and inference args.
    parser.add_argument(
        '--backend',
        choices=['swift', 'danbooru_clip'],
        default='swift',
        help='Embedding backend. Use danbooru_clip for the earlier CLIP checkpoint.',
    )
    parser.add_argument(
        '--device_backend',
        choices=['auto', 'cuda', 'npu'],
        default='auto',
        help='Accelerator backend. auto prefers CUDA if visible, otherwise Ascend NPU.',
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help=(
            'For backend=swift: model checkpoint/path. '
            'For backend=danbooru_clip: .pth checkpoint, e.g. danbooru_clip_best_v6.pth.'
        ),
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='qwen3_5_emb',
        help='swift model type. Ignored for backend=danbooru_clip.',
    )
    parser.add_argument(
        '--clip_model_dir',
        type=str,
        default=None,
        help='Local CLIP model folder for backend=danbooru_clip, e.g. ./danbooru_clip.',
    )
    parser.add_argument(
        '--instruction',
        type=str,
        default=DEFAULT_INSTRUCTION,
        help='System prompt for swift embedding models. Ignored for backend=danbooru_clip.',
    )
    parser.add_argument(
        '--test_dir',
        type=str,
        default=None,
        help='Root folder containing one subfolder per IP.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size per device.',
    )
    parser.add_argument(
        '--num_gpus',
        type=int,
        default=None,
        help='Number of devices. Default: auto-detect. Kept as --num_gpus for backward compatibility.',
    )
    parser.add_argument(
        '--torch_dtype',
        choices=['auto', 'float32', 'float16', 'bfloat16'],
        default='auto',
        help='auto = bfloat16 for swift, float32 for danbooru_clip. For Danbooru on NPU, try float16/bfloat16 if float32 is slow or unsupported.',
    )
    parser.add_argument(
        '--trust_remote_code',
        action='store_true',
        help='Pass trust_remote_code=True when loading Danbooru CLIP.',
    )

    # Output.
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output embeddings JSONL path.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output dir for metrics/logs/plots. Default: same dir as --output or --embeddings.',
    )

    # Re-evaluate from saved embeddings.
    parser.add_argument(
        '--embeddings',
        type=str,
        default=None,
        help='Pre-computed embeddings JSONL. Skips inference.',
    )

    # Eval config.
    parser.add_argument(
        '--K',
        type=int,
        default=10,
        help='K for CMC@K, Recall@K, and mAP@K.',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Workers for folder scanning.',
    )

    return parser.parse_args()


def default_output_path(args) -> str:
    if args.output:
        return args.output

    if args.model:
        model_path = os.path.abspath(args.model.rstrip('/'))
        model_dir = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
        suffix = args.backend
        return os.path.join(model_dir, f'eval_embeddings_{suffix}.jsonl')

    return os.path.join(os.getcwd(), f'eval_embeddings_{args.backend}.jsonl')


def main():
    args = parse_args()

    if args.__worker__:
        worker_args = json.loads(args.__worker__)
        rank_str = str(worker_args['rank'])

        # Set visibility before importing torch / swift inside worker functions.
        configure_visible_device_env(int(rank_str), worker_args['device_backend'])

        run_worker_from_json(worker_args)
        return

    total_t0 = time.time()

    if args.K <= 0:
        raise ValueError('--K must be positive.')
    if args.batch_size <= 0:
        raise ValueError('--batch_size must be positive.')

    if args.embeddings:
        ip_embeddings = load_embeddings(args.embeddings)
        if args.output_dir is None:
            args.output_dir = os.path.dirname(os.path.abspath(args.embeddings))
        model_path_for_meta = '(loaded from embeddings)'
    else:
        if not args.model or not args.test_dir:
            raise SystemExit('Provide either --embeddings, or both --model and --test_dir.')

        if args.backend == 'danbooru_clip' and not args.clip_model_dir:
            raise SystemExit('--clip_model_dir is required when --backend danbooru_clip.')

        args.device_backend = resolve_device_backend(
            args.device_backend,
            prefer_swift=(args.backend == 'swift'),
        )

        if args.num_gpus is None:
            args.num_gpus = detect_device_count(
                args.device_backend,
                prefer_swift=(args.backend == 'swift'),
            )

        if args.num_gpus == 0:
            raise RuntimeError('No accelerator devices detected. Check CUDA/NPU visibility.')

        args.output = default_output_path(args)
        if args.output_dir is None:
            args.output_dir = os.path.dirname(os.path.abspath(args.output))

        print('=' * 60)
        print('  UNIFIED IP RETRIEVAL EVALUATION')
        print('=' * 60)
        print(f'  Backend:      {args.backend}')
        print(f'  Device:       {args.device_backend}')
        print(f'  Model:        {args.model}')
        if args.backend == 'swift':
            print(f'  Model type:   {args.model_type}')
            print(f'  Instruction:  "{args.instruction}"')
            print('  Note: instruction should match the system prompt used during training.')
        else:
            print(f'  CLIP dir:     {args.clip_model_dir}')
            print('  Note: Danbooru CLIP uses image-only CLIP features; --instruction is ignored.')
        print(f'  Test dir:     {args.test_dir}')
        print(f'  Output:       {args.output}')
        print(f'  Output dir:   {args.output_dir}')
        print('=' * 60)

        if os.path.exists(args.output) and os.path.getsize(args.output) > 0:
            print(f'Embeddings file exists, loading: {args.output}')
            ip_embeddings = load_embeddings(args.output)
        else:
            ip_folders = scan_ip_folders(args.test_dir, args.num_workers)
            ip_embeddings = embed_all_images(
                ip_folders=ip_folders,
                backend=args.backend,
                device_backend=args.device_backend,
                model_path=args.model,
                model_type=args.model_type,
                instruction=args.instruction,
                clip_model_dir=args.clip_model_dir,
                batch_size=args.batch_size,
                num_gpus=args.num_gpus,
                output_path=args.output,
                torch_dtype=args.torch_dtype,
                trust_remote_code=args.trust_remote_code,
            )

        model_path_for_meta = args.model

    metadata = {
        'backend': args.backend,
        'device_backend': getattr(args, 'device_backend', None),
        'model_path': model_path_for_meta,
        'model_type': args.model_type if args.backend == 'swift' else None,
        'clip_model_dir': args.clip_model_dir if args.backend == 'danbooru_clip' else None,
        'embeddings_path': args.embeddings,
        'instruction': args.instruction if args.backend == 'swift' else None,
        'test_dir': args.test_dir,
        'torch_dtype': args.torch_dtype,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    metrics, per_query, cmc_at, recall_at, r_precision_list, k_values = evaluate(
        ip_embeddings=ip_embeddings,
        K=args.K,
        metadata=metadata,
    )

    print_results(
        metrics=metrics,
        per_query=per_query,
        cmc_at=cmc_at,
        recall_at=recall_at,
        r_precision_list=r_precision_list,
        K=args.K,
        k_values=k_values,
        output_dir=args.output_dir,
    )

    total_elapsed = time.time() - total_t0
    print(f'\nTotal time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)')


if __name__ == '__main__':
    main()
