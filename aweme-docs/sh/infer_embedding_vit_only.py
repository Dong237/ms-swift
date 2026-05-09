"""
Qwen3.5 ViT-Only Embedding 批量推理脚本
=======================================

功能概述：
  直接从 Qwen3.5 的 ViT 视觉编码器提取图像 embedding，跳过 LLM 的 24 层
  Transformer，仅使用 ViT 的 12 层视觉编码 + PatchMerger 输出。
  用于与 full-pipeline（EOS pooling）方案做性能对比。

与 infer_embedding_multimodal_swift.py 的区别：
  ┌────────────────────────┬───────────────────────┐
  │  Full Pipeline (swift) │  ViT Only (本脚本)     │
  ├────────────────────────┼───────────────────────┤
  │  Image → ViT → LLM    │  Image → ViT → Pool   │
  │  → EOS pooling         │  → Mean/Max pooling   │
  │  ~800M params (0.8B)   │  ~86M params          │
  │  可用文本指令引导       │  纯视觉，无文本输入    │
  │  batch_size=64         │  batch_size=128       │
  │  依赖 ms-swift         │  仅需 transformers    │
  └────────────────────────┴───────────────────────┘

  CLI 和 I/O 格式完全对齐，两个脚本可互换使用做对比实验。

工作原理：
  1. 加载完整 Qwen3.5 模型，提取 .visual（ViT 编码器），释放 LLM 权重
  2. 用 image_processor 预处理图片 → pixel_values + image_grid_thw
  3. ViT 前向：pixel_values → 12 层 Transformer → PatchMerger
     输出 shape: [total_merged_tokens, out_hidden_size]
  4. 按 image_grid_thw 切分每张图的 token，mean/max pooling → [out_hidden_size]
  5. L2 归一化后输出

  ViT 输出维度（= out_hidden_size）：
    - Qwen3.5-0.8B:  1024
    - Qwen3.5-2B:    与 LLM hidden_size 一致
    - 可通过 vit.merger.linear_fc2.out_features 动态获取

输入数据格式（JSONL，与 swift 版完全一致）：
  图片字段（必须有图片，否则返回零向量）：
    - "image_path": "/path/to/image.jpg"
    - "images":     ["/path/to/image.jpg"]

  文本字段（可选，本脚本忽略）：
    - "prompt" / "query" / "messages"

  示例：
    {"prompt": "美食探店", "image_path": "/data/img/001.jpg", "label": "美食"}
    {"image_path": "/data/img/002.jpg", "label": "情感"}

输出数据格式（JSONL，与 swift 版完全一致）：
  原始字段 + "embedding" 字段：
    {"image_path": "...", "label": "...", "embedding": [0.01, -0.02, ...]}

关键参数说明：

  --model            Qwen3.5 模型路径（会自动提取 ViT 部分）
  --input            输入 JSONL 文件路径，支持传入多个
  --output_dir       输出目录（默认: 模型目录同级的 embeddings_vit/）

  速度 / 显存相关参数：
  --batch_size       每个 GPU 每次前向的图片数（默认 128）
                     - ViT 很小（~86M params），可以开大 batch
                     - 高分辨率图片可适当降低
                     - OOM 时降低此值

  --num_gpus         使用的 GPU 数量（默认自动检测全部可用 GPU）
                     - ViT 仅占 ~0.2GB 显存，单卡可轻松运行

  --pooling          池化策略（默认 mean）
                     - mean: 所有 patch token 取平均（推荐）
                     - max:  所有 patch token 逐维取最大值

  --instruction      接受但忽略（为 CLI 兼容性保留，ViT 无文本输入）
  --model_type       接受但忽略（为 CLI 兼容性保留）

用法示例：

  # 基本用法
  python infer_embedding_vit_only.py \\
      --model /path/to/Qwen3.5-0.8B \\
      --input /path/to/data.jsonl

  # 多个文件 + 调参
  python infer_embedding_vit_only.py \\
      --model /path/to/Qwen3.5-0.8B \\
      --input train.jsonl test.jsonl \\
      --batch_size 256 \\
      --num_gpus 4 \\
      --pooling mean
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time

import torch
import torch.nn.functional as F
from PIL import Image


def get_gpu_count() -> int:
    count = torch.cuda.device_count()
    if count == 0:
        raise RuntimeError('No CUDA GPUs detected.')
    return count


def read_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def get_image_path(r: dict) -> str | None:
    if 'image_path' in r:
        return r['image_path']
    if 'images' in r and r['images']:
        return r['images'][0]
    return None


def load_vit_and_processor(model_path: str, dtype: torch.dtype, device: str):
    """从 Qwen3.5 完整模型中提取 ViT 视觉编码器，释放 LLM 权重。

    Qwen3.5 没有独立的 VisionModel.from_pretrained()，
    ViT 权重存储在完整 checkpoint 中，必须加载后提取。

    Returns:
        (vit, processor, out_dim, merge_size)
    """
    from transformers import AutoProcessor, AutoConfig

    print(f'  Loading processor from {model_path} ...')
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    print(f'  Loading full model to extract ViT ...')
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Determine model class from config.architectures
    arch = config.architectures[0] if config.architectures else ''
    if 'Qwen3_5' in arch:
        from transformers import Qwen3_5ForConditionalGeneration as ModelCls
    elif 'Qwen3VL' in arch:
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
    else:
        # Fallback: try AutoModel
        from transformers import AutoModelForImageTextToText as ModelCls

    full_model = ModelCls.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map='cpu',
        trust_remote_code=True,
    )

    # Extract the ViT — try common attribute paths
    vit = None
    for attr_path in ['visual', 'model.visual']:
        obj = full_model
        try:
            for part in attr_path.split('.'):
                obj = getattr(obj, part)
            vit = obj
            break
        except AttributeError:
            continue

    if vit is None:
        raise RuntimeError(
            f'Cannot find visual encoder in model. '
            f'Tried: model.visual, model.model.visual. '
            f'Model type: {type(full_model).__name__}'
        )

    # Move ViT to device
    vit = vit.to(device=device, dtype=dtype).eval()

    # Read output dimension and merge_size dynamically
    out_dim = vit.merger.linear_fc2.out_features
    merge_size = vit.spatial_merge_size

    # Free LLM weights
    del full_model
    gc.collect()
    torch.cuda.empty_cache()

    print(f'  ViT extracted: out_dim={out_dim}, merge_size={merge_size}, '
          f'depth={len(vit.blocks)}, dtype={dtype}')

    return vit, processor, out_dim, merge_size


@torch.inference_mode()
def extract_vit_embeddings(
    vit: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    merge_size: int,
    pooling: str = 'mean',
) -> list[torch.Tensor]:
    """运行 ViT 前向并对每张图的 patch token 做池化。

    Args:
        vit: Qwen3.5 ViT 视觉编码器
        pixel_values: 预处理后的图片 patches
        grid_thw: [num_images, 3] — 每张图的 (temporal, height, width) grid
        merge_size: PatchMerger 的空间合并大小（通常为 2）
        pooling: 'mean' 或 'max'

    Returns:
        每张图一个 L2 归一化后的 [out_hidden_size] 向量
    """
    # ViT forward — return type varies by transformers version:
    #   Qwen3VL (transformers 4.x): raw tuple (hidden_states, deepstack_list)
    #   Qwen3.5 (transformers 5.x): ModelOutput (dict-like, DO NOT tuple-unpack)
    output = vit(pixel_values, grid_thw=grid_thw)

    if isinstance(output, torch.Tensor):
        hidden_states = output
    elif isinstance(output, (tuple, list)):
        hidden_states = output[0]
    else:
        # ModelOutput — use index or attribute access
        if hasattr(output, 'last_hidden_state') and output.last_hidden_state is not None:
            hidden_states = output.last_hidden_state
        elif hasattr(output, 'pooler_output') and output.pooler_output is not None:
            hidden_states = output.pooler_output
        else:
            hidden_states = output[0]
    # hidden_states shape: [total_merged_tokens, out_hidden_size]

    # Split by image: each image produces grid_thw.prod // merge_size^2 tokens
    tokens_per_image = (grid_thw.prod(dim=-1) // (merge_size ** 2))

    results = []
    offset = 0
    for i in range(grid_thw.shape[0]):
        n_tokens = int(tokens_per_image[i].item())
        embeds = hidden_states[offset:offset + n_tokens]
        offset += n_tokens
        # embeds: [num_tokens, out_hidden_size]
        if pooling == 'mean':
            pooled = embeds.mean(dim=0)
        elif pooling == 'max':
            pooled = embeds.max(dim=0).values
        else:
            raise ValueError(f'Unknown pooling method: {pooling}')

        # L2 normalize to unit vector
        pooled = F.normalize(pooled, p=2, dim=0)
        results.append(pooled)

    return results


# ─── Worker 子进程入口 ────────────────────────────────────────────────────────

def run_worker(rank, model, batch_size, pooling, dtype, shard_input, shard_output):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    device = 'cuda'
    torch_dtype = getattr(torch, dtype)

    records = read_jsonl(shard_input)
    total = len(records)
    print(f'[GPU {rank}] {total} records, loading ViT ...')

    vit, processor, out_dim, merge_size = load_vit_and_processor(model, torch_dtype, device)
    zero_embedding = [0.0] * out_dim

    print(f'[GPU {rank}] ViT loaded. Starting inference...')

    t0 = time.time()
    n_done = 0
    n_images = 0
    n_no_image = 0
    warned_no_image = False

    with open(shard_output, 'w', encoding='utf-8') as f_out:
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_records = records[batch_start:batch_end]

            # Collect images from this batch
            pil_images = []
            image_indices = []  # which records in the batch have valid images

            for i, r in enumerate(batch_records):
                img_path = get_image_path(r)
                if img_path and os.path.exists(img_path):
                    try:
                        pil_images.append(Image.open(img_path).convert('RGB'))
                        image_indices.append(i)
                    except Exception as e:
                        n_no_image += 1
                        if not warned_no_image:
                            print(f'\n[GPU {rank}] WARNING: image load failed: {img_path}: {e}')
                            warned_no_image = True
                else:
                    n_no_image += 1
                    if not warned_no_image and img_path:
                        print(f'\n[GPU {rank}] WARNING: image not found: {img_path}, using zero vector')
                        warned_no_image = True

            # Initialize all embeddings as zero vectors
            embeddings = [zero_embedding] * len(batch_records)

            if pil_images:
                # Preprocess images
                media_inputs = processor.image_processor(
                    images=pil_images,
                    return_tensors='pt',
                    do_resize=True,
                )
                pixel_values = media_inputs['pixel_values'].to(device=device, dtype=vit.dtype)
                grid_thw = media_inputs['image_grid_thw'].to(device=device)

                # ViT forward + pooling
                vit_embeds = extract_vit_embeddings(
                    vit, pixel_values, grid_thw, merge_size, pooling
                )

                # Assign embeddings to the correct record positions
                for idx, emb in zip(image_indices, vit_embeds):
                    embeddings[idx] = emb.cpu().float().tolist()

                n_images += len(pil_images)

            # Write output
            for r, emb in zip(batch_records, embeddings):
                r['embedding'] = emb
                f_out.write(json.dumps(r, ensure_ascii=False) + '\n')

            n_done += len(batch_records)
            elapsed = time.time() - t0
            speed = n_done / elapsed if elapsed > 0 else 0
            eta = (total - n_done) / speed if speed > 0 else 0
            print(f'[GPU {rank}] {n_done}/{total} '
                  f'({n_done / total * 100:.1f}%) '
                  f'| {speed:.0f} rows/s | ETA {eta:.0f}s   ', end='\r')

    elapsed = time.time() - t0
    print(f'\n[GPU {rank}] Done: {n_done} rows in {elapsed:.1f}s '
          f'({n_done / elapsed:.0f} rows/s) '
          f'| images={n_images}, no_image={n_no_image}')


# ─── 单文件推理（调度所有 GPU）────────────────────────────────────────────────

def embed_file(model_path, input_path, output_path, args, num_gpus, script_path):
    basename = os.path.basename(input_path)
    name, ext = os.path.splitext(basename)

    tmp_dir = tempfile.mkdtemp(prefix='emb_vit_shards_')

    print(f'\n  Embedding (ViT only): {input_path}')
    records = read_jsonl(input_path)
    total = len(records)
    print(f'    Total records: {total}')
    print(f'    Shard temp dir: {tmp_dir}')

    # 按 round-robin 切分到各 GPU
    shard_inputs, shard_outputs = [], []
    for rank in range(num_gpus):
        shard = records[rank::num_gpus]
        si = os.path.join(tmp_dir, f'{name}_in_shard{rank}{ext}')
        so = os.path.join(tmp_dir, f'{name}_out_shard{rank}{ext}')
        write_jsonl(shard, si)
        shard_inputs.append(si)
        shard_outputs.append(so)
    del records

    for si in shard_inputs:
        assert os.path.exists(si), f'Shard file missing: {si}'

    # 启动子进程
    procs = []
    t0 = time.time()
    for rank in range(num_gpus):
        worker_args = json.dumps({
            'rank': rank,
            'model': model_path,
            'batch_size': args.batch_size,
            'pooling': args.pooling,
            'dtype': args.dtype,
            'shard_input': shard_inputs[rank],
            'shard_output': shard_outputs[rank],
        })
        cmd = [sys.executable, script_path, '--__worker__', worker_args]
        procs.append(subprocess.Popen(cmd))

    for p in procs:
        p.wait()

    failed = [i for i, p in enumerate(procs) if p.returncode != 0]
    if failed:
        print(f'    WARNING: workers {failed} failed')

    elapsed = time.time() - t0
    print(f'    All GPUs done in {elapsed:.1f}s ({total / elapsed:.0f} samples/s)')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Round-robin 交错合并，恢复原始顺序
    print('    Merging shards ...')
    merge_t0 = time.time()
    shard_fhs = []
    for so in shard_outputs:
        shard_fhs.append(open(so, 'r', encoding='utf-8') if os.path.exists(so) else None)

    merged_count = 0
    with open(output_path, 'w', encoding='utf-8') as f_out:
        exhausted = set()
        while len(exhausted) < num_gpus:
            for rank in range(num_gpus):
                if rank in exhausted:
                    continue
                fh = shard_fhs[rank]
                if fh is None:
                    exhausted.add(rank)
                    continue
                line = fh.readline()
                if not line:
                    exhausted.add(rank)
                    continue
                f_out.write(line if line.endswith('\n') else line + '\n')
                merged_count += 1
                if merged_count % 50000 == 0:
                    print(f'      Merged {merged_count}/{total} ...')

    for fh in shard_fhs:
        if fh is not None:
            fh.close()

    merge_elapsed = time.time() - merge_t0
    print(f'    Merged {merged_count}/{total} in {merge_elapsed:.1f}s')
    if merged_count < total:
        print(f'    WARNING: {total - merged_count} records lost')

    # 清理临时文件
    for p in shard_inputs + shard_outputs:
        if os.path.exists(p):
            os.remove(p)
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    print(f'    Saved to: {output_path}')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # 子进程 worker 模式
    if '--__worker__' in sys.argv:
        idx = sys.argv.index('--__worker__')
        w_args = json.loads(sys.argv[idx + 1])
        run_worker(**w_args)
        return

    gpu_count = get_gpu_count()

    parser = argparse.ArgumentParser(
        description='Qwen3.5 ViT-Only Embedding 批量推理',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Qwen3.5 模型路径（会自动提取 ViT 部分）')
    parser.add_argument('--input', type=str, nargs='+', required=True,
                        help='输入 JSONL 文件路径，支持传入多个')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认: 模型目录同级的 embeddings_vit/）')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='每个 GPU 每批图片数 (default: 128)')
    parser.add_argument('--num_gpus', type=int, default=gpu_count,
                        help=f'使用的 GPU 数量 (default: {gpu_count})')
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'max'],
                        help='池化策略 (default: mean)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['bfloat16', 'float16', 'float32'],
                        help='计算精度 (default: bfloat16)')
    # CLI 兼容性参数（接受但忽略）
    parser.add_argument('--instruction', type=str, default='',
                        help='(忽略) ViT 无文本输入，此参数仅保留 CLI 兼容性')
    parser.add_argument('--model_type', type=str, default='',
                        help='(忽略) 自动检测模型架构，此参数仅保留 CLI 兼容性')
    args = parser.parse_args()

    if args.instruction:
        print(f'WARNING: --instruction is ignored in ViT-only mode (ViT has no text input)')

    # 默认输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.model.rstrip('/')), 'embeddings_vit')

    num_gpus = min(args.num_gpus, gpu_count)
    script_path = os.path.abspath(__file__)

    print(f'Detected {gpu_count} GPU(s), using {num_gpus} for data parallelism')
    print(f'Model: {args.model}')
    print(f'Pooling: {args.pooling}')
    print(f'Dtype: {args.dtype}')
    print(f'Batch size per GPU: {args.batch_size}')
    print(f'Input files: {args.input}')
    print(f'Output dir: {args.output_dir}')

    os.makedirs(args.output_dir, exist_ok=True)

    total_t0 = time.time()
    for input_path in args.input:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(args.output_dir, f'{base}_with_embedding.jsonl')

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f'\n  SKIP (exists): {output_path}')
            continue

        embed_file(args.model, input_path, output_path, args, num_gpus, script_path)

    total_elapsed = time.time() - total_t0
    print(f'\n{"=" * 60}')
    print(f'All done in {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)')
    print(f'Results in: {args.output_dir}')


if __name__ == '__main__':
    main()
