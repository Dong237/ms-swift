"""
多模态 Embedding 批量推理脚本（基于 ms-swift TransformersEngine）
================================================================

功能概述：
  对训练好的 Qwen3.5 多模态 Embedding 模型进行批量推理，将文本+图片输入转化为
  固定维度的 embedding 向量（如 0.8B 模型输出 1024 维）。
  支持传入任意数量的 JSONL 文件，每个文件独立生成对应的 embedding 输出。

工作原理：
  1. 数据分片：将输入 JSONL 按 round-robin 方式切分到 N 个 GPU
  2. 子进程并行：每个 GPU 启动独立子进程，各自加载一份完整模型
  3. 批量前向：每个子进程按 batch_size 批量送入模型推理
  4. 合并输出：所有分片按原始顺序 round-robin 交错合并为最终文件

  模型内部流程：
  - 文本经模板格式化为 ChatML 格式，图片通过 <image> 标签注入
  - ViT 视觉编码器处理图片 → 视觉 token
  - 文本 token 和视觉 token 融合后送入 Transformer
  - 取序列末尾 EOS token 的隐状态作为 embedding
  - L2 归一化后输出

输入数据格式（JSONL，每行一条记录）：
  支持以下字段组合，脚本会自动识别：

  文本字段（按优先级，取第一个存在的）：
    - "prompt":  字符串
    - "query":   字符串
    - "messages": [{"role": "user", "content": "..."}]

  图片字段（可选，按优先级）：
    - "image_path": "/path/to/image.jpg"
    - "images":     ["/path/to/image.jpg"]

  示例：
    {"prompt": "美食探店视频", "image_path": "/data/img/001.jpg", "label": "美食"}
    {"query": "今天心情不好", "images": ["/data/img/002.jpg"], "label": "情感"}
    {"prompt": "纯文本内容，没有图片", "label": "其他"}

输出数据格式（JSONL）：
  原始字段 + "embedding" 字段：
    {"prompt": "...", "image_path": "...", "label": "...", "embedding": [0.01, -0.02, ...]}

  embedding 维度取决于模型大小：
    - Qwen3.5-0.8B:  1024 维
    - Qwen3.5-2B:   ~1536 维
    - Qwen3.5-4B:   ~2560 维
    - Qwen3.5-9B:   ~3584 维

关键参数说明：

  --model            模型路径（checkpoint 目录或合并后的模型目录）
  --input            输入 JSONL 文件路径，支持传入多个
  --output_dir       输出目录（默认: 模型目录同级的 embeddings/）

  速度 / 显存相关参数：
  --batch_size       每个 GPU 每次前向的样本数（默认 64）
                     - 增大 → 吞吐量更高，但显存占用更大
                     - 多模态场景因图片处理开销大，建议 32-128
                     - 纯文本场景可以开到 256-512
                     - OOM 时优先降低此值

  --num_gpus         使用的 GPU 数量（默认自动检测全部可用 GPU）
                     - 每个 GPU 加载一份完整模型副本
                     - 0.8B 模型约占 1.6GB 显存，可放心多卡
                     - 9B 模型约占 18GB 显存，注意单卡是否放得下

  --instruction      全局指令，作为 system message 注入模板
                     - 引导模型提取特定维度的语义特征
                     - 例如情感分类任务: "提取该内容的情感特征表示"
                     - 设为空字符串 "" 表示不使用指令

  --model_type       模型类型（默认 qwen3_5_emb）
                     - qwen3_5_emb: Qwen3.5 系列（本分支新增）
                     - qwen3_vl_emb: Qwen3-VL-Embedding 官方模型

用法示例：

  # 单个文件
  python infer_embedding_multimodal_swift.py \\
      --model /path/to/checkpoint-500 \\
      --input /path/to/data.jsonl

  # 多个文件
  python infer_embedding_multimodal_swift.py \\
      --model /path/to/checkpoint-500 \\
      --input train.jsonl test.jsonl val.jsonl

  # 调整速度和指令
  python infer_embedding_multimodal_swift.py \\
      --model /path/to/checkpoint-500 \\
      --input train.jsonl test.jsonl \\
      --batch_size 32 \\
      --num_gpus 4 \\
      --instruction "提取该内容的情感特征表示"
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import torch

# ─── 全局指令（作为 system message 注入，引导 embedding 提取方向）──────────────
# 用户可通过 --instruction 参数覆盖此默认值
# 设为空字符串 "" 表示不使用指令
DEFAULT_INSTRUCTION = "根据输入的图文内容，提取语义特征表示"


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


def get_text(r: dict) -> str:
    if 'prompt' in r:
        return r['prompt']
    if 'query' in r:
        return r['query']
    if 'messages' in r:
        return r['messages'][0]['content']
    # 无文本字段时返回空字符串（纯图片场景）
    return ''


def get_image_path(r: dict) -> str | None:
    if 'image_path' in r:
        return r['image_path']
    if 'images' in r and r['images']:
        return r['images'][0]
    return None


def build_infer_request(text: str, image_path: str | None, instruction: str):
    """构造 InferRequest，使用 OpenAI 多模态 content 格式确保图片正确注入。

    关键：使用 [{'type': 'image', 'image': path}, {'type': 'text', 'text': ...}] 格式，
    ms-swift 的 remove_messages_media() 会自动将其转换为 <image> 标签 + images 列表，
    确保 ViT 视觉编码器正确处理图片。
    """
    from swift.infer_engine import InferRequest

    # 构建 user message 的 content
    if image_path and os.path.exists(image_path):
        # 多模态：图片 + 文本
        content = [
            {'type': 'image', 'image': image_path},
            {'type': 'text', 'text': text},
        ]
    else:
        # 纯文本
        content = text

    messages = []
    if instruction:
        messages.append({'role': 'system', 'content': instruction})
    messages.append({'role': 'user', 'content': content})

    return InferRequest(messages=messages)


# ─── Worker 子进程入口 ────────────────────────────────────────────────────────

def run_worker(rank, model, model_type, batch_size, instruction, shard_input, shard_output):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)

    from swift.infer_engine import TransformersEngine

    records = read_jsonl(shard_input)
    print(f'[GPU {rank}] {len(records)} records, loading model ...')

    engine = TransformersEngine(
        model,
        model_type=model_type,
        task_type='embedding',
        torch_dtype=torch.bfloat16,
    )
    print(f'[GPU {rank}] Model loaded. Starting inference...')

    t0 = time.time()
    n_done = 0
    total = len(records)

    with open(shard_output, 'w', encoding='utf-8') as f_out:
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_records = records[batch_start:batch_end]

            infer_requests = []
            for r in batch_records:
                text = get_text(r)
                img_path = get_image_path(r)
                req = build_infer_request(text, img_path, instruction)
                infer_requests.append(req)

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
            print(f'[GPU {rank}] {n_done}/{total} '
                  f'({n_done / total * 100:.1f}%) '
                  f'| {speed:.0f} rows/s | ETA {eta:.0f}s   ', end='\r')

    elapsed = time.time() - t0
    print(f'\n[GPU {rank}] Done: {n_done} rows in {elapsed:.1f}s '
          f'({n_done / elapsed:.0f} rows/s)')


# ─── 单文件推理（调度所有 GPU）────────────────────────────────────────────────

def embed_file(model_path, input_path, output_path, args, num_gpus, script_path):
    basename = os.path.basename(input_path)
    name, ext = os.path.splitext(basename)

    tmp_dir = tempfile.mkdtemp(prefix='emb_shards_')

    print(f'\n  Embedding: {input_path}')
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
            'model_type': args.model_type,
            'batch_size': args.batch_size,
            'instruction': args.instruction,
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
        description='Qwen3.5 多模态 Embedding 批量推理',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model', type=str, required=True,
                        help='模型路径（checkpoint 目录或合并后的模型目录）')
    parser.add_argument('--input', type=str, nargs='+', required=True,
                        help='输入 JSONL 文件路径，支持传入多个')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认: 模型目录同级的 embeddings/）')
    parser.add_argument('--model_type', type=str, default='qwen3_5_emb',
                        help='模型类型 (default: qwen3_5_emb)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='每个 GPU 每批样本数，OOM 时降低此值 (default: 64)')
    parser.add_argument('--num_gpus', type=int, default=gpu_count,
                        help=f'使用的 GPU 数量 (default: {gpu_count}，即全部可用)')
    parser.add_argument('--instruction', type=str, default=DEFAULT_INSTRUCTION,
                        help='全局指令 (system message)，设为 "" 表示不使用')
    args = parser.parse_args()

    # 默认输出目录: 模型路径同级的 embeddings/
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.model.rstrip('/')), 'embeddings')

    num_gpus = min(args.num_gpus, gpu_count)
    script_path = os.path.abspath(__file__)

    print(f'Detected {gpu_count} GPU(s), using {num_gpus} for data parallelism')
    print(f'Model: {args.model}')
    print(f'Model type: {args.model_type}')
    print(f'Batch size per GPU: {args.batch_size}')
    print(f'Instruction: {args.instruction or "(none)"}')
    print(f'Input files: {args.input}')
    print(f'Output dir: {args.output_dir}')

    os.makedirs(args.output_dir, exist_ok=True)

    total_t0 = time.time()
    for input_path in args.input:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(args.output_dir, f'{base}_with_embedding.jsonl')

        # 跳过已完成的文件
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
