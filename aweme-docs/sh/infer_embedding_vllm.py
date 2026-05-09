"""
Batch embedding inference using vLLM for trained Qwen3-Embedding models.

Usage:
    python infer_embedding_vllm.py \
        --model /mnt/bn/youxiang-hl/models_emb_intent/Qwen3-EMB-0.6B-intent-l1-labels-text/checkpoint-XXX \
        --input /path/to/train.jsonl /path/to/test.jsonl \
        --output_dir /path/to/output/ \
        --tensor_parallel_size 1 \
        --max_model_len 512

Input JSONL format (one per line):
    {"prompt": "Instruct: ...\nQuery: ...", "label": "..."}

Output JSONL format (one per line):
    {"prompt": "...", "label": "...", "embedding": [0.01, -0.02, ...]}
"""

import argparse
import json
import os
import time

from vllm import LLM


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


def main():
    parser = argparse.ArgumentParser(description='Batch embedding inference with vLLM')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, nargs='+', required=True,
                        help='Input JSONL file(s)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save output JSONL files')
    parser.add_argument('--prompt_key', type=str, default='prompt',
                        help='Key in JSONL for the text to encode (default: prompt)')
    parser.add_argument('--max_model_len', type=int, default=512,
                        help='Max sequence length (should match training max_length)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['bfloat16', 'float16', 'float32'],
                        help='Model dtype')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help='Number of GPUs for tensor parallelism')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                        help='GPU memory utilization ratio')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model in pooling mode
    print(f'Loading model from {args.model} ...')
    llm = LLM(
        model=args.model,
        task='embed',
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        override_pooler_config='{"pooling_type": "LAST", "normalize": true}',
    )
    print('Model loaded.')

    # Process each input file
    for input_path in args.input:
        basename = os.path.basename(input_path)
        name, ext = os.path.splitext(basename)
        output_path = os.path.join(args.output_dir, f'{name}_with_embedding{ext}')

        print(f'\nProcessing: {input_path}')
        records = read_jsonl(input_path)
        print(f'  Total records: {len(records)}')

        prompts = [r[args.prompt_key] for r in records]

        # vLLM handles batching/scheduling internally
        t0 = time.time()
        outputs = llm.encode(prompts)
        elapsed = time.time() - t0
        print(f'  Encoding done in {elapsed:.1f}s ({len(prompts) / elapsed:.0f} samples/s)')

        # Attach embeddings to records
        for record, output in zip(records, outputs):
            embedding = output.outputs.data
            # Convert to plain list for JSON serialization
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            else:
                embedding = list(embedding)
            record['embedding'] = embedding

        write_jsonl(records, output_path)
        print(f'  Saved to: {output_path}')

    print('\nAll done.')


if __name__ == '__main__':
    main()
