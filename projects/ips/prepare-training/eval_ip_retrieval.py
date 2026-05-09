"""
IP Embedding Retrieval Evaluation (End-to-End)
================================================

All-in-one script: scan IP folders → multi-GPU embed → save embeddings → evaluate.

Pipeline:
  1. Scan test IP folders, collect all images grouped by IP
  2. Multi-GPU parallel embedding (subprocess per GPU, same pattern as
     infer_embedding_multimodal_swift.py)
  3. Save embeddings as JSONL: one line per IP, keyed by image path
  4. Build query/gallery split (1 query per IP with ≥2 images, rest = gallery)
  5. Compute Recall@K, R-Precision (Precision@K_q), mAP@K
  6. Output metrics to terminal + log file + plots

Evaluation protocol (leave-one-out):
  Every image takes a turn as query against all other images. Deterministic,
  no random seed dependency, maximum statistical power.
  - All images form a single pool
  - Similarity matrix is N×N, diagonal masked (can't retrieve yourself)
  - Images from IPs with ≥2 images serve as queries (n_rel = siblings - 1)
  - Images from 1-image IPs are distractors (in pool but never query)

Embedding JSONL format (one line per IP):
  {"ip_name": "ip_001", "embeddings": {"/path/img1.jpg": [0.01, ...], "/path/img2.jpg": [...]}}

Smoke test tip:
  Before trusting results on real models, verify metric code with synthetic embeddings:
  assign each image embedding = one-hot vector by IP. Expect Recall@K = 1.0, mAP@K = 1.0.

Usage:
  python eval_ip_retrieval.py \\
      --model /path/to/checkpoint \\
      --test_dir /mnt/bn/youxiang-lf/data/facial_ip/IP_image_val \\
      --output /path/to/output_embeddings.jsonl \\
      --K 10

  # Re-evaluate from saved embeddings (skip inference):
  python eval_ip_retrieval.py \\
      --embeddings /path/to/output_embeddings.jsonl \\
      --K 10
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
DEFAULT_INSTRUCTION = '提取该IP角色的身份特征，关注角色本身而非背景或姿态'


# ─── Utilities ───────────────────────────────────────────────────────────────

def read_jsonl(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records, path):
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


# ─── Step 1: Scan IP folders ─────────────────────────────────────────────────

def _scan_one_folder(folder_path):
    name = os.path.basename(folder_path)
    images = []
    try:
        with os.scandir(folder_path) as entries:
            for entry in entries:
                if entry.is_file() and os.path.splitext(entry.name)[1].lower() in IMAGE_EXTS:
                    images.append(entry.path)
    except (PermissionError, OSError):
        pass
    images.sort()
    return name, images


def scan_ip_folders(test_dir, num_workers=8):
    print(f'[Step 1] Scanning {test_dir} ...')
    subdirs = [e.path for e in os.scandir(test_dir) if e.is_dir()]
    print(f'  Found {len(subdirs)} subdirectories')

    ip_folders = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
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
    return ip_folders


# ─── Step 2: Multi-GPU Embedding ─────────────────────────────────────────────

def build_infer_request(image_path, instruction):
    """Build InferRequest for a single image (image-only, no text)."""
    from swift.infer_engine import InferRequest

    # Image-only content — matches training format where content is `<image>`
    # ms-swift's remove_messages_media() converts this to <image> tag + images list
    content = [{'type': 'image', 'image': image_path}]
    messages = []
    if instruction:
        messages.append({'role': 'system', 'content': instruction})
    messages.append({'role': 'user', 'content': content})
    return InferRequest(messages=messages)


def run_worker(rank, model, model_type, batch_size, instruction, shard_input, shard_output):
    """Worker subprocess: load model on one GPU, embed its shard.
    CUDA_VISIBLE_DEVICES is already set by caller before any torch import."""
    import torch
    from swift.infer_engine import TransformersEngine

    records = read_jsonl(shard_input)
    print(f'[GPU {rank}] {len(records)} images, loading model ...')

    engine = TransformersEngine(
        model,
        model_type=model_type,
        task_type='embedding',
        torch_dtype=torch.bfloat16,
    )
    print(f'[GPU {rank}] Model loaded. Starting inference ...')

    t0 = time.time()
    n_done = 0
    total = len(records)

    with open(shard_output, 'w', encoding='utf-8') as f_out:
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_records = records[batch_start:batch_end]

            infer_requests = [
                build_infer_request(r['image_path'], instruction)
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
            print(f'[GPU {rank}] {n_done}/{total} '
                  f'({n_done / total * 100:.1f}%) '
                  f'| {speed:.0f} img/s | ETA {eta:.0f}s   ', end='\r')

    elapsed = time.time() - t0
    print(f'\n[GPU {rank}] Done: {n_done} images in {elapsed:.1f}s '
          f'({n_done / elapsed:.0f} img/s)')


def embed_all_images(ip_folders, model_path, model_type, instruction,
                     batch_size, num_gpus, output_path):
    """
    Embed all images across all IPs using multi-GPU data parallelism.
    Saves result as JSONL: one line per IP with all its embeddings.
    """
    print(f'\n[Step 2] Embedding all images ({num_gpus} GPUs, batch={batch_size}) ...')

    # Flatten to records
    flat_records = []
    for ip_name in sorted(ip_folders.keys()):
        for img_path in ip_folders[ip_name]:
            flat_records.append({'ip_name': ip_name, 'image_path': img_path})

    total = len(flat_records)
    print(f'  Total images to embed: {total}')

    tmp_dir = tempfile.mkdtemp(prefix='ip_eval_shards_')
    script_path = os.path.abspath(__file__)

    # Round-robin shard across GPUs
    shard_inputs, shard_outputs = [], []
    for rank in range(num_gpus):
        shard = flat_records[rank::num_gpus]
        si = os.path.join(tmp_dir, f'shard_in_{rank}.jsonl')
        so = os.path.join(tmp_dir, f'shard_out_{rank}.jsonl')
        write_jsonl(shard, si)
        shard_inputs.append(si)
        shard_outputs.append(so)

    # Launch workers
    procs = []
    t0 = time.time()
    for rank in range(num_gpus):
        worker_args = json.dumps({
            'rank': rank,
            'model': model_path,
            'model_type': model_type,
            'batch_size': batch_size,
            'instruction': instruction,
            'shard_input': shard_inputs[rank],
            'shard_output': shard_outputs[rank],
        })
        cmd = [sys.executable, script_path, '--__worker__', worker_args]
        procs.append(subprocess.Popen(cmd))

    for p in procs:
        p.wait()

    # Hard-fail if any worker crashed
    failed = [i for i, p in enumerate(procs) if p.returncode != 0]
    if failed:
        raise RuntimeError(
            f'GPU workers {failed} failed with non-zero exit codes. '
            f'Check GPU memory (try smaller --batch_size) or model path.')

    elapsed = time.time() - t0
    print(f'  All GPUs done in {elapsed:.1f}s ({total / elapsed:.0f} img/s)')

    # Merge: simple concatenation (order doesn't matter, we group by ip_name)
    print('  Merging shard outputs ...')
    merged = []
    for so in shard_outputs:
        if os.path.exists(so):
            with open(so, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        merged.append(json.loads(line))

    # Verify no data loss
    if len(merged) != total:
        raise RuntimeError(
            f'Data loss during merge: expected {total} records, got {len(merged)}. '
            f'Some GPU workers may have partially failed.')

    print(f'  Merged {len(merged)}/{total} records')

    # Group by IP and save JSONL
    print(f'  Saving embeddings to {output_path} ...')
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ip_embeddings = defaultdict(dict)
    for r in merged:
        ip_embeddings[r['ip_name']][r['image_path']] = r['embedding']

    with open(output_path, 'w', encoding='utf-8') as f:
        for ip_name in sorted(ip_embeddings.keys()):
            line = {
                'ip_name': ip_name,
                'embeddings': ip_embeddings[ip_name],
            }
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

    print(f'  Saved {len(ip_embeddings)} IPs to {output_path}')

    # Cleanup temp files
    for p in shard_inputs + shard_outputs:
        if os.path.exists(p):
            os.remove(p)
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    return ip_embeddings


def load_embeddings(embeddings_path):
    """Load pre-computed embeddings JSONL."""
    print(f'[Step 2] Loading embeddings from {embeddings_path} ...')
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
    return ip_embeddings


# ─── Step 3: Leave-one-out evaluation ─────────────────────────────────────────

def evaluate(ip_embeddings, K, metadata=None):
    """
    Leave-one-out retrieval evaluation.

    Every image takes a turn as query against ALL other images (gallery = everything
    minus self). Deterministic — no random seed needed.

    - All images go into a single pool
    - Similarity matrix is N×N, diagonal masked to -inf (can't retrieve yourself)
    - Only images from IPs with >=2 images serve as queries (1-image IPs are
      distractors in the gallery but have n_rel=0 so can't be evaluated)
    - n_rel per query = (images in same IP) - 1

    Metrics computed for k in {1, 3, 5, K} (deduplicated):
      - CMC@k: fraction of queries with at least 1 correct in top k (binary hit)
      - Recall@k: fraction of relevant items found in top k (coverage)
      - mAP@k, R-Precision
    """
    print(f'\n[Step 3] Leave-one-out evaluation (K={K}) ...')

    # Build flat arrays: all images in one pool
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
    all_ips_arr = np.array(all_ips)

    # Verify embedding dimensions are consistent
    dims = set(len(e) for e in all_embs)
    if len(dims) > 1:
        raise ValueError(f'Inconsistent embedding dimensions across images: {dims}')

    # Identify which images are valid queries (from IPs with >=2 images)
    query_mask = np.array([ip_image_count[ip] >= 2 for ip in all_ips])
    n_queries = int(query_mask.sum())
    n_distractors = sum(1 for ip, cnt in ip_image_count.items() if cnt < 2)

    print(f'  Total images:  {N}')
    print(f'  Queries:       {n_queries} (every image from IPs with >= 2 images)')
    print(f'  Distractor IPs: {n_distractors} (1-image IPs, in gallery only)')

    # Normalize and compute similarity
    # NOTE: N×N matrix. At N=5000 → 100MB (fine). At N=50000 → 10GB.
    # If test set grows past ~20k images, chunk the computation.
    E = np.array(all_embs, dtype=np.float32)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)

    print(f'  Computing similarity matrix ({N} x {N}) ...')
    sim_matrix = E @ E.T
    np.fill_diagonal(sim_matrix, -np.inf)  # can't retrieve yourself

    # K values to report
    k_values = sorted(set([1, 3, 5, K]))
    max_k = max(k_values)

    # Per-query metrics
    cmc_at = {k: [] for k in k_values}  # CMC: binary hit (at least 1 correct in top k)
    recall_at = {k: [] for k in k_values}  # Recall: coverage (fraction of relevant found)
    ap_at = {k: [] for k in k_values}
    r_precision_list = []
    per_query = []

    query_indices = np.where(query_mask)[0]
    print(f'  Evaluating {n_queries} queries ...')

    for idx, i in enumerate(query_indices):
        q_ip = all_ips[i]
        n_rel = ip_image_count[q_ip] - 1  # siblings in the pool (excluding self)
        sims = sim_matrix[i]

        # Get top indices (enough for both max_k and R-Precision)
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
            'n_relevant': n_rel,
        }

        for k in k_values:
            rel_k = relevance[:k]
            n_rel_in_k = int(rel_k.sum())

            # CMC@k: did we find at least 1 correct? (binary)
            cmc = 1.0 if n_rel_in_k > 0 else 0.0
            cmc_at[k].append(cmc)

            # Recall@k: what fraction of all relevant did we find?
            rec = n_rel_in_k / n_rel
            recall_at[k].append(rec)

            # AP@k (truncated)
            ap = 0.0
            n_rel_so_far = 0
            for j in range(min(k, len(relevance))):
                if relevance[j] == 1:
                    n_rel_so_far += 1
                    ap += n_rel_so_far / (j + 1)
            denom = min(n_rel, k)
            ap = ap / denom if denom > 0 else 0.0
            ap_at[k].append(ap)

            query_result[f'cmc@{k}'] = round(cmc, 4)
            query_result[f'recall@{k}'] = round(rec, 4)
            query_result[f'ap@{k}'] = round(ap, 4)

        # R-Precision
        r_prec_rel = relevance[:n_rel]
        r_prec = float(r_prec_rel.sum()) / n_rel
        r_precision_list.append(r_prec)
        query_result['r_precision'] = round(r_prec, 4)

        # Top-5 details
        n_detail = min(5, len(top_indices))
        query_result['top5_ips'] = top_ips[:n_detail].tolist()
        query_result['top5_sims'] = [round(float(sims[top_indices[j]]), 4)
                                     for j in range(n_detail)]
        query_result['top5_correct'] = relevance[:n_detail].tolist()

        per_query.append(query_result)

        if (idx + 1) % 500 == 0 or idx + 1 == n_queries:
            print(f'    {idx+1}/{n_queries} queries done', end='\r')
    print()

    # Aggregate
    metrics = {}
    for k in k_values:
        metrics[f'CMC@{k}'] = round(float(np.mean(cmc_at[k])), 4)
        metrics[f'Recall@{k}'] = round(float(np.mean(recall_at[k])), 4)
        metrics[f'mAP@{k}'] = round(float(np.mean(ap_at[k])), 4)
    metrics['R-Precision'] = round(float(np.mean(r_precision_list)), 4)
    metrics['n_queries'] = n_queries
    metrics['n_total_images'] = N
    metrics['n_distractor_ips'] = n_distractors
    metrics['K'] = K
    if metadata:
        metrics['_metadata'] = metadata

    return metrics, per_query, cmc_at, recall_at, r_precision_list, k_values


# ─── Step 4: Output results ──────────────────────────────────────────────────

def print_results(metrics, per_query, cmc_at, recall_at, r_precision_list, K,
                  k_values, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    report_lines = []
    report_lines.append('=' * 60)
    report_lines.append('IP EMBEDDING RETRIEVAL EVALUATION')
    report_lines.append('=' * 60)
    report_lines.append(f'  Protocol:          Leave-one-out (every image queries all others)')
    report_lines.append(f'  Queries:           {metrics["n_queries"]}')
    report_lines.append(f'  Total images:      {metrics["n_total_images"]}')
    report_lines.append(f'  Distractor IPs:    {metrics["n_distractor_ips"]}')
    report_lines.append(f'  K:                 {K}')

    # CMC table — most important for production (VLM just needs ≥1 correct)
    report_lines.append(f'')
    report_lines.append(f'  ── CMC (hit rate: at least 1 correct in top K) ──')
    for k in k_values:
        marker = '  <-- production cutoff' if k == K else ''
        report_lines.append(f'  CMC@{k:<4d}          {metrics[f"CMC@{k}"]:.4f}{marker}')

    # Recall table
    report_lines.append(f'')
    report_lines.append(f'  ── Recall (coverage: fraction of relevant found in top K) ──')
    for k in k_values:
        marker = '  <-- primary' if k == K else ''
        report_lines.append(f'  Recall@{k:<4d}        {metrics[f"Recall@{k}"]:.4f}{marker}')

    # R-Precision
    report_lines.append(f'')
    report_lines.append(f'  ── R-Precision (Precision@K_q) ──')
    report_lines.append(f'  R-Precision:       {metrics["R-Precision"]:.4f}')

    # mAP table
    report_lines.append(f'')
    report_lines.append(f'  ── Ranking quality ──')
    for k in k_values:
        marker = '  <-- primary' if k == K else ''
        report_lines.append(f'  mAP@{k:<4d}          {metrics[f"mAP@{k}"]:.4f}{marker}')

    # Breakdown by n_relevant
    buckets = defaultdict(list)
    for q in per_query:
        buckets[q['n_relevant']].append(q[f'recall@{K}'])

    report_lines.append(f'')
    report_lines.append(f'  ── Recall@{K} by IP size (n_relevant) ──')
    for n_rel in sorted(buckets.keys()):
        vals = buckets[n_rel]
        report_lines.append(f'    n_rel={n_rel:>2d}: Recall@{K}={np.mean(vals):.4f}  '
                            f'(n={len(vals)} queries)')

    # Failure analysis
    failures = [q for q in per_query if q[f'recall@{K}'] < 1.0]
    report_lines.append(f'')
    report_lines.append(f'  ── Failures: {len(failures)}/{len(per_query)} queries with '
                        f'Recall@{K} < 1.0 ({len(failures)/len(per_query)*100:.1f}%) ──')

    if failures:
        failures_sorted = sorted(failures, key=lambda q: (q[f'recall@{K}'], q['r_precision']))
        top_n = min(15, len(failures_sorted))
        report_lines.append(f'  Top {top_n} worst:')
        report_lines.append(f'    {"IP":>12s}  {"n_rel":>5s}  {"R@"+str(K):>6s}  '
                            f'{"R-Prec":>6s}  {"AP@"+str(K):>6s}')
        report_lines.append('    ' + '-' * 50)
        for q in failures_sorted[:top_n]:
            report_lines.append(f'    {q["query_ip"]:>12s}  {q["n_relevant"]:>5d}  '
                                f'{q[f"recall@{K}"]:>6.3f}  {q["r_precision"]:>6.3f}  '
                                f'{q[f"ap@{K}"]:>6.3f}')

    report = '\n'.join(report_lines)
    print(f'\n{report}')

    # Timestamped filenames to avoid clobbering across runs
    ts = time.strftime('%Y%m%d_%H%M%S')

    log_path = os.path.join(output_dir, f'eval_log_{ts}.txt')
    with open(log_path, 'w') as f:
        f.write(report + '\n')
    print(f'\n  Log saved to {log_path}')

    metrics_path = os.path.join(output_dir, f'eval_metrics_{ts}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'  Metrics saved to {metrics_path}')

    pq_path = os.path.join(output_dir, f'per_query_results_{ts}.jsonl')
    with open(pq_path, 'w') as f:
        for q in per_query:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')
    print(f'  Per-query details saved to {pq_path}')

    # Plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # R-Precision histogram
        axes[0].hist(r_precision_list, bins=20, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=metrics['R-Precision'], color='red', linestyle='--',
                        label=f'Mean={metrics["R-Precision"]:.3f}')
        axes[0].set_xlabel('R-Precision')
        axes[0].set_ylabel('Count')
        axes[0].set_title('R-Precision Distribution')
        axes[0].legend()

        # Recall@K by n_relevant
        x_vals = sorted(buckets.keys())
        y_vals = [np.mean(buckets[x]) for x in x_vals]
        axes[1].bar(x_vals, y_vals, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('n_relevant')
        axes[1].set_ylabel(f'Mean Recall@{K}')
        axes[1].set_title(f'Recall@{K} by IP Size')
        axes[1].set_ylim(0, 1.05)
        for x, y in zip(x_vals, y_vals):
            axes[1].text(x, y + 0.02, f'{y:.2f}', ha='center', fontsize=8)

        # Recall@K histogram
        axes[2].hist(recall_at[K], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        axes[2].axvline(x=metrics[f'Recall@{K}'], color='red', linestyle='--',
                        label=f'Mean={metrics[f"Recall@{K}"]:.3f}')
        axes[2].set_xlabel(f'Recall@{K}')
        axes[2].set_ylabel('Count')
        axes[2].set_title(f'Recall@{K} Distribution')
        axes[2].legend()

        fig.tight_layout()
        plot_path = os.path.join(output_dir, f'eval_plots_{ts}.png')
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Plots saved to {plot_path}')
    except ImportError:
        print('  matplotlib not installed, skipping plots')


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    # Worker subprocess mode — set device visibility BEFORE any torch import
    if '--__worker__' in sys.argv:
        idx = sys.argv.index('--__worker__')
        w_args = json.loads(sys.argv[idx + 1])
        rank_str = str(w_args['rank'])
        # Support both CUDA GPUs and Ascend 910B NPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = rank_str
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = rank_str
        run_worker(**w_args)
        return

    parser = argparse.ArgumentParser(
        description='IP Embedding Retrieval Evaluation (end-to-end)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Inference args (skip if --embeddings provided)
    parser.add_argument('--model', type=str, default=None,
                        help='Model checkpoint path')
    parser.add_argument('--model_type', type=str, default='qwen3_5_emb',
                        help='Model type (default: qwen3_5_emb)')
    parser.add_argument('--instruction', type=str, default=DEFAULT_INSTRUCTION,
                        help='System prompt — MUST match training. '
                             'Only used during inference, ignored with --embeddings.')
    parser.add_argument('--test_dir', type=str, default=None,
                        help='Test IP folders root (e.g. /mnt/.../IP_image_val)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size per GPU (default: 64)')
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='Number of GPUs (default: all available)')

    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Output embeddings JSONL path')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output dir for metrics/logs (default: same dir as --output)')

    # Re-evaluate from saved embeddings
    parser.add_argument('--embeddings', type=str, default=None,
                        help='Pre-computed embeddings JSONL (skip inference)')

    # Eval config
    parser.add_argument('--K', type=int, default=10,
                        help='K for Recall@K and mAP@K (default: 10)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Workers for folder scanning')

    args = parser.parse_args()

    total_t0 = time.time()

    # ── Get embeddings ──
    if args.embeddings:
        ip_embeddings = load_embeddings(args.embeddings)
        if args.output_dir is None:
            args.output_dir = os.path.dirname(os.path.abspath(args.embeddings))
    elif args.model and args.test_dir:
        import torch
        if args.num_gpus is None:
            # Auto-detect: CUDA GPUs or Ascend NPUs
            try:
                from swift.utils import get_device_count
                args.num_gpus = get_device_count()
            except ImportError:
                args.num_gpus = torch.cuda.device_count()
        if args.num_gpus == 0:
            raise RuntimeError('No accelerator devices detected (CUDA GPU or Ascend NPU).')

        if args.output is None:
            model_dir = os.path.dirname(os.path.abspath(args.model.rstrip('/')))
            if os.path.isdir(model_dir):
                args.output = os.path.join(model_dir, 'eval_embeddings.jsonl')
            else:
                args.output = os.path.join(os.getcwd(), 'eval_embeddings.jsonl')
        if args.output_dir is None:
            args.output_dir = os.path.dirname(os.path.abspath(args.output))

        # Print instruction prominently — mismatch with training is a silent killer
        print('=' * 60)
        print(f'  INSTRUCTION (system prompt): "{args.instruction}"')
        print(f'  This MUST match the --system used during training.')
        print(f'  If they differ, metrics will silently underperform.')
        print('=' * 60)

        # Skip embedding if output already exists
        if os.path.exists(args.output) and os.path.getsize(args.output) > 0:
            print(f'Embeddings file exists, loading: {args.output}')
            ip_embeddings = load_embeddings(args.output)
        else:
            ip_folders = scan_ip_folders(args.test_dir, args.num_workers)
            ip_embeddings = embed_all_images(
                ip_folders, args.model, args.model_type, args.instruction,
                args.batch_size, args.num_gpus, args.output)
    else:
        parser.error('Provide either --embeddings, or both --model and --test_dir')

    # ── Evaluate ──
    metadata = {
        'model_path': getattr(args, 'model', None) or '(loaded from embeddings)',
        'embeddings_path': args.embeddings,
        'instruction': args.instruction,
        'test_dir': getattr(args, 'test_dir', None),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    metrics, per_query, cmc_at, recall_at, r_precision_list, k_values = evaluate(
        ip_embeddings, args.K, metadata)

    # ── Output ──
    print_results(metrics, per_query, cmc_at, recall_at, r_precision_list,
                  args.K, k_values, args.output_dir)

    total_elapsed = time.time() - total_t0
    print(f'\nTotal time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)')


if __name__ == '__main__':
    main()
