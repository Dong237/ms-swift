"""
IP 图片 Embedding 训练数据构建脚本
==================================

从 IP 图片文件夹构建 Qwen3.5 Embedding 训练 JSONL。
输入文件夹只包含训练用 IP，评估数据在别处单独处理。

文件夹结构：
  input_dir/
    ip_001/  (img1.jpg, img2.jpg, ...)
    ip_002/  (img1.jpg, img2.jpg, ...)
    ...

输出：
  单个 JSONL 文件，每行一个训练样本：
  {"messages": [{"role": "user", "content": "<image>"}],
   "images": ["path/to/anchor.jpg"],
   "positive_messages": [[{"role": "user", "content": "<image>"}]],
   "positive_images": [["path/to/positive.jpg"]],
   "negative_messages": [[...], ...],
   "negative_images": [[...], ...]}

用法：
  python build_ip_training_data.py \\
      --input_dir /mnt/bn/jinghan-lqa/data/IP/IP_image_train \\
      --output /mnt/bn/jinghan-lqa/data/IP/train.jsonl \\
      --num_negatives 5 \\
      --max_pairs_per_ip 20 \\
      --num_workers 16
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}

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
                self.n += 1
                if self.n - self._last_print >= max(1, self.total // 100):
                    self._last_print = self.n
                    pct = self.n / self.total * 100 if self.total else 0
                    print(f'\r  {self.desc}: {self.n}/{self.total} ({pct:.0f}%)', end='', flush=True)
            print()
        def update(self, n=1):
            self.n += n
            if self.n - self._last_print >= max(1, self.total // 100):
                self._last_print = self.n
                pct = self.n / self.total * 100 if self.total else 0
                print(f'\r  {self.desc}: {self.n}/{self.total} ({pct:.0f}%)', end='', flush=True)
        def close(self):
            print()


# ─── 多进程扫描文件夹 ─────────────────────────────────────────────────────────

def _scan_one_folder(folder_path: str) -> tuple[str, list[str]]:
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


def collect_ip_folders(root_dir: str, min_images: int, num_workers: int) -> dict[str, list[str]]:
    print(f'  Listing subdirectories in {root_dir} ...')
    t0 = time.time()
    subdirs = []
    with os.scandir(root_dir) as entries:
        for entry in entries:
            if entry.is_dir():
                subdirs.append(entry.path)
    print(f'  Found {len(subdirs)} subdirectories in {time.time() - t0:.1f}s')

    print(f'  Scanning images with {num_workers} workers ...')
    ip_folders = {}
    skipped = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_scan_one_folder, d): d for d in subdirs}
        pbar = tqdm(total=len(futures), desc='Scanning folders', unit='dir')
        for future in as_completed(futures):
            name, images = future.result()
            if len(images) >= min_images:
                ip_folders[name] = images
            else:
                skipped += 1
            pbar.update(1)
        pbar.close()

    print(f'  Valid IPs: {len(ip_folders)}, skipped (< {min_images} images): {skipped}')
    return ip_folders


# ─── 构建训练数据 ─────────────────────────────────────────────────────────────

_USER_IMAGE_MSG = {'role': 'user', 'content': '<image>'}


def build_training_jsonl(ip_folders: dict[str, list[str]], ip_names: list[str],
                         num_negatives: int, max_pairs_per_ip: int,
                         output_path: str, seed: int) -> int:
    rng = random.Random(seed)
    n_ips = len(ip_names)
    ip_to_idx = {name: i for i, name in enumerate(ip_names)}

    # 生成所有 positive pairs，全局 shuffle
    print('  Generating positive pairs ...')
    all_triplets = []

    for ip_name in tqdm(ip_names, desc='Pair generation', unit='IP'):
        images = ip_folders[ip_name]
        idx = ip_to_idx[ip_name]

        if len(images) <= max_pairs_per_ip:
            pairs = list(combinations(images, 2))
        else:
            pair_set = set()
            while len(pair_set) < max_pairs_per_ip:
                a, b = rng.sample(images, 2)
                key = (a, b) if a < b else (b, a)
                pair_set.add(key)
            pairs = list(pair_set)

        for anchor, positive in pairs:
            all_triplets.append((idx, anchor, positive))

    print(f'  Total positive pairs: {len(all_triplets):,}')
    print('  Shuffling ...')
    rng.shuffle(all_triplets)

    # 流式写入
    print(f'  Writing JSONL with {num_negatives} negatives per sample ...')
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for ip_idx, anchor, positive in tqdm(all_triplets, desc='Writing', unit='sample'):
            neg_msgs = []
            neg_imgs = []
            sampled = 0
            while sampled < num_negatives:
                neg_idx = rng.randint(0, n_ips - 1)
                if neg_idx == ip_idx:
                    continue
                neg_ip = ip_names[neg_idx]
                neg_img = rng.choice(ip_folders[neg_ip])
                neg_msgs.append([_USER_IMAGE_MSG])
                neg_imgs.append([neg_img])
                sampled += 1

            sample = {
                'messages': [_USER_IMAGE_MSG],
                'images': [anchor],
                'positive_messages': [[_USER_IMAGE_MSG]],
                'positive_images': [[positive]],
                'negative_messages': neg_msgs,
                'negative_images': neg_imgs,
            }
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            count += 1

    return count


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='IP 图片 Embedding 训练数据构建',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='训练 IP 图片根目录（每个子文件夹 = 一个 IP）')
    parser.add_argument('--output', type=str, required=True,
                        help='输出 JSONL 文件路径')
    parser.add_argument('--num_negatives', type=int, default=5,
                        help='每个样本的 explicit negative 数 (default: 5)')
    parser.add_argument('--max_pairs_per_ip', type=int, default=20,
                        help='每个 IP 最多 positive pair 数 (default: 20)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='文件夹扫描并行进程数 (default: 16)')
    parser.add_argument('--min_images', type=int, default=2,
                        help='每个 IP 至少需要的图片数 (default: 2)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    total_t0 = time.time()

    # ── 1. 扫描 ──
    print(f'\n[1/2] Scanning {args.input_dir}')
    ip_folders = collect_ip_folders(args.input_dir, args.min_images, args.num_workers)

    if not ip_folders:
        print('ERROR: No valid IP folders found.')
        sys.exit(1)

    counts = [len(imgs) for imgs in ip_folders.values()]
    print(f'  {len(ip_folders):,} IPs, {sum(counts):,} images')
    print(f'  Images/IP: min={min(counts)}, max={max(counts)}, avg={sum(counts)/len(counts):.1f}')

    # ── 2. 构建 ──
    ip_names = sorted(ip_folders.keys())
    print(f'\n[2/2] Building training data '
          f'(negatives={args.num_negatives}, max_pairs/ip={args.max_pairs_per_ip})')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    n_train = build_training_jsonl(
        ip_folders, ip_names, args.num_negatives, args.max_pairs_per_ip,
        args.output, args.seed)

    elapsed = time.time() - total_t0
    print(f'\n{"=" * 60}')
    print(f'Done in {elapsed:.1f}s ({elapsed/60:.1f}min)')
    print(f'  IPs used:          {len(ip_folders):,}')
    print(f'  Training samples:  {n_train:,}')
    print(f'  Output:            {args.output}')
    print(f'  File size:         {os.path.getsize(args.output) / 1024 / 1024:.1f} MB')
    print(f'\nTo train:')
    print(f'  INFONCE_USE_BATCH=False INFONCE_TEMPERATURE=0.05 \\')
    print(f'  swift sft \\')
    print(f'      --model Qwen/Qwen3.5-0.8B \\')
    print(f'      --model_type qwen3_5_emb \\')
    print(f'      --task_type embedding \\')
    print(f'      --loss_type infonce \\')
    print(f'      --system "提取图片中IP角色的视觉特征表示" \\')
    print(f'      --dataset {args.output} \\')
    print(f'      --attn_impl sdpa \\')
    print(f'      --dataloader_drop_last true')


if __name__ == '__main__':
    main()
