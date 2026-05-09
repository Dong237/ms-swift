#!/usr/bin/env python3
"""
Count IP folder distribution + projected training data per bucket.
Shows both image counts and how many training pairs each bucket produces,
so you can directly feel the data imbalance.
"""

import os
import sys
from collections import Counter
from itertools import combinations
from math import comb
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not installed. Run: pip install tqdm")
    sys.exit(1)

# ---- Config ----
ROOT = Path("/mnt/bn/jinghan-lqa/data/IP/IP_image")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff", ".tif"}
BUCKETS = [0, 1, 2, 3, 5, 10, 20, 50, 100, 500, float("inf")]
MAX_PAIRS_PER_IP = 20  # same as training script default
NUM_NEGATIVES = 5
# ---------------


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def count_images_in_folder(folder: Path) -> int:
    try:
        return sum(1 for p in folder.iterdir() if p.is_file() and is_image(p))
    except (PermissionError, OSError) as e:
        print(f"[WARN] can't read {folder}: {e}", file=sys.stderr)
        return -1


def pairs_from_n(n: int, max_pairs: int) -> int:
    """How many training samples an IP with n images produces."""
    if n < 2:
        return 0
    return min(comb(n, 2), max_pairs)


def bucket_label(n: int) -> str:
    for i in range(len(BUCKETS) - 1):
        lo, hi = BUCKETS[i], BUCKETS[i + 1]
        if lo < n <= hi:
            if hi == float("inf"):
                return f">{int(lo)}"
            return f"{int(lo)+1}-{int(hi)}" if lo + 1 != hi else f"{int(hi)}"
    return str(n)


def main():
    if not ROOT.exists():
        print(f"[ERROR] {ROOT} does not exist")
        sys.exit(1)

    print(f"Scanning {ROOT} ...")
    ip_folders = [p for p in ROOT.iterdir() if p.is_dir()]
    print(f"Found {len(ip_folders)} candidate IP folders\n")

    counts = {}
    errors = []

    for folder in tqdm(ip_folders, desc="Counting", unit="folder"):
        n = count_images_in_folder(folder)
        if n < 0:
            errors.append(folder.name)
        else:
            counts[folder.name] = n

    if not counts:
        print("No folders counted successfully.")
        return

    vals = list(counts.values())
    total_imgs = sum(vals)
    non_empty = [v for v in vals if v > 0]

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total IP folders       : {len(counts)}")
    print(f"Non-empty folders      : {len(non_empty)}")
    print(f"Empty folders          : {len(vals) - len(non_empty)}")
    print(f"Total images           : {total_imgs}")
    if non_empty:
        non_empty_sorted = sorted(non_empty)
        print(f"Min / Max images       : {min(non_empty)} / {max(non_empty)}")
        print(f"Mean / Median images   : "
              f"{sum(non_empty)/len(non_empty):.2f} / "
              f"{non_empty_sorted[len(non_empty)//2]}")
        def pct(p):
            k = max(0, min(len(non_empty_sorted) - 1,
                           int(round(p / 100 * (len(non_empty_sorted) - 1)))))
            return non_empty_sorted[k]
        print(f"P10 / P25 / P75 / P90  : "
              f"{pct(10)} / {pct(25)} / {pct(75)} / {pct(90)}")

    # ── Distribution with training pair projection ──
    print(f"\nDistribution (max_pairs_per_ip={MAX_PAIRS_PER_IP}, num_negatives={NUM_NEGATIVES}):")
    print(f"{'':>10s}  {'IPs':>6s}  {'C(n,2)':>7s}  {'Capped':>7s}  {'Pairs':>10s}  "
          f"{'Pairs%':>7s}  {'Fwd/ep':>10s}  {'Pairs/IP':>8s}  Bar")
    print(f"{'':>10s}  {'':>6s}  {'':>7s}  {'':>7s}  {'(total)':>10s}  "
          f"{'':>7s}  {'(total)':>10s}  {'(avg)':>8s}")
    print("-" * 105)

    # Group by bucket
    order = []
    for i in range(len(BUCKETS) - 1):
        lo, hi = BUCKETS[i], BUCKETS[i + 1]
        if hi == float("inf"):
            order.append((lo, hi, f">{int(lo)}"))
        else:
            order.append((lo, hi, f"{int(lo)+1}-{int(hi)}" if lo + 1 != hi else f"{int(hi)}"))

    bucket_data = []  # (label, n_ips, raw_pairs, capped_pairs, total_pairs, fwd_per_epoch)
    grand_total_pairs = 0
    grand_total_fwd = 0

    for lo, hi, label in order:
        ips_in_bucket = [v for v in vals if lo < v <= hi]
        n_ips = len(ips_in_bucket)
        if n_ips == 0:
            bucket_data.append((label, 0, 0, 0, 0, 0))
            continue

        raw_pairs_list = [comb(v, 2) for v in ips_in_bucket]
        capped_pairs_list = [pairs_from_n(v, MAX_PAIRS_PER_IP) for v in ips_in_bucket]
        total_raw = sum(raw_pairs_list)
        total_capped = sum(capped_pairs_list)
        # Forward passes per epoch: each sample = 1 anchor + 1 pos + N neg = 2+N
        total_fwd = total_capped * (2 + NUM_NEGATIVES)
        avg_pairs = total_capped / n_ips if n_ips else 0

        grand_total_pairs += total_capped
        grand_total_fwd += total_fwd
        bucket_data.append((label, n_ips, total_raw, total_capped, total_fwd, avg_pairs))

    max_pairs_for_bar = max(d[3] for d in bucket_data) if bucket_data else 1
    max_bar = 40

    for label, n_ips, total_raw, total_capped, total_fwd, avg_pairs in bucket_data:
        if n_ips == 0:
            print(f"  {label:>8s}  {0:>6d}  {0:>7d}  {0:>7d}  {0:>10d}  {0:>6.1f}%  {0:>10d}  {0:>8.1f}")
            continue
        pct_pairs = total_capped / grand_total_pairs * 100 if grand_total_pairs else 0
        bar = "█" * int(total_capped / max_pairs_for_bar * max_bar) if total_capped else ""
        print(f"  {label:>8s}  {n_ips:>6d}  {total_raw:>7d}  {total_capped:>7d}  "
              f"{total_capped:>10,d}  {pct_pairs:>6.1f}%  {total_fwd:>10,d}  "
              f"{avg_pairs:>8.1f}  {bar}")

    print("-" * 105)
    print(f"  {'TOTAL':>8s}  {len(vals):>6d}  {'':>7s}  {'':>7s}  "
          f"{grand_total_pairs:>10,d}  {'100%':>7s}  {grand_total_fwd:>10,d}")

    # ── Per-image-count breakdown (the real imbalance view) ──
    print(f"\nPer-image-count detail (n=1..10):")
    print(f"  {'n_imgs':>6s}  {'IPs':>6s}  {'C(n,2)':>6s}  {'Capped':>6s}  "
          f"{'TotalPairs':>10s}  {'Pairs%':>7s}  {'vs n=2':>7s}")
    print("-" * 70)

    pairs_at_2 = 0  # reference: how many pairs does n=2 produce per IP
    detail_data = []
    for n_imgs in range(1, 11):
        n_ips = sum(1 for v in vals if v == n_imgs)
        raw = comb(n_imgs, 2)
        capped = pairs_from_n(n_imgs, MAX_PAIRS_PER_IP)
        total = n_ips * capped
        if n_imgs == 2:
            pairs_at_2 = capped  # = 1
        detail_data.append((n_imgs, n_ips, raw, capped, total))

    for n_imgs, n_ips, raw, capped, total in detail_data:
        pct_pairs = total / grand_total_pairs * 100 if grand_total_pairs else 0
        ratio = f"{capped/pairs_at_2:.0f}x" if pairs_at_2 else "-"
        print(f"  {n_imgs:>6d}  {n_ips:>6d}  {raw:>6d}  {capped:>6d}  "
              f"{total:>10,d}  {pct_pairs:>6.1f}%  {ratio:>7s}")

    # ── Dropped IPs ──
    dropped = sum(1 for v in vals if v < 2)
    trainable = sum(1 for v in vals if v >= 2)
    print(f"\n[Training readiness]")
    print(f"  Trainable IPs (>= 2 images): {trainable:,d} ({trainable/len(vals)*100:.1f}%)")
    print(f"  Dropped IPs   (1 image):     {dropped:,d} ({dropped/len(vals)*100:.1f}%)")
    print(f"  Total training samples:       {grand_total_pairs:,d}")
    print(f"  Forward passes / epoch:       {grand_total_fwd:,d}")

    # ── Imbalance warning ──
    if detail_data:
        max_ratio = max(d[3] for d in detail_data if d[3] > 0)
        min_ratio = min(d[3] for d in detail_data if d[3] > 0)
        print(f"\n[Imbalance] Pairs/IP ranges from {min_ratio} (n=2) to {max_ratio} (n>=7)")
        print(f"  That's a {max_ratio}x imbalance between least and most represented IPs.")
        if max_ratio > 10:
            print(f"  Consider --max_pairs_per_ip {min(10, max_ratio//2)} to reduce to "
                  f"{min(10, max_ratio//2)}x")

    # ── Top/Bottom ──
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 folders by image count:")
    for name, n in sorted_items[:10]:
        p = pairs_from_n(n, MAX_PAIRS_PER_IP)
        print(f"  {n:>5d} imgs  {p:>3d} pairs  {name}")
    print("\nBottom 10 non-empty folders:")
    bottom = [(k, v) for k, v in sorted_items if v > 0][-10:]
    for name, n in bottom:
        p = pairs_from_n(n, MAX_PAIRS_PER_IP)
        print(f"  {n:>5d} imgs  {p:>3d} pairs  {name}")

    if errors:
        print(f"\n[WARN] {len(errors)} folder(s) could not be read, e.g.: {errors[:5]}")

    # ── Save CSV ──
    out_csv = ROOT.parent / "ip_folder_counts.csv"
    try:
        with open(out_csv, "w") as f:
            f.write("folder,image_count,training_pairs\n")
            for name, n in sorted_items:
                f.write(f"{name},{n},{pairs_from_n(n, MAX_PAIRS_PER_IP)}\n")
        print(f"\nPer-folder counts saved to: {out_csv}")
    except OSError as e:
        print(f"\n[WARN] could not write CSV: {e}")


if __name__ == "__main__":
    main()
