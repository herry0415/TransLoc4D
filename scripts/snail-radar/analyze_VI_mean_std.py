#!/usr/bin/env python3
"""
Compute global statistics (mean, std, min, max) for the V and intensity attributes
(across all .npy point-cloud files in specified subfolders of a root directory)
using multiprocessing, and save results to JSON.

– Fourth column (index 3) is the processed V attribute.
– Fifth column (index 4) is the intensity attribute.

Results are written to `norm_stats/<root_folder_name>.json` next to this script.
"""
import os
import sys
import argparse
import json
import numpy as np
import math
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def process_file(file_path):
    """
    Load one .npy file and compute:
      n           – number of points
      sum_v       – sum of V values
      sumsq_v     – sum of squares of V values
      min_v, max_v
      sum_i       – sum of intensity values
      sumsq_i     – sum of squares of intensity values
      min_i, max_i
    Returns a tuple or None on error/empty.
    """
    try:
        data = np.load(file_path)
    except Exception as e:
        print(f"Warning: failed to load {file_path}: {e}", file=sys.stderr)
        return None

    # need at least 5 columns now (x,y,z,V,I,...)
    if data.ndim != 2 or data.shape[1] < 5:
        print(f"Warning: unexpected shape {data.shape} in {file_path}", file=sys.stderr)
        return None

    # V in column index 3, intensity in column index 4
    v = data[:, 3].astype(np.float64)
    i = data[:, 4].astype(np.float64)
    n = v.size
    if n == 0:
        return None

    sum_v   = float(v.sum())
    sumsq_v = float((v * v).sum())
    min_v   = float(v.min())
    max_v   = float(v.max())

    sum_i   = float(i.sum())
    sumsq_i = float((i * i).sum())
    min_i   = float(i.min())
    max_i   = float(i.max())

    return (n, sum_v, sumsq_v, min_v, max_v,
            sum_i, sumsq_i, min_i, max_i)

def main():
    parser = argparse.ArgumentParser(
        description="Compute mean, std, min, max for V and intensity across .npy pointclouds."
    )
    parser.add_argument(
        '--root_dir', type=str, default="/datasets/snail-radar_tl4d",
        help="Root directory containing dataset subfolders."
    )
    parser.add_argument(
        '--rel_dirs', nargs='+', default=["snail/train/query/pointclouds", "snail/train/database/pointclouds"],
        help="List of relative folder names under root_dir to process."
    )
    parser.add_argument(
        '--processes', type=int, default=cpu_count(),
        help="Number of parallel worker processes (default: all CPUs)."
    )
    args = parser.parse_args()

    # gather all .npy files from each specified subfolder (non-recursive)
    files = []
    for rel in args.rel_dirs:
        folder = os.path.join(args.root_dir, rel)
        if not os.path.isdir(folder):
            print(f"Warning: {folder} is not a directory", file=sys.stderr)
            continue
        for fn in os.listdir(folder):
            if fn.lower().endswith('.npy'):
                files.append(os.path.join(folder, fn))

    if not files:
        print("Error: no .npy files found in specified folders.", file=sys.stderr)
        sys.exit(1)

    # global accumulators
    total_n       = 0
    sum_v         = 0.0
    sumsq_v       = 0.0
    global_min_v  = float('inf')
    global_max_v  = float('-inf')
    sum_i         = 0.0
    sumsq_i       = 0.0
    global_min_i  = float('inf')
    global_max_i  = float('-inf')

    with Pool(processes=args.processes) as pool:
        for res in tqdm(pool.imap_unordered(process_file, files, chunksize=10),
                        total=len(files), desc="Processing files"):
            if res is None:
                continue
            (n, sv, sqv, minv, maxv,
             si, sqi, mini, maxi) = res

            total_n += n

            sum_v   += sv
            sumsq_v += sqv
            global_min_v = min(global_min_v, minv)
            global_max_v = max(global_max_v, maxv)

            sum_i   += si
            sumsq_i += sqi
            global_min_i = min(global_min_i, mini)
            global_max_i = max(global_max_i, maxi)

    # compute final stats
    mean_v = sum_v / total_n
    var_v  = sumsq_v / total_n - mean_v**2
    std_v  = math.sqrt(var_v) if var_v > 0 else 0.0

    mean_i = sum_i / total_n
    var_i  = sumsq_i / total_n - mean_i**2
    std_i  = math.sqrt(var_i) if var_i > 0 else 0.0

    # prepare output dict
    stats = {
        'root_dir': args.root_dir,
        'rel_dirs': args.rel_dirs,
        'total_points': total_n,
        'V': {
            'mean': mean_v,
            'std': std_v,
            'min': global_min_v,
            'max': global_max_v
        },
        'intensity': {
            'mean': mean_i,
            'std': std_i,
            'min': global_min_i,
            'max': global_max_i
        }
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = os.path.join(script_dir, "norm_stats")
    os.makedirs(save_folder, exist_ok=True)
    root_name = os.path.basename(os.path.normpath(args.root_dir))
    out_json = os.path.join(save_folder, f"{root_name}.json")
    with open(out_json, 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"Wrote statistics to '{out_json}'")

if __name__ == '__main__':
    main()
