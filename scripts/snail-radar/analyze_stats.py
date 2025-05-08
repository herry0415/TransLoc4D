#!/usr/bin/env python3
"""
Analyze point cloud statistics all over the Snail-Radar dataset.:
    1. --max_range argument to skip points beyond a given range (default 120).
    2. Detailed JSON logging of per-folder, per-file, and overall before/after statistics,
    including files with more out-of-range points under 'norm_stats' folder.
    3. Per-folder listing of files_more_outside and overall count.
"""
import os
import sys
import argparse
import logging
import math
import json
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

data_dict = {
    "bc": ["20230920/1", "20230921/2", "20231007/4", "20231105/6", "20231105_aft/2"],
    "sl": ["20230920/2", "20230921/3", "20230921/5", "20231007/2", "20231019/1",
           "20231105/2", "20231105/3", "20231105_aft/4", "20231109/3"],
    "ss": ["20230921/4", "20231019/2", "20231105/4", "20231105/5", "20231105_aft/5", "20231109/4"],
    "if": ["20231208/4", "20231213/4", "20231213/5", "20240115/3", "20240116/5", "20240116_eve/5", "20240123/3"],
    "iaf": ["20231201/2", "20231201/3", "20231208/5", "20231213/2", "20231213/3",
            "20240113/2", "20240113/3", "20240116_eve/4"],
    "iaef": ["20240113/5", "20240115/2", "20240116/4"],
    "st": ["20231208/1", "20231213/1", "20240113/1"],
    "81r": ["20240116/2", "20240116_eve/3", "20240123/2"]
}

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

def load_pcd(filename):
    """
    Optimized binary-only PCD loader:
    - Reads FIELDS line to get field order.
    - Reads entire DATA block as float32.
    Returns (fields: List[str], points: np.ndarray of shape (N, F)).
    """
    with open(filename, 'rb') as f:
        fields = []
        # parse header
        while True:
            line = f.readline().decode('utf-8')
            if not line:
                break
            if line.startswith("FIELDS"):
                fields = line.split()[1:]
            if "DATA" in line:
                break
        if not fields:
            raise ValueError(f"No FIELDS in {filename!r}")
        raw = f.read()

    num_floats = len(raw) // 4
    arr = np.frombuffer(raw, dtype=np.float32, count=num_floats)
    F = len(fields)
    if num_floats % F != 0:
        raise ValueError(f"Corrupted DATA block in {filename!r}")
    pts = arr.reshape(-1, F)
    return fields, pts

# Worker globals (set once per process)
IX = IY = IZ = None
MAX_RANGE_SQ = None

def init_worker(ix, iy, iz, max_range):
    global IX, IY, IZ, MAX_RANGE_SQ
    IX, IY, IZ = ix, iy, iz
    MAX_RANGE_SQ = float(max_range) ** 2

def process_file(path):
    """
    Load one PCD and return:
      path, sums & sumsqs & counts for all and for in-range points, plus outside count.
    """
    try:
        _, pts = load_pcd(path)
    except Exception:
        return None

    coords = pts[:, (IX, IY, IZ)].astype(np.float64)
    n_all = coords.shape[0]
    if n_all == 0:
        return None

    # squared distances for speed
    dist2 = np.sum(coords * coords, axis=1)
    in_mask = dist2 <= MAX_RANGE_SQ
    n_in = int(in_mask.sum())
    n_out = n_all - n_in

    s_all = coords.sum(axis=0)
    sq_all = (coords * coords).sum(axis=0)

    if n_in > 0:
        coords_in = coords[in_mask]
        s_in = coords_in.sum(axis=0)
        sq_in = (coords_in * coords_in).sum(axis=0)
    else:
        s_in = np.zeros(3, dtype=np.float64)
        sq_in = np.zeros(3, dtype=np.float64)

    return (path,
            s_all, sq_all, n_all,
            s_in,  sq_in,  n_in,
            n_out)

def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Compute & log global mean/std of all point clouds with range filtering"
    )
    parser.add_argument(
        "--base_dir", type=str, default="/datasets/snail-radar",
        help="Root directory containing Snail-Radar place folders"
    )
    parser.add_argument(
        "--radar_rel", type=str, default="eagleg7/enhanced",
        help="Relative path from each place folder to the .pcd files"
    )
    parser.add_argument(
        "--processes", type=int, default=cpu_count(),
        help="Number of worker processes"
    )
    parser.add_argument(
        "--max_range", type=float, default=120.0,
        help="Max distance (in meters) to include points; others are skipped"
    )
    parser.add_argument(
        "--save_folder", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "norm_stats"),
        help="JSON file to write detailed before/after statistics"
    )
    args = parser.parse_args()

    os.makedirs(args.save_folder, exist_ok=True)
    out_json = os.path.join(args.save_folder, f"maxRange{int(args.max_range)}.json")

    # gather all PCD paths
    all_files = []
    for place, folders in data_dict.items():
        for folder in folders:
            folderdir = folder.replace("/", "_")
            ds_root = os.path.join(args.base_dir, place, folderdir)
            pcd_dir = os.path.join(ds_root, args.radar_rel)
            if not os.path.isdir(pcd_dir):
                logging.warning(f"Missing directory: {pcd_dir}")
                continue
            for fn in sorted(os.listdir(pcd_dir)):
                if fn.lower().endswith(".pcd"):
                    all_files.append(os.path.join(pcd_dir, fn))

    if not all_files:
        logging.error("No .pcd files found; exiting.")
        sys.exit(1)

    # read header once to get x,y,z indices
    fields, _ = load_pcd(all_files[0])
    try:
        ix, iy, iz = fields.index('x'), fields.index('y'), fields.index('z')
    except ValueError:
        logging.error("Fields 'x','y','z' not found in PCD header.")
        sys.exit(1)

    # prepare multiprocessing pool
    pool = Pool(
        processes=args.processes,
        initializer=init_worker,
        initargs=(ix, iy, iz, args.max_range)
    )

    # overall accumulators
    sum_all_sums = [[], [], []]
    sum_all_sqs  = [[], [], []]
    total_n_all = 0

    sum_in_sums = [[], [], []]
    sum_in_sqs  = [[], [], []]
    total_n_in = 0

    # per-folder stats container
    folder_stats = {}
    # list of all files with more outside than inside
    all_files_more_outside = []

    # process files in parallel
    for res in tqdm(pool.imap_unordered(process_file, all_files, chunksize=20),
                    total=len(all_files), desc="Aggregating stats"):
        if res is None:
            continue
        (path,
         s_all, sq_all, n_all,
         s_in,  sq_in,  n_in,
         n_out) = res

        # update overall before-filter
        for i in range(3):
            sum_all_sums[i].append(float(s_all[i]))
            sum_all_sqs[i].append(float(sq_all[i]))
        total_n_all += n_all

        # update overall after-filter
        for i in range(3):
            sum_in_sums[i].append(float(s_in[i]))
            sum_in_sqs[i].append(float(sq_in[i]))
        total_n_in += n_in

        # determine folder key
        rel = os.path.relpath(path, args.base_dir)
        parts = rel.split(os.sep)
        folder_key = f"{parts[0]}/{parts[1]}"  # e.g. "bc/20230920_1"

        # init folder_stats entry if needed
        if folder_key not in folder_stats:
            folder_stats[folder_key] = {
                'sum_all_sums': [[], [], []],
                'sum_all_sqs':  [[], [], []],
                'n_all': 0,
                'sum_in_sums': [[], [], []],
                'sum_in_sqs':  [[], [], []],
                'n_in': 0,
                'files_more_outside': []
            }
        fs = folder_stats[folder_key]

        # accumulate per-folder before-filter
        for i in range(3):
            fs['sum_all_sums'][i].append(float(s_all[i]))
            fs['sum_all_sqs'][i].append(float(sq_all[i]))
        fs['n_all'] += n_all

        # accumulate per-folder after-filter
        for i in range(3):
            fs['sum_in_sums'][i].append(float(s_in[i]))
            fs['sum_in_sqs'][i].append(float(sq_in[i]))
        fs['n_in'] += n_in

        # track files_more_outside
        if n_out > n_in:
            fs['files_more_outside'].append(path)
            all_files_more_outside.append(path)

    pool.close()
    pool.join()

    # compute overall stats
    def compute_stats(sums, sqs, total_n):
        sum_vals = [math.fsum(lst) for lst in sums]
        sq_vals  = [math.fsum(lst) for lst in sqs]
        mean = [sum_vals[i]/total_n for i in range(3)]
        var  = [sq_vals[i]/total_n - mean[i]**2 for i in range(3)]
        std  = [math.sqrt(v) for v in var]
        return mean, std

    mean_all, std_all = compute_stats(sum_all_sums, sum_all_sqs, total_n_all)
    mean_in,  std_in  = compute_stats(sum_in_sums,  sum_in_sqs,  total_n_in)

    # build per_folder JSON
    per_folder = {}
    for key, fs in folder_stats.items():
        m_b, s_b = compute_stats(fs['sum_all_sums'], fs['sum_all_sqs'], fs['n_all'])
        if fs['n_in'] > 0:
            m_a, s_a = compute_stats(fs['sum_in_sums'], fs['sum_in_sqs'], fs['n_in'])
        else:
            m_a, s_a = [0,0,0], [0,0,0]
        per_folder[key] = {
            'before_filter': {'mean': m_b, 'std': s_b},
            'after_filter':  {'mean': m_a, 'std': s_a},
            'num_files_more_outside': len(fs['files_more_outside']),
            'files_more_outside': fs['files_more_outside']
        }

    # overall logs
    logging.info(f"[overall] before filter mean={mean_all} std={std_all}")
    logging.info(f"[overall] after  filter mean={mean_in} std={std_in}")
    logging.warning(f"[overall] total files_more_outside count: {len(all_files_more_outside)}")


    # write JSON log
    stats = {
        'overall': {
            'before_filter': {'mean': mean_all, 'std': std_all},
            'after_filter':  {'mean': mean_in,  'std': std_in},
            'num_files_more_outside': len(all_files_more_outside)
        },
        'per_folder': per_folder,
    }
    with open(out_json, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Wrote detailed stats to '{out_json}'.")

if __name__ == "__main__":
    main()
