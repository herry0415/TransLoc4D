#!/usr/bin/env python3
"""
Analyze point cloud statistics all over the Snail-Radar dataset.:
    1. --max_range argument to skip points beyond a given range (default 120).
    2. Detailed JSON logging of per-folder, per-file, and overall before/after statistics,
    including files with more out-of-range points under 'norm_stats' folder.
    3. Per-folder listing of files_more_outside and overall count.
    4. Mean, std, min, and max for Doppler and Power, both before and after filtering.
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
IX = IY = IZ = IDOP = IPOW = None
MAX_RANGE_SQ = None

def init_worker(ix, iy, iz, idop, ipow, max_range):
    global IX, IY, IZ, IDOP, IPOW, MAX_RANGE_SQ
    IX, IY, IZ = ix, iy, iz
    IDOP, IPOW = idop, ipow
    MAX_RANGE_SQ = float(max_range) ** 2

def process_file(path):
    """
    Load one PCD and return:
      path,
      sums & sumsqs & counts for coords before/after,
      doppler sums, sumsqs, mins, maxs, counts before/after,
      power sums, sumsqs, mins, maxs, counts before/after,
      plus outside count.
    """
    try:
        _, pts = load_pcd(path)
    except Exception:
        return None

    coords = pts[:, (IX, IY, IZ)].astype(np.float64)
    dop   = pts[:, IDOP].astype(np.float64)
    pwr   = pts[:, IPOW].astype(np.float64)
    n_all = coords.shape[0]
    if n_all == 0:
        return None

    # squared distances for speed
    dist2 = np.sum(coords * coords, axis=1)
    in_mask = dist2 <= MAX_RANGE_SQ
    n_in = int(in_mask.sum())
    n_out = n_all - n_in

    # coordinate sums
    s_all = coords.sum(axis=0)
    sq_all = (coords * coords).sum(axis=0)
    if n_in > 0:
        coords_in = coords[in_mask]
        s_in = coords_in.sum(axis=0)
        sq_in = (coords_in * coords_in).sum(axis=0)
    else:
        s_in = np.zeros(3, dtype=np.float64)
        sq_in = np.zeros(3, dtype=np.float64)

    # doppler stats
    dop_s_all = float(dop.sum())
    dop_sq_all = float((dop * dop).sum())
    dop_min_all = float(dop.min())
    dop_max_all = float(dop.max())
    if n_in > 0:
        dop_in = dop[in_mask]
        dop_s_in = float(dop_in.sum())
        dop_sq_in = float((dop_in * dop_in).sum())
        dop_min_in = float(dop_in.min())
        dop_max_in = float(dop_in.max())
    else:
        dop_s_in = dop_sq_in = dop_min_in = dop_max_in = 0.0

    # power stats
    pwr_s_all = float(pwr.sum())
    pwr_sq_all = float((pwr * pwr).sum())
    pwr_min_all = float(pwr.min())
    pwr_max_all = float(pwr.max())
    if n_in > 0:
        pwr_in = pwr[in_mask]
        pwr_s_in = float(pwr_in.sum())
        pwr_sq_in = float((pwr_in * pwr_in).sum())
        pwr_min_in = float(pwr_in.min())
        pwr_max_in = float(pwr_in.max())
    else:
        pwr_s_in = pwr_sq_in = pwr_min_in = pwr_max_in = 0.0

    return (
        path,
        s_all, sq_all, n_all,
        s_in,  sq_in,  n_in,
        n_out,
        dop_s_all,    dop_sq_all,    dop_min_all,    dop_max_all,
        dop_s_in,     dop_sq_in,     dop_min_in,     dop_max_in,
        pwr_s_all,    pwr_sq_all,    pwr_min_all,    pwr_max_all,
        pwr_s_in,     pwr_sq_in,     pwr_min_in,     pwr_max_in
    )

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

    # read header once to get indices
    fields, _ = load_pcd(all_files[0])
    try:
        ix, iy, iz = fields.index('x'), fields.index('y'), fields.index('z')
        idop, ipow = fields.index('Doppler'), fields.index('Power')
    except ValueError:
        logging.error("Fields 'x','y','z','Doppler','Power' not found in PCD header.")
        sys.exit(1)

    # prepare multiprocessing pool
    pool = Pool(
        processes=args.processes,
        initializer=init_worker,
        initargs=(ix, iy, iz, idop, ipow, args.max_range)
    )

    # overall accumulators for coordinates
    sum_all_sums = [[], [], []]
    sum_all_sqs  = [[], [], []]
    total_n_all = 0

    sum_in_sums = [[], [], []]
    sum_in_sqs  = [[], [], []]
    total_n_in = 0

    # overall accumulators for Doppler & Power
    dop_all_sums = []
    dop_all_sqs  = []
    dop_all_mins = []
    dop_all_maxs = []
    dop_in_sums  = []
    dop_in_sqs   = []
    dop_in_mins  = []
    dop_in_maxs  = []

    pwr_all_sums = []
    pwr_all_sqs  = []
    pwr_all_mins = []
    pwr_all_maxs = []
    pwr_in_sums  = []
    pwr_in_sqs   = []
    pwr_in_mins  = []
    pwr_in_maxs  = []

    # per-folder stats container
    folder_stats = {}
    # list of all files with more outside than inside
    all_files_more_outside = []

    # helper to compute scalar stats
    def compute_scalar_stats(sums, sqs, total_n, mins, maxs):
        if total_n == 0:
            return 0.0, 0.0, None, None
        total_sum = math.fsum(sums)
        total_sqs = math.fsum(sqs)
        mean = total_sum / total_n
        var  = total_sqs / total_n - mean**2
        std  = math.sqrt(var) if var > 0 else 0.0
        minimum = min(mins) if mins else None
        maximum = max(maxs) if maxs else None
        return mean, std, minimum, maximum

    # process files in parallel
    for res in tqdm(pool.imap_unordered(process_file, all_files, chunksize=20),
                    total=len(all_files), desc="Aggregating stats"):
        if res is None:
            continue
        (
            path,
            s_all, sq_all, n_all,
            s_in, sq_in, n_in,
            n_out,
            dop_s_all, dop_sq_all, dop_min_all, dop_max_all,
            dop_s_in,  dop_sq_in,  dop_min_in,  dop_max_in,
            pwr_s_all, pwr_sq_all, pwr_min_all, pwr_max_all,
            pwr_s_in,  pwr_sq_in,  pwr_min_in,  pwr_max_in
        ) = res

        # update overall before-filter (coords)
        for i in range(3):
            sum_all_sums[i].append(float(s_all[i]))
            sum_all_sqs[i].append(float(sq_all[i]))
        total_n_all += n_all

        # update overall after-filter (coords)
        for i in range(3):
            sum_in_sums[i].append(float(s_in[i]))
            sum_in_sqs[i].append(float(sq_in[i]))
        total_n_in += n_in

        # update overall before-filter (Doppler & Power)
        dop_all_sums.append(dop_s_all)
        dop_all_sqs.append(dop_sq_all)
        dop_all_mins.append(dop_min_all)
        dop_all_maxs.append(dop_max_all)
        pwr_all_sums.append(pwr_s_all)
        pwr_all_sqs.append(pwr_sq_all)
        pwr_all_mins.append(pwr_min_all)
        pwr_all_maxs.append(pwr_max_all)

        # update overall after-filter (Doppler & Power)
        dop_in_sums.append(dop_s_in)
        dop_in_sqs.append(dop_sq_in)
        dop_in_mins.append(dop_min_in)
        dop_in_maxs.append(dop_max_in)
        pwr_in_sums.append(pwr_s_in)
        pwr_in_sqs.append(pwr_sq_in)
        pwr_in_mins.append(pwr_min_in)
        pwr_in_maxs.append(pwr_max_in)

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
                'files_more_outside': [],
                'dop_all_sums': [], 'dop_all_sqs': [], 'dop_all_mins': [], 'dop_all_maxs': [],
                'dop_in_sums':  [], 'dop_in_sqs':  [], 'dop_in_mins':  [], 'dop_in_maxs':  [],
                'pwr_all_sums': [], 'pwr_all_sqs': [], 'pwr_all_mins': [], 'pwr_all_maxs': [],
                'pwr_in_sums':  [], 'pwr_in_sqs':  [], 'pwr_in_mins':  [], 'pwr_in_maxs':  []
            }
        fs = folder_stats[folder_key]

        # accumulate per-folder before-filter (coords)
        for i in range(3):
            fs['sum_all_sums'][i].append(float(s_all[i]))
            fs['sum_all_sqs'][i].append(float(sq_all[i]))
        fs['n_all'] += n_all

        # accumulate per-folder after-filter (coords)
        for i in range(3):
            fs['sum_in_sums'][i].append(float(s_in[i]))
            fs['sum_in_sqs'][i].append(float(sq_in[i]))
        fs['n_in'] += n_in

        # accumulate per-folder before-filter (Doppler & Power)
        fs['dop_all_sums'].append(dop_s_all)
        fs['dop_all_sqs'].append(dop_sq_all)
        fs['dop_all_mins'].append(dop_min_all)
        fs['dop_all_maxs'].append(dop_max_all)
        fs['pwr_all_sums'].append(pwr_s_all)
        fs['pwr_all_sqs'].append(pwr_sq_all)
        fs['pwr_all_mins'].append(pwr_min_all)
        fs['pwr_all_maxs'].append(pwr_max_all)

        # accumulate per-folder after-filter (Doppler & Power)
        fs['dop_in_sums'].append(dop_s_in)
        fs['dop_in_sqs'].append(dop_sq_in)
        fs['dop_in_mins'].append(dop_min_in)
        fs['dop_in_maxs'].append(dop_max_in)
        fs['pwr_in_sums'].append(pwr_s_in)
        fs['pwr_in_sqs'].append(pwr_sq_in)
        fs['pwr_in_mins'].append(pwr_min_in)
        fs['pwr_in_maxs'].append(pwr_max_in)

        # track files_more_outside
        if n_out > n_in:
            fs['files_more_outside'].append(path)
            all_files_more_outside.append(path)

    pool.close()
    pool.join()

    # compute overall stats for coords
    def compute_stats(sums, sqs, total_n):
        sum_vals = [math.fsum(lst) for lst in sums]
        sq_vals  = [math.fsum(lst) for lst in sqs]
        mean = [sum_vals[i]/total_n for i in range(3)]
        var  = [sq_vals[i]/total_n - mean[i]**2 for i in range(3)]
        std  = [math.sqrt(v) for v in var]
        return mean, std

    mean_all, std_all = compute_stats(sum_all_sums, sum_all_sqs, total_n_all)
    mean_in,  std_in  = compute_stats(sum_in_sums,  sum_in_sqs,  total_n_in)

    # compute overall stats for Doppler & Power
    dop_mean_all, dop_std_all, dop_min_all, dop_max_all = compute_scalar_stats(
        dop_all_sums, dop_all_sqs, total_n_all, dop_all_mins, dop_all_maxs)
    dop_mean_in,  dop_std_in,  dop_min_in,  dop_max_in  = compute_scalar_stats(
        dop_in_sums,  dop_in_sqs,  total_n_in,  dop_in_mins,  dop_in_maxs)
    pwr_mean_all, pwr_std_all, pwr_min_all, pwr_max_all = compute_scalar_stats(
        pwr_all_sums, pwr_all_sqs, total_n_all, pwr_all_mins, pwr_all_maxs)
    pwr_mean_in,  pwr_std_in,  pwr_min_in,  pwr_max_in  = compute_scalar_stats(
        pwr_in_sums,  pwr_in_sqs,  total_n_in,  pwr_in_mins,  pwr_in_maxs)

    # build per_folder JSON
    per_folder = {}
    for key, fs in folder_stats.items():
        m_b, s_b = compute_stats(fs['sum_all_sums'], fs['sum_all_sqs'], fs['n_all'])
        db_m, db_s, db_min, db_max = compute_scalar_stats(
            fs['dop_all_sums'], fs['dop_all_sqs'], fs['n_all'], fs['dop_all_mins'], fs['dop_all_maxs'])
        pb_m, pb_s, pb_min, pb_max = compute_scalar_stats(
            fs['pwr_all_sums'], fs['pwr_all_sqs'], fs['n_all'], fs['pwr_all_mins'], fs['pwr_all_maxs'])

        if fs['n_in'] > 0:
            m_a, s_a = compute_stats(fs['sum_in_sums'], fs['sum_in_sqs'], fs['n_in'])
            da_m, da_s, da_min, da_max = compute_scalar_stats(
                fs['dop_in_sums'], fs['dop_in_sqs'], fs['n_in'], fs['dop_in_mins'], fs['dop_in_maxs'])
            pa_m, pa_s, pa_min, pa_max = compute_scalar_stats(
                fs['pwr_in_sums'], fs['pwr_in_sqs'], fs['n_in'], fs['pwr_in_mins'], fs['pwr_in_maxs'])
        else:
            m_a, s_a = [0,0,0], [0,0,0]
            da_m, da_s, da_min, da_max = 0.0, 0.0, None, None
            pa_m, pa_s, pa_min, pa_max = 0.0, 0.0, None, None

        per_folder[key] = {
            'before_filter': {
                'mean': m_b,
                'std': s_b,
                'doppler': {'mean': db_m, 'std': db_s, 'min': db_min, 'max': db_max},
                'power':   {'mean': pb_m, 'std': pb_s, 'min': pb_min, 'max': pb_max},
            },
            'after_filter': {
                'mean': m_a,
                'std': s_a,
                'doppler': {'mean': da_m, 'std': da_s, 'min': da_min, 'max': da_max},
                'power':   {'mean': pa_m, 'std': pa_s, 'min': pa_min, 'max': pa_max},
            },
            'num_files_more_outside': len(fs['files_more_outside']),
            'files_more_outside': fs['files_more_outside']
        }

    # overall logs
    logging.info(f"[overall] before filter mean={mean_all} std={std_all}")
    logging.info(f"[overall] after  filter mean={mean_in} std={std_in}")
    logging.info(
        f"[overall] Doppler before mean={dop_mean_all} std={dop_std_all} "
        f"min={dop_min_all} max={dop_max_all}"
    )
    logging.info(
        f"[overall] Doppler after  mean={dop_mean_in} std={dop_std_in} "
        f"min={dop_min_in} max={dop_max_in}"
    )
    logging.info(
        f"[overall] Power before mean={pwr_mean_all} std={pwr_std_all} "
        f"min={pwr_min_all} max={pwr_max_all}"
    )
    logging.info(
        f"[overall] Power after  mean={pwr_mean_in} std={pwr_std_in} "
        f"min={pwr_min_in} max={pwr_max_in}"
    )
    logging.warning(f"[overall] total files_more_outside count: {len(all_files_more_outside)}")

    # write JSON log
    stats = {
        'overall': {
            'before_filter': {
                'mean': mean_all,
                'std': std_all,
                'doppler': {'mean': dop_mean_all, 'std': dop_std_all, 'min': dop_min_all, 'max': dop_max_all},
                'power':   {'mean': pwr_mean_all, 'std': pwr_std_all, 'min': pwr_min_all, 'max': pwr_max_all},
            },
            'after_filter': {
                'mean': mean_in,
                'std': std_in,
                'doppler': {'mean': dop_mean_in, 'std': dop_std_in, 'min': dop_min_in, 'max': dop_max_in},
                'power':   {'mean': pwr_mean_in, 'std': pwr_std_in, 'min': pwr_min_in, 'max': pwr_max_in},
            },
            'num_files_more_outside': len(all_files_more_outside)
        },
        'per_folder': per_folder
    }
    with open(out_json, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Wrote detailed stats to '{out_json}'.")
    
if __name__ == "__main__":
    main()
