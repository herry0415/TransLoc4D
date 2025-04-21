#!/usr/bin/env python3
import argparse
import os
import numpy as np
import csv
from os.path import join
from tqdm import tqdm
import utm
import bisect
from multiprocessing import Pool
import shutil

from transloc4d.datasets import estimate_ego_vel


def load_pcd(filename):
    """
    Optimized version: Read a PCD file and return the list of fields and a numpy array of points.
    Instead of per-field unpacking, we read the binary block in one call using np.frombuffer.
    """
    with open(filename, 'rb') as file:
        fields = []
        # Read header until "DATA" line is encountered.
        while True:
            line = file.readline().decode('utf-8')
            if not line:
                break
            if line.startswith("FIELDS"):
                fields = line.split()[1:]
            if "DATA" in line:
                break
        if not fields:
            raise ValueError(f"No FIELDS found in file {filename}")
        binary_data = file.read()
        num_floats = len(binary_data) // 4
        data = np.frombuffer(binary_data, dtype=np.float32, count=num_floats)
        if num_floats % len(fields) != 0:
            raise ValueError(
                "Binary data size is not a multiple of the number of fields.")
        points = data.reshape(-1, len(fields))
    return fields, points


def parse_gps_file(gps_file_path):
    """
    Parse the GPS file and return a sorted list of tuples: (time, latitude, longitude).
    """
    gps_entries = []
    with open(gps_file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                gps_time = float(parts[0])
                lat = float(parts[1])
                lon = float(parts[2])
                gps_entries.append((gps_time, lat, lon))
            except ValueError:
                continue
    gps_entries.sort(key=lambda x: x[0])
    return gps_entries


def find_closest_gps(gps_entries, target_time):
    """
    Find the GPS entry with the timestamp closest to target_time using binary search.
    """
    times = [entry[0] for entry in gps_entries]
    pos = bisect.bisect_left(times, target_time)
    if pos == 0:
        return gps_entries[0]
    if pos == len(times):
        return gps_entries[-1]
    before = gps_entries[pos - 1]
    after = gps_entries[pos]
    if abs(before[0] - target_time) <= abs(after[0] - target_time):
        return before
    else:
        return after


def find_closest_image(sorted_image_entries, target_time):
    """
    Find the image entry with the timestamp closest to target_time using binary search.
    sorted_image_entries is a sorted list of tuples: (timestamp, filename).
    """
    image_times = [entry[0] for entry in sorted_image_entries]
    pos = bisect.bisect_left(image_times, target_time)
    if pos == 0:
        return sorted_image_entries[0][1]
    if pos == len(image_times):
        return sorted_image_entries[-1][1]
    before = sorted_image_entries[pos - 1]
    after = sorted_image_entries[pos]
    if abs(before[0] - target_time) <= abs(after[0] - target_time):
        return before[1]
    else:
        return after[1]


def process_timestamp(timestamp_str):
    """
    Convert a timestamp string format to an integer string
    by multiplying the float value by 1e6 and truncating.
    """
    timestamp_float = float(timestamp_str)
    timestamp_int = int(timestamp_float * 1e6)
    return str(timestamp_int)


def normalize_xyz(points):
    """
    Normalize the xyz coordinates of the point cloud.
    Parameters:
        points (np.ndarray): Array of shape (N, 3) representing the x, y, z coordinates.
    Returns:
        np.ndarray: Normalized points.
    """
    centroid = np.mean(points, axis=0)
    points = points - centroid
    furthest_distance = np.max(np.linalg.norm(points, axis=1))
    if furthest_distance > 0:
        points = points / furthest_distance
    return points


def resample_pointcloud(points, target_size=4096):
    """
    Resample the point cloud to have exactly target_size points.
    Parameters:
        points (np.ndarray): Point cloud array of shape (N, D).
        target_size (int): Desired number of points.
    Returns:
        np.ndarray: Resampled point cloud of shape (target_size, D).
    """
    current_size = points.shape[0]
    if current_size > target_size:
        indices = np.random.choice(current_size, target_size, replace=False)
        return points[indices, :]
    elif current_size < target_size:
        indices = np.random.choice(current_size, target_size, replace=True)
        return points[indices, :]
    return points


def process_single_pcd(task):
    """
    Process a single PCD file:
      - Reads and vectorizes the binary point cloud data.
      - Extracts the needed fields.
      - Optionally estimates ego velocity.
      - Normalizes the xyz coordinates and resamples the point cloud to a fixed number of points.
      - Saves the processed point cloud as a .npy file.
      - If image generation is enabled, finds the closest image using find_closest_image
        and copies it to a new folder with the new timestamp.
      - Returns a tuple (processed_timestamp, northing, easting) for GPS synchronization.
    """
    (pcd_file, radar_folder, save_pointclouds_folder, gps_entries,
     generate_original, generate_images, image_folder, sorted_image_entries,
     target_points) = task

    msg = ""
    raw_timestamp = pcd_file[:-4]  # remove ".pcd"
    processed_timestamp = process_timestamp(raw_timestamp)
    pcd_file_path = join(radar_folder, pcd_file)
    try:
        fields, points = load_pcd(pcd_file_path)
    except Exception as e:
        msg = f"Error loading {pcd_file}: {e}"
        return None, msg
    if points.size == 0:
        msg = f"Warning: No data found in file {pcd_file}"
        return None, msg
    try:
        idx_x = fields.index('x')
        idx_y = fields.index('y')
        idx_z = fields.index('z')
        idx_doppler = fields.index('Doppler')
        idx_power = fields.index('Power')
    except ValueError as e:
        msg = f"Error processing {pcd_file}: {e}"
        return None, msg
    radar_scan = points[:, [idx_x, idx_y, idx_z,
                            idx_doppler, idx_power]].astype(np.float32)
    flag, _, processed_scan = estimate_ego_vel(radar_scan)
    if not flag:
        msg = f"Warning: Ego velocity estimation failed for file {pcd_file}"
        processed_scan = radar_scan

    processed_scan[:, :3] = normalize_xyz(processed_scan[:, :3])
    processed_scan = resample_pointcloud(
        processed_scan, target_size=target_points)

    # Save processed point cloud as .npy file.
    npy_save_path = join(save_pointclouds_folder, f"{processed_timestamp}.npy")
    np.save(npy_save_path, processed_scan)
    if generate_original:
        npy_save_path_original = join(
            save_pointclouds_folder, f"{processed_timestamp}_org.npy")
        np.save(npy_save_path_original, radar_scan)
    # GPS synchronization.
    target_time = float(raw_timestamp)
    closest_gps = find_closest_gps(gps_entries, target_time)
    lat, lon = closest_gps[1], closest_gps[2]
    utm_result = utm.from_latlon(lat, lon)
    easting, northing = utm_result[0], utm_result[1]

    # If image generation is enabled, find and copy the closest image.
    if generate_images and image_folder and sorted_image_entries:
        try:
            pcd_time = float(raw_timestamp)
            closest_image = find_closest_image(sorted_image_entries, pcd_time)
            src_image_path = join(image_folder, closest_image)
            dst_image_folder = join(os.path.dirname(
                save_pointclouds_folder), "images")
            os.makedirs(dst_image_folder, exist_ok=True)
            img_ext = src_image_path.split('.')[-1]
            dst_image_path = join(
                dst_image_folder, f"{processed_timestamp}.{img_ext}")
            shutil.copy(src_image_path, dst_image_path)
        except Exception as e:
            msg += f"\nWarning: Failed to process image for {pcd_file}: {e}"

    return (processed_timestamp, northing, easting), msg


def get_args():
    parser = argparse.ArgumentParser(
        description="Preprocess radar dataset, synchronize GPS data, and process point clouds with normalization and resampling."
    )
    parser.add_argument("--dataset_root", type=str, default="/datasets/snail-radar/20240116",
                        help="Root directory of the dataset")
    parser.add_argument("--radar_rel_path", type=str, default="eagleg7/enhanced",
                        help="Relative path to radar .pcd files")
    parser.add_argument("--gps_rel_path", type=str, default="x36d/gnss_ins.txt",
                        help="Relative path to GPS file")
    parser.add_argument("--img_rel_path", type=str, default="zed2i/left",
                        help="Relative path to image camera image files")
    parser.add_argument("--save_folder", type=str, default=None,
                        help="Folder to save preprocessed data")
    parser.add_argument("-o", "--generate_original", action="store_true",
                        help="Generate original point clouds without outlier removal at the same time.")
    parser.add_argument("-i", "--generate_images", action="store_true",
                        help="Copy images from raw data to new folder.")
    parser.add_argument("--target_points", type=int, default=4096,
                        help="The desired number of points in the processed point cloud (for up/down sampling).")
    args = parser.parse_args()
    if args.save_folder is None:
        args.save_folder = f"{args.dataset_root}_preprocessed"
    return args


def main():
    args = get_args()
    radar_folder = join(args.dataset_root, args.radar_rel_path)
    gps_file_path = join(args.dataset_root, args.gps_rel_path)
    os.makedirs(args.save_folder, exist_ok=True)
    save_pointclouds_folder = join(args.save_folder, "pointclouds")
    os.makedirs(save_pointclouds_folder, exist_ok=True)
    # Load and sort GPS data.
    gps_entries = parse_gps_file(gps_file_path)

    # Build tasks list.
    pcd_files = [f for f in os.listdir(radar_folder) if f.endswith('.pcd')]

    # If image generation is enabled, prepare the image folder and a sorted list of images.
    if args.generate_images:
        image_folder = join(args.dataset_root, args.img_rel_path)
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(
            '.jpg') or f.lower().endswith('.png')]
        sorted_image_entries = []
        for f in image_files:
            try:
                ts = float(f[:-4])
                sorted_image_entries.append((ts, f))
            except ValueError:
                continue
        sorted_image_entries.sort(key=lambda x: x[0])
    else:
        image_folder = None
        sorted_image_entries = None

    tasks = []
    for pcd_file in pcd_files:
        tasks.append((pcd_file, radar_folder, save_pointclouds_folder,
                      gps_entries, args.generate_original, args.generate_images,
                      image_folder, sorted_image_entries, args.target_points))

    results = []
    with Pool() as pool:
        for res, msg in tqdm(pool.imap_unordered(process_single_pcd, tasks, chunksize=10),
                             total=len(tasks), desc="Processing PCD files"):
            if msg:
                tqdm.write(msg)
            if res is not None:
                results.append(res)

    results.sort(key=lambda x: int(x[0]))
    # Save GPS synchronization data.
    gps_csv_path = join(args.save_folder, "gps.csv")
    with open(gps_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "northing", "easting"])
        writer.writerows(results)


if __name__ == "__main__":
    main()
