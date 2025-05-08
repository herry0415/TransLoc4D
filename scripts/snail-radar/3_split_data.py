#!/usr/bin/env python3
"""
This script reorganizes the snail dataset into train, validation, and test splits.
The snail dataset is assumed to have the following folder structure for each sequence:

    {dataset_root}/{place}/{sequence_id}_preprocessed/
        ├── pointclouds/    (contains *.npy files)
        └── gps.csv         (CSV file with header: timestamp,northing,easting)

 1. Train and Validation (train, val):
    Data from various places are merged into single folders under a fixed dataset name.
    The merged output folders for train and val are created as:
         {output_dataset_root}/snail/train/query/
         {output_dataset_root}/snail/train/database/
         {output_dataset_root}/snail/val/query/
         {output_dataset_root}/snail/val/database/

 2. Test:
    Data for test are organized on a per-place basis. For each place, separate merged
    folders are created:
         {output_dataset_root}/{place}/test/query/
         {output_dataset_root}/{place}/test/database/

For each merged output folder, the script creates:
  - A 'pointclouds' subfolder containing all copied *.npy files (original filenames are preserved).
  - A 'gps.csv' file: a concatenation of all gps.csv files from the merged sequences.
  - An 'origin_log.txt' file logging the source (place and sequence) of each copied file.

Usage:
------
  python 3_split_data.py [--dataset_root /path/to/source]
                               [--output_dataset_root /path/to/output]
                               [--clean]

  --dataset_root:        Path to the root of the source dataset.
                         (Default: "/datasets/snail-radar")
  --output_dataset_root: Path where the reorganized dataset will be saved.
                         (Default: "/datasets/snail-radar_tl4d")
  --clean:               If specified, delete original source folders after copying.
"""

import os
import shutil
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

DATASET_NAME = "snail"

# Mapping structure: (see inline documentation above)
mapping = {
    "train": {
        "query": {
            "sl": ["20231105_aft_4"],
            "ss": ["20231105_aft_5"],
            "if": ["20231208_4"],
        },
        "database": {
            "sl": ["20231105_2"],
            "ss": ["20231109_4"],
            "if": ["20240116_5"],
            "81r": ["20240123_2"]
        }
    },
    "val": {
        "query": {
            "iaf": ["20231201_3"],
            "if": ["20240115_3"]
        },
        "database": {
            "iaf": ["20231201_2"],
            "if": ["20231213_4"]
        }
    },
    "test": {
        "bc": {
            "query": ["20230921_2", "20231007_4", "20231105_6", "20231105_aft_2"],
            "database": ["20230920_1"]
        },
        "sl": {
            "query": ["20230921_3", "20230921_5", "20231007_2", "20231019_1", "20231105_3", "20231109_3"],
            "database": ["20230920_2"]
        },
        "ss": {
            "query": ["20231019_2", "20231105_4", "20231105_5"],
            "database": ["20230921_4"]
        },
        "if": {
            "query": ["20240116_eve_5", "20240123_3"],
            "database": ["20231213_5"]
        },
        "iaf": {
            "query": ["20231213_2", "20231213_3", "20240113_3", "20240116_eve_4"],
            "database": ["20231208_5"]
        },
        "iaef": {
            "query": ["20240115_2", "20240116_4"],
            "database": ["20240113_5"]
        },
        "st": {
            "query": ["20231213_1", "20240113_1"],
            "database": ["20231208_1"]
        },
        "81r": {
            "query": ["20240116_eve_3"],
            "database": ["20240116_2"]
        }
    }
}


def merge_sequences(split, sequences, dataset_root, merge_output_dir, clean_original, dataset_suffix="preprocessed"):
    """
    For a given list of sequences, copy the pointcloud files concurrently and merge
    the GPS CSV files into a single file in the destination folder.
    If clean_original is True, deletes the source folder after copying.
    """
    pointclouds_dest = os.path.join(merge_output_dir, "pointclouds")
    os.makedirs(pointclouds_dest, exist_ok=True)

    gps_dest_file = os.path.join(merge_output_dir, "gps.csv")
    log_file = os.path.join(merge_output_dir, "origin_log.txt")
    gps_header_written = False

    # A lock for writing log messages to ensure thread safety.
    log_lock = threading.Lock()

    with open(gps_dest_file, 'w') as gps_out, open(log_file, 'w') as log_out:
        for seq_info in tqdm(sequences, desc=f"Merging sequences for {split}", unit="seq", position=1, leave=False):
            place = seq_info["place"]
            sequence_id = seq_info["sequence_id"]
            source_dir = os.path.join(dataset_root, place, f"{sequence_id}_{dataset_suffix}")
            if not os.path.exists(source_dir):
                tqdm.write(f"Warning: Source directory '{source_dir}' does not exist. Skipping {sequence_id}.")
                continue

            # Process pointclouds folder using concurrent copying.
            source_pointclouds = os.path.join(source_dir, "pointclouds")
            if not os.path.exists(source_pointclouds):
                tqdm.write(f"Warning: Pointclouds folder '{source_pointclouds}' not found for {sequence_id}.")
            else:
                # Use os.scandir for faster file listing.
                files = [entry for entry in os.scandir(source_pointclouds) if entry.is_file()]
                with ThreadPoolExecutor(max_workers=8) as executor:
                    future_to_file = {
                        executor.submit(shutil.copy2, entry.path, os.path.join(pointclouds_dest, entry.name)): entry.name
                        for entry in files
                    }
                    for future in tqdm(as_completed(future_to_file), total=len(future_to_file),
                                       desc=f"Copying files for {place}-{sequence_id}", leave=False, unit="file", position=2):
                        filename = future_to_file[future]
                        try:
                            future.result()
                            with log_lock:
                                log_out.write(f"{filename}: from {place}/{sequence_id}_{dataset_suffix}/pointclouds/{filename}\n")
                        except Exception as e:
                            tqdm.write(f"Error copying '{os.path.join(source_pointclouds, filename)}': {e}")

            # Process the gps.csv file.
            source_gps = os.path.join(source_dir, "gps.csv")
            if not os.path.exists(source_gps):
                tqdm.write(f"Warning: GPS file '{source_gps}' not found for {sequence_id}.")
            else:
                try:
                    with open(source_gps, 'r') as gps_in:
                        lines = gps_in.readlines()
                        if not lines:
                            tqdm.write(f"Warning: GPS file '{source_gps}' is empty.")
                        else:
                            if not gps_header_written:
                                gps_out.write(lines[0])
                                gps_header_written = True
                                for line in lines[1:]:
                                    gps_out.write(line)
                            else:
                                for line in lines[1:]:
                                    gps_out.write(line)
                            with log_lock:
                                log_out.write(f"Merged GPS from {place}/{sequence_id}_{dataset_suffix}/gps.csv\n")
                            tqdm.write(f"Merged GPS data from '{source_gps}'.")
                except Exception as e:
                    tqdm.write(f"Error processing GPS file '{source_gps}': {e}")

            # If --clean is specified, delete the entire source folder after copying.
            if clean_original:
                try:
                    shutil.rmtree(source_dir)
                    tqdm.write(f"Cleaned up '{source_dir}'.")
                except Exception as e:
                    tqdm.write(f"Error cleaning '{source_dir}': {e}")

    tqdm.write(f"Merging for {split} complete. Merged data is in '{merge_output_dir}'.")


def process_merged_split(split_name, category, mapping_split, dataset_root, output_dataset_root, clean_original, dataset_suffix="preprocessed"):
    """
    Flattens the mapping (per place) for train/val splits and calls merge_sequences.
    """
    sequences = []
    for place, seq_list in mapping_split.get(category, {}).items():
        for seq in seq_list:
            sequences.append({"place": place, "sequence_id": seq})
    if sequences:
        merge_output_dir = os.path.join(output_dataset_root, DATASET_NAME, split_name, category)
        os.makedirs(merge_output_dir, exist_ok=True)
        tqdm.write(f"Processing {split_name} {category} with {len(sequences)} sequences into '{merge_output_dir}'.")
        merge_sequences(f"{split_name}-{category}", sequences, dataset_root, merge_output_dir, clean_original, dataset_suffix=dataset_suffix)
    else:
        tqdm.write(f"No sequences for {split_name} {category}.")


def process_test_split(mapping_test, dataset_root, output_dataset_root, clean_original, dataset_suffix="preprocessed"):
    """
    Processes the test split (organized per place). For each place and category,
    flattens the list of sequence IDs and calls merge_sequences.
    The merged output for test is stored in:
         {output_dataset_root}/{place}/test/{category}/
    """
    for place, cat_dict in tqdm(list(mapping_test.items()), desc="Processing test places", unit="place"):
        for category, seq_list in tqdm(cat_dict.items(), desc=f"Processing test categories for {place}", unit="cat", leave=False):
            sequences = [{"place": place, "sequence_id": seq} for seq in seq_list]
            merge_output_dir = os.path.join(output_dataset_root, place, "test", category)
            os.makedirs(merge_output_dir, exist_ok=True)
            if sequences:
                tqdm.write(f"Processing test {category} for place '{place}' with {len(sequences)} sequences into '{merge_output_dir}'.")
                merge_sequences(f"test-{place}-{category}", sequences, dataset_root, merge_output_dir, clean_original, dataset_suffix=dataset_suffix)
            else:
                tqdm.write(f"No test sequences for {place} {category}.")


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize snail dataset into train, val, and test splits."
    )
    parser.add_argument('--dataset_root', type=str, default="/datasets/snail-radar",
                        help="Path to the root of the source dataset.")
    parser.add_argument('--output_dataset_root', type=str, default="/datasets/snail-radar_tl4d",
                        help="Path where the reorganized dataset will be saved.")
    parser.add_argument('--clean', action='store_true',
                        help="If set, delete the original source folders after copying.")
    parser.add_argument("--add_suffix", type=str, default="",
                        help="Additional info to describe the dataset.")
    args = parser.parse_args()

    dataset_suffix = "preprocessed"
    if args.add_suffix:
        args.output_dataset_root = f"{args.output_dataset_root}_{args.add_suffix}"
        dataset_suffix = f"{dataset_suffix}_{args.add_suffix}"

    # Process all splits: train, val, and test.
    all_splits = ["train", "val", "test"]
    for split in tqdm(all_splits, desc="Processing splits (train, val, test)", unit="split", position=0, leave=False):
        if split in ["train", "val"]:
            mapping_split = mapping.get(split, {})
            for category in ["query", "database"]:
                process_merged_split(split, category, mapping_split, args.dataset_root, args.output_dataset_root, args.clean, dataset_suffix=dataset_suffix)
        elif split == "test":
            mapping_test = mapping.get("test", {})
            if mapping_test:
                process_test_split(mapping_test, args.dataset_root, args.output_dataset_root, args.clean, dataset_suffix=dataset_suffix)
            else:
                tqdm.write("No test mapping defined.")

    print("Dataset reorganization complete.")


if __name__ == '__main__':
    main()
