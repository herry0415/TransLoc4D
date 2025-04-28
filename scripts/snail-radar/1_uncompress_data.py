#!/usr/bin/env python3
"""
This script processes a dataset grouped by place_setup. Each key in the dictionary represents a group
(e.g. "81r") and its value is a list of date_run strings (formatted as "YYYYMMDD/Run" or "YYYYMMDD_suffix/Run").

For each date_run:
    - The date and run parts are extracted.
    - A folder name is created as {date_part}_{run_part}.
    - It assumes that a source ZIP file exists at:
          {SOURCE_DIR}/{date_part}/data{run_part}.zip
    - A single ref_trajs.zip lives at:
          {SOURCE_DIR}/ref_trajs.zip
      containing folders like YYYYMMDD/data<run>/...
    - The main ZIP is filtered by KEPT_FOLDERS; ref_trajs.zip is **not** filtered.
    - All relevant files are extracted into:
          {DEST_DIR}/{place_setup}/{date_part}_{run_part}
"""

import os
import re
import zipfile
import argparse
from tqdm import tqdm
import shutil  # for copying from ref_trajs.zip

# Specify regular expression pattern(s) to keep files from within each ZIP file.
# If the list is empty, all files in the ZIP are extracted.
KEPT_FOLDERS = [
    "eagleg7/enhanced",
    "x36d",
    "zed2i/left",
]


# -- DICTIONARY-BASED INPUT DATA --
data_dict = {
    "bc": [
        "20230920/1",
        "20230921/2",
        "20231007/4",
        "20231105/6",
        "20231105_aft/2"
    ],
    "sl": [
        "20230920/2",
        "20230921/3",
        "20230921/5",
        "20231007/2",
        "20231019/1",
        "20231105/2",
        "20231105/3",
        "20231105_aft/4",
        "20231109/3"
    ],
    "ss": [
        "20230921/4",
        "20231019/2",
        "20231105/4",
        "20231105/5",
        "20231105_aft/5",
        "20231109/4"
    ],
    "if": [
        "20231208/4",
        "20231213/4",
        "20231213/5",
        "20240115/3",
        "20240116/5",
        "20240116_eve/5",
        "20240123/3"
    ],
    "iaf": [
        "20231201/2",
        "20231201/3",
        "20231208/5",
        "20231213/2",
        "20231213/3",
        "20240113/2",
        "20240113/3",
        "20240116_eve/4"
    ],
    "iaef": [
        "20240113/5",
        "20240115/2",
        "20240116/4"
    ],
    "st": [
        "20231208/1",
        "20231213/1",
        "20240113/1"
    ],
    "81r": [
        "20240116/2",
        "20240116_eve/3",
        "20240123/2"
    ]
}


def process_data(source_dir, dest_dir):
    """
    Process each group in the data dictionary. For each date_run string in each group,
    build the folder structure and attempt to unzip the corresponding source ZIP file.
    Only files whose names match the regular expression patterns specified in KEPT_FOLDERS
    are extracted. If KEPT_FOLDERS is empty, all files are extracted.

    Additionally, if a global ref_trajs.zip exists under source_dir, any entries under
    YYYYMMDD/data<run>/ within it will also be extracted to the same destination.
    Parameters:
        source_dir (str): Base directory for source ZIP files.
        dest_dir (str): Base directory for extracted files.
    """
    # Open global ref_trajs.zip once (if it exists)
    ref_zip_path = os.path.join(source_dir, "ref_trajs.zip")
    if os.path.isfile(ref_zip_path):
        ref_zip = zipfile.ZipFile(ref_zip_path, "r")
        ref_members = [m for m in ref_zip.infolist() if not m.is_dir()]
    else:
        ref_zip = None
        print(f"Note: no ref_trajs.zip found at {ref_zip_path}")

    for group, date_runs in data_dict.items():
        group_dest_dir = os.path.join(dest_dir, group)
        os.makedirs(group_dest_dir, exist_ok=True)

        for date_run in date_runs:
            # Validate the date_run format.
            if "/" not in date_run:
                print(f"WARNING: Invalid date_run format '{date_run}' in group '{group}'. Skipping.")
                continue

            date_part, run_part = date_run.split("/", 1)
            folder_name = f"{date_part}_{run_part}"
            src_zip = os.path.join(source_dir, date_part, f"data{run_part}.zip")
            dest_folder = os.path.join(group_dest_dir, folder_name)

            if os.path.exists(src_zip):
                print(f"Processing group '{group}':")
                print(f"  Date:        {date_part}")
                print(f"  Run:         {run_part}")
                print(f"  Source ZIP:  {src_zip}")
                print(f"  Destination: {dest_folder}")
                try:
                    os.makedirs(dest_folder, exist_ok=True)
                    with zipfile.ZipFile(src_zip, "r") as zip_ref:
                        members = zip_ref.infolist()

                        # If KEPT_FOLDERS contains regex patterns, filter based on those.
                        if KEPT_FOLDERS:
                            filtered_members = [
                                m for m in members
                                if any(re.search(pattern, m.filename) for pattern in KEPT_FOLDERS)
                            ]
                        else:
                            filtered_members = members

                        for member in tqdm(filtered_members, desc="Extracting", unit="file", total=len(filtered_members)):
                            zip_ref.extract(member, dest_folder)
                    print("  Unzipped successfully.")

                    # --- New: extract all from global ref_trajs.zip ---
                    if ref_zip:
                        prefix = f"{date_part}/data{run_part}/"
                        trajs = [m for m in ref_members if m.filename.startswith(prefix)]
                        if trajs:
                            print("  Extracting trajectories from ref_trajs.zip...")
                            for traj in tqdm(trajs, desc="Trajectories", unit="file", total=len(trajs)):
                                rel_path = traj.filename[len(prefix):]
                                out_path = os.path.join(dest_folder, rel_path)
                                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                                with ref_zip.open(traj) as src_f, open(out_path, "wb") as dst_f:
                                    shutil.copyfileobj(src_f, dst_f)
                            print("  Trajectories extracted successfully.")
                        else:
                            print(f"  (no trajectories found for prefix '{prefix}')")

                except Exception as e:
                    print(f"  ERROR: Failed to unzip {src_zip}: {e}")
            else:
                print(f"  WARNING: Source file {src_zip} does not exist.")

            print("-" * 50)

    if ref_zip:
        ref_zip.close()


def main():
    parser = argparse.ArgumentParser(
        description="Process dataset ZIP files based on grouped place_setup and date_run values."
    )
    parser.add_argument(
        "-s",
        "--source-dir",
        default="/datasets/snail-radar",
        help="Base directory for source ZIP files."
    )
    parser.add_argument(
        "-d",
        "--dest-dir",
        default="/datasets/snail-radar",
        help="Base destination directory for extracted files."
    )

    args = parser.parse_args()
    process_data(args.source_dir, args.dest_dir)


if __name__ == "__main__":
    main()
