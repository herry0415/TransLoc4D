#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import logging


script_dir = os.path.dirname(os.path.abspath(__file__))
preprocess_script = os.path.join(script_dir, "pcs_preprocess.py")

if not os.path.isfile(preprocess_script):
    print(f"pcs_preprocess.py not found in the current directory: {script_dir}")
    sys.exit(1)

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

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def main():
    setup_logging()
    
    # Parse command line arguments for the base directory
    parser = argparse.ArgumentParser(
        description="Process dataset folders with pcs_preprocess.py using a specified base directory."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/datasets/snail-radar",
        help="Base directory containing the dataset folders."
    )
    args = parser.parse_args()
    base_dir = args.base_dir

    # Verify that the base directory exists
    if not os.path.isdir(base_dir):
        logging.error(f"The base directory does not exist: {base_dir}")
        sys.exit(1)
    

    for place, folder_list in data_dict.items():
        for folder in folder_list:
            processed_folder = folder.replace("/", "_")
            datasets_root = os.path.join(base_dir, place, processed_folder)
            logging.info(f"Processing folder: {datasets_root}")

            if not os.path.isdir(datasets_root):
                logging.warning(f"Directory does not exist: {datasets_root}. Creating directory.")
                try:
                    os.makedirs(datasets_root, exist_ok=True)
                except Exception as e:
                    logging.error(f"Failed to create directory {datasets_root}: {e}")
                    continue

            command = [sys.executable, preprocess_script, "--dataset_root", datasets_root]
            try:
                result = subprocess.run(
                    command,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                logging.info(f"Success processing {datasets_root}:\n{result.stdout}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Error processing {datasets_root} (exit code {e.returncode}):\n{e.stderr}")
            except Exception as e:
                logging.error(f"Unexpected error processing {datasets_root}: {e}")

    logging.info("All processing tasks have been completed.")

if __name__ == "__main__":
    main()
