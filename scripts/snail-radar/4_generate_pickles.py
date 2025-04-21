import os
import argparse
from tqdm import tqdm

from transloc4d.datasets import build_training_pickle, build_test_pickles

train_config = {
    "dataset_name": "snail",
    "ind_pos_r": 9,
    "ind_nonneg_r": 18
}

test_configs = [
    {
        "datasets_name": ["snail"],
        "subsets": ["val"],
        "valDist": 9
    },
    {
        "datasets_name": ["bc", "sl", "ss", "if", "iaf", "iaef", "st", "81r"],
        "subsets": ["test"],
        "valDist": 9
    }
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate training and test sets.")
    parser.add_argument("--base_path", default="/datasets/snail-radar_tl4d",
                        help="Base path for the dataset")
    args = parser.parse_args()

    assert os.path.exists(
        args.base_path), f"Cannot access dataset root folder: {args.base_path}"

    # Generate training pickle
    tqdm.write("Building training pickle...")
    build_training_pickle(args.base_path, train_config["dataset_name"],
                          ind_pos_r=train_config["ind_pos_r"], ind_nonneg_r=train_config["ind_nonneg_r"])

    for config in tqdm(test_configs, desc="Building test pickles...", position=0, leave=False):
        datasets_name = config["datasets_name"]
        subsets = config["subsets"]
        valDist = config.get("valDist", 25)

        # Construct and save the query and database sets
        build_test_pickles(args.base_path, datasets_name, subsets, valDist=valDist, print_func=tqdm.write)
