import os

from transloc4d.datasets import build_training_pickle


if __name__ == '__main__':

    base_path = "/home/user/datasets"
    dataset_name = "ntu-rsvi"

    assert os.path.exists(
        base_path), f"Cannot access dataset root folder: {base_path}"

    build_training_pickle(base_path, dataset_name,
                          ind_pos_r=10, ind_nonneg_r=50)
