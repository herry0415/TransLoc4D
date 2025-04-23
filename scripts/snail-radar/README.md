# Data preprocessing with [SNAIL-RADAR](https://snail-radar.github.io/)

To use [SNAIL-RADAR](https://snail-radar.github.io/) in this repository, please follow below steps.

## Download
Download the original dataset from [SNAIL-RADAR](https://snail-radar.github.io/).

## Extract data
Run below command to extract data to specific folder, where `-s` is the base directory for source `zip` files, and `-d` is the destination directory for extracted files. 


    python scripts/snail-radar/1_uncompress_data.py -s {source_folder}  -d {destination_folder}


This script only extract specific folders for 4d pointclouds localization, therefore it only extract pointclouds from `eagleg7/enhanced`, GPS read from `x36d`, and images from `zed2i/left`. If you want to extract data from additional sensors, you can change variable `KEPT_FOLDERS` in `1_uncompress_data.py`.

## Preprocess data
Then run following script to preprocess raw 4d pointclouds data.
`base_folder` is the base folder where the extracted data locate.

    python scripts/snail-radar/2_preprocess_data.py --base_dir {base_folder}

For each `.pcd` file, we extract the needed fields from the raw data and estimate ego velocity to remove points from dynamic objects. We also normalizes the xyz coordinates and saved the processed pointclouds as a `.npy` file.

## Data Visualization (Optional)
You can visualize the data to view dynamic points removal perforamnce in notebook `scripts/snail-radar/viz_pcs.ipynb`.


## Split data
Then you can re-organize the processed data to train/val/test splits.
Here we follow the splits implemented in [SNAIL-RADAR](https://snail-radar.github.io/).
Each route is considered one seperate test set, while for train/val set, different routes are merged.

    python scripts/snail-radar/3_split_data.py --dataset_root {based_folder} --output_dataset_root {destination_folder}


## Generate pickles files
Finally, generate `.pickle` files for offline positives mining. 

    python scripts/snail-radar/4_generate_pickles.py --base_path {base_folder}