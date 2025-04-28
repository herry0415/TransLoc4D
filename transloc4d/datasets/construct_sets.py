import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import pickle
from tqdm import tqdm

from .base_datasets import TrainingTuple


def load_dataframe(base_path, dataset_name, subset, data_type):
    """
    Load the DataFrame from a specific CSV file in the dataset structure.
    """
    csv_filename = "gps.csv"
    folder_name = "pointclouds"
    csv_path = os.path.join(base_path, dataset_name,
                            subset, data_type, csv_filename)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    point_folder = os.path.join(
        base_path, dataset_name, subset, data_type, folder_name)
    ext = sorted(os.listdir(point_folder))[0].split(".")[-1]
    df = pd.read_csv(csv_path)
    df["file"] = df["timestamp"].apply(
        lambda x: os.path.join(
            point_folder, f"{x}.{ext}"
        )
    )
    return df


def output_to_file(output, base_path, filename, print_func=print):
    """
    Saves the given data to a pickle file.
    """
    file_path = os.path.join(base_path, filename)
    with open(file_path, "wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print_func(f"Completed: {filename}")


def construct_training_dict(df_query, df_db, base_path, filename,
                            ind_pos_r=10, ind_nonneg_r=50,
                            angle_threshold_deg=None, print_func=print):
    df_combined = pd.concat([df_query, df_db]).reset_index(drop=True)

    # Precompute heading in degrees if available
    yaw_exists = 'yaw' in df_combined.columns
    if yaw_exists:
        yaw_arr = np.degrees(df_combined['yaw'].values)

    # Build combined KDTree and query neighbors in parallel
    tree_combined = KDTree(df_combined[['northing', 'easting']])
    ind_positive = tree_combined.query_radius(
        df_combined[['northing', 'easting']], r=ind_pos_r)
    ind_nonneg = tree_combined.query_radius(
        df_combined[['northing', 'easting']], r=ind_nonneg_r)

    queries = {}
    for anchor_ndx in tqdm(range(len(ind_positive)), desc="Processing"):
        positives = ind_positive[anchor_ndx]
        non_negatives = ind_nonneg[anchor_ndx]
        if len(positives) == 0 or len(non_negatives) == 0:
            continue

        # Filter positives by heading difference using vectorized mask
        if yaw_exists and angle_threshold_deg is not None:
            anchor_yaw = yaw_arr[anchor_ndx]
            mask = np.abs(yaw_arr[positives] - anchor_yaw) <= angle_threshold_deg
            positives = positives[mask]
            if positives.size == 0:
                continue

        anchor_pos = np.array(
            df_combined.iloc[anchor_ndx][['northing', 'easting']])
        timestamp = df_combined.iloc[anchor_ndx]["timestamp"]
        scan_filename = df_combined.iloc[anchor_ndx]["file"]
        assert os.path.isfile(
            scan_filename), 'point cloud file {} is found'.format(scan_filename)

        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)

        queries[anchor_ndx] = TrainingTuple(
            id=anchor_ndx,
            timestamp=timestamp,
            rel_scan_filepath=scan_filename,
            positives=positives,
            non_negatives=non_negatives,
            position=anchor_pos
        )

    with open(os.path.join(base_path, filename), 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print_func(f"Completed: {filename}")


def build_test_pickles(base_path, datasets_name, subsets,
                       valDist=25, angle_threshold_deg=30,
                       print_func=print):
    """
    Build and saves the query and database sets pickles for each subset in the dataset
    """
    assert callable(print_func), "print_func should be a callable function"

    for dataset_name in datasets_name:
        for subset in subsets:
            df_query = load_dataframe(
                base_path, dataset_name, subset, "query"
            )
            df_database = load_dataframe(
                base_path, dataset_name, subset, "database"
            )

            # Precompute heading in degrees if available
            yaw_exists = 'yaw' in df_query.columns and 'yaw' in df_database.columns
            if yaw_exists:
                df_query['yaw_deg'] = np.degrees(df_query['yaw'].values)
                df_database['yaw_deg'] = np.degrees(df_database['yaw'].values)

            # Build KDTree for the database
            tree_database = KDTree(df_database[["northing", "easting"]])

            database_sets = []
            test_sets = []

            for index, row in tqdm(
                df_query.iterrows(),
                total=df_query.shape[0],
                desc=f"Processing {subset} subset",
            ):
                coor = np.array([[row["northing"], row["easting"]]])
                # Radius search for positives in parallel
                indices = tree_database.query_radius(
                    coor, r=valDist)[0].tolist()

                # Filter positives by heading difference
                if yaw_exists:
                    anchor_yaw = row['yaw_deg']
                    yaw_arr_db = df_database['yaw_deg'].values
                    mask = np.abs(yaw_arr_db[indices] - anchor_yaw) <= angle_threshold_deg
                    indices = list(np.array(indices)[mask])

                test = {
                    "file": row["file"],
                    "northing": row["northing"],
                    "easting": row["easting"],
                    "positives": indices,
                }
                test_sets.append(test)

            for _, row in df_database.iterrows():
                database_sets.append({
                    "file": row["file"],
                    "northing": row["northing"],
                    "easting": row["easting"],
                })

            suffix = f"_{angle_threshold_deg}" if yaw_exists else ""

            output_to_file(
                database_sets,
                base_path,
                f"{dataset_name}_{subset}_evaluation_database_{valDist}{suffix}.pickle",
                print_func=print_func
            )
            output_to_file(
                test_sets,
                base_path,
                f"{dataset_name}_{subset}_evaluation_query_{valDist}{suffix}.pickle",
                print_func=print_func
            )


def build_training_pickle(base_path, dataset_name,
                          ind_pos_r=10, ind_nonneg_r=50,
                          angle_threshold_deg=30,
                          print_func=print):
    """
    Build the training pickle file for the dataset.
    """
    assert callable(print_func), "print_func should be a callable function"

    df_train_query = load_dataframe(base_path, dataset_name, "train", "query")
    df_train_db = load_dataframe(base_path, dataset_name, "train", "database")

    yaw_exists = 'yaw' in df_train_query.columns and 'yaw' in df_train_db.columns
    suffix = f"_{angle_threshold_deg}" if yaw_exists else ""

    construct_training_dict(
        df_train_query, df_train_db, base_path,
        f"train_queries_{dataset_name}_pos{ind_pos_r}_nonneg{ind_nonneg_r}{suffix}.pickle",
        ind_pos_r=ind_pos_r, ind_nonneg_r=ind_nonneg_r,
        angle_threshold_deg=angle_threshold_deg,
        print_func=print_func
    )
