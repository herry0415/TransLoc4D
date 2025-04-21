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


def construct_training_dict(df_query, df_db, base_path, filename, ind_pos_r=10, ind_nonneg_r=50, print_func=print):
    df_combined = pd.concat([df_query, df_db]).reset_index(drop=True)
    tree_db2q = KDTree(df_query[['northing', 'easting']])
    ind_positive_db2q = tree_db2q.query_radius(
        df_combined[['northing', 'easting']], r=ind_pos_r)

    tree_q2db = KDTree(df_db[['northing', 'easting']])
    ind_positive_q2db = tree_q2db.query_radius(
        df_query[['northing', 'easting']], r=ind_pos_r)

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

        if anchor_ndx < len(df_query):
            positives_for_train = ind_positive_q2db[anchor_ndx]
            positives_for_train = [i+len(df_query)
                                   for i in positives_for_train]
        else:
            positives_for_train = ind_positive_db2q[anchor_ndx]

        anchor_pos = np.array(
            df_combined.iloc[anchor_ndx][['northing', 'easting']])
        timestamp = df_combined.iloc[anchor_ndx]["timestamp"]
        scan_filename = df_combined.iloc[anchor_ndx]["file"]
        assert os.path.isfile(
            scan_filename), 'point cloud file {} is found'.format(scan_filename)

        # Sort ascending order
        positives_for_train = np.sort(positives_for_train)
        non_negatives = np.sort(non_negatives)

        queries[anchor_ndx] = TrainingTuple(
            id=anchor_ndx,
            timestamp=timestamp,
            rel_scan_filepath=scan_filename,
            positives=positives_for_train,
            non_negatives=non_negatives,
            position=anchor_pos
        )

    with open(os.path.join(base_path, filename), 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print_func(f"Completed: {filename}")


def build_test_pickles(base_path, datasets_name, subsets, valDist=25, print_func=print):
    """
    Build and saves the query and database sets pickles for each subset in the dataset
    """
    assert callable(print_func), "print_func should be a callable function"

    for dataset_name in datasets_name:
        for subset in subsets:
            df_query = load_dataframe(
                base_path, dataset_name, subset, f"query"
            )
            df_database = load_dataframe(
                base_path, dataset_name, subset, f"database"
            )

            # Build KDTree for the database
            tree_database = KDTree(df_database[["northing", "easting"]])

            # Initialize containers for the structured output
            database_sets = []
            test_sets = []

            # Process queries to find positive matches within the database
            for index, row in tqdm(
                df_query.iterrows(),
                total=df_query.shape[0],
                desc=f"Processing {subset} subset",
            ):
                coor = np.array([[row["northing"], row["easting"]]])
                # Radius search for positives
                indices = tree_database.query_radius(
                    coor, r=valDist)[0].tolist()

                # Assuming the same structuring as your first script, adjust as necessary
                test = {
                    "file": row["file"],
                    "northing": row["northing"],
                    "easting": row["easting"],
                    "positives": indices,  # Indices of the positive matches
                }
                test_sets.append(test)

            # Process the database entries similarly if needed
            for _, row in df_database.iterrows():
                database = {
                    "file": row["file"],
                    "northing": row["northing"],
                    "easting": row["easting"],
                }
                database_sets.append(database)

            # Output to files, following naming convention similar to the first script
            output_to_file(
                database_sets,
                base_path,
                f"{dataset_name}_{subset}_evaluation_database_{valDist}.pickle",
                print_func=print_func
            )
            output_to_file(
                test_sets,
                base_path,
                f"{dataset_name}_{subset}_evaluation_query_{valDist}.pickle",
                print_func=print_func
            )


def build_training_pickle(base_path, dataset_name, ind_pos_r=10, ind_nonneg_r=50, print_func=print):
    """
    Build the training pickle file for the dataset.
    """
    assert callable(print_func), "print_func should be a callable function"

    df_train_query = load_dataframe(base_path, dataset_name, "train", f"query")
    df_train_db = load_dataframe(base_path, dataset_name, "train", f"database")

    construct_training_dict(df_train_query, df_train_db, base_path,
                            f"train_queries_{dataset_name}.pickle", ind_pos_r=ind_pos_r, ind_nonneg_r=ind_nonneg_r, print_func=print_func)
