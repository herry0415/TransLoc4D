import os

from transloc4d.datasets import build_test_pickles



if __name__ == "__main__":
    tasks = [
        {
            "base_path": "/home/user/datasets",
            "datasets_name": ["ntu-rsvi"],
            "subsets": ["val"],
            "valDist": 25,
        },
        {
            "base_path": "/home/user/datasets",
            "datasets_name": ["nyl-night-rsvi", "nyl-rain-rsvi", "src-night-rsvi"],
            "subsets": ["test"],
            "valDist": 25,
        },
        {
            "base_path": "/home/user/datasets",
            "datasets_name": ["sjtu-rsvi"],
            "subsets": ["test_a", "test_b"],
            "valDist": 25,
        }
    ]

    for task in tasks:
        base_path = task["base_path"]
        datasets_name = task["datasets_name"]
        subsets = task["subsets"]
        valDist = task.get("valDist", 25)

        # Ensure the dataset's base path exists
        assert os.path.exists(
            base_path
        ), f"Cannot access dataset root folder: {base_path}"

        # Construct and save the query and database sets
        build_test_pickles(
            base_path, datasets_name, subsets, valDist=valDist
        )
