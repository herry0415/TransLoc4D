#!/usr/bin/env python
import subprocess
from os.path import join
import argparse
import json
import os

# Change the current working directory to the parent directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)
print(f"Changed working directory to: {parent_dir}")

def run_evaluation(task, gpu_id):
    datasets = task['datasets']
    weights_paths = task['weights_paths']
    posDist = task.get('posDist', 25)
    posHeading = task.get('posHeading', None)
    model_config = task.get('model_config', None)
    config = task.get('config', None)
    dataset_root = task.get('dataset_root', '/home/user/datasets')
    
    for dataset in datasets:
        database_pickle = join(dataset_root, f'{dataset}_evaluation_database_{posDist}_{posHeading}.pickle' if posHeading else f'{dataset}_evaluation_database_{posDist}.pickle')
        query_pickle = join(dataset_root, f'{dataset}_evaluation_query_{posDist}_{posHeading}.pickle' if posHeading else f'{dataset}_evaluation_query_{posDist}.pickle')
        
        for weights_path in weights_paths:
            command = [
                'python', 'scripts/eval.py',
                '--database_pickle', database_pickle,
                '--query_pickle', query_pickle,
                '--weights', weights_path,
                '--gpu_id', gpu_id
            ]

            if model_config is not None:
                command += ['--model_config', model_config]
            if config is not None:
                command += ['--config', config]

            
            print(f"Running: {' '.join(command)}")
            subprocess.run(command)

def main():
    parser = argparse.ArgumentParser(description="Run evaluation with given GPU ID and configuration.")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use for the evaluation.")
    parser.add_argument("--config_path", type=str, default="config/test/snail.json", help="Path to the configuration JSON file.")
    
    args = parser.parse_args()
    gpu_id = args.gpu_id
    
    with open(args.config_path, 'r') as config_file:
        tasks = json.load(config_file)
    
    for task in tasks:
        run_evaluation(task, gpu_id)

if __name__ == "__main__":
    main()