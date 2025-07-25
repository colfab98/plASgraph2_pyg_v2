# In scripts/train.py

import argparse
import os
import shutil
import yaml
import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split

# Import modules from our library
from plasgraph.data import Dataset_Pytorch
from plasgraph.config import config as Config
from plasgraph.engine import train_final_model

from contextlib import contextmanager
from accelerate import Accelerator





def main():
    """
    Main function to train a final k-fold ensemble model.

    This script orchestrates the final model training process after
    hyperparameter optimization has been completed. It performs the following steps:
    1. Loads a base configuration file and updates it with the best
       hyperparameters found during a previous optimization stage.

    3. Loads and processes the training data into a graph format.
    4. Sets up k-fold cross-validation splits from the labeled training data.
    5. Calls the `train_final_model` engine, which iteratively trains a model for
       each fold and saves it to disk.
    6. Saves the final, consolidated configuration file for later use in
       evaluation and prediction.
    """

    parser = argparse.ArgumentParser(description="Train a final plASgraph2 model.")
    parser.add_argument("config_file", help="Base YAML configuration file")
    parser.add_argument("best_params_file", help="YAML file with best HPO parameters")
    parser.add_argument("train_file_list", help="CSV file listing training samples")
    parser.add_argument("file_prefix", help="Common prefix for all filenames")
    parser.add_argument("model_output_dir", help="Output folder for final model and logs")
    parser.add_argument("--data_cache_dir", required=True, help="Directory to store/load the processed graph data.")
    parser.add_argument("--training_mode", choices=['k-fold', 'single-fold'], default='k-fold', help="Choose between k-fold ensemble or single-fold training.")
    args = parser.parse_args()

    # load base config and update it with the best hyperparameters from HPO
    parameters = Config(args.config_file)
    with open(args.best_params_file, 'r') as f:
        best_params = yaml.safe_load(f)
    parameters._params.update(best_params)

    if isinstance(parameters['features'], str):
        parameters._params['features'] = tuple(parameters['features'].split(','))


    # directory for logs related to the final model training
    log_dir = os.path.join(args.model_output_dir, "final_training_logs")
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    accelerator = Accelerator()

    # load and process the training data
    accelerator.print("✅ Loading data...")
    all_graphs = Dataset_Pytorch(
        root=args.data_cache_dir,
        file_prefix=args.file_prefix,
        train_file_list=args.train_file_list,
        parameters=parameters,
        accelerator=accelerator 
    )

    # extract the graph data object, the NetworkX graph, and the node list
    data = all_graphs[0]
    G = all_graphs.G
    node_list = all_graphs.node_list

    print(f"✅ Using model architecture: {parameters['model_type']}")
    
    
    all_sample_ids = np.array(sorted(list(set(G.nodes[node_id]["sample"] for node_id in node_list))))
    print(f"✅ Found {len(all_sample_ids)} unique samples for splitting.")

    # Get indices of all nodes that have a label
    labeled_indices = np.array([i for i, node_id in enumerate(node_list) if G.nodes[node_id]["text_label"] != "unlabeled"])

    if args.training_mode == 'k-fold':
        print(f"✅ Setting up {parameters['k_folds']}-fold cross-validation based on samples.")
        kf = KFold(n_splits=parameters["k_folds"], shuffle=True, random_state=parameters["random_seed"])
        # Create splits based on the indices of the sample IDs array
        splits = list(kf.split(all_sample_ids))
    else: # single-fold
        print("✅ Setting up single-fold training with an 80/20 train/validation split based on samples.")
        # Create a single 80/20 split of sample IDs
        train_s_idx, val_s_idx = train_test_split(
            np.arange(len(all_sample_ids)), # Split the indices of the sample_ids array
            test_size=0.2,
            random_state=parameters["random_seed"],
            shuffle=True
        )
        # The train_final_model function expects a list of splits
        splits = [(train_s_idx, val_s_idx)]


    # train the K-fold ensemble
    train_final_model(
        accelerator, parameters, data, splits, all_sample_ids, labeled_indices, log_dir, G, node_list
    )

    # save the final base configuration file for future predictions
    base_params_path = os.path.join(args.model_output_dir, "base_model_config.yaml")
    parameters.write_yaml(base_params_path)

    ensemble_dir = os.path.join(log_dir, "ensemble_models")
    print(f"\n✅ Training complete. Ensemble models and their thresholds saved in: {ensemble_dir}")
    print(f"✅ Base config saved to: {base_params_path}")


if __name__ == "__main__":
    main()