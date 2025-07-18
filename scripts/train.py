# In scripts/train.py

import argparse
import os
import yaml
import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split

# Import modules from our library
from plasgraph.data import Dataset_Pytorch
from plasgraph.config import config as Config
from plasgraph.engine import train_final_model

# 💡 Define a dummy class to satisfy the Dataset_Pytorch constructor
class DummyAccelerator:
    """A mock class that mimics the accelerator.print() method for single-GPU execution."""
    def print(self, message):
        print(message)

def main():
    """
    Main function to train a final k-fold ensemble model.

    This script orchestrates the final model training process after
    hyperparameter optimization has been completed. It performs the following steps:
    1. Loads a base configuration file and updates it with the best
       hyperparameters found during a previous optimization stage.
    2. Initializes a `DummyAccelerator` to allow the use of the distributed-aware
       `Dataset_Pytorch` class in a single-device environment.
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

    # directory for logs related to the final model training
    log_dir = os.path.join(args.model_output_dir, "final_training_logs")
    os.makedirs(log_dir, exist_ok=True)

    # dummy accelerator to pass to the data loader
    dummy_accelerator = DummyAccelerator()

    # load and process the training data
    dummy_accelerator.print("✅ Loading data...")
    all_graphs = Dataset_Pytorch(
        root=args.data_cache_dir,
        file_prefix=args.file_prefix,
        train_file_list=args.train_file_list,
        parameters=parameters,
        accelerator=dummy_accelerator 
    )

    # extract the graph data object, the NetworkX graph, and the node list
    data = all_graphs[0]
    G = all_graphs.G
    node_list = all_graphs.node_list

    print(f"✅ Using model architecture: {parameters['model_type']}")
    labeled_indices = np.array([i for i, node_id in enumerate(node_list) if G.nodes[node_id]["text_label"] != "unlabeled"])

    if args.training_mode == 'k-fold':
        print(f"✅ Setting up {parameters['k_folds']}-fold cross-validation.")
        kf = KFold(n_splits=parameters["k_folds"], shuffle=True, random_state=parameters["random_seed"])
        splits = list(kf.split(labeled_indices))
    else:
        print("✅ Setting up single-fold training with an 80/20 train/validation split.")
        # Create a single 80/20 split
        train_idx, val_idx = train_test_split(
            np.arange(len(labeled_indices)), # Split the indices of the labeled_indices array
            test_size=0.2,
            random_state=parameters["random_seed"],
            shuffle=True
        )
        # The train_final_model function expects a list of splits
        splits = [(train_idx, val_idx)]

    # train the K-fold ensemble
    train_final_model(
        parameters, data, splits, labeled_indices, log_dir, G, node_list
    )

    # save the final base configuration file for future predictions
    base_params_path = os.path.join(args.model_output_dir, "base_model_config.yaml")
    parameters.write_yaml(base_params_path)

    ensemble_dir = os.path.join(log_dir, "ensemble_models")
    print(f"\n✅ Training complete. Ensemble models and their thresholds saved in: {ensemble_dir}")
    print(f"✅ Base config saved to: {base_params_path}")


if __name__ == "__main__":
    main()