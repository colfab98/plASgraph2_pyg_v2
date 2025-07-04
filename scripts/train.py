# In scripts/train.py

import argparse
import os
import yaml
import numpy as np
import torch
from sklearn.model_selection import KFold

# Import modules from our library
from plasgraph.data import Dataset_Pytorch
from plasgraph.config import config as Config
from plasgraph.engine import train_final_model # <-- Import the new engine function

# REPLACE the main function in scripts/train.py with this new version.

from plasgraph.engine import train_final_model # <-- Import the new engine functions

from accelerate import Accelerator


def main():
    parser = argparse.ArgumentParser(description="Train a final plASgraph2 model.")
    parser.add_argument("config_file", help="Base YAML configuration file")
    parser.add_argument("best_params_file", help="YAML file with best HPO parameters")
    parser.add_argument("train_file_list", help="CSV file listing training samples")
    parser.add_argument("file_prefix", help="Common prefix for all filenames")
    parser.add_argument("model_output_dir", help="Output folder for final model and logs")
    parser.add_argument("--data_cache_dir", required=True, help="Directory to store/load the processed graph data.")
    args = parser.parse_args()

    accelerator = Accelerator()


    # 1. Load base config and update with best HPO params
    parameters = Config(args.config_file)
    parameters.config_file_path = args.config_file # Store path for re-loading
    with open(args.best_params_file, 'r') as f:
        best_params = yaml.safe_load(f)
    parameters._params.update(best_params)
    
    log_dir = os.path.join(args.model_output_dir, "final_training_logs")
    os.makedirs(log_dir, exist_ok=True)

    # 2. Load Data
    all_graphs = Dataset_Pytorch(
        root=args.data_cache_dir,
        file_prefix=args.file_prefix,
        train_file_list=args.train_file_list,
        parameters=parameters
    )
    data = all_graphs[0]
    G = all_graphs.G
    node_list = all_graphs.node_list
    
    # 3. Setup Device and Cross-Validation Splits
    accelerator.print(f"Using device: {accelerator.device}")
    accelerator.print(f"✅ Using model architecture: {parameters['model_type']}")


    labeled_indices = np.array([i for i, node_id in enumerate(node_list) if G.nodes[node_id]["text_label"] != "unlabeled"])
    kf = KFold(n_splits=parameters["k_folds"], shuffle=True, random_state=parameters["random_seed"])
    splits = list(kf.split(labeled_indices))

    # 4. Train the K-fold ensemble and determine fold-specific thresholds
    train_final_model(
        accelerator, parameters, data, splits, labeled_indices, log_dir, G, node_list
    )

    # 5. Save the base configuration file for future predictions
    if accelerator.is_main_process:
        # Save the base parameters used for the run (without fold-specific thresholds)
        base_params_path = os.path.join(args.model_output_dir, "base_model_config.yaml")
        parameters.write_yaml(base_params_path)
        
        ensemble_dir = os.path.join(log_dir, "ensemble_models")
        accelerator.print(f"\n✅ Training complete. Ensemble models and their thresholds saved in: {ensemble_dir}")
        accelerator.print(f"✅ Base config saved to: {base_params_path}")


if __name__ == "__main__":
    main()