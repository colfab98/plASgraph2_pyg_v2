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

from plasgraph.engine import tune_thresholds, train_final_model # <-- Import the new engine functions

def main():
    parser = argparse.ArgumentParser(description="Train a final plASgraph2 model.")
    parser.add_argument("config_file", help="Base YAML configuration file")
    parser.add_argument("best_params_file", help="YAML file with best HPO parameters")
    parser.add_argument("train_file_list", help="CSV file listing training samples")
    parser.add_argument("file_prefix", help="Common prefix for all filenames")
    parser.add_argument("model_output_dir", help="Output folder for final model and logs")
    parser.add_argument("--data_cache_dir", required=True, help="Directory to store/load the processed graph data.")
    args = parser.parse_args()

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
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print(f"Using device: {device}")
    data = data.to(device)

    labeled_indices = np.array([i for i, node_id in enumerate(node_list) if G.nodes[node_id]["text_label"] != "unlabeled"])
    kf = KFold(n_splits=parameters["k_folds"], shuffle=True, random_state=parameters["random_seed"])
    splits = list(kf.split(labeled_indices))

    # 4. NEW STEP: Tune thresholds using cross-validation
    threshold_log_dir = os.path.join(args.model_output_dir, "threshold_tuning_logs")
    os.makedirs(threshold_log_dir, exist_ok=True)
    
    avg_plasmid_thresh, avg_chromosome_thresh = tune_thresholds(
        parameters, data, device, splits, labeled_indices, threshold_log_dir
    )
    
    # Add the averaged thresholds to the final parameters object
    parameters['plasmid_threshold'] = avg_plasmid_thresh
    parameters['chromosome_threshold'] = avg_chromosome_thresh

    # 5. REVISED STEP: Train the final model on all data
    final_model = train_final_model(parameters, data, device, splits, labeled_indices, log_dir)

    # 6. Save the final artifacts
    final_model_path = os.path.join(args.model_output_dir, "final_model.pt")
    torch.save(final_model.state_dict(), final_model_path)

    final_parameters_path = os.path.join(args.model_output_dir, "final_model_config_with_thresholds.yaml")
    parameters.write_yaml(final_parameters_path)
    
    print(f"\n✅ Final model and config (with tuned thresholds) saved to {args.model_output_dir}")


if __name__ == "__main__":
    main()