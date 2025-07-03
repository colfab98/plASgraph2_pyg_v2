# In scripts/evaluate_ensemble.py

import argparse
import os
import torch
import glob
import pandas as pd

# Import from our library
from plasgraph.data import Dataset_Pytorch
from plasgraph.config import config as Config
from plasgraph.models import GCNModel, GGNNModel
from plasgraph.metrics import calculate_and_print_metrics
from plasgraph.utils import plot_f1_violin

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained plASgraph2 model ensemble.")
    parser.add_argument("model_dir", help="Directory containing the trained models and config")
    parser.add_argument("test_file_list", help="CSV file listing test samples")
    parser.add_argument("file_prefix", help="Common prefix for all filenames")
    parser.add_argument("log_dir", help="Output folder for evaluation logs and plots")
    args = parser.parse_args()

    # 1. Load Config and Find Ensemble Models
    config_path = os.path.join(args.model_dir, "final_model_config_with_thresholds.yaml")
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return
    parameters = Config(config_path)

    # Locate all saved model files for the ensemble
    ensemble_dir = os.path.join(args.model_dir, "final_training_logs", "ensemble_models")
    model_paths = sorted(glob.glob(os.path.join(ensemble_dir, "fold_*_model.pt")))

    if not model_paths:
        print(f"Error: No ensemble models found in {ensemble_dir}")
        return
    print(f"Found {len(model_paths)} models for the ensemble evaluation.")

    # 2. Setup Device and Load Models
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print(f"Using device: {device}")

    models = []
    for model_path in model_paths:
        if parameters['model_type'] == 'GCNModel':
            model = GCNModel(parameters).to(device)
        elif parameters['model_type'] == 'GGNNModel':
            model = GGNNModel(parameters).to(device)
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)

    # 3. Load Test Data
    test_data_root = os.path.join(args.log_dir, "test_data_processed")
    all_test_graphs = Dataset_Pytorch(
        root=test_data_root,
        file_prefix=args.file_prefix,
        train_file_list=args.test_file_list,
        parameters=parameters
    )
    data_test = all_test_graphs[0].to(device)
    G_test = all_test_graphs.G
    node_list_test = all_test_graphs.node_list

    # 4. Create Test Masks
    masks_test_values = []
    for node_id in node_list_test:
        label = G_test.nodes[node_id]["text_label"]
        if label == "unlabeled": masks_test_values.append(0.0)
        elif label == "chromosome": masks_test_values.append(1.0)
        else: masks_test_values.append(float(parameters["plasmid_ambiguous_weight"]))
    masks_test = torch.tensor(masks_test_values, dtype=torch.float32, device=device)

    # 5. Run Inference with the Ensemble
    all_probs = []
    with torch.no_grad():
        for model in models:
            # Get raw logits and apply sigmoid to get probabilities
            logits = model(data_test)
            probs = torch.sigmoid(logits)
            all_probs.append(probs)

    # Average the probabilities from all models to get the final ensemble prediction
    ensemble_outputs = torch.stack(all_probs).mean(dim=0)

    # 6. Calculate Metrics and Generate Plots
    os.makedirs(args.log_dir, exist_ok=True)
    plasmid_f1s, chromosome_f1s = calculate_and_print_metrics(
        ensemble_outputs, data_test, masks_test, G_test, node_list_test, parameters
    )
    plot_f1_violin(plasmid_f1s, chromosome_f1s, args.log_dir)

if __name__ == "__main__":
    main()