# In scripts/evaluate.py

import argparse
import os
import torch

# Import from our library
from plasgraph.data import Dataset_Pytorch
from plasgraph.config import config as Config
from plasgraph.models import GCNModel, GGNNModel
from plasgraph.metrics import calculate_and_print_metrics
from plasgraph.utils import plot_f1_violin

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained plASgraph2 model.")
    parser.add_argument("model_dir", help="Directory containing the trained model and config")
    parser.add_argument("test_file_list", help="CSV file listing test samples")
    parser.add_argument("file_prefix", help="Common prefix for all filenames")
    parser.add_argument("log_dir", help="Output folder for evaluation logs and plots")
    args = parser.parse_args()

    # 1. Load Config and Model
    config_path = os.path.join(args.model_dir, "final_model_config_with_thresholds.yaml")
    parameters = Config(config_path)
    
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print(f"Using device: {device}")

    if parameters['model_type'] == 'GCNModel':
        model = GCNModel(parameters).to(device)
    elif parameters['model_type'] == 'GGNNModel':
        model = GGNNModel(parameters).to(device)

    model_weights_path = os.path.join(args.model_dir, "final_model.pt")
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    # 2. Load Test Data
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

    # 3. Create Test Masks (from original plASgraph2_test.py)
    masks_test_values = []
    for node_id in node_list_test:
        label = G_test.nodes[node_id]["text_label"]
        if label == "unlabeled": masks_test_values.append(0.0)
        elif label == "chromosome": masks_test_values.append(1.0)
        else: masks_test_values.append(float(parameters["plasmid_ambiguous_weight"]))
    masks_test = torch.tensor(masks_test_values, dtype=torch.float32, device=device)

    # 4. Run Inference
    with torch.no_grad():
        all_nodes_outputs = torch.sigmoid(model(data_test))

    # 5. Calculate Metrics and Generate Plots
    os.makedirs(args.log_dir, exist_ok=True)
    plasmid_f1s, chromosome_f1s = calculate_and_print_metrics(
        all_nodes_outputs, data_test, masks_test, G_test, node_list_test, parameters
    )
    plot_f1_violin(plasmid_f1s, chromosome_f1s, args.log_dir)

if __name__ == "__main__":
    main()