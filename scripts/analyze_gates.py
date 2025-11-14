# In scripts/analyze_gates.py

import argparse
import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

# Import from our library
from plasgraph.data import Dataset_Pytorch
from plasgraph.config import config as Config
from plasgraph.models import GGNNModel, activation_map # Import original GGNN for structure
from plasgraph.utils import pair_to_label

class DummyAccelerator:
    """A mock class to satisfy the Dataset_Pytorch constructor."""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Add num_processes attribute for compatibility with Dataset_Pytorch
        self.num_processes = 1
        self.is_main_process = True

    def print(self, message):
        print(message)


# Define a device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- MODIFICATION START ---
# The function signature is changed to accept the full 'parameters' object
def create_plots(df, layer_num, output_dir, parameters):
    """Generates and saves a set of plots for a layer's gate data."""
    
    # --- Plot 1: Gate Distribution (Unchanged) ---
    plt.figure(figsize=(10, 6))
    sns.histplot(df['gate_value'], bins=50, kde=True)
    plt.title(f'Layer {layer_num}: Distribution of Edge Gate Values')
    plt.xlabel('Gate Value (0=Closed, 1=Open)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'gate_dist_layer_{layer_num}.png'))
    plt.close()

    # --- Plot 2: Gate vs. Similarity ---
    feature_method = parameters['feature_generation_method']
    if feature_method == 'emb':
        sim_label = 'Edge Attribute (DNABERT Cosine Similarity)'
    elif feature_method == 'kmer':
        sim_label = 'Edge Attribute (K-mer Dot Product)'
    else:
        sim_label = 'Edge Similarity Attribute'

    plt.figure(figsize=(10, 6))
    # Sample the dataframe once for all scatter plots
    sample_df = df.sample(n=min(5000, len(df)))
    
    # Use the 'edge_similarity' column
    sns.scatterplot(data=sample_df, x='edge_similarity', y='gate_value', alpha=0.5, s=10)
    plt.title(f'Layer {layer_num}: Gate Value vs. Edge Similarity')
    plt.xlabel(sim_label)
    plt.ylabel('Gate Value')
    plt.grid(True, alpha=0.3)
    # Save with a new, more specific name
    plt.savefig(os.path.join(output_dir, f'gate_vs_similarity_layer_{layer_num}.png'))
    plt.close()

    # --- Plot 3: Gate vs. Read Support (NEW PLOT) ---
    # This plot is only generated if the 'use_edge_read_counts' parameter is True
    if parameters['use_edge_read_counts']:
        plt.figure(figsize=(10, 6))
        # Use the 'edge_read_support' column
        sns.scatterplot(data=sample_df, x='edge_read_support', y='gate_value', alpha=0.5, s=10)
        plt.title(f'Layer {layer_num}: Gate Value vs. Read Support')
        # The feature is log-transformed in data.py
        plt.xlabel('Edge Attribute (Log Read Support)') 
        plt.ylabel('Gate Value')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'gate_vs_read_support_layer_{layer_num}.png'))
        plt.close()
    
    # --- Plot 4: Gate by Type (Previously Plot 3) ---
    if 'connection_type' in df.columns:
        plt.figure(figsize=(12, 7))
        order = sorted(df['connection_type'].unique())
        sns.boxplot(data=df, x='connection_type', y='gate_value', order=order)
        plt.title(f'Layer {layer_num}: Gate Values by Node Connection Type')
        plt.xlabel('Connection Type (Source-Target)')
        plt.ylabel('Gate Value')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'gate_by_type_layer_{layer_num}.png'))
        plt.close()
# --- MODIFICATION END ---


def main():
    parser = argparse.ArgumentParser(description="Analyze edge gate values of a trained plASgraph2 GGNN model.")
    parser.add_argument("--run_name", required=True, help="Unique name of the experiment run to analyze.")
    parser.add_argument("train_file_list", help="CSV file listing training samples (for data loading).")
    parser.add_argument("file_prefix", help="Common prefix for all data filenames.")
    args = parser.parse_args()

    data_cache_dir = os.path.join("processed_data", args.run_name, "train")
    run_dir = os.path.join("runs", args.run_name)
    model_dir = os.path.join(run_dir, "final_model")
    output_dir = os.path.join(run_dir, "gate_analysis")
    csv_dir = os.path.join(output_dir, "csv_reports")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"📊 Analysis results will be saved to: {output_dir}")

    config_path = os.path.join(model_dir, "base_model_config.yaml")
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    parameters = Config(config_path)
    
    if isinstance(parameters['features'], str):
        parameters._params['features'] = tuple(parameters['features'].split(','))

    if parameters['model_type'] != 'GGNNModel' or not parameters['use_edge_gate']:
        print("Error: This analysis script is designed for a GGNNModel trained with 'use_edge_gate: true'.")
        return

    print("Loading dataset...")
    dummy_accelerator = DummyAccelerator()
    all_graphs = Dataset_Pytorch(
        root=data_cache_dir,
        file_prefix=args.file_prefix,
        train_file_list=args.train_file_list,
        parameters=parameters,
        accelerator=dummy_accelerator
    )
    data = all_graphs[0]
    G = all_graphs.G
    node_list = all_graphs.node_list
    print("Dataset loaded.")

    print("Loading trained model...")
    ensemble_dir = os.path.join(model_dir, "final_training_logs", "ensemble_models")
    fold_models = [f for f in os.listdir(ensemble_dir) if f.startswith('fold_') and f.endswith('_model.pt')]
    if not fold_models:
        print(f"Error: No fold models found in {ensemble_dir}")
        return
    model_path = os.path.join(ensemble_dir, sorted(fold_models)[0])
    print(f"Using model: {model_path}")

    model = GGNNModel(parameters)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Model loaded.")

    print("\nRunning gate analysis...")
    model.set_analysis_mode(True)

    with torch.no_grad():
        _ = model(data.to(DEVICE))

    gate_tensors_per_layer = model.get_gate_data()
    model.set_analysis_mode(False)
    
    print("Analysis complete. Generating reports...")
    
    node_labels = [G.nodes[node_id]["text_label"] for node_id in node_list]

    for i, gate_values_tensor in enumerate(gate_tensors_per_layer):
        layer_num = i + 1
        print(f"\n--- Processing Layer {layer_num} ---")
        
        edge_index_sl, edge_attr_sl = add_self_loops(
            data.edge_index, edge_attr=data.edge_attr, num_nodes=data.num_nodes, fill_value=1.
        )

        # --- MODIFICATION START ---
        # This block dynamically creates the dataframe columns
        # based on the 'use_edge_read_counts' parameter.
        
        edge_attr_data = edge_attr_sl.cpu().numpy()

        df_data = {
            'source_idx': edge_index_sl[0].cpu().numpy(),
            'target_idx': edge_index_sl[1].cpu().numpy(),
            'gate_value': gate_values_tensor.squeeze(-1).cpu().numpy()
        }

        # The model was trained with both similarity and read counts
        if parameters['use_edge_read_counts']:
            # edge_attr_data is shape [N, 2]
            df_data['edge_similarity'] = edge_attr_data[:, 0]
            df_data['edge_read_support'] = edge_attr_data[:, 1]
        else:
            # edge_attr_data is shape [N, 1]
            df_data['edge_similarity'] = edge_attr_data.squeeze(-1)

        df = pd.DataFrame(df_data)
        # --- MODIFICATION END ---


        df['source_label'] = df['source_idx'].map(lambda idx: node_labels[idx])
        df['target_label'] = df['target_idx'].map(lambda idx: node_labels[idx])
        
        df['connection_type'] = df.apply(
            lambda row: "-".join(sorted([row['source_label'], row['target_label']])), axis=1
        )
        
        csv_path = os.path.join(csv_dir, f'gate_analysis_layer_{layer_num}.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved detailed data to {csv_path}")

        print("Gate Value Summary:")
        print(df['gate_value'].describe())

        # --- MODIFICATION START ---
        # Pass the full parameters object to the plotting function
        create_plots(df, layer_num, plots_dir, parameters)
        # --- MODIFICATION END ---
        
        print(f"✅ Saved plots for Layer {layer_num}")
        
    print("\nAll done!")

if __name__ == "__main__":
    main()