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

# ==============================================================================
# 1. SPECIALIZED MODEL FOR ANALYSIS
# ==============================================================================

class AnalyzableGGNNConv(MessagePassing):
    """A modified GGNNConv layer that stores gate values during message passing."""
    def __init__(self, in_channels, out_channels, parameters):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation_map[parameters['gcn_activation']]
        self.dropout = nn.Dropout(parameters['dropout_rate'])
        edge_gate_input_dim = in_channels * 2 + 1
        edge_gate_hidden_dim = parameters['edge_gate_hidden_dim']
        edge_gate_depth = parameters['edge_gate_depth'] 

        gate_layers = []
        gate_layers.append(nn.Linear(edge_gate_input_dim, edge_gate_hidden_dim))
        gate_layers.append(nn.ReLU())
        for _ in range(edge_gate_depth - 1):
            gate_layers.append(nn.Linear(edge_gate_hidden_dim, edge_gate_hidden_dim))
            gate_layers.append(nn.ReLU())
        gate_layers.append(nn.Linear(edge_gate_hidden_dim, 1))

        self.edge_gate_network = nn.Sequential(*gate_layers)
        self.lin_z = nn.Linear(in_channels + out_channels, out_channels)
        self.lin_r = nn.Linear(in_channels + out_channels, out_channels)
        self.lin_h = nn.Linear(in_channels + out_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        edge_index_with_self_loops, edge_attr_with_self_loops = add_self_loops(
            edge_index, edge_attr=edge_attr, num_nodes=x.size(0), fill_value=1.
        )
        row, col = edge_index_with_self_loops
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        return self.propagate(edge_index_with_self_loops, x=x, norm=norm, edge_attr=edge_attr_with_self_loops)

    def message(self, x_i, x_j, norm, edge_attr):
        if edge_attr is None:
            edge_attr_expanded = torch.zeros((x_i.size(0), 1), device=x_i.device)
        else:
            edge_attr_expanded = edge_attr
        
        edge_gate_input = torch.cat([x_i, x_j, edge_attr_expanded], dim=-1)
        edge_gate_logit = self.edge_gate_network(edge_gate_input)
        edge_gate_value = torch.sigmoid(edge_gate_logit)
        
        if hasattr(self, '_gate_storage'):
            self._gate_storage.append(edge_gate_value.detach())
            
        original_message = norm.view(-1, 1) * x_j
        gated_message = edge_gate_value * original_message
        return gated_message

    def update(self, aggr_out, x):
        z_input = torch.cat([x, aggr_out], dim=-1)
        r_input = torch.cat([x, aggr_out], dim=-1)
        z = torch.sigmoid(self.lin_z(z_input))
        r = torch.sigmoid(self.lin_r(r_input))
        h_candidate_input = torch.cat([r * x, aggr_out], dim=-1)
        h_candidate = self.lin_h(h_candidate_input)
        h_candidate = self.activation(h_candidate)
        out = (1 - z) * x + z * h_candidate
        return self.dropout(out)

class AnalyzableGGNNModel(GGNNModel):
    """
    A wrapper around the original GGNNModel that uses the AnalyzableGGNNConv
    and provides a method to perform a forward pass while capturing gate values.
    """
    def __init__(self, parameters):
        super().__init__(parameters)
        
        # Overwrite the GGNN layers with our analyzable version
        # This assumes the model uses edge gates; the script is named analyze_gates
        if self['tie_gnn_layers']:
            self.ggnn_layer = AnalyzableGGNNConv(
                self['n_channels'], self['n_channels'], parameters=parameters
            )
        else:
            self.ggnn_layers = nn.ModuleList([
                AnalyzableGGNNConv(
                    self['n_channels'], self['n_channels'], parameters=parameters
                ) for _ in range(self['n_gnn_layers'])
            ])

    @torch.no_grad()
    def analyze_gates(self, data):
        self.eval()
        data = data.to(DEVICE)
        gate_data_per_layer = []

        # --- Manual forward pass to capture intermediate states ---
        # This now mirrors GGNNModel.forward() from models.py
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.preproc(x)
        if self['use_GraphNorm']:
            # CORRECTED: LayerNorm does not take a batch argument
            x = self.norm_preproc(x)
        x = self.preproc_activation(x)

        # CORRECTED: Aligned with the actual model's forward pass
        node_identity = self.fully_connected_activation(self.fc_input_1(x))
        h = self.fully_connected_activation(self.fc_input_2(x))

        for i in range(self['n_gnn_layers']):
            h = self.dropout(h)
            
            # Determine the current GNN and dense layers (handles tied weights)
            current_layer = self.ggnn_layer if self['tie_gnn_layers'] else self.ggnn_layers[i]
            current_dense = self.dense_layer if self['tie_gnn_layers'] else self.dense_layers[i]

            current_layer._gate_storage = []
            h = current_layer(h, edge_index, edge_attr=edge_attr)
            
            gate_values_tensor = torch.cat(current_layer._gate_storage)
            edge_index_sl, edge_attr_sl = add_self_loops(
                edge_index, edge_attr=edge_attr, num_nodes=data.num_nodes, fill_value=1.
            )

            df = pd.DataFrame({
                'source_idx': edge_index_sl[0].cpu().numpy(),
                'target_idx': edge_index_sl[1].cpu().numpy(),
                'edge_attr': edge_attr_sl.squeeze(-1).cpu().numpy() if edge_attr_sl is not None else 0,
                'gate_value': gate_values_tensor.squeeze(-1).cpu().numpy()
            })
            gate_data_per_layer.append(df)
            
            delattr(current_layer, '_gate_storage')

            # Continue the forward pass, mirroring the original model
            if self['use_GraphNorm']:
                h = self.norm_ggnn(h)
            
            h = torch.cat([node_identity, h], dim=1)
            h = self.dropout(h) # Dropout is applied here in the original model
            h = self.fully_connected_activation(current_dense(h))
            
        return gate_data_per_layer


# ==============================================================================
# 2. ANALYSIS AND VISUALIZATION FUNCTIONS
# ==============================================================================

def create_plots(df, layer_num, output_dir):
    """Generates and saves a set of plots for a layer's gate data."""
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['gate_value'], bins=50, kde=True)
    plt.title(f'Layer {layer_num}: Distribution of Edge Gate Values')
    plt.xlabel('Gate Value (0=Closed, 1=Open)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'gate_dist_layer_{layer_num}.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    sample_df = df.sample(n=min(5000, len(df)))
    sns.scatterplot(data=sample_df, x='edge_attr', y='gate_value', alpha=0.5, s=10)
    plt.title(f'Layer {layer_num}: Gate Value vs. Edge Attribute')
    plt.xlabel('Edge Attribute (e.g., K-mer Dot Product)')
    plt.ylabel('Gate Value')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'gate_vs_attr_layer_{layer_num}.png'))
    plt.close()

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


def main():
    parser = argparse.ArgumentParser(description="Analyze edge gate values of a trained plASgraph2 GGNN model.")
    parser.add_argument("--run_name", required=True, help="Unique name of the experiment run to analyze.")
    parser.add_argument("train_file_list", help="CSV file listing training samples (for data loading).")
    parser.add_argument("file_prefix", help="Common prefix for all data filenames.")
    args = parser.parse_args()

    data_cache_dir = os.path.join("processed_data", args.run_name, "train")

    model_dir = os.path.join("runs", args.run_name, "final_model")

    output_dir = os.path.join(model_dir, "gate_analysis")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📊 Analysis results will be saved to: {output_dir}")

    config_path = os.path.join(model_dir, "base_model_config.yaml")
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    parameters = Config(config_path)
    
    # Handle tuple feature from older configs
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
    # More robustly find the first fold model
    ensemble_dir = os.path.join(model_dir, "final_training_logs", "ensemble_models")
    fold_models = [f for f in os.listdir(ensemble_dir) if f.startswith('fold_') and f.endswith('_model.pt')]
    if not fold_models:
        print(f"Error: No fold models found in {ensemble_dir}")
        return
    model_path = os.path.join(ensemble_dir, sorted(fold_models)[0])
    print(f"Using model: {model_path}")
    
    model = AnalyzableGGNNModel(parameters)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    print("Model loaded.")

    print("\nRunning gate analysis...")
    gate_data_per_layer = model.analyze_gates(data)
    print("Analysis complete. Generating reports...")

    node_labels = [G.nodes[node_id]["text_label"] for node_id in node_list]

    for i, df in enumerate(gate_data_per_layer):
        layer_num = i + 1
        print(f"\n--- Processing Layer {layer_num} ---")
        
        df['source_label'] = df['source_idx'].map(lambda idx: node_labels[idx])
        df['target_label'] = df['target_idx'].map(lambda idx: node_labels[idx])
        
        df['connection_type'] = df.apply(
            lambda row: "-".join(sorted([row['source_label'], row['target_label']])), axis=1
        )
        
        csv_path = os.path.join(output_dir, f'gate_analysis_layer_{layer_num}.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved detailed data to {csv_path}")

        print("Gate Value Summary:")
        print(df['gate_value'].describe())

        create_plots(df, layer_num, output_dir)
        print(f"✅ Saved plots for Layer {layer_num}")
        
    print("\nAll done!")

if __name__ == "__main__":
    main()