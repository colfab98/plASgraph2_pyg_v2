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

# Define a device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. SPECIALIZED MODEL FOR ANALYSIS
# We create a new set of classes that inherit from the originals but are
# modified to capture and return the internal gate values during a forward pass.
# ==============================================================================

class AnalyzableGGNNConv(MessagePassing):
    """A modified GGNNConv layer that stores gate values during message passing."""
    def __init__(self, in_channels, out_channels, parameters, activation, dropout_rate, edge_gate_hidden_dim):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation_map[activation]
        self.dropout = nn.Dropout(dropout_rate)
        edge_gate_input_dim = in_channels * 2 + 1
        edge_gate_hidden_dim = parameters['edge_gate_hidden_dim']
        edge_gate_depth = parameters['edge_gate_depth'] 

        gate_layers = []
        # Input layer
        gate_layers.append(nn.Linear(edge_gate_input_dim, edge_gate_hidden_dim))
        gate_layers.append(nn.ReLU())

        # Hidden layers (if edge_gate_depth > 1)
        for _ in range(edge_gate_depth - 1):
            gate_layers.append(nn.Linear(edge_gate_hidden_dim, edge_gate_hidden_dim))
            gate_layers.append(nn.ReLU())

        # Output layer
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
        
        # This is the main propagation call that triggers message() and update()
        return self.propagate(edge_index_with_self_loops, x=x, norm=norm, edge_attr=edge_attr_with_self_loops)

    def message(self, x_i, x_j, norm, edge_attr):
        if edge_attr is None:
            edge_attr_expanded = torch.zeros((x_i.size(0), 1), device=x_i.device)
        else:
            edge_attr_expanded = edge_attr
        
        edge_gate_input = torch.cat([x_i, x_j, edge_attr_expanded], dim=-1)
        edge_gate_logit = self.edge_gate_network(edge_gate_input)
        edge_gate_value = torch.sigmoid(edge_gate_logit)
        
        # --- HOOK: This is where we store the gate value for analysis ---
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
        # First, call the original __init__ to build the model structure
        super().__init__(parameters)
        
        # Now, overwrite the GGNN layers with our analyzable version
        if self['tie_gnn_layers']:
            self.ggnn_layer = AnalyzableGGNNConv(
                self['n_channels'], self['n_channels'], parameters=parameters, activation=self['gcn_activation'],
                dropout_rate=self['dropout_rate'], edge_gate_hidden_dim=self['edge_gate_hidden_dim']
            )
        else:
            self.ggnn_layers = nn.ModuleList([
                AnalyzableGGNNConv(
                    self['n_channels'], self['n_channels'], parameters=parameters, activation=self['gcn_activation'],
                    dropout_rate=self['dropout_rate'], edge_gate_hidden_dim=self['edge_gate_hidden_dim']
                ) for _ in range(self['n_gnn_layers'])
            ])

    @torch.no_grad()
    def analyze_gates(self, data):
        self.eval()
        data = data.to(DEVICE)
        gate_data_per_layer = []

        # --- Manual forward pass to capture intermediate states ---
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.preproc(x)
        x = self.norm_preproc(x, batch)
        x = self.preproc_activation(x)
        h_0 = self.fully_connected_activation(self.norm_initial(self.initial_node_transform(x), batch))
        
        h = h_0
        for i in range(self['n_gnn_layers']):
            current_layer = self.ggnn_layers[i] if not self['tie_gnn_layers'] else self.ggnn_layer
            
            # Attach a temporary storage list to the layer for this pass
            current_layer._gate_storage = []
            
            # Execute the forward pass for this layer, which will populate the storage
            h = current_layer(h, edge_index, edge_attr=data.edge_attr)
            
            # --- Process the captured gate values ---
            gate_values_tensor = torch.cat(current_layer._gate_storage)
            
            # We need the edge_index that includes self-loops, as that's what propagate uses
            edge_index_sl, edge_attr_sl = add_self_loops(
                edge_index, edge_attr=data.edge_attr, num_nodes=data.num_nodes, fill_value=1.
            )

            df = pd.DataFrame({
                'source_idx': edge_index_sl[0].cpu().numpy(),
                'target_idx': edge_index_sl[1].cpu().numpy(),
                'edge_attr': edge_attr_sl.squeeze(-1).cpu().numpy(),
                'gate_value': gate_values_tensor.squeeze(-1).cpu().numpy()
            })
            gate_data_per_layer.append(df)
            
            # Clean up the storage hook
            delattr(current_layer, '_gate_storage')

            # Continue the forward pass
            h = self.norm_ggnn(h, batch)
            
        return gate_data_per_layer


# ==============================================================================
# 2. ANALYSIS AND VISUALIZATION FUNCTIONS
# ==============================================================================

def create_plots(df, layer_num, output_dir):
    """Generates and saves a set of plots for a layer's gate data."""
    
    # --- Plot 1: Histogram of Gate Values ---
    plt.figure(figsize=(10, 6))
    sns.histplot(df['gate_value'], bins=50, kde=True)
    plt.title(f'Layer {layer_num}: Distribution of Edge Gate Values')
    plt.xlabel('Gate Value (0=Closed, 1=Open)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'gate_dist_layer_{layer_num}.png'))
    plt.close()

    # --- Plot 2: Scatter Plot of Gate Value vs. Edge Attribute ---
    plt.figure(figsize=(10, 6))
    # Sample to avoid overplotting if there are too many points
    sample_df = df.sample(n=min(5000, len(df)))
    sns.scatterplot(data=sample_df, x='edge_attr', y='gate_value', alpha=0.5, s=10)
    plt.title(f'Layer {layer_num}: Gate Value vs. Edge Attribute')
    plt.xlabel('Edge Attribute (e.g., K-mer Dot Product)')
    plt.ylabel('Gate Value')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'gate_vs_attr_layer_{layer_num}.png'))
    plt.close()

    # --- Plot 3: Boxplot by Connection Type ---
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
    parser.add_argument("model_dir", help="Directory containing the final trained model and config file.")
    parser.add_argument("train_file_list", help="CSV file listing training samples (for data loading).")
    parser.add_argument("file_prefix", help="Common prefix for all data filenames.")
    parser.add_argument("--data_cache_dir", required=True, help="Directory where processed graph data is stored.")
    args = parser.parse_args()

    # --- Setup Output Directory ---
    output_dir = os.path.join(args.model_dir, "gate_analysis")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📊 Analysis results will be saved to: {output_dir}")

    # --- Load Config and Data ---
    config_path = os.path.join(args.model_dir, "base_model_config.yaml")
    parameters = Config(config_path)
    
    if parameters['model_type'] != 'GGNNModel':
        print("Error: This analysis script is designed for GGNNModel only.")
        return

    print("Loading dataset...")
    all_graphs = Dataset_Pytorch(
        root=args.data_cache_dir,
        file_prefix=args.file_prefix,
        train_file_list=args.train_file_list,
        parameters=parameters
    )
    data = all_graphs[0]
    G = all_graphs.G
    node_list = all_graphs.node_list
    print("Dataset loaded.")

    # --- Load Model ---
    print("Loading trained model...")
    model_path = os.path.join(args.model_dir, "final_training_logs", "ensemble_models", "fold_1_model.pt")
    
    # Instantiate our special analyzable model and load the trained weights into it
    model = AnalyzableGGNNModel(parameters)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    print("Model loaded.")

    # --- Run Analysis ---
    print("\nRunning gate analysis...")
    gate_data_per_layer = model.analyze_gates(data)
    print("Analysis complete. Generating reports...")

    # --- Process and Save Results ---
    node_labels = [G.nodes[node_id]["text_label"] for node_id in node_list]

    for i, df in enumerate(gate_data_per_layer):
        layer_num = i + 1
        print(f"\n--- Processing Layer {layer_num} ---")
        
        # Add labels to the dataframe
        df['source_label'] = df['source_idx'].map(lambda idx: node_labels[idx])
        df['target_label'] = df['target_idx'].map(lambda idx: node_labels[idx])
        
        # Create a connection type string (e.g., "plasmid-chromosome")
        df['connection_type'] = df.apply(
            lambda row: "-".join(sorted([row['source_label'], row['target_label']])), axis=1
        )
        
        # Save to CSV
        csv_path = os.path.join(output_dir, f'gate_analysis_layer_{layer_num}.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved detailed data to {csv_path}")

        # Print summary stats
        print("Gate Value Summary:")
        print(df['gate_value'].describe())

        # Generate plots
        create_plots(df, layer_num, output_dir)
        print(f"✅ Saved plots for Layer {layer_num}")
        
    print("\nAll done!")

if __name__ == "__main__":
    main()