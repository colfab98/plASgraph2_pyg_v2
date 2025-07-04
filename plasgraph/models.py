import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GCNConv, GraphNorm
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import numpy as np
import random
import os

from .utils import apply_thresholds


activation_map = {
    'relu': F.relu,
    'sigmoid': torch.sigmoid,
    None: lambda x: x, # Use the Python object None as the key
    'none': lambda x: x  # Optional: Keep for backward compatibility
}

class GCNModel(torch.nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self._params = parameters

        self.preproc = nn.Linear(self['n_input_features'], self['n_channels_preproc'])  
        self.preproc_activation = activation_map[self['preproc_activation']]

        self.fc_input_1 = nn.Linear(self['n_channels_preproc'], self['n_channels'])
        self.fc_input_2 = nn.Linear(self['n_channels_preproc'], self['n_channels'])
        self.fully_connected_activation = activation_map[self['fully_connected_activation']]
        self.gcn_activation = activation_map[self['gcn_activation']]

        self.dropout = nn.Dropout(self['dropout_rate'])

        if self['tie_gnn_layers']:
            self.gcn = GCNConv(self['n_channels'], self['n_channels'])
            self.dense = nn.Linear(self['n_channels'] * 2, self['n_channels'])
        else:
            self.gcn_layers = nn.ModuleList([
                GCNConv(self['n_channels'], self['n_channels']) for _ in range(self['n_gnn_layers'])
            ])
            self.dense_layers = nn.ModuleList([
                nn.Linear(self['n_channels'] * 2, self['n_channels']) for _ in range(self['n_gnn_layers'])
            ])

        self.final_fc1 = nn.Linear(self['n_channels'] * 2, self['n_channels'])
        self.final_fc2 = nn.Linear(self['n_channels'], 2)

        self.output_activation = activation_map[self['output_activation']]


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.preproc(x)
        x = self.preproc_activation(x)
        node_identity = self.fully_connected_activation(self.fc_input_1(x))
        x = self.fully_connected_activation(self.fc_input_2(x))

        for i in range(self['n_gnn_layers']):
            x = self.dropout(x)
            if self['tie_gnn_layers']:
                x = self.gcn_activation(self.gcn(x, edge_index))
                x = torch.cat([node_identity, x], dim=1)
                x = self.dropout(x)
                x = self.fully_connected_activation(self.dense(x))
            else:
                x = self.gcn_activation(self.gcn_layers[i](x, edge_index))
                x = torch.cat([node_identity, x], dim=1)
                x = self.dropout(x)
                x = self.fully_connected_activation(self.dense_layers[i](x))

        x = torch.cat([node_identity, x], dim=1)
        x = self.dropout(x)
        x = self.fully_connected_activation(self.final_fc1(x))
        x = self.dropout(x)
        x = self.final_fc2(x)

        return x
    
    def __getitem__(self, key):
        return self._params[key]
    

# NEW WITH BATCHNORM1D


# Define a custom GGNN layer
class GGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, parameters):
        super().__init__(aggr='add') # aggregation for sum of messages

        activation = parameters['gcn_activation']
        dropout_rate = parameters['dropout_rate']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation_map[activation]
        self.dropout = nn.Dropout(dropout_rate)

        # Edge Gate layer; +1 for kmer dot product between contigs 
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

        # GRU-like gates
        self.lin_z = nn.Linear(in_channels + out_channels, out_channels) 
        self.lin_r = nn.Linear(in_channels + out_channels, out_channels) 
        self.lin_h = nn.Linear(in_channels + out_channels, out_channels) 

    def forward(self, x, edge_index, edge_attr=None): 

        # add self-loops to all nodes in the graph
        # The fill_value for self-loops is now 1.0, representing perfect self-similarity.
        edge_index_with_self_loops, edge_attr_with_self_loops = add_self_loops(
            edge_index, edge_attr=edge_attr, num_nodes=x.size(0), fill_value=1.
        )

        # normalization (use the edge_index with self-loops)
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

        # features of the target node (x_i), source node (x_j), and the edge attribute (edge_attr_expanded, 
        # which is the k-mer dot product for that edge) are concatenated
        edge_gate_input = torch.cat([x_i, x_j, edge_attr_expanded], dim=-1) 
        edge_gate_logit = self.edge_gate_network(edge_gate_input)

        # edge gate value between 0 and 1 for each edge
        edge_gate_value = torch.sigmoid(edge_gate_logit)

        original_message = norm.view(-1, 1) * x_j

        # message that would have been passed from node j is multiplied by this learned edge_gate_value
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


class GGNNModel(torch.nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self._params = parameters

        self.preproc = nn.Linear(self['n_input_features'], self['n_channels_preproc'])
        # REPLACE BatchNorm1d with GraphNorm
        self.norm_preproc = GraphNorm(self['n_channels_preproc'])
        self.preproc_activation = activation_map[self['preproc_activation']]

        self.initial_node_transform = nn.Linear(self['n_channels_preproc'], self['n_channels'])
        # REPLACE BatchNorm1d with GraphNorm
        self.norm_initial = GraphNorm(self['n_channels'])
        self.fully_connected_activation = activation_map[self['fully_connected_activation']]

        self.dropout = nn.Dropout(self['dropout_rate'])

        self.edge_gate_hidden_dim = self['edge_gate_hidden_dim']
        # REPLACE BatchNorm1d with GraphNorm for the GNN layers
        self.norm_ggnn = GraphNorm(self['n_channels'])

        if self['tie_gnn_layers']:
            self.ggnn_layer = GGNNConv(self['n_channels'],
                                    self['n_channels'],
                                    parameters=self._params)
        else:
            self.ggnn_layers = nn.ModuleList([
                GGNNConv(self['n_channels'],
                        self['n_channels'],
                        parameters=self._params)
                for _ in range(self['n_gnn_layers'])
            ])

        self.final_fc1 = nn.Linear(self['n_channels'] * 2, self['n_channels']) 
        # REPLACE BatchNorm1d with GraphNorm
        self.norm_final_fc1 = GraphNorm(self['n_channels'])
        self.final_fc2 = nn.Linear(self['n_channels'], 2)

        self.output_activation = activation_map[self['output_activation']]

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.preproc(x)
        # Apply GraphNorm using x and the batch vector from the data object
        x = self.norm_preproc(x, data.batch)
        x = self.preproc_activation(x)

        h_0 = self.initial_node_transform(x)
        # Apply GraphNorm
        h_0 = self.norm_initial(h_0, data.batch)
        h_0 = self.fully_connected_activation(h_0) 
        
        node_identity = h_0 

        h = h_0 

        for i in range(self['n_gnn_layers']):
            current_edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

            if self['tie_gnn_layers']:
                h = self.ggnn_layer(h, edge_index, edge_attr=current_edge_attr) 
            else:
                h = self.ggnn_layers[i](h, edge_index, edge_attr=current_edge_attr) 
            # Apply GraphNorm after each GNN layer
            h = self.norm_ggnn(h, data.batch)

        x = torch.cat([node_identity, h], dim=1) 
        x = self.dropout(x) 
        
        x = self.final_fc1(x) 
        # Apply GraphNorm
        x = self.norm_final_fc1(x, data.batch)
        x = self.fully_connected_activation(x) 

        x = self.dropout(x) 
        x = self.final_fc2(x) 

        return x

    def __getitem__(self, key):
        return self._params[key]



