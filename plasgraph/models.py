import random
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GraphNorm, MessagePassing
from torch_geometric.utils import add_self_loops, degree

from .utils import apply_thresholds


# dictionary to map string names to activation function objects
activation_map = {
    'relu': F.relu,
    'sigmoid': torch.sigmoid,
    'none': lambda x: x  
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
    


class GGNNConv(MessagePassing):
    """
    A Gated Graph Neural Network (GGNN) layer.

    This layer uses a gated mechanism to control information flow across edges
    and a GRU-like mechanism to update node hidden states.
    """
    
    def __init__(self, in_channels, out_channels, parameters):
        # 'add' aggregation
        super().__init__(aggr='add')

        activation = parameters['gcn_activation']
        dropout_rate = parameters['dropout_rate']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation_map[activation]
        self.dropout = nn.Dropout(dropout_rate)

        # input dimension is the size of the source and target node features, 
        # plus 1 for the edge attribute (cosine similarity of embeddings or a k-mer dot product)
        edge_gate_input_dim = in_channels * 2 + 1       

        edge_gate_hidden_dim = parameters['edge_gate_hidden_dim']
        edge_gate_depth = parameters['edge_gate_depth'] 

        gate_layers = []
        # input layer
        gate_layers.append(nn.Linear(edge_gate_input_dim, edge_gate_hidden_dim))
        gate_layers.append(nn.ReLU())
        # hidden layers 
        for _ in range(edge_gate_depth - 1):
            gate_layers.append(nn.Linear(edge_gate_hidden_dim, edge_gate_hidden_dim))
            gate_layers.append(nn.ReLU())
        # output layer which produces a single logit for the gate
        gate_layers.append(nn.Linear(edge_gate_hidden_dim, 1))
        self.edge_gate_network = nn.Sequential(*gate_layers)

        # update step 'z'
        self.lin_z = nn.Linear(in_channels + out_channels, out_channels) 
        # reset gate 'r'
        self.lin_r = nn.Linear(in_channels + out_channels, out_channels) 
        # candidate hidden state 'h_candidate'
        self.lin_h = nn.Linear(in_channels + out_channels, out_channels) 

    def forward(self, x, edge_index, edge_attr=None): 
        """
        The forward pass of the layer.

        This method defines the overall computation flow: adding self-loops,
        normalizing, and then calling the propagate method which orchestrates
        the message, aggregate, and update steps.
        """

        # add self-loops to the graph to allow nodes to consider their own features
        # and perfect self-similarity for the edge attribute
        edge_index_with_self_loops, edge_attr_with_self_loops = add_self_loops(
            edge_index, edge_attr=edge_attr, num_nodes=x.size(0), fill_value=1.)

        # --- graph normalization ---
        # helps stabilize training by scaling node features based on their degree
        row, col = edge_index_with_self_loops
        # degree of each node
        deg = degree(col, x.size(0), dtype=x.dtype)
        # inverse square root of the degree
        deg_inv_sqrt = deg.pow(-0.5)
        # infinite values (from nodes with degree 0) to 0
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0 
        #normalization factor for each edge
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index_with_self_loops, x=x, norm=norm, edge_attr=edge_attr_with_self_loops)


    def message(self, x_i, x_j, norm, edge_attr): 
        """
        Computes messages from source nodes (j) to target nodes (i).

        This method calculates the learned gate value for each edge and uses it
        to scale the message being passed.
        """

        # if no edge attributes are provided, create a zero tensor as a placeholder
        if edge_attr is None: 
            edge_attr_expanded = torch.zeros((x_i.size(0), 1), device=x_i.device) 
        else:
            edge_attr_expanded = edge_attr

        # concatenate features of the target node (i), source node (j), and the edge between them
        edge_gate_input = torch.cat([x_i, x_j, edge_attr_expanded], dim=-1) 
        # pass the concatenated features through the edge gate network to get a raw logit
        edge_gate_logit = self.edge_gate_network(edge_gate_input)
        # apply a sigmoid to get a gate value between 0 and 1
        edge_gate_value = torch.sigmoid(edge_gate_logit)

        # calculate the standard GCN message, scaled by the normalization factor
        original_message = norm.view(-1, 1) * x_j
        # message that would have been passed from node j is multiplied by this learned edge_gate_value
        gated_message = edge_gate_value * original_message
        return gated_message

    def update(self, aggr_out, x):
        """
        Updates node embeddings using a GRU-like mechanism.

        This function takes the aggregated messages from neighbors and the original
        node features to compute the new node representation.
        """
        # concatenate the original node features (x) with the aggregated messages
        z_input = torch.cat([x, aggr_out], dim=-1)
        r_input = torch.cat([x, aggr_out], dim=-1)
        
        # calculate the update gate 'z' (determines how much of the old state to keep)
        z = torch.sigmoid(self.lin_z(z_input)) 
        # calculate the reset gate 'r' (determines how much of the old state to use for the new candidate)
        r = torch.sigmoid(self.lin_r(r_input)) 

        # create the input for the candidate hidden state
        h_candidate_input = torch.cat([r * x, aggr_out], dim=-1)
        # calculate the candidate hidden state
        h_candidate = self.lin_h(h_candidate_input)
        # apply the main activation function
        h_candidate = self.activation(h_candidate) 

        # combine the old state and the new candidate state using the update gate 'z'
        out = (1 - z) * x + z * h_candidate
        # apply dropout for regularization
        return self.dropout(out)


class GGNNModel(torch.nn.Module):
    """
    A complete Gated Graph Neural Network model for graph classification tasks.

    This model consists of an initial preprocessing block, a series of GGNN layers
    for message passing, and a final prediction head to generate output scores.
    It uses GraphNorm for normalization and includes a skip connection from the
    initial node embeddings to the final layers.
    """

    def __init__(self, parameters):
        super().__init__()
        self._params = parameters

        # --- input preprocessing block ---
        # linear layer to project the raw input node features to a new dimension
        self.preproc = nn.Linear(self['n_input_features'], self['n_channels_preproc'])
        # graph-aware batch normalization layer
        self.norm_preproc = GraphNorm(self['n_channels_preproc'])
        self.preproc_activation = activation_map[self['preproc_activation']]

        # --- initial node transformation ---
        # linear layer to transform preprocessed features into the main hidden dimension for the GNN
        self.initial_node_transform = nn.Linear(self['n_channels_preproc'], self['n_channels'])
        # graph-aware normalization for the initial hidden states
        self.norm_initial = GraphNorm(self['n_channels'])
        self.fully_connected_activation = activation_map[self['fully_connected_activation']]

        # standard dropout layer for regularization
        self.dropout = nn.Dropout(self['dropout_rate'])

        # --- GGNN message passing layers ---
        # graph-aware normalization to be applied after each GNN layer
        self.norm_ggnn = GraphNorm(self['n_channels'])
        # if 'tie_gnn_layers' is True, a single GGNN layer is created and its weights are shared across all steps
        if self['tie_gnn_layers']:
            self.ggnn_layer = GGNNConv(self['n_channels'],
                                    self['n_channels'],
                                    parameters=self._params)
        # otherwise, create a list of unique GGNN layers
        else:
            self.ggnn_layers = nn.ModuleList([
                GGNNConv(self['n_channels'],
                        self['n_channels'],
                        parameters=self._params)
                for _ in range(self['n_gnn_layers'])
            ])

        # --- final prediction head ---
        # linear layer that takes the concatenated initial and final node embeddings
        self.final_fc1 = nn.Linear(self['n_channels'] * 2, self['n_channels']) 
        self.norm_final_fc1 = GraphNorm(self['n_channels'])
        # final output layer that produces 2 logits (for plasmid and chromosome classes)
        self.final_fc2 = nn.Linear(self['n_channels'], 2)

    def forward(self, data):
        """
        Defines the forward pass of the model.
        """

        x, edge_index = data.x, data.edge_index
        # apply the input preprocessing block
        x = self.preproc(x)
        # apply GraphNorm using the batch vector to normalize features on a per-graph basis
        x = self.norm_preproc(x, data.batch)
        x = self.preproc_activation(x)

        # apply the initial node transformation to get the first hidden state (h_0)
        h_0 = self.initial_node_transform(x)
        h_0 = self.norm_initial(h_0, data.batch)
        h_0 = self.fully_connected_activation(h_0) 
        
        # store the initial hidden state for the final skip connection
        node_identity = h_0 

        h = h_0 

        # --- message passing loop ---
        for i in range(self['n_gnn_layers']):
            # get the edge attributes for the current layer
            current_edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
            # pass the hidden state through the GGNN layer
            if self['tie_gnn_layers']:
                h = self.ggnn_layer(h, edge_index, edge_attr=current_edge_attr) 
            else:
                h = self.ggnn_layers[i](h, edge_index, edge_attr=current_edge_attr) 
            # apply GraphNorm after each message passing step
            h = self.norm_ggnn(h, data.batch)

        # --- prediction head ---
        # concatenate the initial node embeddings (skip connection) with the final GNN embeddings
        x = torch.cat([node_identity, h], dim=1) 
        # apply dropout for regularization
        x = self.dropout(x) 
        
        # pass through the first fully connected layer of the prediction head
        x = self.final_fc1(x) 
        x = self.norm_final_fc1(x, data.batch)
        x = self.fully_connected_activation(x) 

        # apply dropout again before the final output layer
        x = self.dropout(x) 
        # final classification logits
        x = self.final_fc2(x) 

        return x

    def __getitem__(self, key):
        return self._params[key]



