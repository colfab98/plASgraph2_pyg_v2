# In scripts/explain.py

import argparse
import os
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Import from our library
from plasgraph.config import config as Config
from plasgraph.models import GCNModel, GGNNModel
from plasgraph.data import read_single_graph
# --- NEW: Import the main Explainer class ---
from torch_geometric.explain import Explainer, GNNExplainer
import torch_geometric

def load_model_and_config(model_dir):
    """Helper function to load a trained model and its config."""
    config_path = os.path.join(model_dir, "final_model_config_with_thresholds.yaml")
    parameters = Config(config_path)

    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print(f"Using device: {device}")

    if parameters['model_type'] == 'GCNModel':
        model = GCNModel(parameters).to(device)
    elif parameters['model_type'] == 'GGNNModel':
        model = GGNNModel(parameters).to(device)
    else:
        raise ValueError(f"Unsupported model type: {parameters['model_type']}")

    weights_path = os.path.join(model_dir, "final_model.pt")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return parameters, model, device

def prepare_graph_data(parameters, graph_file, file_prefix, sample_id):
    """Loads and prepares a single graph for PyTorch Geometric."""
    G = read_single_graph(file_prefix, graph_file, sample_id, parameters['minimum_contig_length'])
    node_list = list(G)

    for u, v in G.edges():
        kmer_u = np.array(G.nodes[u]["kmer_counts_norm"])
        kmer_v = np.array(G.nodes[v]["kmer_counts_norm"])
        dot_product = np.dot(kmer_u, kmer_v)
        G.edges[u, v]["kmer_dot_product"] = dot_product
    
    features = parameters["features"]
    x = np.array([[G.nodes[node_id][f] for f in features] for node_id in node_list])

    node_to_idx = {node_id: i for i, node_id in enumerate(node_list)}
    edge_sources, edge_targets, edge_attrs = [], [], []
    for u, v, data in G.edges(data=True):
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        edge_sources.extend([u_idx, v_idx])
        edge_targets.extend([v_idx, u_idx])
        dot_product = data.get("kmer_dot_product", 0.0)
        edge_attrs.extend([[dot_product], [dot_product]])

    data_obj = torch_geometric.data.Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(np.vstack((edge_sources, edge_targets)), dtype=torch.long),
        edge_attr=torch.tensor(edge_attrs, dtype=torch.float)
    )
    return G, data_obj, node_list, node_to_idx

def visualize_explanation(G, node_idx, node_list, edge_mask, edge_index, threshold, output_path):
    """Visualizes the explanation subgraph."""
    node_id = node_list[node_idx]
    
    neighbors_and_self = list(G.neighbors(node_id)) + [node_id]
    subgraph = G.subgraph(neighbors_and_self)
    
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(subgraph, seed=42)
    nx.draw_networkx_nodes(subgraph, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(subgraph, pos, font_size=8)

    nx.draw_networkx_nodes(subgraph, pos, nodelist=[node_id], node_color='red', node_size=600)

    for i, (u, v) in enumerate(edge_index.T):
        u_node, v_node = node_list[u], node_list[v]
        if u_node in subgraph.nodes and v_node in subgraph.nodes:
            if edge_mask[i] > threshold:
                nx.draw_networkx_edges(
                    subgraph, pos, edgelist=[(u_node, v_node)],
                    alpha=edge_mask[i].item(), width=2.5, edge_color='orange'
                )

    plt.title(f"Explanation for Contig: {G.nodes[node_id]['contig']}")
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Explanation visualization saved to: {output_path}")

def explain_node(args):
    """Generates explanation for a single node."""
    parameters, model, device = load_model_and_config(args.model_dir)
    G, data_obj, node_list, node_to_idx = prepare_graph_data(
        parameters, args.graph_file, args.file_prefix, args.sample_id
    )
    data_obj = data_obj.to(device)

    full_node_id = f"{args.sample_id}:{args.contig_id}"
    if full_node_id not in node_to_idx:
        raise ValueError(f"Contig ID '{args.contig_id}' not found in sample '{args.sample_id}'.")
    node_idx = node_to_idx[full_node_id]

    print(f"\n🔬 Explaining prediction for node: {full_node_id} (Index: {node_idx})")

    # +++ NEW API FOR EXPLAINER FRAMEWORK +++
    explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='regression', # Corrected mode
        task_level='node',
        return_type='raw' 
    ),
)
    # +++ END OF NEW API SETUP +++

    # The call is now simpler as the model is already configured in the explainer
    explanation = explainer(
        x=data_obj.x,
        edge_index=data_obj.edge_index,
        edge_attr=data_obj.edge_attr,
        index=node_idx
    )

    node_feat_mask = explanation.node_feat_mask
    edge_mask = explanation.edge_mask

    print("\n--- GNNExplainer Results ---")
    
    feature_names = parameters['features']
    feat_importance = pd.Series(node_feat_mask.cpu().numpy().flatten(), index=feature_names).sort_values(ascending=False)
    print("\n📊 Top 5 Most Important Features:")
    print(feat_importance.head(5))

    print("\n🔗 Top 5 Most Important Edges (Neighbors):")
    source_edges = (data_obj.edge_index[0] == node_idx)
    target_edges = (data_obj.edge_index[1] == node_idx)
    connected_edge_indices = torch.where(source_edges | target_edges)[0]

    top_edges = torch.topk(edge_mask[connected_edge_indices], k=min(5, len(connected_edge_indices)))

    for i in range(len(top_edges.values)):
        edge_idx = connected_edge_indices[top_edges.indices[i]].item()
        u, v = data_obj.edge_index[:, edge_idx]
        neighbor_node_idx = v.item() if u.item() == node_idx else u.item()
        neighbor_node_id = node_list[neighbor_node_idx]
        neighbor_contig = G.nodes[neighbor_node_id]['contig']
        importance = top_edges.values[i].item()
        print(f"  - Neighbor: {neighbor_contig} (Importance: {importance:.4f})")
    
    output_path = os.path.join(args.output_dir, f"explanation_{args.contig_id.replace(':', '_')}.png")
    os.makedirs(args.output_dir, exist_ok=True)
    visualize_explanation(G, node_idx, node_list, edge_mask, data_obj.edge_index, args.vis_threshold, output_path)

def explain_global(args):
    """Generates a global feature importance summary."""
    parameters, model, device = load_model_and_config(args.model_dir)
    _, data_obj, _, _ = prepare_graph_data(
        parameters, args.graph_file, args.file_prefix, args.sample_id
    )
    data_obj = data_obj.to(device)
    
    print(f"\n🔬 Generating Global Feature Importance Explanation...")
    print(f"This will explain {args.global_nodes} random nodes and aggregate the results.")

    num_nodes_to_explain = min(args.global_nodes, data_obj.num_nodes)
    node_indices = np.random.choice(data_obj.num_nodes, num_nodes_to_explain, replace=False)

    # +++ NEW API FOR EXPLAINER FRAMEWORK +++
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multilabel_classification',
            task_level='node',
            return_type='raw'
        ),
    )
    # +++ END OF NEW API SETUP +++
    
    all_feat_masks = []
    for i, node_idx in enumerate(node_indices):
        print(f"  - Explaining node {i+1}/{num_nodes_to_explain} (Index: {node_idx})...")
        explanation = explainer(
            x=data_obj.x,
            edge_index=data_obj.edge_index,
            edge_attr=data_obj.edge_attr,
            index=node_idx
        )
        all_feat_masks.append(explanation.node_feat_mask)

    global_feat_importance = torch.stack(all_feat_masks).mean(dim=0)
    
    feature_names = parameters['features']
    feat_importance_series = pd.Series(global_feat_importance.cpu().numpy().flatten(), index=feature_names).sort_values(ascending=False)
    
    print("\n" + "="*40)
    print("📊 Global Feature Importance (Aggregated)")
    print("="*40)
    print(feat_importance_series)
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_csv = os.path.join(args.output_dir, "global_feature_importance.csv")
    feat_importance_series.to_csv(output_csv, header=['average_importance'])
    print(f"\n✅ Global feature importance saved to: {output_csv}")

def list_contigs(args):
    """Loads a graph and lists all available contig IDs."""
    print("Loading graph to list available contigs...")
    parameters = Config(os.path.join(args.model_dir, "final_model_config_with_thresholds.yaml"))
    
    G, _, node_list, _ = prepare_graph_data(
        parameters, args.graph_file, args.file_prefix, args.sample_id
    )
    
    print("\n" + "="*40)
    print(f"Available contigs for sample '{args.sample_id}'")
    print(f"(after filtering by min_length={parameters['minimum_contig_length']})")
    print("="*40)
    
    contig_ids = sorted([G.nodes[node_id]['contig'] for node_id in node_list])
    for cid in contig_ids:
        print(cid)
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explain or list contigs from a plASgraph2 model.")
    subparsers = parser.add_subparsers(dest='command', required=True)

    p_list = subparsers.add_parser('list-contigs', help="List all available contig IDs from a GFA file after processing.")
    p_list.add_argument("model_dir", help="Directory with the trained model and config (to get settings like min_length)")
    p_list.add_argument("graph_file", help="GFA file for the sample")
    p_list.add_argument("sample_id", help="Sample ID corresponding to the GFA file")
    p_list.add_argument("--file_prefix", default="", help="Common path prefix for filenames")
    p_list.set_defaults(func=list_contigs)

    p_node = subparsers.add_parser('node', help="Explain the prediction for a single contig.")
    p_node.add_argument("model_dir", help="Directory with the trained model and config")
    p_node.add_argument("graph_file", help="GFA file for the sample to explain")
    p_node.add_argument("sample_id", help="Sample ID corresponding to the GFA file")
    p_node.add_argument("contig_id", help="The contig ID within the sample to explain")
    p_node.add_argument("output_dir", help="Directory to save explanation plots and reports")
    p_node.add_argument("--file_prefix", default="", help="Common path prefix for filenames")
    p_node.add_argument("--vis_threshold", type=float, default=0.1, help="Edge importance threshold for visualization")
    p_node.set_defaults(func=explain_node)


    p_global = subparsers.add_parser('global', help="Generate a global feature importance report.")
    p_global.add_argument("model_dir", help="Directory with the trained model and config")
    p_global.add_argument("graph_file", help="GFA file for a representative sample")
    p_global.add_argument("sample_id", help="Sample ID for the representative sample")
    p_global.add_argument("output_dir", help="Directory to save the global report")
    p_global.add_argument("--file_prefix", default="", help="Common path prefix for filenames")
    p_global.add_argument("--global_nodes", type=int, default=50, help="Number of random nodes to aggregate for global explanation")
    p_global.set_defaults(func=explain_global)

    args = parser.parse_args()
    args.func(args)