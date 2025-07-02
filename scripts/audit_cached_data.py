# In audit_cached_data.py
import argparse
import torch
import networkx as nx
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import the necessary classes from your project structure
from plasgraph.data import Dataset_Pytorch
from plasgraph.config import config as Config

def analyze_and_plot_feature_distributions(graph: nx.Graph, features_to_analyze: list[str], output_dir: str):
    """
    Analyzes and plots the distribution of specified node features.
    
    For each feature, it calculates summary statistics and saves a histogram plot.
    """
    print(f"📊 Analyzing feature distributions. Plots will be saved to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    node_data = {node: data for node, data in graph.nodes(data=True)}
    df = pd.DataFrame.from_dict(node_data, orient='index')

    # --- NEW: Explicitly identify and report ghost nodes ---
    ghost_nodes_mask = df['length'].isna()
    if ghost_nodes_mask.any():
        print(f"\n👻 Found {ghost_nodes_mask.sum()} ghost nodes (with missing 'length' attribute).")
        print("IDs of ghost nodes:", df[ghost_nodes_mask].index.tolist())
    else:
        print("\n✅ No ghost nodes with missing attributes found.")

    for feature in features_to_analyze:
        if feature not in df.columns:
            print(f"\n⚠️ Feature '{feature}' not found in the graph attributes. Skipping.")
            continue
        
        # Fill missing values with 0 to include ghost nodes in the audit
        feature_series = df[feature].fillna(0)

        if feature_series.empty:
            print(f"\n⚠️ Feature '{feature}' has no data. Skipping.")
            continue
            
        print(f"\n--- Analysis for feature: '{feature}' (including ghost nodes as 0) ---")

        print("Summary Statistics:")
        print(feature_series.describe())

        plt.figure(figsize=(10, 6))
        sns.histplot(feature_series, kde=True, bins=50)
        plt.title(f"Distribution of Node Feature: '{feature}' (Ghost Nodes Included)", fontsize=16)
        plt.xlabel(f"Value of {feature}", fontsize=12)
        plt.ylabel("Frequency (Number of Nodes)", fontsize=12)

        if feature_series.max() > 1000 and feature_series.skew() > 2:
             plt.yscale('log')
             plt.ylabel("Frequency (Log Scale)", fontsize=12)

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f"distribution_{feature}_with_ghosts.png")
        plt.savefig(plot_path)
        plt.close()

        print(f"✅ Histogram saved to: {plot_path}")

# --- NEW: Function to analyze edge attributes ---
def analyze_edge_attributes(graph: nx.Graph, output_dir: str):
    """
    Analyzes and plots the distribution of the primary edge attribute.
    """
    print("\n" + "="*50)
    print("🔬 Analyzing Edge Attributes")
    print("="*50)

    # Check for the 'evo' embedding attribute first, then the 'kmer' one
    edge_attrs = list(nx.get_edge_attributes(graph, "embedding_cosine_similarity").values())
    attr_name = "embedding_cosine_similarity"

    if not edge_attrs:
        edge_attrs = list(nx.get_edge_attributes(graph, "kmer_dot_product").values())
        attr_name = "kmer_dot_product"

    if not edge_attrs:
        print("\n⚠️ No edge attributes ('embedding_cosine_similarity' or 'kmer_dot_product') found. Skipping analysis.")
        return

    edge_series = pd.Series(edge_attrs)

    print(f"\n--- Analysis for edge attribute: '{attr_name}' ---")
    
    print("Summary Statistics:")
    print(edge_series.describe())

    # Specifically count the number of edges with an attribute of exactly 0
    zero_count = (edge_series == 0).sum()
    total_edges = len(edge_series)
    if total_edges > 0:
        zero_percentage = (zero_count / total_edges) * 100
        print(f"\n🔍 Found {zero_count} edges with an attribute of exactly 0 ({zero_percentage:.2f}% of total).")

    plt.figure(figsize=(10, 6))
    sns.histplot(edge_series, bins=50)
    plt.title(f"Distribution of Edge Attribute: '{attr_name}'", fontsize=16)
    plt.xlabel("Attribute Value", fontsize=12)
    plt.ylabel("Frequency (Number of Edges)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"distribution_edge_{attr_name}.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"✅ Edge attribute histogram saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a comprehensive analysis on cached plASgraph2 data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("config_file", help="YAML configuration file.")
    parser.add_argument("train_file_list", help="CSV file listing training samples.")
    parser.add_argument("file_prefix", help="Common prefix for all filenames.")
    parser.add_argument("--data_cache_dir", required=True, help="Directory where the processed graph data is cached.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the analysis plots.")
    parser.add_argument("--feature", type=str, default=None, help="Optional: Specify a single feature to analyze.")
    args = parser.parse_args()

    cache_file_path = os.path.join(args.data_cache_dir, "processed", "all_graphs.pt")
    if not os.path.exists(cache_file_path):
        print(f"❌ Error: Cache file not found at {cache_file_path}")
        return

    print(f"✅ Cache found. Loading processed data from: {cache_file_path}")
    parameters = Config(args.config_file)
    all_graphs_dataset = Dataset_Pytorch(
        root=args.data_cache_dir,
        file_prefix=args.file_prefix,
        train_file_list=args.train_file_list,
        parameters=parameters
    )
    
    G = all_graphs_dataset.G
    print(f"Successfully loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    if args.feature:
        features_to_audit = [args.feature]
        print(f"\n🔬 Auditing single specified feature: '{args.feature}'")
    else:
        features_to_audit = list(parameters["features"]) + ["degree", "length", "gc"]
        print(f"\n🔬 Auditing all model and graph features...")
    
    analyze_and_plot_feature_distributions(G, features_to_audit, args.output_dir)

    # --- NEW: Call the edge analysis function ---
    analyze_edge_attributes(G, args.output_dir)


if __name__ == "__main__":
    main()