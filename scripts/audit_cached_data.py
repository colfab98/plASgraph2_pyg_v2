import argparse
import torch
import networkx as nx
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from plasgraph.data import Dataset_Pytorch
from plasgraph.config import config as Config


class DummyAccelerator:
    """
    a mock class that satisfies the Dataset_Pytorch constructor.
    allows loading the dataset without initializing a full distributed environment.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def print(self, message):
        print(message)


def analyze_and_plot_feature_distributions(graph: nx.Graph, features_to_analyze: list[str], output_dir: str):
    """
    analyzes and plots the distribution of specified node features.
    
    for each feature, it calculates summary statistics and saves a histogram plot.
    """
    print(f" Analyzing feature distributions. Plots will be saved to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    node_data = {node: data for node, data in graph.nodes(data=True)}
    df = pd.DataFrame.from_dict(node_data, orient='index')

    for feature in features_to_analyze:
        if feature not in df.columns:
            print(f"\n Feature '{feature}' not found in the graph attributes. Skipping.")
            continue
        
        feature_series = df[feature].dropna()

        if feature_series.empty:
            print(f"\n Feature '{feature}' has no data. Skipping.")
            continue
            
        print(f"\n--- Analysis for feature: '{feature}' ---")

        print("Summary Statistics:")
        print(feature_series.describe())

        plt.figure(figsize=(10, 6))
        sns.histplot(feature_series, kde=True, bins=50)
        plt.title(f"Distribution of Node Feature: '{feature}'", fontsize=16)
        
        plt.xlabel(f"Value of {feature}", fontsize=12)
        plt.ylabel("Frequency (Number of Nodes)", fontsize=12)

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f"distribution_{feature}.png")
        plt.savefig(plot_path)
        plt.close()

        print(f" Histogram saved to: {plot_path}")


def analyze_edge_attributes(graph: nx.Graph, output_dir: str):
    """
    analyzes and plots the distribution of the primary edge attribute.
    this helps confirm if edge features (like embedding similarity) are providing meaningful signals.
    """
    print("\n" + "="*50)
    print(" Analyzing Edge Attributes")
    print("="*50)

    edge_attrs = list(nx.get_edge_attributes(graph, "embedding_cosine_similarity").values())
    attr_name = "embedding_cosine_similarity"

    if not edge_attrs:
        edge_attrs = list(nx.get_edge_attributes(graph, "kmer_dot_product").values())
        attr_name = "kmer_dot_product"

    if not edge_attrs:
        print("\n No edge attributes ('embedding_cosine_similarity' or 'kmer_dot_product') found. Skipping analysis.")
        return

    edge_series = pd.Series(edge_attrs)

    print(f"\n--- Analysis for edge attribute: '{attr_name}' ---")
    
    print("Summary Statistics:")
    print(edge_series.describe())

    zero_count = (edge_series == 0).sum()
    total_edges = len(edge_series)
    if total_edges > 0:
        zero_percentage = (zero_count / total_edges) * 100
        print(f"\n Found {zero_count} edges with an attribute of exactly 0 ({zero_percentage:.2f}% of total).")

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

    print(f" Edge attribute histogram saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a comprehensive analysis on cached plASgraph2 data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("config_file", help="YAML configuration file.")
    parser.add_argument("train_file_list", help="CSV file listing training samples.")
    parser.add_argument("file_prefix", help="Common prefix for all filenames.")
    parser.add_argument("--run_name", required=True, help="Unique name for the experiment run.")
    parser.add_argument("--feature", type=str, default=None, help="Optional: Specify a single feature to analyze.")
    args = parser.parse_args()

    data_cache_dir = os.path.join("processed_data", args.run_name, "train")
    output_dir = os.path.join("runs", args.run_name, "feature_analysis")
    os.makedirs(output_dir, exist_ok=True)

    cache_file_path = os.path.join(data_cache_dir, "processed", "all_graphs.pt")
    if not os.path.exists(cache_file_path):
        print(f" Error: Cache file not found at {cache_file_path}")
        return

    print(f" Cache found. Loading processed data from: {cache_file_path}")
    dummy_accelerator = DummyAccelerator()
    parameters = Config(args.config_file)
    all_graphs_dataset = Dataset_Pytorch(
        root=data_cache_dir,
        file_prefix=args.file_prefix,
        train_file_list=args.train_file_list,
        parameters=parameters,
        accelerator=dummy_accelerator
    )
    
    G = all_graphs_dataset.G
    print(f"Successfully loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    if args.feature:
        features_to_audit = [args.feature]
        print(f"\n Auditing single specified feature: '{args.feature}'")
    else:
        features_to_audit = list(parameters["features"]) + ["degree", "length", "gc"]
        print(f"\n Auditing all model and graph features...")
    
    analyze_and_plot_feature_distributions(G, features_to_audit, output_dir)
    analyze_edge_attributes(G, output_dir)

if __name__ == "__main__":
    main()