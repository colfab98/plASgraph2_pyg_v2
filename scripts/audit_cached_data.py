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

    # Convert node attributes to a pandas DataFrame for easier analysis
    # This is more efficient than iterating through the graph for each feature
    node_data = {node: data for node, data in graph.nodes(data=True)}
    df = pd.DataFrame.from_dict(node_data, orient='index')

    for feature in features_to_analyze:
        if feature not in df.columns:
            print(f"\n⚠️ Feature '{feature}' not found in the graph attributes. Skipping.")
            continue

        feature_series = df[feature].dropna()

        if feature_series.empty:
            print(f"\n⚠️ Feature '{feature}' has no data. Skipping.")
            continue
            
        print(f"\n--- Analysis for feature: '{feature}' ---")

        # 1. Print Summary Statistics
        print("Summary Statistics:")
        print(feature_series.describe())

        # 2. Generate and Save Histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(feature_series, kde=True, bins=50)
        plt.title(f"Distribution of Node Feature: '{feature}'", fontsize=16)
        plt.xlabel(f"Value of {feature}", fontsize=12)
        plt.ylabel("Frequency (Number of Nodes)", fontsize=12)

        # Use a log scale for y-axis if the data is highly skewed
        if feature_series.max() > 1000 and feature_series.skew() > 2:
             plt.yscale('log')
             plt.ylabel("Frequency (Log Scale)", fontsize=12)

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f"distribution_{feature}.png")
        plt.savefig(plot_path)
        plt.close()

        print(f"✅ Histogram saved to: {plot_path}")

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

    # Verify cache existence
    cache_file_path = os.path.join(args.data_cache_dir, "processed", "all_graphs.pt")
    if not os.path.exists(cache_file_path):
        print(f"❌ Error: Cache file not found at {cache_file_path}")
        return

    # Load data
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

    # Determine which features to analyze
    if args.feature:
        features_to_audit = [args.feature]
        print(f"\n🔬 Auditing single specified feature: '{args.feature}'")
    else:
        # Analyze all features defined in the config + some fundamental graph properties
        features_to_audit = list(parameters["features"]) + ["degree", "length", "gc"]
        print(f"\n🔬 Auditing all model and graph features...")
    
    # Run the comprehensive analysis
    analyze_and_plot_feature_distributions(G, features_to_audit, args.output_dir)

if __name__ == "__main__":
    main()