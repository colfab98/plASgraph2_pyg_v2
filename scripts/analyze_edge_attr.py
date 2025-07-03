import argparse
import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Import necessary modules from the project
from plasgraph.data import Dataset_Pytorch
from plasgraph.config import config as Config

def run_analysis(parameters, data_cache_dir, file_prefix, train_file_list, output_dir):
    """
    Performs a comprehensive analysis of graph edge attributes, including
    summary statistics, visualizations, and a detailed report on zero-value similarities.
    """
    print("--- Starting Edge Attribute Analysis ---")
    os.makedirs(output_dir, exist_ok=True)

    # Force feature generation to 'evo' for this analysis
    parameters['feature_generation_method'] = 'evo'
    print(f"Forcing feature generation method to: {parameters['feature_generation_method']}")

    # --- 1. Load Data ---
    print("Loading dataset and generating/loading graph features...")
    try:
        dataset = Dataset_Pytorch(
            root=data_cache_dir,
            file_prefix=file_prefix,
            train_file_list=train_file_list,
            parameters=parameters
        )
        G = dataset.G
        print("Graph data loaded successfully.")
    except Exception as e:
        print(f"🚨 Error loading data: {e}")
        return

    # --- 2. Comprehensive Analysis of All Edge Similarities ---
    print("\n" + "="*50)
    print("--- Comprehensive Analysis of All Edge Similarities ---")
    print("="*50)

    all_similarities = [
        edge_data.get("embedding_cosine_similarity")
        for u, v, edge_data in G.edges(data=True)
        if "embedding_cosine_similarity" in edge_data
    ]

    if not all_similarities:
        print("No 'embedding_cosine_similarity' attributes found on edges. Cannot perform analysis.")
        return

    sim_series = pd.Series(all_similarities)
    
    print("\n📈 Summary Statistics for Edge Cosine Similarity:")
    print(sim_series.describe().round(4))

    # --- 3. Generate and Save Visualizations ---
    print(f"\n🖼️  Generating and saving visualizations to: {output_dir}")

    # Plot 1: Histogram
    plt.figure(figsize=(12, 7))
    sns.histplot(sim_series, kde=True, bins=50, color="skyblue")
    plt.title('Distribution of Edge Cosine Similarities (Evo Embeddings)', fontsize=16)
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    histogram_path = os.path.join(output_dir, "edge_similarity_histogram.png")
    plt.savefig(histogram_path)
    plt.close()
    print(f"✅ Histogram saved to: {histogram_path}")

    # Plot 2: Box Plot
    plt.figure(figsize=(12, 7))
    sns.boxplot(x=sim_series, color="lightcoral")
    plt.title('Box Plot of Edge Cosine Similarities', fontsize=16)
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    boxplot_path = os.path.join(output_dir, "edge_similarity_boxplot.png")
    plt.savefig(boxplot_path)
    plt.close()
    print(f"✅ Box plot saved to: {boxplot_path}")

    # --- 4. Detailed Analysis of Zero-Similarity Edges ---
    print("\n" + "="*50)
    print("--- Detailed Analysis of Zero-Similarity Edges ---")
    print("="*50)

    zero_similarity_edges = []
    node_features_to_report = parameters['features'] + ('sequence', 'length')

    for u, v, edge_data in G.edges(data=True):
        if edge_data.get("embedding_cosine_similarity") == 0.0:
            edge_info = {"Edge": f"({u}, {v})"}
            for feature in node_features_to_report:
                edge_info[f"U_{feature}"] = G.nodes[u].get(feature, 'N/A')
                edge_info[f"V_{feature}"] = G.nodes[v].get(feature, 'N/A')
            zero_similarity_edges.append(edge_info)

    if not zero_similarity_edges:
        print("\n✅ No edges with a cosine similarity of 0 were found.")
    else:
        print(f"\nFound {len(zero_similarity_edges)} edges with exactly zero cosine similarity.")
        df_zero = pd.DataFrame(zero_similarity_edges)
        
        print("\n🔬 Investigating the primary cause for zero-similarity edges...")
        df_zero['U_is_empty_seq'] = df_zero['U_sequence'].apply(lambda x: x == '')
        df_zero['V_is_empty_seq'] = df_zero['V_sequence'].apply(lambda x: x == '')
        
        empty_seq_causes = df_zero[df_zero['U_is_empty_seq'] | df_zero['V_is_empty_seq']]
        
        if not empty_seq_causes.empty:
            print("🚨 Root cause identified: All zero-similarity edges involve at least one node with an EMPTY sequence.")
            print("   This results in a zero-vector embedding, leading to a cosine similarity of 0.")
        else:
            print("   The cause might be more complex, possibly related to the Evo model's handling of specific short sequences.")

def main():
    parser = argparse.ArgumentParser(
        description="Run a comprehensive analysis of Evo embedding edge attributes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("config_file", help="Path to the YAML configuration file.")
    parser.add_argument("train_file_list", help="Path to the CSV file listing training samples.")
    parser.add_argument("file_prefix", help="Common prefix for all data filenames.")
    parser.add_argument("--data_cache_dir", required=True, help="Directory to store or load the processed graph data.")
    parser.add_argument("--output_dir", required=True, help="Directory to save analysis plots and reports.")
    args = parser.parse_args()

    parameters = Config(args.config_file)
    run_analysis(
        parameters=parameters,
        data_cache_dir=args.data_cache_dir,
        file_prefix=args.file_prefix,
        train_file_list=args.train_file_list,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()