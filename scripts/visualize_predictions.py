# In scripts/visualize_predictions.py

import argparse
import os
import glob
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Re-use the GFA reading function from your data module
from plasgraph.data import read_single_graph
from plasgraph.utils import get_node_id

def load_data(args):
    """Loads the graph, predictions, and optional ground truth data."""
    # Construct paths
    run_dir = os.path.join("runs", args.run_name)
    prediction_path = os.path.join(run_dir, "classify", args.prediction_file)
    
    # Find the corresponding GFA file
    gfa_pattern = os.path.join(args.gfa_prefix, f"{args.sample_id}*.gfa*")
    gfa_files = glob.glob(gfa_pattern)
    if not gfa_files:
        raise FileNotFoundError(f"Could not find GFA file for sample '{args.sample_id}' with pattern: {gfa_pattern}")
    gfa_path = gfa_files[0]
    
    # 1. Load the NetworkX graph
    # We set minimum_contig_length to 0 to ensure all nodes from predictions are present
    G = read_single_graph(args.gfa_prefix, os.path.basename(gfa_path), args.sample_id, 0)
    
    # 2. Load predictions
    pred_df = pd.read_csv(prediction_path)
    # Create a dictionary mapping node_id -> predicted_label
    pred_labels = {
        get_node_id(row['sample'], row['contig']): row['label']
        for _, row in pred_df.iterrows()
    }

    # 3. Load ground truth if provided
    true_labels = None
    if args.label_file:
        label_df = pd.read_csv(args.label_file)
        # Filter for the current sample if the file contains multiple samples
        if 'sample' in label_df.columns:
            label_df = label_df[label_df['sample'] == args.sample_id]
        
        true_labels = {
            get_node_id(args.sample_id, row['contig']): row['label']
            for _, row in label_df.iterrows()
        }
        
    return G, pred_labels, true_labels

def draw_graph(ax, G, node_labels_dict, title, color_map):
    """Draws a single graph on a given matplotlib axis."""
    # Set node colors based on labels
    node_colors = [color_map.get(node_labels_dict.get(n, 'unlabeled'), 'grey') for n in G.nodes()]

    # Set node sizes based on contig length (log-scaled)
    lengths = np.array([G.nodes[n]['length'] for n in G.nodes()])
    node_sizes = np.log1p(lengths) * 20  # np.log1p is log(1+x)
    
    print(f"Drawing '{title}'...")
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)
    
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, 
        linewidths=0.5, edgecolors='black'
    )
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=0.7)
    
    ax.set_title(title, fontsize=16)
    ax.axis('off')

def main():
    """Main function to generate and save graph visualizations."""
    parser = argparse.ArgumentParser(description="Visualize plASgraph2 classification results for a sample.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run_name", required=True, help="Unique name of the experiment run.")
    parser.add_argument("--sample_id", required=True, help="The ID of the sample to visualize.")
    parser.add_argument("--prediction_file", required=True, help="Filename of the prediction CSV inside the 'classify' folder.")
    parser.add_argument("--gfa_prefix", required=True, help="Path prefix for the original GFA files.")
    parser.add_argument("--output_filename", required=True, help="Name for the output image file (e.g., 'visualization.png').")
    parser.add_argument("--label_file", help="Optional: Path to the ground truth label CSV file for side-by-side comparison.")
    args = parser.parse_args()

    # --- Setup Directories and Load Data ---
    output_dir = os.path.join("runs", args.run_name, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.output_filename)

    G, pred_labels, true_labels = load_data(args)

    # --- Plotting ---
    num_plots = 2 if true_labels else 1
    fig, axes = plt.subplots(1, num_plots, figsize=(12 * num_plots, 12))
    
    color_map = {
        'plasmid': '#FF6B6B',      # A vibrant red
        'chromosome': '#4ECDC4',   # A calming teal
        'ambiguous': '#45B7D1',    # A distinct blue
        'unlabeled': '#C7C7C7'     # Grey
    }

    # Draw the prediction graph
    ax1 = axes[0] if num_plots > 1 else axes
    draw_graph(ax1, G, pred_labels, f"'{args.sample_id}': Predicted Labels", color_map)

    # Draw the ground truth graph if available
    if true_labels:
        ax2 = axes[1]
        draw_graph(ax2, G, true_labels, f"'{args.sample_id}': Ground Truth Labels", color_map)

    # Create a shared legend
    legend_handles = [plt.Rectangle((0,0),1,1, color=color_map[label]) for label in color_map]
    fig.legend(legend_handles, color_map.keys(), loc='lower center', ncol=len(color_map), fontsize=14, bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to make space for the legend
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    print(f"\n✅ Visualization saved to: {output_path}")

if __name__ == "__main__":
    main()



# python scripts/visualize_predictions.py \
#     --run_name "eskapee_v1" \
#     --sample_id "SAMPLE_01" \
#     --prediction_file "SAMPLE_01_predictions.csv" \
#     --gfa_prefix "path/to/gfa_files/" \
#     --output_filename "sample_01_prediction_viz.png"

# python scripts/visualize_predictions.py \
#     --run_name "eskapee_v1" \
#     --sample_id "SAMPLE_01" \
#     --prediction_file "SAMPLE_01_predictions.csv" \
#     --gfa_prefix "path/to/gfa_files/" \
#     --output_filename "sample_01_comparison_viz.png" \
#     --label_file "path/to/your/labels.csv"