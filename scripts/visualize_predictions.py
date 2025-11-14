import argparse
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from plasgraph.data import read_single_graph
from plasgraph.utils import get_node_id

def load_data(args):
    """
    loads the graph structure, predicted labels, and optional ground truth.
    aligns data by inferring the sample_id from the prediction file.
    """
    # load predictions and infer sample_id
    pred_df = pd.read_csv(args.prediction_file)
    if pred_df.empty:
        raise ValueError("Prediction file is empty.")
    sample_id = pred_df['sample'].iloc[0] # Get sample_id from the file

    # create a dictionary mapping unique node ids to predicted labels
    pred_labels = {
        get_node_id(row['sample'], row['contig']): row['label']
        for _, row in pred_df.iterrows()
    }

    # load the networkx graph topology from the raw gfa file
    gfa_prefix = os.path.dirname(args.gfa_file) + '/'
    gfa_filename = os.path.basename(args.gfa_file)
    G = read_single_graph(gfa_prefix, gfa_filename, sample_id, 0)

    # load ground truth if provided (for side-by-side comparison)
    true_labels = None
    if args.label_file:
        label_df = pd.read_csv(args.label_file)
        if 'sample' in label_df.columns:
            label_df = label_df[label_df['sample'] == sample_id]
        true_labels = {
            get_node_id(sample_id, row['contig']): row['label']
            for _, row in label_df.iterrows()
        }
        
    return G, pred_labels, true_labels, sample_id 

def draw_graph(ax, G, node_labels_dict, title, color_map):
   """
    draws a single graph on a matplotlib axis using force-directed layout.
    encodes biological properties (length, classification) into visual elements (size, color).
    """
    # map node labels to colors, defaulting to 'grey' for unlabeled nodes
    node_colors = [color_map.get(node_labels_dict.get(n, 'unlabeled'), 'grey') for n in G.nodes()]

    # visual encoding: map contig length to node size using log-scaling
    # log(1+x) is used to handle large variance in contig lengths (hundreds vs millions of bp)
    lengths = np.array([G.nodes[n]['length'] for n in G.nodes()])
    node_sizes = np.log1p(lengths) * 20  
    
    print(f"Drawing '{title}'...")
    pos = nx.kamada_kawai_layout(G)
    
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, 
        linewidths=0.5, edgecolors='black'
    )
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=0.7)
    
    ax.set_title(title, fontsize=16)
    ax.axis('off')

def main():
    """
    main function to generate prediction visualizations.
    supports single-plot (prediction only) or dual-plot (prediction vs truth) modes.
    """
    parser = argparse.ArgumentParser(description="Visualize plASgraph2 classification results for a sample.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run_name", required=True, help="Unique name of the experiment run.")
    parser.add_argument("--prediction_file", required=True, help="Full path to the prediction CSV file.")
    parser.add_argument("--gfa_file", required=True, help="Full path to the corresponding GFA file.")
    parser.add_argument("--label_file", help="Optional: Full path to the ground truth label CSV file.")
    args = parser.parse_args()

    # load data first to retrieve the sample_id needed for file naming
    G, pred_labels, true_labels, sample_id = load_data(args)

    output_dir = os.path.join("runs", args.run_name, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    suffix = "comparison" if args.label_file else "prediction"
    filename = f"{sample_id}_{suffix}_visual.png"
    output_path = os.path.join(output_dir, filename)

    num_plots = 2 if true_labels else 1
    fig, axes = plt.subplots(1, num_plots, figsize=(12 * num_plots, 12), squeeze=False)
    
    color_map = {
        'plasmid': '#e60049',      # Intense magenta-red
        'chromosome': '#0bb4ff',   # Vibrant cyan-blue
        'ambiguous': '#9b19f5',    # Strong purple
        'unlabeled': '#C7C7C7'     # Neutral grey
    }

    # always draw the prediction graph in the first column
    ax1 = axes[0, 0]
    draw_graph(ax1, G, pred_labels, f"'{sample_id}': Predicted Labels", color_map)

    # conditionally draw the ground truth graph in the second column
    if true_labels:
        ax2 = axes[0, 1]
        draw_graph(ax2, G, true_labels, f"'{sample_id}': Ground Truth Labels", color_map)
        
    # create a unified legend for the entire figure
    legend_handles = [plt.Rectangle((0,0),1,1, color=color_map[label]) for label in color_map]
    fig.legend(legend_handles, color_map.keys(), loc='lower center', ncol=len(color_map), fontsize=14, bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1]) 
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    print(f"\n✅ Visualization saved to: {output_path}")

if __name__ == "__main__":
    main()