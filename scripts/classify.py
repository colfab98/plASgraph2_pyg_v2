import argparse
import os
import pandas as pd
import torch
import glob
import yaml
import numpy as np
import torch_geometric

from plasgraph.config import config as Config
from plasgraph.models import GCNModel, GGNNModel
from plasgraph.data import read_single_graph
from plasgraph.utils import pair_to_label, apply_thresholds

def load_ensemble_and_config(model_dir):
    """
    loads the base configuration and discovers all trained fold models.
    essential for preparing the ensemble inference pipeline.
    """
    config_path = os.path.join(model_dir, "base_model_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Base config file not found at {config_path}")
    parameters = Config(config_path)
    
    parameters.config_file_path = config_path
    
    ensemble_dir = os.path.join(model_dir, "final_training_logs", "ensemble_models")
    model_paths = sorted(glob.glob(os.path.join(ensemble_dir, "fold_*_model.pt")))
    threshold_paths = sorted(glob.glob(os.path.join(ensemble_dir, "fold_*_thresholds.yaml")))

    if not model_paths or len(model_paths) != len(threshold_paths):
        raise FileNotFoundError(f"Mismatch or missing model/threshold files in {ensemble_dir}")
    
    print(f"Found {len(model_paths)} models for ensembling.")
    return parameters, model_paths, threshold_paths

def classify_with_ensemble(parameters, model_paths, threshold_paths, graph_file, file_prefix, sample_id):
    """
    classifies a single graph using the full model ensemble, correctly applying
    per-fold thresholds before averaging.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # load and process the graph data once
    G = read_single_graph(file_prefix, graph_file, sample_id, parameters['minimum_contig_length'])
    node_list = list(G)

    # calculate k-mer dot products for edges (needed for GGNN edge gates)
    for u, v in G.edges():
        kmer_u = np.array(G.nodes[u]["kmer_counts_norm"])
        kmer_v = np.array(G.nodes[v]["kmer_counts_norm"])
        dot_product = np.dot(kmer_u, kmer_v)
        G.edges[u, v]["kmer_dot_product"] = dot_product

    # construct the pytorch geometric data object
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
        edge_attr=torch.tensor(edge_attrs, dtype=torch.float),
        batch=torch.zeros(x.shape[0], dtype=torch.long)
    ).to(device)

    # run inference with each model and apply its specific thresholds
    all_final_scores = []
    
    with torch.no_grad():
        for model_path, thresh_path in zip(model_paths, threshold_paths):
            temp_params = Config(parameters.config_file_path)
            temp_params._params.update(parameters._params)

            # load the specific thresholds optimized for this fold during training
            with open(thresh_path, 'r') as f:
                thresholds = yaml.safe_load(f)
            temp_params['plasmid_threshold'] = thresholds['plasmid_threshold']
            temp_params['chromosome_threshold'] = thresholds['chromosome_threshold']

            # instantiate the model architecture
            if temp_params['model_type'] == 'GCNModel':
                model = GCNModel(temp_params).to(device)
            else: # GGNNModel
                model = GGNNModel(temp_params).to(device)
            
            # load weights and set to eval mode
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            # forward pass to get logits
            logits = model(data_obj)
            
            # handle output activation (some models output raw logits, others probs)
            if temp_params['output_activation'] is None:
                probs = torch.sigmoid(logits) # Convert logits to probs
            else:
                probs = logits 
            
            # scale the raw probabilities using the fold-specific thresholds
            final_scores = torch.from_numpy(
                apply_thresholds(probs.cpu().numpy(), temp_params)
            ).to(device)
            all_final_scores.append(final_scores)

    # average the final scaled scores from all models
    ensemble_final_scores = torch.stack(all_final_scores).mean(dim=0).cpu().numpy()

    # format the results into a dataframe
    output_rows = []
    for i, node_id in enumerate(node_list):
        plasmid_score = ensemble_final_scores[i, 0]
        chrom_score = ensemble_final_scores[i, 1]
        label = pair_to_label([round(plasmid_score), round(chrom_score)])
        
        output_rows.append([
            sample_id,
            G.nodes[node_id]["contig"],
            G.nodes[node_id]["length"],
            plasmid_score,
            chrom_score,
            label
        ])
    
    return pd.DataFrame(
        output_rows,
        columns=["sample", "contig", "length", "plasmid_score", "chrom_score", "label"]
    )

def main_set(args):
    """classifies a set of GFA files listed in a CSV."""
    model_dir = os.path.join("runs", args.run_name, "final_model")
    classify_dir = os.path.join("runs", args.run_name, "classify")
    os.makedirs(classify_dir, exist_ok=True)
    output_path = os.path.join(classify_dir, args.output_filename)

    parameters, model_paths, threshold_paths = load_ensemble_and_config(model_dir)
    test_files = pd.read_csv(args.test_file_list, names=('graph', 'csv', 'sample_id'), header=None)
    
    all_dfs = []
    for idx, row in test_files.iterrows():
        print(f"Processing sample: {row['sample_id']}...")
        prediction_df = classify_with_ensemble(
            parameters, model_paths, threshold_paths,
            graph_file=row['graph'],
            file_prefix=args.file_prefix,
            sample_id=row['sample_id']
        )
        all_dfs.append(prediction_df)
    
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv(output_path, header=True, index=False, mode='w')
    print(f"\n✅ Classification complete. Results saved to {output_path}")

def main_gfa(args):
    """Classifies a single GFA file."""
    model_dir = os.path.join("runs", args.run_name, "final_model")
    classify_dir = os.path.join("runs", args.run_name, "classify")
    os.makedirs(classify_dir, exist_ok=True)

    base_gfa_name = os.path.basename(args.graph_file)

    if base_gfa_name.endswith(".gfa.gz"):
        sample_id = base_gfa_name[:-7] 
    elif base_gfa_name.endswith(".gfa"):
        sample_id = base_gfa_name[:-4] 
    else:
        sample_id = base_gfa_name.split('.')[0]
    
    if args.output_filename:
        output_filename = args.output_filename
    else:
        output_filename = f"{sample_id}.csv"
        print(f"No output filename provided. Auto-generating: {output_filename}")
    
    output_path = os.path.join(classify_dir, output_filename)
    
    parameters, model_paths, threshold_paths = load_ensemble_and_config(model_dir)
    
    print(f"Processing sample: {sample_id}...")
    prediction_df = classify_with_ensemble(
        parameters, model_paths, threshold_paths,
        graph_file=args.graph_file,
        file_prefix='', 
        sample_id=sample_id 
    )
    
    prediction_df.to_csv(output_path, header=True, index=False, mode='w')
    print(f"\n✅ Classification complete. Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify contigs using a trained plASgraph2 model ensemble.", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_name", required=True, help="Unique name of the experiment run to use for classification.")
    subparsers = parser.add_subparsers(dest='command', required=True)

    p_set = subparsers.add_parser('set', help="Classify a set of GFA files from a list.")
    p_set.add_argument("test_file_list", help="CSV file listing samples to classify (e.g., 'test_files.csv').")
    p_set.add_argument("file_prefix", help="Common path prefix for filenames in the list (e.g., 'data/graphs/').")
    p_set.add_argument("output_filename", help="Name for the output CSV file (e.g., 'results.csv').")
    p_set.set_defaults(func=main_set)

    p_gfa = subparsers.add_parser('gfa', help="Classify a single GFA file.")
    p_gfa.add_argument("graph_file", help="Path to the input GFA or GFA.gz file.")
    
    p_gfa.add_argument(
        "output_filename", 
        help="Name for the output CSV file. If omitted, it will be auto-generated based on the GFA filename (e.g., 'my_sample.csv').",
        nargs='?',  
        default=None 
    )
    
    p_gfa.set_defaults(func=main_gfa)

    args = parser.parse_args()
    args.func(args)