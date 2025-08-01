import argparse
import os
import pandas as pd
import torch
import glob
import yaml
import numpy as np

from plasgraph.config import config as Config
from plasgraph.models import GCNModel, GGNNModel
from plasgraph.engine import classify_graph
from plasgraph.utils import pair_to_label, apply_thresholds

def load_ensemble_and_config(model_dir):
    """
    Loads the base configuration and finds all model and threshold files
    for the k-fold ensemble.
    """
    config_path = os.path.join(model_dir, "base_model_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Base config file not found at {config_path}")
    parameters = Config(config_path)

    ensemble_dir = os.path.join(model_dir, "final_training_logs", "ensemble_models")
    model_paths = sorted(glob.glob(os.path.join(ensemble_dir, "fold_*_model.pt")))
    threshold_paths = sorted(glob.glob(os.path.join(ensemble_dir, "fold_*_thresholds.yaml")))

    if not model_paths or len(model_paths) != len(threshold_paths):
        raise FileNotFoundError(f"Mismatch or missing model/threshold files in {ensemble_dir}")
    
    print(f"Found {len(model_paths)} models for ensembling.")
    return parameters, model_paths, threshold_paths

def classify_with_ensemble(parameters, model_paths, threshold_paths, graph_file, file_prefix, sample_id):
    """
    Classifies a single graph using the full model ensemble, averaging their predictions.
    """
    if torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")

    all_raw_probs = []
    
    # This loop is inspired by the evaluation script's ensemble logic
    for model_path, thresh_path in zip(model_paths, threshold_paths):
        temp_params = Config(parameters.config_file_path)
        temp_params._params.update(parameters._params)
        with open(thresh_path, 'r') as f:
            thresholds = yaml.safe_load(f)
        temp_params['plasmid_threshold'] = thresholds['plasmid_threshold']
        temp_params['chromosome_threshold'] = thresholds['chromosome_threshold']

        if temp_params['model_type'] == 'GCNModel':
            model = GCNModel(temp_params).to(device)
        else: # GGNNModel
            model = GGNNModel(temp_params).to(device)
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        from plasgraph.data import read_single_graph
        import torch_geometric

        G = read_single_graph(file_prefix, graph_file, sample_id, temp_params['minimum_contig_length'])
        node_list = list(G)

        for u, v in G.edges():
            kmer_u = np.array(G.nodes[u]["kmer_counts_norm"])
            kmer_v = np.array(G.nodes[v]["kmer_counts_norm"])
            dot_product = np.dot(kmer_u, kmer_v)
            G.edges[u, v]["kmer_dot_product"] = dot_product
        
        features = temp_params["features"]
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
        
        model.eval()
        with torch.no_grad():
            logits = model(data_obj)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_raw_probs.append(probs)

    ensemble_probs = np.mean(all_raw_probs, axis=0)
    final_scores = apply_thresholds(ensemble_probs, parameters)

    output_rows = []
    for i, node_id in enumerate(node_list):
        plasmid_score = final_scores[i, 0]
        chrom_score = final_scores[i, 1]
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
    """Classifies a set of GFA files listed in a CSV."""
    model_dir = os.path.join("runs", args.run_name, "final_model")
    classify_dir = os.path.join("runs", args.run_name, "classify")
    os.makedirs(classify_dir, exist_ok=True)
    output_path = os.path.join(classify_dir, args.output_filename)

    parameters, model_paths, threshold_paths = load_ensemble_and_config(model_dir)
    test_files = pd.read_csv(args.test_file_list, names=('graph', 'csv', 'sample_id'), header=None)
    
    for idx, row in test_files.iterrows():
        print(f"Processing sample: {row['sample_id']}...")
        prediction_df = classify_with_ensemble(
            parameters, model_paths, threshold_paths,
            graph_file=row['graph'],
            file_prefix=args.file_prefix,
            sample_id=row['sample_id']
        )
        
        if idx == 0:
            prediction_df.to_csv(output_path, header=True, index=False, mode='w')
        else:
            prediction_df.to_csv(output_path, header=False, index=False, mode='a')
    print(f"\n✅ Classification complete. Results saved to {output_path}")

def main_gfa(args):
    """Classifies a single GFA file."""
    model_dir = os.path.join("runs", args.run_name, "final_model")
    classify_dir = os.path.join("runs", args.run_name, "classify")
    os.makedirs(classify_dir, exist_ok=True)
    output_path = os.path.join(classify_dir, args.output_filename)

    parameters, model_paths, threshold_paths = load_ensemble_and_config(model_dir)
    sample_id = os.path.splitext(os.path.basename(args.graph_file))[0].replace('.gfa', '')
    
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
    p_set.add_argument("test_file_list", help="CSV file listing samples to classify")
    p_set.add_argument("file_prefix", help="Common path prefix for filenames in the list")
    p_set.add_argument("output_filename", help="Name for the output CSV file (e.g., 'results.csv')")
    p_set.set_defaults(func=main_set)

    p_gfa = subparsers.add_parser('gfa', help="Classify a single GFA file.")
    p_gfa.add_argument("graph_file", help="Input GFA or GFA.gz file")
    p_gfa.add_argument("output_filename", help="Name for the output CSV file (e.g., 'sample_x_results.csv')")
    p_gfa.set_defaults(func=main_gfa)

    args = parser.parse_args()
    args.func(args)