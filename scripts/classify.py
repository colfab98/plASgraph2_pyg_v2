# In scripts/classify.py

import argparse
import os
import pandas as pd
import torch

# Import from our library
from plasgraph.config import config as Config
from other_code.models_no_bn import GCNModel, GGNNModel
from plasgraph.engine import classify_graph # <-- The new engine function does all the work

def main_set(args):
    """Classifies a set of GFA files listed in a CSV."""
    parameters, model = load_model_and_config(args.model_dir)
    test_files = pd.read_csv(args.test_file_list, names=('graph', 'csv', 'sample_id'), header=None)
    
    for idx, row in test_files.iterrows():
        print(f"Processing sample: {row['sample_id']}...")
        prediction_df = classify_graph(
            model=model,
            parameters=parameters,
            graph_file=row['graph'],
            file_prefix=args.file_prefix,
            sample_id=row['sample_id']
        )
        
        # Append results to the output file
        if idx == 0:
            prediction_df.to_csv(args.output_file, header=True, index=False, mode='w')
        else:
            prediction_df.to_csv(args.output_file, header=False, index=False, mode='a')
    print(f"\n✅ Classification complete. Results saved to {args.output_file}")

def main_gfa(args):
    """Classifies a single GFA file."""
    parameters, model = load_model_and_config(args.model_dir)
    sample_id = os.path.basename(args.graph_file).split('.')[0]
    
    print(f"Processing sample: {sample_id}...")
    prediction_df = classify_graph(
        model=model,
        parameters=parameters,
        graph_file=args.graph_file,
        file_prefix='', # No prefix for a single file path
        sample_id=sample_id
    )
    
    prediction_df.to_csv(args.output_file, header=True, index=False, mode='w')
    print(f"\n✅ Classification complete. Results saved to {args.output_file}")

def load_model_and_config(model_dir):
    """Helper function to load a trained model and its config."""
    config_path = os.path.join(model_dir, "final_model_config_with_thresholds.yaml")
    parameters = Config(config_path)
    
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")

    if parameters['model_type'] == 'GCNModel':
        model = GCNModel(parameters).to(device)
    elif parameters['model_type'] == 'GGNNModel':
        model = GGNNModel(parameters).to(device)
    
    weights_path = os.path.join(model_dir, "final_model.pt")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return parameters, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify contigs using a trained plASgraph2 model.", formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Sub-command for processing a set of files
    p_set = subparsers.add_parser('set', help="Classify a set of GFA files from a list.")
    p_set.add_argument("model_dir", help="Directory with the trained model and config")
    p_set.add_argument("test_file_list", help="CSV file listing samples to classify")
    p_set.add_argument("file_prefix", help="Common path prefix for filenames in the list")
    p_set.add_argument("output_file", help="Output CSV file for classification results")
    p_set.set_defaults(func=main_set)

    # Sub-command for processing a single gfa file
    p_gfa = subparsers.add_parser('gfa', help="Classify a single GFA file.")
    p_gfa.add_argument("model_dir", help="Directory with the trained model and config")
    p_gfa.add_argument("graph_file", help="Input GFA or GFA.gz file")
    p_gfa.add_argument("output_file", help="Output CSV file for classification results")
    p_gfa.set_defaults(func=main_gfa)

    args = parser.parse_args()
    args.func(args)