# In scripts/train.py

import argparse
import os
import shutil
import yaml
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd

import subprocess
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import modules from our library
from plasgraph.data import Dataset_Pytorch
from plasgraph.config import config as Config
from plasgraph.engine import train_final_model
from plasgraph.utils import set_all_seeds

from contextlib import contextmanager
from accelerate import Accelerator

def plot_gpu_utilization(csv_file, output_png):
    """Reads the nvidia-smi CSV log and saves a utilization plot."""
    try:
        # Read the CSV, skipping the header row
        df = pd.read_csv(csv_file, header=0)
        
        # Clean up column names (remove spaces)
        df.columns = df.columns.str.strip()
        
        # --- Clean the data ---
        # Convert timestamp to datetime objects
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Remove ' %' and convert to integer
        df['utilization.gpu [%]'] = df['utilization.gpu [%]'].str.replace(' %', '').astype(int)
        
        # --- Create 'Elapsed Time' ---
        # Get the first timestamp
        start_time = df['timestamp'].iloc[0]
        # Calculate time in seconds from the start
        df['Elapsed Time (s)'] = (df['timestamp'] - start_time).dt.total_seconds()

        # --- Generate Plot ---
        plt.figure(figsize=(12, 6))
        plt.plot(df['Elapsed Time (s)'], df['utilization.gpu [%]'], label='GPU Utilization')
        
        # Filtered data for average calculation (ignoring 0%)
        active_util = df[df['utilization.gpu [%]'] > 0]['utilization.gpu [%]']
        avg_util = active_util.mean() if not active_util.empty else 0
        
        # Add average line to the plot
        plt.axhline(y=avg_util, color='r', linestyle='--', label=f'Avg. (when >0%): {avg_util:.2f}%')
        
        plt.title(f'GPU Utilization Over Time\n(File: {os.path.basename(csv_file)})')
        plt.xlabel('Elapsed Time (seconds)')
        plt.ylabel('GPU Utilization (%)')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 105) # Set Y-axis from 0 to 105%
        
        # Save the figure
        plt.savefig(output_png, dpi=150)
        plt.close() # Close the figure to free memory
        
        print(f"    > Plot saved to: {output_png}")

    except Exception as e:
        print(f"    > FAILED to generate plot: {e}")

@contextmanager
def log_gpu_utilization(output_file, accelerator):
    """
    A context manager to run nvidia-smi in the background
    and log utilization to a file.
    """
    process = None
    if accelerator.is_main_process:
        accelerator.print(f"Starting nvidia-smi logging to: {output_file}")
        # Start nvidia-smi in the background
        cmd = f"nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 1 > {output_file}"
        # Using shell=True to handle the file redirection '>'
        process = subprocess.Popen(cmd, shell=True)
        time.sleep(2) # Give it a second to start
    
    try:
        # This 'yield' runs the code inside the 'with' block
        yield
    finally:
        # This code runs after the 'with' block exits
        if process and accelerator.is_main_process:
            accelerator.print(f"Stopping nvidia-smi (PID: {process.pid})...")
            # Stop the background process
            process.terminate()
            # Wait for it to fully stop
            process.wait()
            
            # --- 3. ADD PLOTTING CALL ---
            # Now that the CSV is closed, generate the plot
            plot_png = os.path.splitext(output_file)[0] + ".png"
            plot_gpu_utilization(output_file, plot_png)


def main():
    """
    Main function to train a final k-fold ensemble model.

    This script orchestrates the final model training process after
    hyperparameter optimization has been completed. It performs the following steps:
    1. Loads a base configuration file and updates it with the best
       hyperparameters found during a previous optimization stage.

    3. Loads and processes the training data into a graph format.
    4. Sets up k-fold cross-validation splits from the labeled training data.
    5. Calls the `train_final_model` engine, which iteratively trains a model for
       each fold and saves it to disk.
    6. Saves the final, consolidated configuration file for later use in
       evaluation and prediction.
    """

    parser = argparse.ArgumentParser(description="Train a final plASgraph2 model.")
    parser.add_argument("config_file", help="Base YAML configuration file")
    parser.add_argument("--run_name", required=True, help="Unique name for the experiment run to train.")
    parser.add_argument("train_file_list", help="CSV file listing training samples")
    parser.add_argument("file_prefix", help="Common prefix for all filenames")
    args = parser.parse_args()

    parameters = Config(args.config_file)
    set_all_seeds(parameters['random_seed'])

    data_cache_dir = os.path.join("processed_data", args.run_name, "train")

    run_dir = os.path.join("runs", args.run_name)
    best_params_file = os.path.join(run_dir, "hpo_study", "best_arch_params.yaml")
    model_output_dir = os.path.join(run_dir, "final_model")


    with open(best_params_file, 'r') as f:
        best_params = yaml.safe_load(f)
    parameters._params.update(best_params)

    if isinstance(parameters['features'], str):
        parameters._params['features'] = tuple(parameters['features'].split(','))



    # directory for logs related to the final model training
    log_dir = os.path.join(model_output_dir, "final_training_logs")
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    gpu_metrics_dir = os.path.join(model_output_dir, "gpu_metrics")
    os.makedirs(gpu_metrics_dir, exist_ok=True)

    accelerator = Accelerator()

    if accelerator.is_main_process:
        print("----------------------------------------")
        print(f"RUN_NAME: {args.run_name}")
        print(f"Dataset: {os.path.basename(args.train_file_list)}")
        print("----------------------------------------")

    data_util_log = os.path.join(gpu_metrics_dir, "gpu_util_data_processing.csv")
    
    with log_gpu_utilization(data_util_log, accelerator):
        all_graphs = Dataset_Pytorch(
            root=data_cache_dir,
            file_prefix=args.file_prefix,
            train_file_list=args.train_file_list,
            parameters=parameters,
            accelerator=accelerator 
        )

    # extract the graph data object, the NetworkX graph, and the node list
    data = all_graphs[0]
    G = all_graphs.G
    node_list = all_graphs.node_list

    print(f"✅ Using model architecture: {parameters['model_type']}")
    
    
    all_sample_ids = np.array(sorted(list(set(G.nodes[node_id]["sample"] for node_id in node_list))))
    print(f"✅ Found {len(all_sample_ids)} unique samples for splitting.")

    # Get indices of all nodes that have a label
    labeled_indices = np.array([i for i, node_id in enumerate(node_list) if G.nodes[node_id]["text_label"] != "unlabeled"])

    splits = None # Initialize splits

    if parameters['validation_split_mode'] == 'stratified':
        # --- START OF EXISTING LOGIC (UNCHANGED, JUST INDENTED) ---
        print("🔬 Analyzing sample characteristics for stratified splitting...")
        sample_plasmid_ratios = []
        for sample_id in all_sample_ids:
            sample_nodes = [nid for nid in node_list if G.nodes[nid]["sample"] == sample_id]
            
            # Count plasmid vs. total labeled nodes for this sample
            plasmid_count = 0
            labeled_count = 0
            for nid in sample_nodes:
                # Check if the node has a definitive label
                if G.nodes[nid]["text_label"] != "unlabeled":
                    labeled_count += 1
                    # Check if the label is 'plasmid' ([1, 0])
                    if G.nodes[nid]["plasmid_label"] == 1 and G.nodes[nid]["chrom_label"] == 0:
                        plasmid_count += 1
            
            # Calculate the ratio
            ratio = plasmid_count / labeled_count if labeled_count > 0 else 0.0
            sample_plasmid_ratios.append(ratio)

        # Discretize ratios into bins for stratification
        try:
            # Use quantiles to create balanced bins
            stratification_y = pd.qcut(sample_plasmid_ratios, q=5, labels=False, duplicates='drop')
        except ValueError:
            # Fallback to simple binning if quantiles fail (e.g., too few unique values)
            stratification_y = pd.cut(sample_plasmid_ratios, bins=5, labels=False, duplicates='drop')
        print("  > Stratification groups created based on plasmid ratios.")


        # --- This if/else block was modified ---
        if parameters['training_mode'] == 'k-fold':
            print(f"✅ Setting up stratified {parameters['k_folds']}-fold cross-validation based on samples.")
            skf = StratifiedKFold(n_splits=parameters["k_folds"], shuffle=True, random_state=parameters["random_seed"])
            # Create splits based on the indices of the sample IDs array, stratified by y
            splits = list(skf.split(all_sample_ids, stratification_y))
        else: # single_fold
            print("✅ Setting up stratified single-fold training with an 80/20 train/validation split.")
            # Create a single 80/20 stratified split of sample IDs
            train_s_idx, val_s_idx = train_test_split(
                np.arange(len(all_sample_ids)), # Split the indices of the sample_ids array
                test_size=0.2,
                random_state=parameters["random_seed"],
                shuffle=True,
                stratify=stratification_y # Use the generated strata
            )
            # The train_final_model function expects a list of splits
            splits = [(train_s_idx, val_s_idx)]
        # --- END OF EXISTING LOGIC ---

    elif parameters['validation_split_mode'] == 'node_level_random':
        # --- START OF NEW LOGIC ---
        print("🔬 Using node-level random 80/20 split (Original TF-style).")
        if parameters['training_mode'] == 'k-fold':
            raise ValueError(f"validation_split_mode 'node_level_random' is not compatible with training_mode 'k-fold'.")
        # Pass [None] to signal to the engine to run its loop once and generate the node-level split internally.
        splits = [None]
        # --- END OF NEW LOGIC ---
    
    else:
        raise ValueError(f"Unknown validation_split_mode: {parameters['validation_split_mode']}")


    train_util_log = os.path.join(gpu_metrics_dir, "gpu_util_training.csv")
    
    accelerator.print("\n--- 🚀 Starting Final Model Training ---")
    with log_gpu_utilization(train_util_log, accelerator):
        train_final_model(
            accelerator, parameters, data, splits, all_sample_ids, labeled_indices, log_dir, G, node_list
        )
    accelerator.print("--- ✅ Finished Final Model Training ---")


    # save the final base configuration file for future predictions
    base_params_path = os.path.join(model_output_dir, "base_model_config.yaml")
    parameters.write_yaml(base_params_path)

    ensemble_dir = os.path.join(log_dir, "ensemble_models")
    print(f"\n✅ Training complete. Ensemble models and their thresholds saved in: {ensemble_dir}")
    print(f"✅ Base config saved to: {base_params_path}")


if __name__ == "__main__":
    main()