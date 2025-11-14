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

from accelerate.utils import InitProcessGroupKwargs  
from datetime import timedelta

from plasgraph.data import Dataset_Pytorch
from plasgraph.config import config as Config
from plasgraph.engine import train_final_model
from plasgraph.utils import set_all_seeds

from contextlib import contextmanager
from accelerate import Accelerator

def plot_gpu_utilization(csv_file, output_png):
    """Reads the nvidia-smi CSV log and saves a utilization plot."""
    try:
        df = pd.read_csv(csv_file, header=0)
        df.columns = df.columns.str.strip()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['utilization.gpu [%]'] = df['utilization.gpu [%]'].str.replace(' %', '').astype(int)
        start_time = df['timestamp'].iloc[0]
        df['Elapsed Time (s)'] = (df['timestamp'] - start_time).dt.total_seconds()

        plt.figure(figsize=(12, 6))
        plt.plot(df['Elapsed Time (s)'], df['utilization.gpu [%]'], label='GPU Utilization')
        
        active_util = df[df['utilization.gpu [%]'] > 0]['utilization.gpu [%]']
        avg_util = active_util.mean() if not active_util.empty else 0
        
        plt.axhline(y=avg_util, color='r', linestyle='--', label=f'Avg. (when >0%): {avg_util:.2f}%')
        
        plt.title(f'GPU Utilization Over Time')
        plt.xlabel('Elapsed Time (seconds)')
        plt.ylabel('GPU Utilization (%)')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 105) 
        
        plt.savefig(output_png, dpi=150)
        plt.close()
        
        print(f"    > Plot saved to: {output_png}")

    except Exception as e:
        print(f"    > FAILED to generate plot: {e}")


@contextmanager
def log_gpu_utilization(output_file, accelerator):
    """
    context manager that runs nvidia-smi in the background to log gpu usage 
    to a csv file during the execution of a code block
    """
    process = None
    if accelerator.is_main_process:
        accelerator.print(f"Starting nvidia-smi logging to: {output_file}")
        # start nvidia-smi in the background, querying timestamp, utilization and memory
        cmd = f"nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 1 > {output_file}"
        # using shell=True to handle the file redirection '>'
        process = subprocess.Popen(cmd, shell=True)
        time.sleep(2) # give it a second to start recording
    
    try:
        # runs the code inside the 'with' block
        yield
    finally:
        # runs after the 'with' block exits
        if process and accelerator.is_main_process:
            accelerator.print(f"Stopping nvidia-smi (PID: {process.pid})...")
            # stop the background process
            process.terminate()
            process.wait()
            
            # generate the visualization plot
            plot_png = os.path.splitext(output_file)[0] + ".png"
            plot_gpu_utilization(output_file, plot_png)


def main():
    """
    main function to train a final k-fold ensemble model.

    orchestrates the entire training pipeline: loading config, setting up the 
    accelerator for multi-gpu support, preparing data, determining stratified 
    splits, and invoking the training engine.
    """

    parser = argparse.ArgumentParser(description="Train a final plASgraph2 model.")
    parser.add_argument("config_file", help="Base YAML configuration file")
    parser.add_argument("--run_name", required=True, help="Unique name for the experiment run to train.")
    parser.add_argument("train_file_list", help="CSV file listing training samples")
    parser.add_argument("file_prefix", help="Common prefix for all filenames")
    args = parser.parse_args()

    # load base configuration and set seeds for reproducibility
    parameters = Config(args.config_file)
    set_all_seeds(parameters['random_seed'])

    # define paths for caching processed graph data
    data_cache_dir = os.path.join("processed_data", args.run_name, "train")

    # define paths for loading best hyperparameters from the optimization step
    run_dir = os.path.join("runs", args.run_name)
    best_params_file = os.path.join(run_dir, "hpo_study", "best_arch_params.yaml")
    model_output_dir = os.path.join(run_dir, "final_model")

    with open(best_params_file, 'r') as f:
        best_params = yaml.safe_load(f)
    parameters._params.update(best_params)

    if isinstance(parameters['features'], str):
        parameters._params['features'] = tuple(parameters['features'].split(','))

    log_dir = os.path.join(model_output_dir, "final_training_logs")
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    gpu_metrics_dir = os.path.join(model_output_dir, "gpu_metrics")
    os.makedirs(gpu_metrics_dir, exist_ok=True)

    # timeout for distributed process groups (prevent timeouts during long data loading)
    ipg_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=60))

    # initialize the accelerator to handle distributed training details automatically
    accelerator = Accelerator(kwargs_handlers=[ipg_kwargs])

    if accelerator.is_main_process:
        print("----------------------------------------")
        print(f"RUN_NAME: {args.run_name}")
        print(f"Dataset: {os.path.basename(args.train_file_list)}")
        print("----------------------------------------")

    data_util_log = os.path.join(gpu_metrics_dir, "gpu_util_data_processing.csv")
    
    # load and process the graph dataset inside the gpu logging context
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
    
    # identify unique samples in the dataset for sample-level operations
    all_sample_ids = np.array(sorted(list(set(G.nodes[node_id]["sample"] for node_id in node_list))))
    print(f"✅ Found {len(all_sample_ids)} unique samples for splitting.")

    # get indices of all nodes that have a valid label (excluding 'unlabeled')
    labeled_indices = np.array([i for i, node_id in enumerate(node_list) if G.nodes[node_id]["text_label"] != "unlabeled"])

    splits = None # initialize splits

    if parameters['validation_split_mode'] == 'stratified':
        print("🔬 Analyzing sample characteristics for stratified splitting...")
        sample_plasmid_ratios = []
        # calculate the ratio of plasmid nodes per sample to ensure balanced folds
        for sample_id in all_sample_ids:
            sample_nodes = [nid for nid in node_list if G.nodes[nid]["sample"] == sample_id]
            plasmid_count = 0
            labeled_count = 0
            for nid in sample_nodes:
                if G.nodes[nid]["text_label"] != "unlabeled":
                    labeled_count += 1
                    if G.nodes[nid]["plasmid_label"] == 1 and G.nodes[nid]["chrom_label"] == 0:
                        plasmid_count += 1
            ratio = plasmid_count / labeled_count if labeled_count > 0 else 0.0
            sample_plasmid_ratios.append(ratio)

        # discretize ratios into bins for stratification
        try:
            # use quantiles to create balanced bins
            stratification_y = pd.qcut(sample_plasmid_ratios, q=5, labels=False, duplicates='drop')
        except ValueError:
            stratification_y = pd.cut(sample_plasmid_ratios, bins=5, labels=False, duplicates='drop')
        print("  > Stratification groups created based on plasmid ratios.")


        # setup k-fold or single-fold based on the configuration
        if parameters['training_mode'] == 'k-fold':
            print(f"✅ Setting up stratified {parameters['k_folds']}-fold cross-validation based on samples.")
            skf = StratifiedKFold(n_splits=parameters["k_folds"], shuffle=True, random_state=parameters["random_seed"])
            # create splits based on the indices of the sample ids array, stratified by the calculated bins
            splits = list(skf.split(all_sample_ids, stratification_y))
        else: # single_fold
            print("✅ Setting up stratified single-fold training with an 80/20 train/validation split.")
            # create a single 80/20 stratified split of sample ids
            train_s_idx, val_s_idx = train_test_split(
                np.arange(len(all_sample_ids)), # split the indices of the sample_ids array
                test_size=0.2,
                random_state=parameters["random_seed"],
                shuffle=True,
                stratify=stratification_y # use the generated strata
            )
            splits = [(train_s_idx, val_s_idx)]

    elif parameters['validation_split_mode'] == 'node_level_random':
        print("🔬 Using node-level random 80/20 split (Original TF-style).")
        if parameters['training_mode'] == 'k-fold':
            raise ValueError(f"validation_split_mode 'node_level_random' is not compatible with training_mode 'k-fold'.")
        splits = [None]
    
    else:
        raise ValueError(f"Unknown validation_split_mode: {parameters['validation_split_mode']}")


    train_util_log = os.path.join(gpu_metrics_dir, "gpu_util_training.csv")
    
    accelerator.print("\n--- 🚀 Starting Final Model Training ---")
    # execute the final training loop, which handles k-fold ensemble creation
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