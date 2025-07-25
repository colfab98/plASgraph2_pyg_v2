import argparse
import os
import yaml
import numpy as np
import optuna
import torch
from sklearn.model_selection import KFold

from plasgraph.data import Dataset_Pytorch
from plasgraph.config import config as Config
from plasgraph.engine import objective

import optuna.visualization as vis
import matplotlib.pyplot as plt
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate
)

from accelerate import Accelerator

def main():
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter search for plASgraph2.")
    parser.add_argument("config_file", help="YAML configuration file")
    parser.add_argument("train_file_list", help="CSV file listing training samples")
    parser.add_argument("file_prefix", help="Common prefix for all filenames")
    parser.add_argument("model_output_dir", help="Output folder for Optuna study and results")
    parser.add_argument("--data_cache_dir", required=True, help="Directory to store/load the processed graph data.")
    args = parser.parse_args()

    # automatically detects hardware setup (single GPU, multiple GPUs, TPUs, etc.) and prepares 
    # PyTorch training components (model, optimizer, data loaders) to run seamlessly in that environment
    accelerator = Accelerator()

    # creates the main configuration object for the script by reading a user-provided YAML file (plasgraph_config.yaml)
    parameters = Config(args.config_file)
    # attach the path of the loaded configuration file to the parameters object itself
    parameters.config_file_path = args.config_file

    # instantiates the custom Dataset_Pytorch class from data.py
    # uses a caching mechanism: if a pre-processed graph exists in 'root' it loads it
    all_graphs = Dataset_Pytorch(
        root=args.data_cache_dir,               # directory for caching the processed dataset
        file_prefix=args.file_prefix,           # path prefix for raw data files
        train_file_list=args.train_file_list,   # CSV listing the graphs to load
        parameters=parameters,                  # main configuration object
        accelerator=accelerator                 # accelerator for distributed processing
    )

    # get the first (and only) item, which is the combined Data object
    data = all_graphs[0]
    # underlying NetworkX graph object
    G = all_graphs.G
    # ordered list of node IDs
    node_list = all_graphs.node_list

    # update the configuration with the actual number of node features from the processed data
    parameters['n_input_features'] = data.num_node_features

    # output directory
    os.makedirs(args.model_output_dir, exist_ok=True)

    # --- setup for cross-validation ---
    all_sample_ids = np.array(sorted(list(set(G.nodes[node_id]["sample"] for node_id in node_list))))
    accelerator.print(f"✅ Found {len(all_sample_ids)} unique samples for splitting.")

    # Get indices of all nodes that have a label
    labeled_indices = np.array([i for i, node_id in enumerate(node_list) if G.nodes[node_id]["text_label"] != "unlabeled"])

    # Perform the K-Fold split on the sample IDs
    kf = KFold(n_splits=parameters["k_folds"], shuffle=True, random_state=parameters["random_seed"])
    splits = list(kf.split(all_sample_ids))


    accelerator.print(f"Using device: {accelerator.device}")

    # --- run optuna study ---
    # pruner to stop unpromising trials early
    pruner = optuna.pruners.MedianPruner(n_startup_trials=parameters['n_startup_trials'])
    study_name = "plasgraph-hpo-study"
    # storage location for the study database (using SQLite)
    storage_name = f"sqlite:///{os.path.join(args.model_output_dir, 'hpo_study.db')}"
    # configure the sampler, which suggests hyperparameter values
    sampler = optuna.samplers.TPESampler(multivariate=True)

    # check if this is the main process (in a multi-GPU setup, process_index 0)
    if accelerator.is_main_process:
        accelerator.print(f"Main process creating/loading Optuna study '{study_name}'...")
        # the main process creates the study and the corresponding database file
        optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="maximize",   # maximize the objective (AUROC)
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True  # allows resuming the study if the database file already exists
        )

    # synchronize all processes to ensure the main process has finished creating the study database
    accelerator.wait_for_everyone()

    # all processes can safely load the study from the shared database file
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_name
    )
    accelerator.print(f"Process {accelerator.process_index} loaded study with {len(study.trials)} trials.")
    total_trials = parameters["optuna_n_trials"]
    n_trials_per_process = total_trials // accelerator.num_processes
    
    # if the total number of trials is not perfectly divisible by the number of processes,
    # assign the remaining trials to the main process
    if accelerator.is_main_process:
        n_trials_per_process += total_trials % accelerator.num_processes

    accelerator.print(f"Process {accelerator.process_index} will run {n_trials_per_process} trials.")

    # start the optimization process
    study.optimize(
        lambda trial: objective(trial, accelerator, parameters, data, splits, all_sample_ids, labeled_indices, node_list, G),
        n_trials=n_trials_per_process,
        show_progress_bar=accelerator.is_main_process
    )

    # save results (only on the main process) ---
    if accelerator.is_main_process:
        accelerator.print("\nOptuna study finished.")
        # retrieve the best trial from the study
        best_trial = study.best_trial
        accelerator.print(f"  Value (Best Avg AUROC): {best_trial.value:.4f}")

        best_arch_params = parameters._params.copy()
        # update the copy with the hyperparameters from the best trial
        best_arch_params.update(study.best_params)

        # handle the 'features' parameter, which is a tuple, to be saved as a comma-separated string in YAML
        if 'features' in best_arch_params and isinstance(best_arch_params['features'], tuple):
            best_arch_params['features'] = ",".join(best_arch_params['features'])

        best_params_path = os.path.join(args.model_output_dir, "best_arch_params.yaml")
        # write the best hyperparameters to the YAML file
        with open(best_params_path, 'w') as f:
            yaml.dump(best_arch_params, f, sort_keys=False)
        accelerator.print(f"\nBest architecture parameters saved to: {best_params_path}")

        # --- generate and save visualizations ---
        print("Generating Optuna visualizations (matplotlib backend)...")
        optuna_viz_dir = os.path.join(args.model_output_dir, "optuna_visualizations")
        os.makedirs(optuna_viz_dir, exist_ok=True)

        try:
            # optimization history
            ax = plot_optimization_history(study)
            fig = ax.get_figure()
            fig.set_size_inches(8, 6)
            fig.tight_layout()
            fig.savefig(os.path.join(optuna_viz_dir, "optimization_history.png"))
            plt.close(fig)

            # parameter importances
            ax = plot_param_importances(study)
            fig = ax.get_figure()
            fig.set_size_inches(8, 6)
            fig.tight_layout()
            fig.savefig(os.path.join(optuna_viz_dir, "param_importances.png"))
            plt.close(fig)

            # parallel coordinates
            ax = plot_parallel_coordinate(study)
            fig = ax.get_figure()
            ax.grid(False)  
            fig.set_size_inches(10, 6)
            fig.tight_layout()
            fig.savefig(os.path.join(optuna_viz_dir, "parallel_coordinate.png"))
            plt.close(fig)

            print(f"Visualizations saved to: {optuna_viz_dir}")

        except Exception as e:
            print(f"Could not generate all Optuna visualizations: {e}")


if __name__ == "__main__":
    main()