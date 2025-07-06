# In scripts/tune_hpo.py

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

    accelerator = Accelerator()

    # 1. Load Config and Data
    parameters = Config(args.config_file)
    parameters.config_file_path = args.config_file

    all_graphs = Dataset_Pytorch(
        root=args.data_cache_dir,
        file_prefix=args.file_prefix,
        train_file_list=args.train_file_list,
        parameters=parameters,
        accelerator=accelerator
    )
    data = all_graphs[0]
    G = all_graphs.G
    node_list = all_graphs.node_list

    parameters['n_input_features'] = data.num_node_features

    os.makedirs(args.model_output_dir, exist_ok=True)

    # 2. Setup for Cross-Validation
    labeled_indices = np.array([i for i, node_id in enumerate(node_list) if G.nodes[node_id]["text_label"] != "unlabeled"])
    kf = KFold(n_splits=parameters["k_folds"], shuffle=True, random_state=parameters["random_seed"])
    splits = list(kf.split(labeled_indices))

    # 3. Setup Device
    accelerator.print(f"Using device: {accelerator.device}")


    # 4. Run Optuna Study
    pruner = optuna.pruners.MedianPruner(n_startup_trials=parameters['n_startup_trials'])
    study_name = "plasgraph-hpo-study"
    storage_name = f"sqlite:///{os.path.join(args.model_output_dir, 'hpo_study.db')}"
    sampler = optuna.samplers.TPESampler(multivariate=True)


    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )
    study.optimize(
        lambda trial: objective(trial, accelerator, parameters, data, splits, labeled_indices),
        n_trials=parameters["optuna_n_trials"],
        show_progress_bar=True
    )

    # 5. Save Results (only on the main process)
    if accelerator.is_main_process:
        accelerator.print("\nOptuna study finished.")
        best_trial = study.best_trial
        accelerator.print(f"  Value (Best Avg AUROC): {best_trial.value:.4f}")

        best_arch_params = parameters._params.copy()
        best_arch_params.update(study.best_params)

        if 'features' in best_arch_params and isinstance(best_arch_params['features'], tuple):
            best_arch_params['features'] = ",".join(best_arch_params['features'])

        best_params_path = os.path.join(args.model_output_dir, "best_arch_params.yaml")
        with open(best_params_path, 'w') as f:
            yaml.dump(best_arch_params, f, sort_keys=False)
        accelerator.print(f"\n✅ Best architecture parameters saved to: {best_params_path}")


        # 6. Generate and save visualizations
        print("Generating Optuna visualizations (matplotlib backend)...")
        optuna_viz_dir = os.path.join(args.model_output_dir, "optuna_visualizations")
        os.makedirs(optuna_viz_dir, exist_ok=True)

        try:
            # Optimization history
            ax = plot_optimization_history(study)
            fig = ax.get_figure()
            fig.set_size_inches(8, 6)
            fig.tight_layout()
            fig.savefig(os.path.join(optuna_viz_dir, "optimization_history.png"))
            plt.close(fig)


            # Parameter importances
            ax = plot_param_importances(study)
            fig = ax.get_figure()
            fig.set_size_inches(8, 6)
            fig.tight_layout()
            fig.savefig(os.path.join(optuna_viz_dir, "param_importances.png"))
            plt.close(fig)

            # Parallel coordinates
            ax = plot_parallel_coordinate(study)
            fig = ax.get_figure()
            ax.grid(False)  # <--- turn off the grid
            fig.set_size_inches(10, 6)
            fig.tight_layout()
            fig.savefig(os.path.join(optuna_viz_dir, "parallel_coordinate.png"))
            plt.close(fig)

            print(f"✅ Visualizations saved to: {optuna_viz_dir}")

        except Exception as e:
            print(f"⚠️ Could not generate all Optuna visualizations: {e}")



if __name__ == "__main__":
    main()