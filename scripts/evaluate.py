import argparse
import os
import torch
import glob
import pandas as pd
import yaml
import numpy as np

from plasgraph.data import Dataset_Pytorch
from plasgraph.config import config as Config
from plasgraph.models import GCNModel, GGNNModel
from plasgraph.metrics import calculate_and_print_metrics
from plasgraph.utils import plot_f1_violin, apply_thresholds

from accelerate import Accelerator


def main():
    """
    Main function to evaluate a trained k-fold model ensemble on a test dataset.

    This script performs the following steps:
    1. Initializes an Accelerator for handling hardware (CPU/GPU) and distributed setup.
    2. Loads the base model configuration and the test data, using the Accelerator
       to parallelize data processing if multiple GPUs are available.
    3. On the main process, it loads each model from the k-fold ensemble.
    4. For each model, it loads its corresponding fold-specific classification thresholds.
    5. It runs inference with each model on the test set, generating both raw probabilities
       and final scores (scaled by the fold-specific thresholds).
    6. It averages the predictions from all models in the ensemble.
    7. It calculates and prints aggregate performance metrics (F1, AUROC, etc.) and
       generates a violin plot of the F1 scores.
    """

    parser = argparse.ArgumentParser(description="Evaluate a trained plASgraph2 model ensemble.")
    parser.add_argument("--run_name", required=True, help="Unique name of the experiment run to evaluate.")
    parser.add_argument("test_file_list", help="CSV file listing test samples")
    parser.add_argument("file_prefix", help="Common prefix for all filenames")
    args = parser.parse_args()

    run_dir = os.path.join("runs", args.run_name)
    model_dir = os.path.join(run_dir, "final_model")
    log_dir = os.path.join(run_dir, "evaluation_results")
    os.makedirs(log_dir, exist_ok=True)

    accelerator = Accelerator()

    if accelerator.is_main_process:
        print("----------------------------------------")
        print(f"RUN_NAME: {args.run_name}")
        print(f"Evaluation Dataset: {os.path.basename(args.test_file_list)}")
        print("----------------------------------------")


    config_path = os.path.join(model_dir, "base_model_config.yaml")
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return
    parameters = Config(config_path)

    test_data_root = os.path.join("processed_data", args.run_name, "test")
    all_test_graphs = Dataset_Pytorch(
        root=test_data_root,
        file_prefix=args.file_prefix,
        train_file_list=args.test_file_list,
        parameters=parameters,
        accelerator=accelerator  # pass the accelerator to enable distributed data processing
    )

    accelerator.wait_for_everyone()

    # restrict evaluation to the main process.
    if accelerator.is_main_process:
        device = accelerator.device
        accelerator.print(f"Main process using device: {device}")

        # Locate all saved model files for the ensemble
        ensemble_dir = os.path.join(model_dir, "final_training_logs", "ensemble_models")
        model_paths = sorted(glob.glob(os.path.join(ensemble_dir, "fold_*_model.pt")))
        threshold_paths = sorted(glob.glob(os.path.join(ensemble_dir, "fold_*_thresholds.yaml")))

        if not model_paths or len(model_paths) != len(threshold_paths):
            accelerator.print(f"Error: Mismatch between the {len(model_paths)} models and {len(threshold_paths)} threshold files found in {ensemble_dir}")
            return
        accelerator.print(f"Found {len(model_paths)} models and threshold files for the ensemble evaluation.")

        # load the processed test data onto the main device
        data_test = all_test_graphs[0].to(device)
        G_test = all_test_graphs.G
        node_list_test = all_test_graphs.node_list

        masks_test_values = []
        for node_id in node_list_test:
            label = G_test.nodes[node_id]["text_label"]
            if label == "unlabeled": masks_test_values.append(0.0)      # ignore unlabeled nodes
            elif label == "chromosome": masks_test_values.append(1.0)
            else: masks_test_values.append(float(parameters["plasmid_ambiguous_weight"]))
        masks_test = torch.tensor(masks_test_values, dtype=torch.float32, device=device)

        all_final_scores = []
        all_raw_probs = []
        temp_params = Config(config_path)

        with torch.no_grad():
            for model_path, thresh_path in zip(model_paths, threshold_paths):
                if parameters['model_type'] == 'GCNModel':
                    model = GCNModel(parameters).to(device)
                else: # GGNNModel
                    model = GGNNModel(parameters).to(device)

                # load the saved weights for the current fold's model
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()

                # load the optimal thresholds specifically determined for this model during training
                with open(thresh_path, 'r') as f:
                    thresholds = yaml.safe_load(f)
                temp_params['plasmid_threshold'] = thresholds['plasmid_threshold']
                temp_params['chromosome_threshold'] = thresholds['chromosome_threshold']

                # get raw probabilities (for AUROC calculation)
                logits = model(data_test)
                probs = torch.sigmoid(logits)
                all_raw_probs.append(probs)

                # apply this model's specific thresholds to get final, scaled scores
                final_scores = torch.from_numpy(
                    apply_thresholds(probs.cpu().numpy(), temp_params)
                ).to(device)
                all_final_scores.append(final_scores)

        # average the raw probabilities (for AUROC) and the final scaled scores (for F1, etc.)
        ensemble_raw_probs = torch.stack(all_raw_probs).mean(dim=0)
        ensemble_final_scores = torch.stack(all_final_scores).mean(dim=0)

        # calculate metrics and generate plots
        plasmid_f1s, chromosome_f1s = calculate_and_print_metrics(
            ensemble_final_scores, ensemble_raw_probs, data_test, masks_test, G_test, node_list_test
        )
        plot_f1_violin(plasmid_f1s, chromosome_f1s, log_dir)
        accelerator.print(f"✅ Evaluation complete. Results saved to: {log_dir}")


if __name__ == "__main__":
    main()