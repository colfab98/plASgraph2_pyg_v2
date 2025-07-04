import numpy as np
import matplotlib.pyplot as plt
import torch
import optuna
import os 

from . import config
from .models import GCNModel, GGNNModel 
from .utils import fix_gradients

from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.optim as optim

from . import utils

import copy
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from .metrics import calculate_and_print_metrics
import yaml


def objective(trial, accelerator, parameters, data, splits, labeled_indices):
    """
    Optuna objective function with K-fold cross-validation.
    """
    trial_params_dict = parameters._params.copy() 

    trial_params_dict['l2_reg'] = trial.suggest_float("l2_reg", 1e-8, 1e-6, log=True)
    trial_params_dict['n_channels'] = trial.suggest_int("n_channels", 210, 310, step=16)
    trial_params_dict['n_gnn_layers'] = trial.suggest_int("n_gnn_layers", 8, 12)
    trial_params_dict['dropout_rate'] = trial.suggest_float("dropout_rate", 0.1, 0.4)
    trial_params_dict['gradient_clipping'] = trial.suggest_float("gradient_clipping", 0.0, 0.3)
    trial_params_dict['edge_gate_hidden_dim'] = trial.suggest_int("edge_gate_hidden_dim", 64, 120, step=8)
    trial_params_dict['n_channels_preproc'] = trial.suggest_int("n_channels_preproc", 20, 50, step=5)
    trial_params_dict['edge_gate_depth'] = trial.suggest_int("edge_gate_depth", 2, 6)
    # trial_params_dict['learning_rate'] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    
    trial_config_obj = config.config(parameters.config_file_path)
    trial_config_obj._params = trial_params_dict 

    data = data.to(accelerator.device)

    fold_aurocs = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        
        if trial_config_obj['model_type'] == 'GCNModel':
            model = GCNModel(trial_config_obj)
        elif trial_config_obj['model_type'] == 'GGNNModel':
            model = GGNNModel(trial_config_obj)

        optimizer = torch.optim.Adam(model.parameters(), lr=trial_config_obj['learning_rate'], weight_decay=trial_config_obj['l2_reg'])
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')


        model, optimizer = accelerator.prepare(model, optimizer)

        train_fold_indices = labeled_indices[train_idx]
        val_fold_indices = labeled_indices[val_idx]

        masks_train = torch.zeros(data.num_nodes, dtype=torch.float32, device=accelerator.device)
        masks_train[train_fold_indices] = 1.0 

        masks_validate = torch.zeros(data.num_nodes, dtype=torch.float32, device=accelerator.device)
        masks_validate[val_fold_indices] = 1.0

        best_val_loss_for_fold = float("inf")
        patience_for_fold = 0

        best_model_state_for_fold = None

        
        for epoch in range(parameters["epochs_trials"]):
            model.train()
            optimizer.zero_grad()
            outputs = model(data) 
            
            valid_label_mask = data.y.sum(dim=1) != 0
            masked_weights_train = masks_train[valid_label_mask]
            
            loss_per_node_train = criterion(outputs[valid_label_mask], data.y[valid_label_mask])
            train_loss = (loss_per_node_train.sum(dim=1) * masked_weights_train).sum() / masked_weights_train.sum()

            accelerator.backward(train_loss)
            fix_gradients(trial_config_obj, model) 
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_outputs = model(data)
                masked_weights_val = masks_validate[valid_label_mask]
                loss_per_node_val = criterion(val_outputs[valid_label_mask], data.y[valid_label_mask])
                val_loss = (loss_per_node_val.sum(dim=1) * masked_weights_val).sum() / masked_weights_val.sum()

            if val_loss.item() < best_val_loss_for_fold:
                best_val_loss_for_fold = val_loss.item()
                patience_for_fold = 0
                # ADDED: Save the best model state
                unwrapped_model = accelerator.unwrap_model(model)
                best_model_state_for_fold = copy.deepcopy(unwrapped_model.state_dict())
            else:
                patience_for_fold += 1
                if patience_for_fold >= parameters["early_stopping_patience"]:
                    break
        
        if best_model_state_for_fold:
            # Create a new model instance for inference to avoid issues with the accelerator-wrapped model
            if trial_config_obj['model_type'] == 'GCNModel':
                inference_model = GCNModel(trial_config_obj)
            else:
                inference_model = GGNNModel(trial_config_obj)
            
            inference_model.load_state_dict(best_model_state_for_fold)
            inference_model = inference_model.to(accelerator.device)
            inference_model.eval()

            with torch.no_grad():
                # Get raw logits from the model
                outputs = inference_model(data)
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(outputs)

            y_true_fold = data.y[val_fold_indices].cpu().numpy()
            y_probs_fold = probs[val_fold_indices].cpu().numpy()

            auroc_plasmid = 0.5
            auroc_chromosome = 0.5

            # Calculate AUROC only if both classes are present in the true labels
            if len(np.unique(y_true_fold[:, 0])) > 1:
                auroc_plasmid = roc_auc_score(y_true_fold[:, 0], y_probs_fold[:, 0])
            if len(np.unique(y_true_fold[:, 1])) > 1:
                auroc_chromosome = roc_auc_score(y_true_fold[:, 1], y_probs_fold[:, 1])

            # Use the average of plasmid and chromosome AUROC as the fold's score
            avg_auroc_for_fold = (auroc_plasmid + auroc_chromosome) / 2.0
            fold_aurocs.append(avg_auroc_for_fold)
        else:
            # If no model was saved (e.g., training failed), append a score of 0
            fold_aurocs.append(0.0)
        
        trial.report(np.median(fold_aurocs), fold_idx)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    average_auroc = np.median(fold_aurocs)
    
    return average_auroc



def train_final_model(accelerator, parameters, data, splits, labeled_indices, log_dir, G, node_list):
    """
    MODIFIED: Trains K-fold models and saves each one for ensembling.
    This function no longer retrains a single final model. It evaluates and saves
    the model from each CV fold.
    """
    accelerator.print("\n" + "="*60)
    accelerator.print("💾 Training K-Fold Ensemble Models")
    accelerator.print("="*60)

    data = data.to(accelerator.device)

    # Create a sub-directory for the ensemble models
    ensemble_models_dir = os.path.join(log_dir, "ensemble_models")
    os.makedirs(ensemble_models_dir, exist_ok=True)
    accelerator.print(f"Ensemble models will be saved to: {ensemble_models_dir}")

    # Create a sub-directory for fold-specific plots
    cv_plots_dir = os.path.join(log_dir, "cv_fold_plots")
    os.makedirs(cv_plots_dir, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        accelerator.print(f"\n--- Running Fold {fold_idx + 1}/{len(splits)} ---")

        # 1. Initialize Model for this fold
        if parameters['model_type'] == 'GCNModel':
            model = GCNModel(parameters)
        elif parameters['model_type'] == 'GGNNModel':
            model = GGNNModel(parameters)

        optimizer = optim.Adam(model.parameters(), lr=parameters['learning_rate'], weight_decay=parameters['l2_reg'])
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        model, optimizer = accelerator.prepare(model, optimizer)

        # 2. Create masks for this fold's train/val split
        train_fold_indices = labeled_indices[train_idx]
        val_fold_indices = labeled_indices[val_idx]
        masks_train = torch.zeros(data.num_nodes, dtype=torch.float32, device=accelerator.device)
        masks_train[train_fold_indices] = 1.0
        masks_validate = torch.zeros(data.num_nodes, dtype=torch.float32, device=accelerator.device)
        masks_validate[val_fold_indices] = 1.0

        # 3. Train with early stopping
        best_val_loss_for_fold = float("inf")
        patience_for_fold = 0
        best_epoch_for_fold = 0
        best_model_state_for_fold = None
        
        train_losses_fold, val_losses_fold = [], []

        for epoch in range(parameters["epochs"]):
            model.train()
            optimizer.zero_grad()
            outputs = model(data)

            valid_label_mask = data.y.sum(dim=1) != 0
            masked_weights_train = masks_train[valid_label_mask]
            loss_per_node_train = criterion(outputs[valid_label_mask], data.y[valid_label_mask])
            train_loss = (loss_per_node_train.sum(dim=1) * masked_weights_train).sum() / masked_weights_train.sum()
            train_losses_fold.append(train_loss.item())
            
            accelerator.backward(train_loss)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_outputs = model(data)
                masked_weights_val = masks_validate[valid_label_mask]
                loss_per_node_val = criterion(val_outputs[valid_label_mask], data.y[valid_label_mask])
                val_loss = (loss_per_node_val.sum(dim=1) * masked_weights_val).sum() / masked_weights_val.sum()
                val_losses_fold.append(val_loss.item())

            if val_loss.item() < best_val_loss_for_fold:
                best_val_loss_for_fold = val_loss.item()
                best_epoch_for_fold = epoch + 1
                patience_for_fold = 0
                best_model_state_for_fold = copy.deepcopy(accelerator.unwrap_model(model).state_dict())
            else:
                patience_for_fold += 1
                if patience_for_fold >= parameters["early_stopping_patience_retrain"]:
                    break
        
        accelerator.print(f"Fold {fold_idx + 1} finished. Best epoch: {best_epoch_for_fold} with val loss: {best_val_loss_for_fold:.4f}")

        if best_model_state_for_fold:
            model_save_path = os.path.join(ensemble_models_dir, f"fold_{fold_idx + 1}_model.pt")
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                torch.save(best_model_state_for_fold, model_save_path)
            accelerator.print(f"  > Saved best model for fold {fold_idx + 1} to: {model_save_path}")

        # --- NEW: Determine and save thresholds for this specific fold ---
        accelerator.print(f"  > Determining thresholds for fold {fold_idx + 1}...")

        # Create a new model instance for inference
        if parameters['model_type'] == 'GCNModel':
            inference_model = GCNModel(parameters).to(accelerator.device)
        else:
            inference_model = GGNNModel(parameters).to(accelerator.device)
        inference_model.load_state_dict(best_model_state_for_fold)
        inference_model.eval()

        # Create a temporary config object to hold this fold's thresholds
        fold_params = copy.deepcopy(parameters)
        
        # Find optimal thresholds using the validation mask for this fold
        utils.set_thresholds(inference_model, data, masks_validate, fold_params) 
        
        p_thresh = fold_params['plasmid_threshold']
        c_thresh = fold_params['chromosome_threshold']

        # Save these specific thresholds to a file
        if accelerator.is_main_process:
            threshold_data = {
                'plasmid_threshold': p_thresh,
                'chromosome_threshold': c_thresh
            }
            threshold_save_path = os.path.join(ensemble_models_dir, f"fold_{fold_idx + 1}_thresholds.yaml")
            with open(threshold_save_path, 'w') as f:
                yaml.dump(threshold_data, f)

        accelerator.print(f"  > Fold {fold_idx+1} thresholds (P:{p_thresh:.4f}, C:{c_thresh:.4f}) saved to: {threshold_save_path}")

        # Plotting for the current fold
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses_fold, label='Train Loss')
        plt.plot(val_losses_fold, label='Validation Loss')
        plt.axvline(x=best_epoch_for_fold - 1, color='r', linestyle='--', label=f'Best Epoch ({best_epoch_for_fold})')
        plt.title(f'Fold {fold_idx + 1} Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(cv_plots_dir, f"cv_loss_fold_{fold_idx + 1}.png")
        plt.savefig(plot_path)
        plt.close()

        # Evaluation for the current fold
        accelerator.print(f"\n--- Evaluating Fold {fold_idx + 1} Model on its Validation Set ---")
        if best_model_state_for_fold:
            # The 'inference_model' is already loaded with the best state and on the correct device
            inference_model.eval()

            with torch.no_grad():
                # Get raw probabilities
                outputs = inference_model(data)
                raw_probs_fold = torch.sigmoid(outputs)
                
                # Use the 'fold_params' which contain the just-calculated thresholds for this fold
                final_scores_fold = torch.from_numpy(
                    utils.apply_thresholds(raw_probs_fold.cpu().numpy(), fold_params)
                ).to(accelerator.device)

            calculate_and_print_metrics(
                final_scores_fold,
                raw_probs_fold,
                data,
                masks_validate,
                G,
                node_list,
                verbose=False
            )
        else:
            accelerator.print("No best model state found for this fold, skipping evaluation.")

    # The function now concludes after the loop, as there's no single final model to train.
    # Return None as the concept of a single "final model" no longer applies.
    accelerator.print("\n✅ Ensemble model training complete. Models for each fold saved.")
    return 




# In plasgraph/engine.py
# ... (existing functions are above) ...
import pandas as pd
import torch_geometric

# Import from our own library
from .data import read_single_graph
from .utils import apply_thresholds, pair_to_label

def predict(model, graph, parameters):
    """
    Applies the model to a graph to get predictions.
    (This function was formerly apply_to_graph in architecture.py)
    """
    device = next(model.parameters()).device
    graph = graph.to(device)
    model.eval()
    with torch.no_grad():
        output = model(graph)
    
    preds_np = output.cpu().numpy()
    preds_clipped = np.clip(preds_np, 0, 1)
    
    # Apply the custom thresholds to get final scores
    preds_final = apply_thresholds(preds_clipped, parameters)
    preds_final = np.clip(preds_final, 0, 1)
    
    return preds_final

def classify_graph(model, parameters, graph_file, file_prefix, sample_id):
    """
    Loads a single graph file, classifies it, and returns a results DataFrame.
    (This logic comes from test_one in plASgraph2_classify.py)
    """
    # 1. Load and process the graph data
    G = read_single_graph(file_prefix, graph_file, sample_id, parameters['minimum_contig_length'])
    node_list = list(G)

    # Calculate k-mer dot products for edges
    for u, v in G.edges():
        kmer_u = np.array(G.nodes[u]["kmer_counts_norm"])
        kmer_v = np.array(G.nodes[v]["kmer_counts_norm"])
        dot_product = np.dot(kmer_u, kmer_v)
        G.edges[u, v]["kmer_dot_product"] = dot_product
    
    # 2. Prepare graph for PyTorch Geometric
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
        # Add a batch tensor of all zeros for the single graph
        batch=torch.zeros(x.shape[0], dtype=torch.long)
    )


    # 3. Run prediction using the model
    preds = predict(model, data_obj, parameters)

    # 4. Format results into a DataFrame
    output_rows = []
    for i, node_id in enumerate(node_list):
        plasmid_score = preds[i, 0]
        chrom_score = preds[i, 1]
        label = pair_to_label([round(plasmid_score), round(chrom_score)])
        
        output_rows.append([
            sample_id,
            G.nodes[node_id]["contig"],
            G.nodes[node_id]["length"],
            plasmid_score,
            chrom_score,
            label
        ])
    
    prediction_df = pd.DataFrame(
        output_rows,
        columns=["sample", "contig", "length", "plasmid_score", "chrom_score", "label"]
    )
    return prediction_df