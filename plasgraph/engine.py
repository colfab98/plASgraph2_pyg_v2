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

# Add these imports at the top of plasgraph/engine.py
from torch_geometric.loader import NeighborLoader
from torch.profiler import profile, record_function, ProfilerActivity
import torch.profiler

import torch.distributed as dist


def objective(trial, accelerator, parameters, data, splits, labeled_indices):
    """
    Optuna objective function with K-fold cross-validation.
    """
    trial.set_user_attr("pid", os.getpid())

    trial_params_dict = parameters._params.copy()
    trial_params_dict['l2_reg'] = trial.suggest_float("l2_reg", 1e-6, 1e-2, log=True)
    trial_params_dict['n_channels'] = trial.suggest_int("n_channels", 16, 128, step=16)
    trial_params_dict['n_gnn_layers'] = trial.suggest_int("n_gnn_layers", 4, 8)
    trial_params_dict['dropout_rate'] = trial.suggest_float("dropout_rate", 0.0, 0.3)
    trial_params_dict['gradient_clipping'] = trial.suggest_float("gradient_clipping", 0.0, 0.3)
    trial_params_dict['edge_gate_hidden_dim'] = trial.suggest_int("edge_gate_hidden_dim", 16, 64, step=8)
    trial_params_dict['n_channels_preproc'] = trial.suggest_int("n_channels_preproc", 5, 20, step=5)
    trial_params_dict['edge_gate_depth'] = trial.suggest_int("edge_gate_depth", 2, 6)
    trial_params_dict['batch_size'] = trial.suggest_categorical('batch_size', [2048, 4096, 8192])

    n_layers = trial_params_dict['n_gnn_layers']
    neighbors_list = [15] + [10] * (n_layers - 1)
    
    trial_config_obj = config.config(parameters.config_file_path)
    trial_config_obj._params = trial_params_dict

    device = accelerator.device
    data = data.to(device)

    fold_aurocs = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):

        train_fold_indices = labeled_indices[train_idx]
        val_fold_indices = labeled_indices[val_idx]
        
        if trial_config_obj['model_type'] == 'GCNModel':
            model = GCNModel(trial_config_obj).to(device)
        elif trial_config_obj['model_type'] == 'GGNNModel':
            model = GGNNModel(trial_config_obj).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=trial_config_obj['learning_rate'], weight_decay=trial_config_obj['l2_reg'])
        criterion = torch.nn.BCEWithLogitsLoss()

        train_loader = NeighborLoader(
            data,
            input_nodes=torch.from_numpy(train_fold_indices).to(data.x.device),
            num_neighbors=neighbors_list, # Use all neighbors. Can be tuned e.g., [15, 10]
            shuffle=True,
            # Use the new tunable batch_size and num_workers from config
            batch_size=trial_config_obj['batch_size'],
            num_workers=trial_config_obj['num_workers'],
            pin_memory=True,
        )

        val_loader = NeighborLoader(
            data,
            input_nodes=torch.from_numpy(val_fold_indices).to(data.x.device),
            num_neighbors=neighbors_list,
            shuffle=False,
            batch_size=trial_config_obj['batch_size'],
            num_workers=trial_config_obj['num_workers'],
            pin_memory=True,
        )


        best_val_loss_for_fold = float("inf")
        patience_for_fold = 0

        best_model_state_for_fold = None

        
        for epoch in range(parameters["epochs_trials"]):
            model.train()
            for batch in train_loader:
                batch = batch.to(device) # Add this line
                optimizer.zero_grad()
                # The model now receives a mini-batch, not the full graph
                outputs = model(batch)
                # Loss is calculated on the output and labels of the mini-batch
                loss = criterion(outputs, batch.y)
                loss.backward()
                fix_gradients(trial_config_obj, model)
                optimizer.step()

            # Validation loop
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    outputs = model(batch)
                    val_loss = criterion(outputs, batch.y)
                    total_val_loss += val_loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)

            if avg_val_loss < best_val_loss_for_fold:
                best_val_loss_for_fold = avg_val_loss
                patience_for_fold = 0
                best_model_state_for_fold = copy.deepcopy(model.state_dict())
            else:
                patience_for_fold += 1
                if patience_for_fold >= parameters["early_stopping_patience"]:
                    break
            

        # --- Final evaluation for the fold ---
        # Only proceed if a model was successfully saved during training
        if best_model_state_for_fold is not None:
            # 1. Initialize a new model for inference
            if trial_config_obj['model_type'] == 'GCNModel':
                inference_model = GCNModel(trial_config_obj).to(device)
            else:
                inference_model = GGNNModel(trial_config_obj).to(device)
            
            # 2. Load the best state from training
            inference_model.load_state_dict(best_model_state_for_fold)
            inference_model.eval() # Set model to evaluation mode

            # 3. Get predictions on the validation set
            all_probs, all_true = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device) 
                    outputs = inference_model(batch)
                    all_probs.append(torch.sigmoid(outputs))
                    all_true.append(batch.y)
            
            y_probs_fold = torch.cat(all_probs).cpu().numpy()
            y_true_fold = torch.cat(all_true).cpu().numpy()

            # 4. Calculate AUROC
            auroc_plasmid = 0.5
            auroc_chromosome = 0.5
            if len(np.unique(y_true_fold[:, 0])) > 1:
                auroc_plasmid = roc_auc_score(y_true_fold[:, 0], y_probs_fold[:, 0])
            if len(np.unique(y_true_fold[:, 1])) > 1:
                auroc_chromosome = roc_auc_score(y_true_fold[:, 1], y_probs_fold[:, 1])

            avg_auroc_for_fold = (auroc_plasmid + auroc_chromosome) / 2.0
            fold_aurocs.append(avg_auroc_for_fold)
        else:
            # If no model was saved (e.g., training was unstable), report a score of 0
            fold_aurocs.append(0.0)

        # Report the result of this fold to Optuna for pruning decisions
        # This is outside the if/else, but inside the loop for the fold
        trial.report(np.median(fold_aurocs), fold_idx)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # The loop continues to the next fold...

    # After all folds are complete:
    average_auroc = np.median(fold_aurocs)
    return average_auroc



# In plasgraph/engine.py

def train_final_model(parameters, data, splits, labeled_indices, log_dir, G, node_list):
    """
    Trains K-fold models on a single GPU and saves each one for ensembling.
    This version is aligned with the trial training logic for consistency.
    """
    print("\n" + "="*60)
    print("💾 Training K-Fold Ensemble Models (Single-GPU)")
    print("="*60)

    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    data = data.to(device)

    # Create directories for models and plots
    ensemble_models_dir = os.path.join(log_dir, "ensemble_models")
    os.makedirs(ensemble_models_dir, exist_ok=True)
    print(f"Ensemble models will be saved to: {ensemble_models_dir}")

    cv_plots_dir = os.path.join(log_dir, "cv_fold_plots")
    os.makedirs(cv_plots_dir, exist_ok=True)

    # Define neighbor list based on final parameters
    n_layers = parameters['n_gnn_layers']
    neighbors_list = [15] + [10] * (n_layers - 1)

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- Running Fold {fold_idx + 1}/{len(splits)} ---")

        train_fold_indices = labeled_indices[train_idx]
        val_fold_indices = labeled_indices[val_idx]

        # 1. Initialize Model for this fold
        if parameters['model_type'] == 'GCNModel':
            model = GCNModel(parameters).to(device)
        elif parameters['model_type'] == 'GGNNModel':
            model = GGNNModel(parameters).to(device)

        optimizer = optim.Adam(model.parameters(), lr=parameters['learning_rate'], weight_decay=parameters['l2_reg'])
        criterion = torch.nn.BCEWithLogitsLoss()

        # 2. Create DataLoaders for this fold
        train_loader = NeighborLoader(
            data,
            input_nodes=torch.from_numpy(train_fold_indices).to(device),
            num_neighbors=neighbors_list,
            shuffle=True,
            batch_size=parameters['batch_size'],
            num_workers=parameters['num_workers'],
            pin_memory=True,
        )

        val_loader = NeighborLoader(
            data,
            input_nodes=torch.from_numpy(val_fold_indices).to(device),
            num_neighbors=neighbors_list,
            shuffle=False,
            batch_size=parameters['batch_size'],
            num_workers=parameters['num_workers'],
            pin_memory=True,
        )

        # 3. Train with early stopping
        best_val_loss_for_fold = float("inf")
        patience_for_fold = 0
        best_epoch_for_fold = 0
        best_model_state_for_fold = None
        
        train_losses_fold, val_losses_fold = [], []

        for epoch in range(parameters["epochs"]):
            model.train()
            total_train_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, batch.y)
                loss.backward()
                utils.fix_gradients(parameters, model)
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses_fold.append(avg_train_loss)

            # Validation loop
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    outputs = model(batch)
                    val_loss = criterion(outputs, batch.y)
                    total_val_loss += val_loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses_fold.append(avg_val_loss)

            if (epoch + 1) % 10 == 0 or (epoch + 1) == parameters["epochs"]:
                print(f"Epoch {epoch + 1}/{parameters['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss_for_fold:
                best_val_loss_for_fold = avg_val_loss
                best_epoch_for_fold = epoch + 1
                patience_for_fold = 0
                # Use deepcopy to ensure the state is not a reference
                best_model_state_for_fold = copy.deepcopy(model.state_dict())
            else:
                patience_for_fold += 1
                if patience_for_fold >= parameters["early_stopping_patience_retrain"]:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
        
        print(f"Fold {fold_idx + 1} finished. Best epoch: {best_epoch_for_fold} with val loss: {best_val_loss_for_fold:.4f}")

        # --- Save model, plot, and determine thresholds ---
        if best_model_state_for_fold:
            # Save the model
            model_save_path = os.path.join(ensemble_models_dir, f"fold_{fold_idx + 1}_model.pt")
            torch.save(best_model_state_for_fold, model_save_path)
            print(f"  > Saved best model for fold {fold_idx + 1} to: {model_save_path}")

            # Save the training history plot
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
            print(f"  > Saved loss plot to: {plot_path}")

            # --- Evaluation and Thresholding ---
            # Create a temporary model to load the best state for evaluation
            if parameters['model_type'] == 'GCNModel':
                inference_model = GCNModel(parameters).to(device)
            else:
                inference_model = GGNNModel(parameters).to(device)
            
            inference_model.load_state_dict(best_model_state_for_fold)
            inference_model.eval()

            # Get predictions on the validation set for this fold
            all_probs_val, all_true_val = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    outputs = inference_model(batch)
                    all_probs_val.append(torch.sigmoid(outputs))
                    all_true_val.append(batch.y)
            
            y_probs_val_fold = torch.cat(all_probs_val).cpu()
            y_true_val_fold = torch.cat(all_true_val).cpu()

            # Determine and save thresholds
            fold_params = copy.deepcopy(parameters)
            utils.set_thresholds_from_predictions(y_true_val_fold, y_probs_val_fold, fold_params, log_dir)
            
            threshold_save_path = os.path.join(ensemble_models_dir, f"fold_{fold_idx + 1}_thresholds.yaml")
            threshold_data = {
                'plasmid_threshold': fold_params['plasmid_threshold'],
                'chromosome_threshold': fold_params['chromosome_threshold']
            }
            with open(threshold_save_path, 'w') as f:
                yaml.dump(threshold_data, f)
            print(f"  > Fold {fold_idx+1} thresholds saved to: {threshold_save_path}")

            # 💡 ADDED: Calculate and print evaluation metrics for the fold
            print(f"\n--- Evaluating Fold {fold_idx + 1} Model on its Full Validation Set ---")

            # Apply thresholds to get final binary predictions
            final_scores_val_fold = torch.from_numpy(
                utils.apply_thresholds(y_probs_val_fold.numpy(), fold_params)
            ).to(device)
            
            # Move probability scores to the correct device
            y_probs_val_fold = y_probs_val_fold.to(device)

            # Get the global indices for the current validation fold
            val_fold_indices_global = torch.from_numpy(val_fold_indices).to(device)

            # Create tensors to hold predictions for the entire graph
            raw_probs_full = torch.zeros_like(data.y, dtype=torch.float, device=device)
            final_scores_full = torch.zeros_like(data.y, dtype=torch.float, device=device)

            # Place the validation predictions into the full tensors at the correct indices
            raw_probs_full.scatter_(0, val_fold_indices_global.unsqueeze(1).expand(-1, 2), y_probs_val_fold)
            final_scores_full.scatter_(0, val_fold_indices_global.unsqueeze(1).expand(-1, 2), final_scores_val_fold)

            # Create a mask for the validation nodes
            masks_validate = torch.zeros(data.num_nodes, dtype=torch.float32, device=device)
            masks_validate[val_fold_indices_global] = 1.0

            # Call the metrics function
            calculate_and_print_metrics(
                final_scores_full, raw_probs_full, data, masks_validate, G, node_list, verbose=False
            )




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