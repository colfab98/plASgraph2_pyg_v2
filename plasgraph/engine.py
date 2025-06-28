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


def objective(trial, accelerator, parameters, data, splits, labeled_indices):
    """
    Optuna objective function with K-fold cross-validation.
    """
    trial_params_dict = parameters._params.copy() 

    trial_params_dict['l2_reg'] = trial.suggest_float("l2_reg", 1e-7, 1e-2, log=True)
    trial_params_dict['n_channels'] = trial.suggest_int("n_channels", 64, 160, step=16)
    trial_params_dict['n_gnn_layers'] = trial.suggest_int("n_gnn_layers", 6, 12)
    trial_params_dict['dropout_rate'] = trial.suggest_float("dropout_rate", 0.0, 0.5)
    trial_params_dict['gradient_clipping'] = trial.suggest_float("gradient_clipping", 0.0, 0.7)
    trial_params_dict['edge_gate_hidden_dim'] = trial.suggest_int("edge_gate_hidden_dim", 8, 200, step=8)
    trial_params_dict['n_channels_preproc'] = trial.suggest_int("n_channels_preproc", 2, 25, step=2)
    
    trial_config_obj = config.config(parameters.config_file_path)
    trial_config_obj._params = trial_params_dict 

    data = data.to(accelerator.device)

    fold_losses = []
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
            else:
                patience_for_fold += 1
                if patience_for_fold >= parameters["early_stopping_patience"]:
                    break
        
        fold_losses.append(best_val_loss_for_fold)
        
        trial.report(np.mean(fold_losses), fold_idx)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    average_val_loss = np.mean(fold_losses)
    
    return average_val_loss


# ADD THIS CORRECTED FUNCTION TO plasgraph/engine.py

def tune_thresholds(accelerator, parameters, data, splits, labeled_indices, log_dir):
    """
    Performs a new K-fold cross-validation using the best hyperparameters
    to determine the optimal classification thresholds.
    This version is fully compatible with Accelerate.
    """
    accelerator.print("\n" + "="*60)
    accelerator.print("🎯 Tuning Classification Thresholds via Cross-Validation")
    accelerator.print("="*60)

    # Move data to the correct device once
    data = data.to(accelerator.device)
    
    fold_thresholds = {'plasmid': [], 'chromosome': []}

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        accelerator.print(f"\n--- Threshold Tuning: Fold {fold_idx + 1}/{len(splits)} ---")

        # 1. Initialize Model for this fold
        if parameters['model_type'] == 'GCNModel':
            model = GCNModel(parameters)
        elif parameters['model_type'] == 'GGNNModel':
            model = GGNNModel(parameters)

        optimizer = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'], weight_decay=parameters['l2_reg'])
        # Corrected criterion definition
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

        # Use Accelerate to prepare the model and optimizer
        model, optimizer = accelerator.prepare(model, optimizer)

        # 2. Create masks for this fold's train/val split on the correct device
        train_fold_indices = labeled_indices[train_idx]
        val_fold_indices = labeled_indices[val_idx]
        masks_train = torch.zeros(data.num_nodes, dtype=torch.float32, device=accelerator.device)
        masks_train[train_fold_indices] = 1.0
        masks_validate = torch.zeros(data.num_nodes, dtype=torch.float32, device=accelerator.device)
        masks_validate[val_fold_indices] = 1.0

        # 3. Train the model on this fold's training data
        best_val_loss_for_fold = float("inf")
        patience_for_fold = 0
        best_model_state_for_fold = None

        for epoch in range(parameters["epochs"]): # Use full epochs for robust training
            model.train()
            optimizer.zero_grad()
            outputs = model(data)

            valid_label_mask = data.y.sum(dim=1) != 0
            masked_weights_train = masks_train[valid_label_mask]
            loss_per_node_train = criterion(outputs[valid_label_mask], data.y[valid_label_mask])
            train_loss = (loss_per_node_train.sum(dim=1) * masked_weights_train).sum() / masked_weights_train.sum()

            accelerator.backward(train_loss)
            utils.fix_gradients(parameters, model)
            optimizer.step()

            # Validation for early stopping
            model.eval()
            with torch.no_grad():
                val_outputs = model(data)
                masked_weights_val = masks_validate[valid_label_mask]
                loss_per_node_val = criterion(val_outputs[valid_label_mask], data.y[valid_label_mask])
                val_loss = (loss_per_node_val.sum(dim=1) * masked_weights_val).sum() / masked_weights_val.sum()

            if val_loss.item() < best_val_loss_for_fold:
                best_val_loss_for_fold = val_loss.item()
                # Unwrap the model before saving its state
                unwrapped_model = accelerator.unwrap_model(model)
                best_model_state_for_fold = unwrapped_model.state_dict()
                patience_for_fold = 0
            else:
                patience_for_fold += 1
                if patience_for_fold >= parameters["early_stopping_patience"]:
                    accelerator.print(f"Stopping early at epoch {epoch+1}.")
                    break
        
        # 4. Determine thresholds on this fold's validation data
        accelerator.print(f"Loading best model for fold {fold_idx+1} to determine thresholds...")
        # Create a new model instance for loading the state, as the original is wrapped by accelerate
        if parameters['model_type'] == 'GCNModel':
            inference_model = GCNModel(parameters).to(accelerator.device)
        else:
            inference_model = GGNNModel(parameters).to(accelerator.device)
        inference_model.load_state_dict(best_model_state_for_fold)
        
        temp_params_for_fold = copy.deepcopy(parameters)

        
        fold_log_dir = os.path.join(log_dir, f"threshold_tuning_fold_{fold_idx+1}")
        os.makedirs(fold_log_dir, exist_ok=True)
        
        utils.set_thresholds(inference_model, data, masks_validate, temp_params_for_fold, fold_log_dir) 
        
        p_thresh = temp_params_for_fold['plasmid_threshold']
        c_thresh = temp_params_for_fold['chromosome_threshold']
        accelerator.print(f"Fold {fold_idx+1} thresholds: Plasmid={p_thresh:.2f}, Chromosome={c_thresh:.2f}")
        
        fold_thresholds['plasmid'].append(p_thresh)
        fold_thresholds['chromosome'].append(c_thresh)

    # 5. Average the thresholds across all folds
    avg_plasmid_thresh = float(np.mean(fold_thresholds['plasmid']))
    avg_chromosome_thresh = float(np.mean(fold_thresholds['chromosome']))

    accelerator.print("\n" + "="*60)
    accelerator.print(f"✅ Average thresholds determined: Plasmid={avg_plasmid_thresh:.4f}, Chromosome={avg_chromosome_thresh:.4f}")
    accelerator.print("="*60)
    
    return avg_plasmid_thresh, avg_chromosome_thresh


# REPLACE the existing train_final_model function in engine.py with this one.
def train_final_model(accelerator, parameters, data, splits, labeled_indices, log_dir):
    """
    Trains the final model on all labeled data using the best hyperparameters.
    Early stopping is monitored using a combined validation set from all folds.
    """
    accelerator.print("\n" + "="*60)
    accelerator.print("💾 Training Final Model on ALL Labeled Data")
    accelerator.print("="*60)

    data = data.to(accelerator.device)

    # 1. Create a combined validation mask from all folds for plotting and early stopping
    all_val_indices = np.concatenate([val_idx for _, val_idx in splits])
    val_indices_for_plot = labeled_indices[all_val_indices]
    masks_val_final_plot = torch.zeros(data.num_nodes, dtype=torch.float32, device=accelerator.device)
    masks_val_final_plot[val_indices_for_plot] = 1.0

    # 2. Initialize Model, Optimizer, Scheduler, Criterion
    if parameters['model_type'] == 'GCNModel':
        final_model = GCNModel(parameters)
    elif parameters['model_type'] == 'GGNNModel':
        final_model = GGNNModel(parameters)

    optimizer = optim.Adam(final_model.parameters(), lr=parameters['learning_rate'], weight_decay=parameters['l2_reg'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=parameters['scheduler_factor'],
                                  patience=parameters['scheduler_patience'])
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')


    final_model, optimizer, scheduler = accelerator.prepare(final_model, optimizer, scheduler)


    # 3. Create mask for ALL labeled training data
    masks_train_all = torch.zeros(data.num_nodes, dtype=torch.float32, device=accelerator.device)
    masks_train_all[labeled_indices] = 1.0

    # 4. The main training loop
    best_val_loss_final = float('inf')
    patience_final = 0
    best_model_state_final = None
    train_losses_final = []
    val_losses_final = []
    best_epoch = 0

    for epoch in range(parameters["epochs"]):
        final_model.train()
        optimizer.zero_grad()
        outputs = final_model(data)

        valid_label_mask = data.y.sum(dim=1) != 0
        masked_weights_train_all = masks_train_all[valid_label_mask]
        loss_per_node = criterion(outputs[valid_label_mask], data.y[valid_label_mask])
        loss = (loss_per_node.sum(dim=1) * masked_weights_train_all).sum() / masked_weights_train_all.sum()

        accelerator.backward(loss)

        if parameters['gradient_clipping'] > 0:
            accelerator.clip_grad_value_(final_model.parameters(), parameters['gradient_clipping'])

        unwrapped_model = accelerator.unwrap_model(final_model)
        grad_magnitudes = utils.get_gradient_magnitudes(unwrapped_model)
        utils.plot_gradient_magnitudes(grad_magnitudes, epoch, log_dir, plot_frequency=100)

        optimizer.step()

        # Validation on the combined validation set
        final_model.eval()
        with torch.no_grad():
            val_outputs = final_model(data)
            masked_weights_val = masks_val_final_plot[valid_label_mask]
            val_loss = 0.0
            if masked_weights_val.sum() > 0:
                loss_per_node_val = criterion(val_outputs[valid_label_mask], data.y[valid_label_mask])
                val_loss = (loss_per_node_val.sum(dim=1) * masked_weights_val).sum() / masked_weights_val.sum()
                val_losses_final.append(val_loss.item())
            else:
                val_losses_final.append(0.0) # Should not happen if there are validation folds
        train_losses_final.append(loss.item())

        scheduler.step(val_loss)

        if val_loss < best_val_loss_final:
            best_val_loss_final = val_loss
            unwrapped_model = accelerator.unwrap_model(final_model)
            best_model_state_final = unwrapped_model.state_dict()
            patience_final = 0
        else:
            patience_final += 1

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Final training... Epoch {epoch+1}/{parameters['epochs']}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr}")

        if patience_final >= parameters['early_stopping_patience_retrain']:
            print(f"Stopping early at epoch {epoch+1} due to no improvement in validation loss.")
            break

    # 5. Load best model and set thresholds
    accelerator.print(f"\nTraining finished. Loading best model state from epoch {epoch+1-patience_final}.")
    # Accelerate handles loading the state dict into the wrapped model correctly.
    final_model.load_state_dict(best_model_state_final)

    # --- PLOTTING (NO THRESHOLDING) ---
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses_final, label='Final Train Loss (on all data)')
    plt.plot(val_losses_final, label=f"Final Validation Loss (on combined val folds)")
    plt.axvline(x=best_epoch - 1, color='r', linestyle='--', label='Best Model Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Final Model Training (Initial LR: {parameters['learning_rate']})")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(log_dir, "final_model_training_loss.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\n✅ Final model training loss plot saved to: {plot_path}")

    # Return only the trained model. Parameters are handled in the main script.
    return accelerator.unwrap_model(final_model), parameters




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