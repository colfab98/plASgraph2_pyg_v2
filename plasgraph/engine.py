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


def objective(trial, accelerator, parameters, data, sample_splits, all_sample_ids, labeled_indices, node_list, G):

    """
    Optuna objective function with K-fold cross-validation.
    """

    # set a user attribute to store the process ID, useful for debugging in parallel environments
    trial.set_user_attr("pid", os.getpid())

    # use the 'trial' object to suggest hyperparameter values for Optuna to optimize
    trial_params_dict = parameters._params.copy()
    trial_params_dict['l2_reg'] = trial.suggest_float("l2_reg", 1e-5, 1e-3, log=True)
    trial_params_dict['n_channels'] = trial.suggest_int("n_channels", 8, 64, step=16)
    trial_params_dict['n_gnn_layers'] = trial.suggest_int("n_gnn_layers", 2, 6)
    trial_params_dict['dropout_rate'] = trial.suggest_float("dropout_rate", 0.0, 0.3)
    trial_params_dict['gradient_clipping'] = trial.suggest_float("gradient_clipping", 1.0, 10.0, log=True)
    trial_params_dict['edge_gate_hidden_dim'] = trial.suggest_int("edge_gate_hidden_dim", 8, 32, step=8)
    trial_params_dict['n_channels_preproc'] = trial.suggest_int("n_channels_preproc", 10, 25, step=5)
    trial_params_dict['edge_gate_depth'] = trial.suggest_int("edge_gate_depth", 2, 6)
    trial_params_dict['batch_size'] = trial.suggest_categorical('batch_size', [64, 128 , 256, 512, 1024])
    first_hop_val = trial.suggest_int("neighbors_first_hop", 10, 50, step=10)
    trial_params_dict['first_hop_neighbors'] = first_hop_val
    trial_params_dict['subsequent_hop_neighbors'] = trial.suggest_int("neighbors_subsequent_hops", 10, first_hop_val, step=5)

    trial_params_dict['learning_rate'] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    n_layers = trial_params_dict['n_gnn_layers']
    first_hop_neighbors = trial_params_dict['first_hop_neighbors']
    subsequent_hop_neighbors = trial_params_dict['subsequent_hop_neighbors']
    neighbors_list = [first_hop_neighbors] + [subsequent_hop_neighbors] * (n_layers - 1)

    
    # temporary config object using the suggested hyperparameters for this trial
    trial_config_obj = config.config(parameters.config_file_path)
    trial_config_obj._params = trial_params_dict

    device = accelerator.device
    # move the graph data object to the correct device
    data = data.to(device)

    unique_samples_from_graph = sorted(list(set(node_id.split(':')[0] for node_id in node_list)))

    sample_to_batch_idx = {sample_id: i for i, sample_id in enumerate(unique_samples_from_graph)}

    fold_aurocs = []

    labeled_nodes_set = set(labeled_indices)

    # k-fold cross-validation loop
    for fold_idx, (train_sample_indices, val_sample_indices) in enumerate(sample_splits):

        train_samples = all_sample_ids[train_sample_indices]
        val_samples = all_sample_ids[val_sample_indices]

        train_batch_indices = [sample_to_batch_idx[sid] for sid in train_samples]
        val_batch_indices = [sample_to_batch_idx[sid] for sid in val_samples]

        train_mask = torch.isin(data.batch, torch.tensor(train_batch_indices, device=data.batch.device))
        val_mask = torch.isin(data.batch, torch.tensor(val_batch_indices, device=data.batch.device))

        train_node_indices = torch.where(train_mask)[0]
        val_node_indices = torch.where(val_mask)[0]

        train_data = data.subgraph(train_node_indices)
        val_data = data.subgraph(val_node_indices)

        train_nodes_global_set = set(train_node_indices.cpu().numpy())
        val_nodes_global_set = set(val_node_indices.cpu().numpy())

        train_fold_labeled_global = torch.tensor(sorted(list(train_nodes_global_set.intersection(labeled_nodes_set))), dtype=torch.long)
        val_fold_labeled_global = torch.tensor(sorted(list(val_nodes_global_set.intersection(labeled_nodes_set))), dtype=torch.long)

        global_to_local_train_map = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(train_node_indices)}
        global_to_local_val_map = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(val_node_indices)}

        local_train_seed_indices = torch.tensor([global_to_local_train_map[global_idx.item()] for global_idx in train_fold_labeled_global], dtype=torch.long)
        local_val_seed_indices = torch.tensor([global_to_local_val_map[global_idx.item()] for global_idx in val_fold_labeled_global], dtype=torch.long)


        if len(local_train_seed_indices) == 0 or len(local_val_seed_indices) == 0:
            continue
        
        # initialize the model
        if trial_config_obj['model_type'] == 'GCNModel':
            model = GCNModel(trial_config_obj).to(device)
        elif trial_config_obj['model_type'] == 'GGNNModel':
            model = GGNNModel(trial_config_obj).to(device)

        # Adam optimizer with the trial's learning rate and L2 regularization
        optimizer = torch.optim.Adam(model.parameters(), lr=trial_config_obj['learning_rate'], weight_decay=trial_config_obj['l2_reg'])
        # learning rate scheduler to reduce the learning rate on validation loss plateau
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=trial_config_obj['scheduler_factor'], patience=trial_config_obj['scheduler_patience'])
        # loss function
        criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')

        # NeighborLoader for the training set of this fold
        train_loader = NeighborLoader(
            train_data,
            input_nodes=local_train_seed_indices.to(device),
            num_neighbors=neighbors_list,                                       # neighborhood sampling sizes
            shuffle=True,                                                       # shuffle the data at each epoch
            batch_size=trial_config_obj['batch_size'],                          # use the batch size suggested by Optuna
            num_workers=trial_config_obj['num_workers'],
            pin_memory=True,                                                    # for faster data transfer to GPU
        )

        # NeighborLoader for the validation set of this fold
        val_loader = NeighborLoader(
            val_data,
            input_nodes=local_val_seed_indices.to(device),
            num_neighbors=neighbors_list,
            shuffle=False,                                                      # no need to shuffle validation data
            batch_size=trial_config_obj['batch_size'],
            num_workers=trial_config_obj['num_workers'],
            pin_memory=True,
        )

        # initialize variables for early stopping within the fold
        best_val_loss_for_fold = float("inf")
        patience_for_fold = 0
        best_model_state_for_fold = None

        # start the training loop for the number of epochs specified for HPO trials
        for epoch in range(parameters["epochs_trials"]):
            # set the model to training mode
            model.train()
            for batch in train_loader:
                batch = batch.to(device)                    # move the current mini-batch to the device
                optimizer.zero_grad()                       # clear gradients from the previous step
                outputs = model(batch)                      # perform a forward pass on the mini-batch
                loss = criterion(outputs[:batch.batch_size], batch.y[:batch.batch_size])
                loss.backward()                             # perform backpropagation to compute gradients
                fix_gradients(trial_config_obj, model)      # apply gradient clipping/fixing
                optimizer.step()                            # update the model weights

            # validation loop
            model.eval()
            total_val_loss = 0
            with torch.no_grad():                           # disable gradient computation for validation
                for batch in val_loader:
                    batch = batch.to(device)
                    outputs = model(batch)
                    val_loss = criterion(outputs[:batch.batch_size], batch.y[:batch.batch_size])
                    total_val_loss += val_loss.item()
            
            # calculate the average validation loss for the epoch
            avg_val_loss = total_val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            # check for improvement in validation loss for early stopping
            if avg_val_loss < best_val_loss_for_fold:
                best_val_loss_for_fold = avg_val_loss                           # update the best loss
                patience_for_fold = 0                                           # reset patience
                best_model_state_for_fold = copy.deepcopy(model.state_dict())   # save the best model state
            else:
                patience_for_fold += 1
                # if no improvement for a set number of epochs, stop training for this fold
                if patience_for_fold >= parameters["early_stopping_patience"]:
                    break
            

        # --- final evaluation for the fold ---

        if best_model_state_for_fold is not None:
            if trial_config_obj['model_type'] == 'GCNModel':
                inference_model = GCNModel(trial_config_obj).to(device)
            else:
                inference_model = GGNNModel(trial_config_obj).to(device)
            
            # load the best model state found during training
            inference_model.load_state_dict(best_model_state_for_fold)
            inference_model.eval() 

            all_probs_fold, all_true_fold = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    outputs = inference_model(batch)
                    all_probs_fold.append(torch.sigmoid(outputs[:batch.batch_size]))
                    all_true_fold.append(batch.y[:batch.batch_size])

            if not all_probs_fold:
                continue
            
            y_probs_fold = torch.cat(all_probs_fold).cpu()
            y_true_fold = torch.cat(all_true_fold).cpu()

            # --- MODIFIED: Simplified and robust AUROC calculation ---
            auroc_p, auroc_c = 0.5, 0.5 # Default to 0.5 if calculation is not possible

            # Plasmid AUROC
            y_true_p = y_true_fold[:, 0].numpy()
            y_probs_p = y_probs_fold[:, 0].numpy()
            if len(np.unique(y_true_p)) > 1:
                auroc_p = roc_auc_score(y_true_p, y_probs_p)

            # Chromosome AUROC
            y_true_c = y_true_fold[:, 1].numpy()
            y_probs_c = y_probs_fold[:, 1].numpy()
            if len(np.unique(y_true_c)) > 1:
                auroc_c = roc_auc_score(y_true_c, y_probs_c)

            # Average AUROC for the fold
            avg_auroc_for_fold = (auroc_p + auroc_c) / 2.0
            fold_aurocs.append(avg_auroc_for_fold)

            trial.report(avg_auroc_for_fold, fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()


    final_objective_value = np.mean(fold_aurocs)
    return final_objective_value




def train_final_model(accelerator, parameters, data, sample_splits, all_sample_ids, labeled_indices, log_dir, G, node_list):
    
    """
    Trains K-fold models on a single GPU and saves each one for ensembling.
    This version is aligned with the trial training logic for consistency.
    """
    print("\n" + "="*60)
    print("💾 Training K-Fold Ensemble Models (Single-GPU)")
    print("="*60)

    device = accelerator.device
    data = data.to(device)

    # create a directory to store the trained model files for the ensemble
    ensemble_models_dir = os.path.join(log_dir, "ensemble_models")
    os.makedirs(ensemble_models_dir, exist_ok=True)
    print(f"Ensemble models will be saved to: {ensemble_models_dir}")

    # create a directory to store plots of the training history for each cross-validation fold
    cv_plots_dir = os.path.join(log_dir, "cv_fold_plots")
    os.makedirs(cv_plots_dir, exist_ok=True)

    unique_samples_from_graph = sorted(list(set(node_id.split(':')[0] for node_id in node_list)))
    sample_to_batch_idx = {sample_id: i for i, sample_id in enumerate(unique_samples_from_graph)}
    labeled_nodes_set = set(labeled_indices)

    # define the neighborhood sampling sizes
    n_layers = parameters['n_gnn_layers']
    first_hop_neighbors = parameters['neighbors_first_hop']
    subsequent_hop_neighbors = parameters['neighbors_subsequent_hops']

    neighbors_list = [first_hop_neighbors] + [subsequent_hop_neighbors] * (n_layers - 1)

    # start the main loop to iterate through each cross-validation fold
    for fold_idx, (train_sample_indices, val_sample_indices) in enumerate(sample_splits):
        print(f"\n--- Running Fold {fold_idx + 1}/{len(sample_splits)} ---")

        train_samples = all_sample_ids[train_sample_indices]
        val_samples = all_sample_ids[val_sample_indices]
        train_batch_indices = [sample_to_batch_idx[sid] for sid in train_samples]
        val_batch_indices = [sample_to_batch_idx[sid] for sid in val_samples]

        train_mask = torch.isin(data.batch, torch.tensor(train_batch_indices, device=data.batch.device))
        val_mask = torch.isin(data.batch, torch.tensor(val_batch_indices, device=data.batch.device))
        train_node_indices = torch.where(train_mask)[0]
        val_node_indices = torch.where(val_mask)[0]

        train_data = data.subgraph(train_node_indices)
        val_data = data.subgraph(val_node_indices)
        print(f"  Train samples: {len(train_samples)}, Nodes: {train_data.num_nodes} | Val samples: {len(val_samples)}, Nodes: {val_data.num_nodes}")

        train_nodes_global_set = set(train_node_indices.cpu().numpy())
        val_nodes_global_set = set(val_node_indices.cpu().numpy())

        train_fold_labeled_global = torch.tensor(sorted(list(train_nodes_global_set.intersection(labeled_nodes_set))), dtype=torch.long)
        val_fold_labeled_global = torch.tensor(sorted(list(val_nodes_global_set.intersection(labeled_nodes_set))), dtype=torch.long)

        global_to_local_train_map = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(train_node_indices)}
        global_to_local_val_map = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(val_node_indices)}

        local_train_seed_indices = torch.tensor([global_to_local_train_map[global_idx.item()] for global_idx in train_fold_labeled_global], dtype=torch.long)
        local_val_seed_indices = torch.tensor([global_to_local_val_map[global_idx.item()] for global_idx in val_fold_labeled_global], dtype=torch.long)


        if parameters['model_type'] == 'GCNModel':
            model = GCNModel(parameters).to(device)
        elif parameters['model_type'] == 'GGNNModel':
            model = GGNNModel(parameters).to(device)

        optimizer = optim.Adam(model.parameters(), lr=parameters['learning_rate'], weight_decay=parameters['l2_reg'])
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=parameters['scheduler_factor'], patience=parameters['scheduler_patience'], verbose=True)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')

        num_workers = parameters['num_workers']

        train_loader = NeighborLoader(
            train_data,
            input_nodes=local_train_seed_indices.to(device),
            num_neighbors=neighbors_list,
            shuffle=True,
            batch_size=parameters['batch_size'],
            num_workers=num_workers,
            pin_memory=True,
        )

        val_loader = NeighborLoader(
            val_data,
            input_nodes=local_val_seed_indices.to(device),
            num_neighbors=neighbors_list,
            shuffle=False,
            batch_size=parameters['batch_size'],
            num_workers=num_workers,
            pin_memory=True,
        )

        best_val_loss_for_fold, patience_for_fold, best_epoch_for_fold = float("inf"), 0, 0
        best_model_state_for_fold = None
        train_losses_fold, val_losses_fold = [], []
        plot_frequency = max(1, parameters["epochs"] // 10)

        for epoch in range(parameters["epochs"]):
            model.train()
            total_train_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs[:batch.batch_size], batch.y[:batch.batch_size])
                loss.backward()
                utils.fix_gradients(parameters, model)
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
            train_losses_fold.append(avg_train_loss)

            if fold_idx == 0: 
                utils.plot_gradient_magnitudes(utils.get_gradient_magnitudes(model), epoch + 1, cv_plots_dir, plot_frequency=plot_frequency)
            
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    outputs = model(batch)
                    val_loss = criterion(outputs[:batch.batch_size], batch.y[:batch.batch_size])
                    total_val_loss += val_loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
            val_losses_fold.append(avg_val_loss)
            scheduler.step(avg_val_loss)

            if (epoch + 1) % 10 == 0 or (epoch + 1) == parameters["epochs"]:
                print(f"Epoch {epoch + 1}/{parameters['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss_for_fold:
                best_val_loss_for_fold = avg_val_loss
                best_epoch_for_fold = epoch + 1
                patience_for_fold = 0
                best_model_state_for_fold = copy.deepcopy(model.state_dict())
            else:
                patience_for_fold += 1
                if patience_for_fold >= parameters["early_stopping_patience_retrain"]:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
        
        print(f"Fold {fold_idx + 1} finished. Best epoch: {best_epoch_for_fold} with val loss: {best_val_loss_for_fold:.4f}")

        # --- save the best model, plot training history, and determine thresholds ---
        if best_model_state_for_fold:
            model_save_path = os.path.join(ensemble_models_dir, f"fold_{fold_idx + 1}_model.pt")
            torch.save(best_model_state_for_fold, model_save_path)
            print(f"  > Saved best model for fold {fold_idx + 1} to: {model_save_path}")

            plt.figure(figsize=(10, 6)); plt.plot(train_losses_fold, label='Train Loss'); plt.plot(val_losses_fold, label='Validation Loss'); plt.axvline(x=best_epoch_for_fold - 1, color='r', linestyle='--', label=f'Best Epoch ({best_epoch_for_fold})'); plt.title(f'Fold {fold_idx + 1} Training History'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True); plot_path = os.path.join(cv_plots_dir, f"cv_loss_fold_{fold_idx + 1}.png"); plt.savefig(plot_path); plt.close()
            print(f"  > Saved loss plot to: {plot_path}")

            if parameters['model_type'] == 'GCNModel':
                inference_model = GCNModel(parameters).to(device)
            else:
                inference_model = GGNNModel(parameters).to(device)
            
            inference_model.load_state_dict(best_model_state_for_fold)
            inference_model.eval()

            all_probs_val, all_true_val = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    outputs = inference_model(batch)
                    all_probs_val.append(torch.sigmoid(outputs[:batch.batch_size]))
                    all_true_val.append(batch.y[:batch.batch_size])

            
            y_probs_val_fold = torch.cat(all_probs_val).cpu()
            y_true_val_fold = torch.cat(all_true_val).cpu()

            fold_params = copy.deepcopy(parameters)
            utils.set_thresholds_from_predictions(y_true_val_fold, y_probs_val_fold, fold_params, log_dir)
            
            threshold_save_path = os.path.join(ensemble_models_dir, f"fold_{fold_idx + 1}_thresholds.yaml")
            with open(threshold_save_path, 'w') as f:
                yaml.dump({'plasmid_threshold': fold_params['plasmid_threshold'], 'chromosome_threshold': fold_params['chromosome_threshold']}, f)
            print(f"  > Fold {fold_idx+1} thresholds saved to: {threshold_save_path}")

            print(f"\n--- Evaluating Fold {fold_idx + 1} Model on its Full Validation Set ---")
            
            # 9. Scatter validation predictions back to full-size tensors for metric calculation
            final_scores_val_fold = torch.from_numpy(utils.apply_thresholds(y_probs_val_fold.numpy(), fold_params))
            
            # Global indices of the nodes that were in the validation subgraph
            val_node_indices_global = val_node_indices.cpu()

            raw_probs_full = torch.zeros_like(data.y, dtype=torch.float, device='cpu')
            final_scores_full = torch.zeros_like(data.y, dtype=torch.float, device='cpu')

            raw_probs_full.scatter_(0, val_fold_labeled_global.cpu().unsqueeze(1).expand(-1, 2), y_probs_val_fold)
            final_scores_full.scatter_(0, val_fold_labeled_global.cpu().unsqueeze(1).expand(-1, 2), final_scores_val_fold)


            masks_validate = torch.zeros(data.num_nodes, dtype=torch.float32, device='cpu')
            masks_validate[val_fold_labeled_global] = 1.0

            calculate_and_print_metrics(final_scores_full.to(device), raw_probs_full.to(device), data, masks_validate.to(device), G, node_list, verbose=False)




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