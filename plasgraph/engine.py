import numpy as np
import matplotlib.pyplot as plt
import torch
import optuna
import os 
import random
import time

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
    optuna objective function for hyperparameter optimization using k-fold cross-validation
    """
    # store the process id for debugging in parallel environments
    trial.set_user_attr("pid", os.getpid())

    # create a copy of base parameters to modify for this specific trial
    trial_params_dict = parameters._params.copy()
    # use the 'trial' object to suggest hyperparameter values for optuna to optimize
    trial_params_dict['l2_reg'] = trial.suggest_float("l2_reg", 1e-5, 1e-3, log=True)
    trial_params_dict['n_channels'] = trial.suggest_int("n_channels", 8, 64, step=16)
    trial_params_dict['n_gnn_layers'] = trial.suggest_int("n_gnn_layers", 2, 6)
    trial_params_dict['dropout_rate'] = trial.suggest_float("dropout_rate", 0.0, 0.3)
    trial_params_dict['gradient_clipping'] = trial.suggest_float("gradient_clipping", 1.0, 10.0, log=True)
    trial_params_dict['edge_gate_hidden_dim'] = trial.suggest_int("edge_gate_hidden_dim", 8, 32, step=8)
    trial_params_dict['n_channels_preproc'] = trial.suggest_int("n_channels_preproc", 10, 25, step=5)
    trial_params_dict['edge_gate_depth'] = trial.suggest_int("edge_gate_depth", 2, 6)
    trial_params_dict['batch_size'] = trial.suggest_categorical('batch_size', [64, 128 , 256, 512, 1024])
    # ensure subsequent hop neighbors are less than or equal to the first hop
    first_hop_val = trial.suggest_int("neighbors_first_hop", 10, 50, step=10)
    trial_params_dict['first_hop_neighbors'] = first_hop_val
    trial_params_dict['subsequent_hop_neighbors'] = trial.suggest_int("neighbors_subsequent_hops", 10, first_hop_val, step=5)
    trial_params_dict['learning_rate'] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    # construct the list of neighbors to sample for each gnn layer
    n_layers = trial_params_dict['n_gnn_layers']
    first_hop_neighbors = trial_params_dict['first_hop_neighbors']
    subsequent_hop_neighbors = trial_params_dict['subsequent_hop_neighbors']
    neighbors_list = [first_hop_neighbors] + [subsequent_hop_neighbors] * (n_layers - 1)

    # create a temporary config object for this trial's hyperparameters
    trial_config_obj = config.config(parameters.config_file_path)
    trial_config_obj._params = trial_params_dict

    device = accelerator.device
    # move the graph data object to the correct device
    data = data.to(device)

    # create mappings from sample id to its index in the batch tensor
    unique_samples_from_graph = sorted(list(set(node_id.split(':')[0] for node_id in node_list)))
    sample_to_batch_idx = {sample_id: i for i, sample_id in enumerate(unique_samples_from_graph)}

    fold_aurocs = []    # list to store the final score for each fold
    labeled_nodes_set = set(labeled_indices)

    # Determine the loop structure based on the split mode
    if trial_config_obj['validation_split_mode'] == 'node_level_random':
        if trial_config_obj['training_mode'] == 'k-fold':
            trial.set_user_attr("warning", "node_level_random not compatible with k-fold in HPO. Running a single 80/20 node split.")
        
        splits_to_iterate = [('node_level', None, None)]
        
    elif trial_config_obj['validation_split_mode'] == 'stratified':
        splits_to_iterate = [('sample_level', split_data) for split_data in sample_splits]
    
    else:
        raise ValueError(f"Unknown validation_split_mode: {trial_config_obj['validation_split_mode']}")

    for fold_idx, split_info in enumerate(splits_to_iterate):
        
        # --- ADD THIS UNPACKING LOGIC ---
        if split_info[0] == 'sample_level':
            # --- This is the EXISTING sample-level logic (UNCHANGED, just indented) ---
            train_sample_indices, val_sample_indices = split_info[1]
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
            
        elif split_info[0] == 'node_level':
            print("  Using node-level random 80/20 split (TF-style mimic).")
            # Set seed just like the TF script did, using the `random` module
            random.seed(trial_config_obj["random_seed"])

            train_node_indices_split = []
            val_node_indices_split = []

            # Iterate over all labeled indices and split using random.random()
            for node_idx in labeled_indices:
                if random.random() > 0.8:
                    val_node_indices_split.append(node_idx) # Goes to validation
                else:
                    train_node_indices_split.append(node_idx) # Goes to train

            train_data = data # Use full graph
            val_data = data   # Use full graph
            local_train_seed_indices = torch.tensor(train_node_indices_split, dtype=torch.long)
            local_val_seed_indices = torch.tensor(val_node_indices_split, dtype=torch.long)
            # --- END OF BLOCK ---

        if len(local_train_seed_indices) == 0 or len(local_val_seed_indices) == 0:
            continue
        
        # initialize the model based on the config
        if trial_config_obj['model_type'] == 'GCNModel':
            model = GCNModel(trial_config_obj).to(device)
        elif trial_config_obj['model_type'] == 'GGNNModel':
            model = GGNNModel(trial_config_obj).to(device)

        # setup optimizer, scheduler, and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=trial_config_obj['learning_rate'], weight_decay=trial_config_obj['l2_reg'])
        # learning rate scheduler to reduce the learning rate on validation loss plateau
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=trial_config_obj['scheduler_factor'], patience=trial_config_obj['scheduler_patience'])
        # loss function
        if trial_config_obj['output_activation'] == 'none':
            criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
        else:
            criterion = torch.nn.BCELoss(reduction='sum')

        # --- MODIFICATION START: Conditional data loading and training loop ---
        
        # initialize variables for early stopping within the fold
        best_val_loss_for_fold = float("inf")
        patience_for_fold = 0
        best_model_state_for_fold = None
        y_probs_fold = torch.empty(0) # Init empty tensors
        y_true_fold = torch.empty(0)
        
        if trial_config_obj['training_style'] == 'neighbor_sampling':
            # NeighborLoader for the training set of this fold
            train_loader = NeighborLoader(
                train_data,     # the subgraph for this training fold
                input_nodes=local_train_seed_indices.to(device),    # seed nodes are the labeled nodes
                num_neighbors=neighbors_list,   # neighborhood sampling sizes per layer
                shuffle=True,    # shuffle the data at each epoch
                batch_size=trial_config_obj['batch_size'],    # use the batch size suggested by Optuna
                num_workers=trial_config_obj['num_workers'],
                pin_memory=True,     # for faster data transfer to GPU
            )

            # NeighborLoader for the validation set of this fold
            val_loader = NeighborLoader(
                val_data,
                input_nodes=local_val_seed_indices.to(device),
                num_neighbors=neighbors_list,
                shuffle=False,     # no need to shuffle validation data
                batch_size=trial_config_obj['batch_size'],
                num_workers=trial_config_obj['num_workers'],
                pin_memory=True,
            )

            # --- training loop for the number of epochs specified for hpo trials ---
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

                # --- validation loop ---
                model.eval()
                total_val_loss = 0
                with torch.no_grad():                           # disable gradient computation for validation
                    for batch in val_loader:
                        batch = batch.to(device)
                        outputs = model(batch)
                        val_loss = criterion(outputs[:batch.batch_size], batch.y[:batch.batch_size])
                        total_val_loss += val_loss.item()
                
                # --- THIS IS LINE 1 (FIXED) ---
                avg_val_loss = total_val_loss     # calculate the total validation loss for the epoch
                scheduler.step(avg_val_loss)    # update learning rate scheduler

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
                
            # --- final evaluation for the fold using the best model state ---
            if best_model_state_for_fold is not None:
                # create a new model instance for inference
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
                        # convert logits to probabilities using sigmoid
                        if trial_config_obj['output_activation'] == 'none':
                            probs = torch.sigmoid(outputs[:batch.batch_size])
                        else:
                            probs = outputs[:batch.batch_size] # Already probabilities
                        all_probs_fold.append(probs)
                        all_true_fold.append(batch.y[:batch.batch_size])

                if not all_probs_fold:
                    continue
                
                # concatenate results from all validation batches
                y_probs_fold = torch.cat(all_probs_fold).cpu()
                y_true_fold = torch.cat(all_true_fold).cpu()
        
        elif trial_config_obj['training_style'] == 'full_graph':
            # --- FULL GRAPH TRAINING ---
            
            # Move full subgraphs to device
            train_data = train_data.to(device)
            val_data = val_data.to(device)

            # --- training loop for the number of epochs specified for hpo trials ---
            for epoch in range(parameters["epochs_trials"]):
                # set the model to training mode
                model.train()
                optimizer.zero_grad()
                outputs = model(train_data) # Full forward pass
                # Calculate loss ONLY on labeled nodes
                loss = criterion(outputs[local_train_seed_indices], train_data.y[local_train_seed_indices])
                loss.backward()
                fix_gradients(trial_config_obj, model)
                optimizer.step()

                # --- validation loop ---
                model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    outputs = model(val_data) # Full forward pass
                    val_loss = criterion(outputs[local_val_seed_indices], val_data.y[local_val_seed_indices])
                    total_val_loss = val_loss.item()
                
                avg_val_loss = total_val_loss # Not an average, just the total loss for the epoch
                scheduler.step(avg_val_loss)

                # check for improvement in validation loss for early stopping
                if avg_val_loss < best_val_loss_for_fold:
                    best_val_loss_for_fold = avg_val_loss
                    patience_for_fold = 0
                    best_model_state_for_fold = copy.deepcopy(model.state_dict())
                else:
                    patience_for_fold += 1
                    if patience_for_fold >= parameters["early_stopping_patience"]:
                        break

            # --- final evaluation for the fold using the best model state ---
            if best_model_state_for_fold is not None:
                if trial_config_obj['model_type'] == 'GCNModel':
                    inference_model = GCNModel(trial_config_obj).to(device)
                else:
                    inference_model = GGNNModel(trial_config_obj).to(device)
                
                inference_model.load_state_dict(best_model_state_for_fold)
                inference_model.eval() 

                all_probs_fold, all_true_fold = [], []
                with torch.no_grad():
                    outputs = inference_model(val_data)
                    # Get probs and true labels ONLY for labeled nodes
                    if trial_config_obj['output_activation'] == 'none':
                        probs = torch.sigmoid(outputs[local_val_seed_indices])
                    else:
                        probs = outputs[local_val_seed_indices] # Already probabilities
                    all_probs_fold.append(probs)
                    all_true_fold.append(val_data.y[local_val_seed_indices])

                if not all_probs_fold:
                    continue

                y_probs_fold = torch.cat(all_probs_fold).cpu()
                y_true_fold = torch.cat(all_true_fold).cpu()
        
        else:
            raise ValueError(f"Unknown training_style: {trial_config_obj['training_style']}")

        # --- MODIFICATION END ---
            
        # --- This part (AUROC calculation) stays outside the if/else ---
        if best_model_state_for_fold is not None and y_probs_fold.numel() > 0:
            # --- calculate auroc robustly, defaulting to 0.5 if a class is missing ---
            auroc_p, auroc_c = 0.5, 0.5  
            # plasmid AUROC
            y_true_p = y_true_fold[:, 0].numpy()
            y_probs_p = y_probs_fold[:, 0].numpy()
            if len(np.unique(y_true_p)) > 1:
                auroc_p = roc_auc_score(y_true_p, y_probs_p)
            # chromosome AUROC
            y_true_c = y_true_fold[:, 1].numpy()
            y_probs_c = y_probs_fold[:, 1].numpy()
            if len(np.unique(y_true_c)) > 1:
                auroc_c = roc_auc_score(y_true_c, y_probs_c)

             # use average of plasmid and chromosome auroc
            avg_auroc_for_fold = (auroc_p + auroc_c) / 2.0
            fold_aurocs.append(avg_auroc_for_fold)

            # report intermediate value to optuna for pruning
            trial.report(avg_auroc_for_fold, fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    # final objective value is the mean auroc across all folds
    final_objective_value = np.mean(fold_aurocs) if fold_aurocs else 0.0
    return final_objective_value




def train_final_model(accelerator, parameters, data, sample_splits, all_sample_ids, labeled_indices, log_dir, G, node_list):
    """
    Trains k-fold models in parallel across available processes (e.g., multiple GPUs)
    and saves each model for ensembling.
    """
    start_time_kfold = time.perf_counter()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(accelerator.device)

    accelerator.print("\n" + "="*60)
    accelerator.print("💾 Training K-Fold Ensemble Models (Parallel / Distributed)")
    accelerator.print("="*60)

    device = accelerator.device
    data = data.to(device)

    # create a directory to store the trained model files for the ensemble
    ensemble_models_dir = os.path.join(log_dir, "ensemble_models")
    # create a directory to store plots of the training history for each cross-validation fold
    cv_plots_dir = os.path.join(log_dir, "cv_fold_plots")

    if accelerator.is_main_process:
        os.makedirs(ensemble_models_dir, exist_ok=True)
        accelerator.print(f"Ensemble models will be saved to: {ensemble_models_dir}")
        os.makedirs(cv_plots_dir, exist_ok=True)
    
    # All processes wait here to ensure directories are created
    # before they try to save files.
    accelerator.wait_for_everyone()

    # setup for data splitting (same logic as in `objective` function)
    unique_samples_from_graph = sorted(list(set(node_id.split(':')[0] for node_id in node_list)))
    sample_to_batch_idx = {sample_id: i for i, sample_id in enumerate(unique_samples_from_graph)}
    labeled_nodes_set = set(labeled_indices)

    # define the neighborhood sampling sizes
    n_layers = parameters['n_gnn_layers']
    first_hop_neighbors = parameters['neighbors_first_hop']
    subsequent_hop_neighbors = parameters['neighbors_subsequent_hops']
    neighbors_list = [first_hop_neighbors] + [subsequent_hop_neighbors] * (n_layers - 1)

    for fold_idx, split_data in enumerate(sample_splits):
        # --- ADD THIS BLOCK ---
        # Assign this fold to a specific process.
        # This makes each process take a different subset of folds.
        if fold_idx % accelerator.num_processes != accelerator.process_index:
            continue
        # --- END OF ADDED BLOCK ---
        
        accelerator.print(f"\n--- Running Fold {fold_idx + 1}/{len(sample_splits)} ---")
        
        # --- ADD THIS CONDITIONAL LOGIC ---
        if split_data is None:
            accelerator.print("  Using node-level random 80/20 split (TF-style mimic).")
            
            # Set seed just like the TF script did, using the `random` module
            random.seed(parameters["random_seed"])

            train_node_indices = []
            val_node_indices = []
            
            # Iterate over all labeled indices and split using random.random()
            for node_idx in labeled_indices:
                if random.random() > 0.8:
                    val_node_indices.append(node_idx) # Goes to validation
                else:
                    train_node_indices.append(node_idx) # Goes to train

            train_data = data # Use full graph
            val_data = data   # Use full graph
            
            local_train_seed_indices = torch.tensor(train_node_indices, dtype=torch.long)
            local_val_seed_indices = torch.tensor(val_node_indices, dtype=torch.long)
            
            val_fold_labeled_global = local_val_seed_indices
            
            accelerator.print(f"  Total Labeled Nodes: {len(labeled_indices)} | Train Nodes: {len(local_train_seed_indices)} | Val Nodes: {len(local_val_seed_indices)}")
            # --- END OF BLOCK ---

        else:
            # --- START OF EXISTING SAMPLE-LEVEL SPLIT LOGIC (UNCHANGED, JUST INDENTED) ---
            accelerator.print("  Using stratified sample-level 80/20 split.")
            train_sample_indices, val_sample_indices = split_data
            # get sample ids and batch indices for this fold
            train_samples = all_sample_ids[train_sample_indices]
            val_samples = all_sample_ids[val_sample_indices]
            train_batch_indices = [sample_to_batch_idx[sid] for sid in train_samples]
            val_batch_indices = [sample_to_batch_idx[sid] for sid in val_samples]
            # create node masks and subgraphs for this fold
            train_mask = torch.isin(data.batch, torch.tensor(train_batch_indices, device=data.batch.device))
            val_mask = torch.isin(data.batch, torch.tensor(val_batch_indices, device=data.batch.device))
            train_node_indices = torch.where(train_mask)[0]
            val_node_indices = torch.where(val_mask)[0]
            train_data = data.subgraph(train_node_indices)
            val_data = data.subgraph(val_node_indices)
            accelerator.print(f"  Train samples: {len(train_samples)}, Nodes: {train_data.num_nodes} | Val samples: {len(val_samples)}, Nodes: {val_data.num_nodes}")
            # get local seed indices for the neighbor loaders
            train_nodes_global_set = set(train_node_indices.cpu().numpy())
            val_nodes_global_set = set(val_node_indices.cpu().numpy())
            train_fold_labeled_global = torch.tensor(sorted(list(train_nodes_global_set.intersection(labeled_nodes_set))), dtype=torch.long)
            val_fold_labeled_global = torch.tensor(sorted(list(val_nodes_global_set.intersection(labeled_nodes_set))), dtype=torch.long)
            global_to_local_train_map = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(train_node_indices)}
            global_to_local_val_map = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(val_node_indices)}
            local_train_seed_indices = torch.tensor([global_to_local_train_map[global_idx.item()] for global_idx in train_fold_labeled_global], dtype=torch.long)
            local_val_seed_indices = torch.tensor([global_to_local_val_map[global_idx.item()] for global_idx in val_fold_labeled_global], dtype=torch.long)

        # initialize model, optimizer, scheduler, and criterion
        if parameters['model_type'] == 'GCNModel':
            model = GCNModel(parameters).to(device)
        elif parameters['model_type'] == 'GGNNModel':
            model = GGNNModel(parameters).to(device)

        optimizer = optim.Adam(model.parameters(), lr=parameters['learning_rate'], weight_decay=parameters['l2_reg'])
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=parameters['scheduler_factor'], patience=parameters['scheduler_patience'], verbose=True)
        if parameters['output_activation'] == 'none':
            criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
        else:
            criterion = torch.nn.BCELoss(reduction='sum')

        # variables for training loop and early stopping
        best_val_loss_for_fold, patience_for_fold, best_epoch_for_fold = float("inf"), 0, 0
        best_model_state_for_fold = None
        train_losses_fold, val_losses_fold = [], []
        plot_frequency = max(1, parameters["epochs"] // 10)
        
        num_workers = parameters['num_workers']

        # --- MODIFICATION START: Conditional data loading and training loop ---
        
        # We need val_loader for threshold setting later, regardless of training style
        val_loader = NeighborLoader(
            val_data,
            input_nodes=local_val_seed_indices.to(device),
            num_neighbors=neighbors_list,
            shuffle=False,
            batch_size=parameters['batch_size'],
            num_workers=num_workers,
            pin_memory=True,
        )

        if parameters['training_style'] == 'neighbor_sampling':
            # setup data loaders for this fold
            train_loader = NeighborLoader(
                train_data,
                input_nodes=local_train_seed_indices.to(device),
                num_neighbors=neighbors_list,
                shuffle=True,
                batch_size=parameters['batch_size'],
                num_workers=num_workers,
                pin_memory=True,
            )

            # --- main training loop (NEIGHBOR SAMPLING) ---
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
                
                # --- THIS IS LINE 2 (FIXED) ---
                avg_train_loss = total_train_loss
                train_losses_fold.append(avg_train_loss)

                # plot gradient magnitudes periodically for the first fold to diagnose training
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
                
                # --- THIS IS LINE 3 (FIXED) ---
                avg_val_loss = total_val_loss
                val_losses_fold.append(avg_val_loss)
                scheduler.step(avg_val_loss)

                # logging progress
                if (epoch + 1) % 10 == 0 or (epoch + 1) == parameters["epochs"]:
                    accelerator.print(f"Epoch {epoch + 1}/{parameters['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

                # early stopping logic
                if avg_val_loss < best_val_loss_for_fold:
                    best_val_loss_for_fold = avg_val_loss
                    best_epoch_for_fold = epoch + 1
                    patience_for_fold = 0
                    best_model_state_for_fold = copy.deepcopy(model.state_dict())
                else:
                    patience_for_fold += 1
                    if patience_for_fold >= parameters["early_stopping_patience_retrain"]:
                        accelerator.print(f"Early stopping triggered at epoch {epoch + 1}")
                        break
        
        elif parameters['training_style'] == 'full_graph':
            # --- MAIN TRAINING LOOP (FULL GRAPH) ---
            
            # Move full subgraphs to device
            train_data = train_data.to(device)
            val_data = val_data.to(device)
            
            for epoch in range(parameters["epochs"]):
                model.train()
                optimizer.zero_grad()
                outputs = model(train_data)
                loss = criterion(outputs[local_train_seed_indices], train_data.y[local_train_seed_indices])
                loss.backward()
                utils.fix_gradients(parameters, model)
                optimizer.step()
                
                avg_train_loss = loss.item() # This is the total loss, not avg per node
                train_losses_fold.append(avg_train_loss)

                # plot gradient magnitudes periodically
                if fold_idx == 0: 
                    utils.plot_gradient_magnitudes(utils.get_gradient_magnitudes(model), epoch + 1, cv_plots_dir, plot_frequency=plot_frequency)
                
                model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    outputs = model(val_data)
                    val_loss = criterion(outputs[local_val_seed_indices], val_data.y[local_val_seed_indices])
                    total_val_loss = val_loss.item()

                avg_val_loss = total_val_loss # This is the total loss
                val_losses_fold.append(avg_val_loss)
                scheduler.step(avg_val_loss)

                # logging progress
                if (epoch + 1) % 10 == 0 or (epoch + 1) == parameters["epochs"]:
                    accelerator.print(f"Epoch {epoch + 1}/{parameters['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

                # early stopping logic
                if avg_val_loss < best_val_loss_for_fold:
                    best_val_loss_for_fold = avg_val_loss
                    best_epoch_for_fold = epoch + 1
                    patience_for_fold = 0
                    best_model_state_for_fold = copy.deepcopy(model.state_dict())
                else:
                    patience_for_fold += 1
                    if patience_for_fold >= parameters["early_stopping_patience_retrain"]:
                        accelerator.print(f"Early stopping triggered at epoch {epoch + 1}")
                        break
        else:
            raise ValueError(f"Unknown training_style: {parameters['training_style']}")
        
        # --- MODIFICATION END ---
        
        accelerator.print(f"Fold {fold_idx + 1} finished. Best epoch: {best_epoch_for_fold} with val loss: {best_val_loss_for_fold:.4f}")

        # --- after training a fold: save model, plot history, and determine thresholds ---
        if best_model_state_for_fold:
            # save the best model state for this fold
            model_save_path = os.path.join(ensemble_models_dir, f"fold_{fold_idx + 1}_model.pt")
            torch.save(best_model_state_for_fold, model_save_path)
            accelerator.print(f"  > Saved best model for fold {fold_idx + 1} to: {model_save_path}")
            # plot and save the training and validation loss history for this fold
            plt.figure(figsize=(10, 6)); plt.plot(train_losses_fold, label='Train Loss'); plt.plot(val_losses_fold, label='Validation Loss'); plt.axvline(x=best_epoch_for_fold - 1, color='r', linestyle='--', label=f'Best Epoch ({best_epoch_for_fold})'); plt.title(f'Fold {fold_idx + 1} Training History'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True); plot_path = os.path.join(cv_plots_dir, f"cv_loss_fold_{fold_idx + 1}.png"); plt.savefig(plot_path); plt.close()
            accelerator.print(f"  > Saved loss plot to: {plot_path}")

            # --- determine and save optimal classification thresholds on the validation set ---
            if parameters['model_type'] == 'GCNModel':
                inference_model = GCNModel(parameters).to(device)
            else:
                inference_model = GGNNModel(parameters).to(device)
            inference_model.load_state_dict(best_model_state_for_fold)
            inference_model.eval()

            # --- MODIFICATION START: Conditional validation data gathering ---
            all_probs_val, all_true_val = [], []
            with torch.no_grad():
                if parameters['training_style'] == 'neighbor_sampling':
                    for batch in val_loader: # Use the val_loader defined above
                        batch = batch.to(device)
                        outputs = inference_model(batch)
                        if parameters['output_activation'] == 'none':
                            probs = torch.sigmoid(outputs[:batch.batch_size])
                        else:
                            probs = outputs[:batch.batch_size]
                        all_probs_val.append(probs)
                        all_true_val.append(batch.y[:batch.batch_size])
                elif parameters['training_style'] == 'full_graph':
                    val_data = val_data.to(device) # Ensure val_data is on device
                    outputs = inference_model(val_data)
                    if parameters['output_activation'] == 'none':
                        probs = torch.sigmoid(outputs[local_val_seed_indices])
                    else:
                        probs = outputs[local_val_seed_indices]
                    all_probs_val.append(probs)
                    all_true_val.append(val_data.y[local_val_seed_indices])
            # --- MODIFICATION END ---
            
            y_probs_val_fold = torch.cat(all_probs_val).cpu()
            y_true_val_fold = torch.cat(all_true_val).cpu()

            # find and set the best thresholds based on f1-score
            fold_params = copy.deepcopy(parameters)
            utils.set_thresholds_from_predictions(y_true_val_fold, y_probs_val_fold, fold_params, log_dir)
            
            # save the determined thresholds to a yaml file for this fold
            threshold_save_path = os.path.join(ensemble_models_dir, f"fold_{fold_idx + 1}_thresholds.yaml")
            with open(threshold_save_path, 'w') as f:
                yaml.dump({'plasmid_threshold': fold_params['plasmid_threshold'], 'chromosome_threshold': fold_params['chromosome_threshold']}, f)
            accelerator.print(f"  > Fold {fold_idx+1} thresholds saved to: {threshold_save_path}")

            accelerator.print(f"\n--- Evaluating Fold {fold_idx + 1} Model on its Full Validation Set ---")
            
            # evaluate the fold's model on its validation set using the new thresholds
            final_scores_val_fold = torch.from_numpy(utils.apply_thresholds(y_probs_val_fold.numpy(), fold_params))
            
            # create full-size tensors to hold the validation predictions for the metrics function
            raw_probs_full = torch.zeros_like(data.y, dtype=torch.float, device='cpu')
            final_scores_full = torch.zeros_like(data.y, dtype=torch.float, device='cpu')
            # scatter the validation predictions back to their original global node indices
            raw_probs_full.scatter_(0, val_fold_labeled_global.cpu().unsqueeze(1).expand(-1, 2), y_probs_val_fold)
            final_scores_full.scatter_(0, val_fold_labeled_global.cpu().unsqueeze(1).expand(-1, 2), final_scores_val_fold)
            # create a mask for the validation nodes
            masks_validate = torch.zeros(data.num_nodes, dtype=torch.float32, device='cpu')
            masks_validate[val_fold_labeled_global] = 1.0
            # calculate and print the final metrics for this fold
            calculate_and_print_metrics(final_scores_full.to(device), raw_probs_full.to(device), data, masks_validate.to(device), G, node_list, verbose=False)

    # After the loop, wait for all processes to finish their assigned folds
    # before allowing the script to continue.
    accelerator.wait_for_everyone()
    end_time_kfold = time.perf_counter()

    # --- ADD THIS BLOCK ---
    # Get peak memory for this process
    local_peak_mem_bytes = 0
    if torch.cuda.is_available():
        local_peak_mem_bytes = torch.cuda.max_memory_allocated(accelerator.device)
    
    # Create a tensor on the device for reduction
    local_peak_mem_tensor = torch.tensor(local_peak_mem_bytes, device=accelerator.device, dtype=torch.float)
    
    # Reduce across all processes to find the maximum peak
    global_peak_mem_tensor = accelerator.reduce(local_peak_mem_tensor, reduction='max')
    # --- END OF ADDED BLOCK ---

    if accelerator.is_main_process:
        elapsed_seconds = end_time_kfold - start_time_kfold
        
        # --- THIS PART IS NEW ---
        peak_mem_gb = global_peak_mem_tensor.item() / (1024**3)
        accelerator.print("\n" + "="*60)
        accelerator.print(f"⏱️ [PERF] Total k-fold training (task-parallel):")
        accelerator.print(f"  > Total Time: {elapsed_seconds:.2f} seconds")
        accelerator.print(f"  > Max Peak VRAM: {peak_mem_gb:.2f} GB")
        accelerator.print("="*60)
        # --- END OF NEW PART ---

    accelerator.print("✅ All parallel fold-training processes finished. Synchronization complete.")
    accelerator.print("="*66)



# --- functions for inference on new, unseen graphs ---
import pandas as pd
import torch_geometric

from .data import read_single_graph
from .utils import apply_thresholds, pair_to_label

def predict(model, graph, parameters):
    """applies a trained model to a graph to get predictions"""
    device = next(model.parameters()).device
    graph = graph.to(device)
    model.eval()
    with torch.no_grad():
        output = model(graph)
    
    preds_np = output.cpu().numpy()
    preds_clipped = np.clip(preds_np, 0, 1)
    
    preds_final = apply_thresholds(preds_clipped, parameters)
    preds_final = np.clip(preds_final, 0, 1)
    
    return preds_final

def classify_graph(model, parameters, graph_file, file_prefix, sample_id):
    """
    loads a single graph file, classifies its nodes, and returns a results dataframe
    """
    G = read_single_graph(file_prefix, graph_file, sample_id, parameters['minimum_contig_length'])
    node_list = list(G)

    for u, v in G.edges():
        kmer_u = np.array(G.nodes[u]["kmer_counts_norm"])
        kmer_v = np.array(G.nodes[v]["kmer_counts_norm"])
        dot_product = np.dot(kmer_u, kmer_v)
        G.edges[u, v]["kmer_dot_product"] = dot_product

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
        batch=torch.zeros(x.shape[0], dtype=torch.long)
    )

    preds = predict(model, data_obj, parameters)

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

