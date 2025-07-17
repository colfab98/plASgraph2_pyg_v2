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

    # set a user attribute to store the process ID, useful for debugging in parallel environments
    trial.set_user_attr("pid", os.getpid())

    # use the 'trial' object to suggest hyperparameter values for Optuna to optimize
    trial_params_dict = parameters._params.copy()
    # trial_params_dict['l2_reg'] = trial.suggest_float("l2_reg", 1e-6, 1e-2, log=True)
    trial_params_dict['n_channels'] = trial.suggest_int("n_channels", 16, 64, step=16)
    trial_params_dict['n_gnn_layers'] = trial.suggest_int("n_gnn_layers", 4, 8)
    # trial_params_dict['dropout_rate'] = trial.suggest_float("dropout_rate", 0.0, 0.3)
    trial_params_dict['gradient_clipping'] = trial.suggest_float("gradient_clipping", 0.0, 0.3)
    trial_params_dict['edge_gate_hidden_dim'] = trial.suggest_int("edge_gate_hidden_dim", 16, 64, step=8)
    trial_params_dict['n_channels_preproc'] = trial.suggest_int("n_channels_preproc", 5, 15, step=5)
    trial_params_dict['edge_gate_depth'] = trial.suggest_int("edge_gate_depth", 2, 6)
    trial_params_dict['batch_size'] = trial.suggest_categorical('batch_size', [2048, 4096, 8192])

    # define the neighborhood sampling sizes based on the suggested number of GNN layers
    n_layers = trial_params_dict['n_gnn_layers']
    neighbors_list = [15] + [10] * (n_layers - 1)
    
    # temporary config object using the suggested hyperparameters for this trial
    trial_config_obj = config.config(parameters.config_file_path)
    trial_config_obj._params = trial_params_dict

    device = accelerator.device
    # move the graph data object to the correct device
    data = data.to(device)

    fold_aurocs = []

    # k-fold cross-validation loop
    for fold_idx, (train_idx, val_idx) in enumerate(splits):

        # get the actual node indices for the training and validation sets for this fold
        train_fold_indices = labeled_indices[train_idx]
        val_fold_indices = labeled_indices[val_idx]
        
        # initialize the model
        if trial_config_obj['model_type'] == 'GCNModel':
            model = GCNModel(trial_config_obj).to(device)
        elif trial_config_obj['model_type'] == 'GGNNModel':
            model = GGNNModel(trial_config_obj).to(device)

        # Adam optimizer with the trial's learning rate and L2 regularization
        optimizer = torch.optim.Adam(model.parameters(), lr=trial_config_obj['learning_rate'], weight_decay=trial_config_obj['l2_reg'])
        # loss function
        criterion = torch.nn.BCEWithLogitsLoss()

        # NeighborLoader for the training set of this fold
        train_loader = NeighborLoader(
            data,
            input_nodes=torch.from_numpy(train_fold_indices).to(data.x.device), # specify seed nodes
            num_neighbors=neighbors_list,                                       # neighborhood sampling sizes
            shuffle=True,                                                       # shuffle the data at each epoch
            batch_size=trial_config_obj['batch_size'],                          # use the batch size suggested by Optuna
            num_workers=trial_config_obj['num_workers'],
            pin_memory=True,                                                    # for faster data transfer to GPU
        )

        # NeighborLoader for the validation set of this fold
        val_loader = NeighborLoader(
            data,
            input_nodes=torch.from_numpy(val_fold_indices).to(data.x.device),
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
                loss = criterion(outputs, batch.y)          # calculate the loss on the mini-batch
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
                    val_loss = criterion(outputs, batch.y)
                    total_val_loss += val_loss.item()
            
            # calculate the average validation loss for the epoch
            avg_val_loss = total_val_loss / len(val_loader)

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
            
            # get predictions (probabilities) on the entire validation set for this fold
            all_probs, all_true = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device) 
                    outputs = inference_model(batch)
                    all_probs.append(torch.sigmoid(outputs))
                    all_true.append(batch.y)
            
            # concatenate batch results into single numpy arrays
            y_probs_fold = torch.cat(all_probs).cpu().numpy()
            y_true_fold = torch.cat(all_true).cpu().numpy()

            # calculate the AUROC score for both plasmid and chromosome predictions
            auroc_plasmid = 0.5
            auroc_chromosome = 0.5
            # ensure there are both positive and negative samples before calculating AUROC
            if len(np.unique(y_true_fold[:, 0])) > 1:
                auroc_plasmid = roc_auc_score(y_true_fold[:, 0], y_probs_fold[:, 0])
            if len(np.unique(y_true_fold[:, 1])) > 1:
                auroc_chromosome = roc_auc_score(y_true_fold[:, 1], y_probs_fold[:, 1])

            # calculate the average AUROC for the fold
            avg_auroc_for_fold = (auroc_plasmid + auroc_chromosome) / 2.0
            fold_aurocs.append(avg_auroc_for_fold)
        else:
            fold_aurocs.append(0.0)

        # report the median of the AUROCs collected so far to Optuna's pruner
        trial.report(np.median(fold_aurocs), fold_idx)
        # check if the pruner suggests stopping this trial early
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # after all folds are complete, calculate the final score for the trial
    # the median AUROC across all folds is used as the robust performance metric
    average_auroc = np.median(fold_aurocs)
    return average_auroc



def train_final_model(parameters, data, splits, labeled_indices, log_dir, G, node_list):
    """
    Trains K-fold models on a single GPU and saves each one for ensembling.
    This version is aligned with the trial training logic for consistency.
    """
    print("\n" + "="*60)
    print("💾 Training K-Fold Ensemble Models (Single-GPU)")
    print("="*60)

    # determine the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    data = data.to(device)

    # create a directory to store the trained model files for the ensemble
    ensemble_models_dir = os.path.join(log_dir, "ensemble_models")
    os.makedirs(ensemble_models_dir, exist_ok=True)
    print(f"Ensemble models will be saved to: {ensemble_models_dir}")

    # create a directory to store plots of the training history for each cross-validation fold
    cv_plots_dir = os.path.join(log_dir, "cv_fold_plots")
    os.makedirs(cv_plots_dir, exist_ok=True)

    # define the neighborhood sampling sizes
    n_layers = parameters['n_gnn_layers']
    neighbors_list = [15] + [10] * (n_layers - 1)

    # start the main loop to iterate through each cross-validation fold
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- Running Fold {fold_idx + 1}/{len(splits)} ---")

        # get the global indices for the training and validation nodes for the current fold
        train_fold_indices = labeled_indices[train_idx]
        val_fold_indices = labeled_indices[val_idx]

        if parameters['model_type'] == 'GCNModel':
            model = GCNModel(parameters).to(device)
        elif parameters['model_type'] == 'GGNNModel':
            model = GGNNModel(parameters).to(device)

        optimizer = optim.Adam(model.parameters(), lr=parameters['learning_rate'], weight_decay=parameters['l2_reg'])
        criterion = torch.nn.BCEWithLogitsLoss()

        # on MPS, move data to CPU for the loader. On CUDA, this does nothing
        data = data.to('cpu') if device.type == 'mps' else data

        # If on Mac/MPS, disable multiprocessing in DataLoader to prevent crash.
        # On HPC (CUDA), the original num_workers from config will be used.
        num_workers = parameters['num_workers']
        if device.type == 'mps':
            num_workers = 0

        train_loader = NeighborLoader(
            data,
            input_nodes=torch.from_numpy(train_fold_indices).to(device),
            num_neighbors=neighbors_list,
            shuffle=True,
            batch_size=parameters['batch_size'],
            num_workers=num_workers,
            pin_memory=True,
        )

        val_loader = NeighborLoader(
            data,
            input_nodes=torch.from_numpy(val_fold_indices).to(device),
            num_neighbors=neighbors_list,
            shuffle=False,
            batch_size=parameters['batch_size'],
            num_workers=num_workers,
            pin_memory=True,
        )

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

            # gradient magnitude plotting (first fold only)
            if fold_idx == 0: 
                grad_data = utils.get_gradient_magnitudes(model)
                utils.plot_gradient_magnitudes(grad_data, epoch + 1, cv_plots_dir)
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses_fold.append(avg_train_loss)

            # validation loop
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

            if parameters['model_type'] == 'GCNModel':
                inference_model = GCNModel(parameters).to(device)
            else:
                inference_model = GGNNModel(parameters).to(device)
            
            inference_model.load_state_dict(best_model_state_for_fold)
            inference_model.eval()

            # get predictions (probabilities) on the validation set for this fold
            all_probs_val, all_true_val = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    outputs = inference_model(batch)
                    all_probs_val.append(torch.sigmoid(outputs))
                    all_true_val.append(batch.y)
            
            # concatenate the lists of tensors into single tensors
            y_probs_val_fold = torch.cat(all_probs_val).cpu()
            y_true_val_fold = torch.cat(all_true_val).cpu()

            # determine the optimal classification thresholds from the validation predictions
            fold_params = copy.deepcopy(parameters)
            utils.set_thresholds_from_predictions(y_true_val_fold, y_probs_val_fold, fold_params, log_dir)
            
            # save the determined thresholds to a YAML file for later use
            threshold_save_path = os.path.join(ensemble_models_dir, f"fold_{fold_idx + 1}_thresholds.yaml")
            threshold_data = {
                'plasmid_threshold': fold_params['plasmid_threshold'],
                'chromosome_threshold': fold_params['chromosome_threshold']
            }
            with open(threshold_save_path, 'w') as f:
                yaml.dump(threshold_data, f)
            print(f"  > Fold {fold_idx+1} thresholds saved to: {threshold_save_path}")

            print(f"\n--- Evaluating Fold {fold_idx + 1} Model on its Full Validation Set ---")

            # apply the just-determined thresholds to get final binary predictions
            final_scores_val_fold = torch.from_numpy(
                utils.apply_thresholds(y_probs_val_fold.numpy(), fold_params)
            ).to(device)
            
            y_probs_val_fold = y_probs_val_fold.to(device)
            val_fold_indices_global = torch.from_numpy(val_fold_indices).to(device)

            raw_probs_full = torch.zeros_like(data.y, dtype=torch.float, device=device)
            final_scores_full = torch.zeros_like(data.y, dtype=torch.float, device=device)

            # place the validation predictions into the full tensors at their correct global indices
            raw_probs_full.scatter_(0, val_fold_indices_global.unsqueeze(1).expand(-1, 2), y_probs_val_fold)
            final_scores_full.scatter_(0, val_fold_indices_global.unsqueeze(1).expand(-1, 2), final_scores_val_fold)

            # create a mask to identify which nodes were part of the validation set
            masks_validate = torch.zeros(data.num_nodes, dtype=torch.float32, device=device)
            masks_validate[val_fold_indices_global] = 1.0

            # call the metrics function to calculate and print detailed performance scores
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