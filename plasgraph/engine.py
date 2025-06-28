import numpy as np
import matplotlib.pyplot as plt
import torch
import optuna
import os 

from . import config
from .models import GCNModel, GGNNModel 
from .utils import fix_gradients

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from . import utils


def objective(trial, parameters, data, device, splits, labeled_indices):
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

    fold_losses = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        
        if trial_config_obj['model_type'] == 'GCNModel':
            model = GCNModel(trial_config_obj).to(device)
        elif trial_config_obj['model_type'] == 'GGNNModel':
            model = GGNNModel(trial_config_obj).to(device)
        else:
            raise ValueError(f"Unsupported model type in trial {trial.number}: {trial_config_obj['model_type']}")

        optimizer = torch.optim.Adam(model.parameters(), lr=trial_config_obj['learning_rate'], weight_decay=trial_config_obj['l2_reg'])
        criterion = torch.nn.BCELoss(reduction='none')

        train_fold_indices = labeled_indices[train_idx]
        val_fold_indices = labeled_indices[val_idx]

        masks_train = torch.zeros(data.num_nodes, dtype=torch.float32, device=device)
        masks_train[train_fold_indices] = 1.0 

        masks_validate = torch.zeros(data.num_nodes, dtype=torch.float32, device=device)
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

            train_loss.backward()
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



def train_final_model(parameters, data, device, splits, labeled_indices, log_dir):
    """
    Trains the final model on all labeled data using the best hyperparameters.
    """
    print("\n" + "="*60)
    print("💾 Training Final Model on ALL Labeled Data")
    print("="*60)

    # --- This entire block is from your original plASgraph2_train.py ---
    
    # 1. Setup validation set for plotting and early stopping
    _, val_indices_for_plot = splits[-1]
    masks_val_final_plot = torch.zeros(data.num_nodes, dtype=torch.float32, device=device)
    masks_val_final_plot[labeled_indices[val_indices_for_plot]] = 1.0

    # 2. Initialize Model, Optimizer, Scheduler, Criterion
    if parameters['model_type'] == 'GCNModel':
        final_model = GCNModel(parameters).to(device)
    elif parameters['model_type'] == 'GGNNModel':
        final_model = GGNNModel(parameters).to(device)

    optimizer = optim.Adam(final_model.parameters(), lr=parameters['learning_rate'], weight_decay=parameters['l2_reg'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=parameters['scheduler_factor'],
                                  patience=parameters['scheduler_patience'])
    criterion = torch.nn.BCELoss(reduction='none')

    # 3. Create mask for all labeled data
    masks_train_all = torch.zeros(data.num_nodes, dtype=torch.float32, device=device)
    masks_train_all[labeled_indices] = 1.0

    # 4. The main training loop
    best_val_loss_final = float('inf')
    patience_final = 0
    best_model_state_final = None
    train_losses_final = []
    val_losses_final = []

    for epoch in range(parameters["epochs"]):
        final_model.train()
        optimizer.zero_grad()
        outputs = final_model(data)

        valid_label_mask = data.y.sum(dim=1) != 0
        masked_weights_train_all = masks_train_all[valid_label_mask]
        loss_per_node = criterion(outputs[valid_label_mask], data.y[valid_label_mask])
        loss = (loss_per_node.sum(dim=1) * masked_weights_train_all).sum() / masked_weights_train_all.sum()

        loss.backward()
        
        # We need to move the gradient functions to utils.py
        grad_magnitudes = utils.get_gradient_magnitudes(final_model)
        utils.plot_gradient_magnitudes(grad_magnitudes, epoch, log_dir, plot_frequency=100)
        utils.fix_gradients(parameters, final_model)
        
        optimizer.step()

        # --- (Copy the rest of the loop: validation, early stopping, printing) ---
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
                val_losses_final.append(0.0)
        train_losses_final.append(loss.item())
        
        scheduler.step(val_loss)

        if val_loss < best_val_loss_final:
            best_val_loss_final = val_loss
            best_model_state_final = final_model.state_dict()
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
    print(f"\nTraining finished. Loading best model state from epoch {epoch+1-patience_final}.")
    final_model.load_state_dict(best_model_state_final)

    print("Determining final thresholds on the validation set of the last fold...")
    utils.set_thresholds(final_model, data, masks_val_final_plot, parameters, log_dir)
    print(f"Optimal thresholds determined: Plasmid={parameters['plasmid_threshold']:.2f}, Chromosome={parameters['chromosome_threshold']:.2f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses_final, label='Final Train Loss (on all data)')
    plt.plot(val_losses_final, label=f"Final Validation Loss (on fold {parameters['k_folds']})")
    plt.axvline(x=epoch+1-patience_final, color='r', linestyle='--', label='Best Model Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Final Model Training (Initial LR: {parameters['learning_rate']})")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    
    # Save the plot to the log directory
    plot_path = os.path.join(log_dir, "final_model_training_loss.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\n✅ Final model training loss plot saved to: {plot_path}")
    
    # Return the trained model and the parameters (which now include thresholds)
    return final_model, parameters



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