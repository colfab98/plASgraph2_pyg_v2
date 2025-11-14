import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from collections import Counter

def calculate_and_print_metrics(final_scores, raw_probs, data, masks_test, G, node_list, verbose=True):
    """
    Calculates and prints per-sample and aggregate metrics for the test set.

    This function iterates through each sample (graph) in the test set, calculates
    a suite of classification metrics (F1, Accuracy, Precision, Recall, AUROC)
    for both plasmid and chromosome classes, and then reports the median of these
    metrics across all samples.
    """

    # create a mapping from a node's string ID to its integer index for efficient lookups
    node_to_idx_map = {node_id: i for i, node_id in enumerate(node_list)}
    # unique sample IDs present in the graph
    sample_ids = sorted(list(set(G.nodes[node_id]["sample"] for node_id in node_list)))

    all_f1_plasmid, all_acc_plasmid, all_prec_plasmid, all_rec_plasmid, all_auroc_plasmid = [], [], [], [], []
    all_f1_chrom, all_acc_chrom, all_prec_chrom, all_rec_chrom, all_auroc_chrom = [], [], [], [], []

    # iterate through each sample to evaluate its performance individually
    for sample_id in sample_ids:
        # get the global tensor indices for all nodes belonging to the current sample
        current_sample_node_indices_global = [
            node_to_idx_map[node_id] 
            for node_id in node_list
            if G.nodes[node_id]["sample"] == sample_id
        ]
        
        # if True, prints detailed metrics for each individual sample in addition to the final aggregate results
        if verbose:
            print(f"\n--- Sample {sample_id} ---")

        # convert the list of indices to a tensor for slicing
        current_sample_node_indices_global = torch.tensor(current_sample_node_indices_global, dtype=torch.long)

        # slice the main tensors to get data only for the current sample
        y_sample = data.y[current_sample_node_indices_global]
        scores_sample = final_scores[current_sample_node_indices_global]
        probs_sample = raw_probs[current_sample_node_indices_global]
        masks_sample = masks_test[current_sample_node_indices_global]

        # boolean mask to select only the labeled test nodes within this sample
        active_mask_for_sample = masks_sample > 0

        if not torch.any(active_mask_for_sample):
            if verbose:
                print("  No labeled contigs to evaluate. Skipping.")
            continue

        # --- plasmid evaluation ---
        y_true_p = y_sample[active_mask_for_sample.cpu(), 0].cpu().numpy()
        # skip evaluation for this sample if all labels belong to a single class
        if len(np.unique(y_true_p)) > 1:
            # get binary predictions by thresholding the final scores at 0.5
            y_pred_p = (scores_sample[active_mask_for_sample, 0].cpu().numpy() >= 0.5).astype(int)
            # get the raw probabilities for calculating AUROC
            y_prob_p = probs_sample[active_mask_for_sample, 0].cpu().numpy()
            
            all_f1_plasmid.append(f1_score(y_true_p, y_pred_p, zero_division=0))
            all_acc_plasmid.append(accuracy_score(y_true_p, y_pred_p))
            all_prec_plasmid.append(precision_score(y_true_p, y_pred_p, zero_division=0))
            all_rec_plasmid.append(recall_score(y_true_p, y_pred_p, zero_division=0))
            all_auroc_plasmid.append(roc_auc_score(y_true_p, y_prob_p))
            if verbose:
                print(f"  PLASMID   | F1: {all_f1_plasmid[-1]:.4f} | Acc: {all_acc_plasmid[-1]:.4f} | Prec: {all_prec_plasmid[-1]:.4f} | Rec: {all_rec_plasmid[-1]:.4f} | AUROC: {all_auroc_plasmid[-1]:.4f}")
        
        # --- chromosome evaluation ---
        y_true_c = y_sample[active_mask_for_sample.cpu(), 1].cpu().numpy()
        if len(np.unique(y_true_c)) > 1:
            y_pred_c = (scores_sample[active_mask_for_sample, 1].cpu().numpy() >= 0.5).astype(int)
            y_prob_c = probs_sample[active_mask_for_sample, 1].cpu().numpy()

            all_f1_chrom.append(f1_score(y_true_c, y_pred_c, zero_division=0))
            all_acc_chrom.append(accuracy_score(y_true_c, y_pred_c))
            all_prec_chrom.append(precision_score(y_true_c, y_pred_c, zero_division=0))
            all_rec_chrom.append(recall_score(y_true_c, y_pred_c, zero_division=0))
            all_auroc_chrom.append(roc_auc_score(y_true_c, y_prob_c))
            if verbose:
                print(f"  CHROMOSOME| F1: {all_f1_chrom[-1]:.4f} | Acc: {all_acc_chrom[-1]:.4f} | Prec: {all_prec_chrom[-1]:.4f} | Rec: {all_rec_chrom[-1]:.4f} | AUROC: {all_auroc_chrom[-1]:.4f}")

    # --- print aggregate metrics ---
    print("\n" + "="*60)
    print("📊 Aggregate Test Metrics (Median of valid samples)")
    print("\n--- Median PLASMID Metrics ---")
    print(f"  F1-Score:  {np.median(all_f1_plasmid):.4f}" if all_f1_plasmid else "N/A")
    print(f"  Accuracy:  {np.median(all_acc_plasmid):.4f}" if all_acc_plasmid else "N/A")
    print(f"  Precision: {np.median(all_prec_plasmid):.4f}" if all_prec_plasmid else "N/A")
    print(f"  Recall:    {np.median(all_rec_plasmid):.4f}" if all_rec_plasmid else "N/A")
    print(f"  AUROC:     {np.median(all_auroc_plasmid):.4f}" if all_auroc_plasmid else "N/A")

    print("\n--- Median CHROMOSOME Metrics ---")
    print(f"  F1-Score:  {np.median(all_f1_chrom):.4f}" if all_f1_chrom else "N/A")
    print(f"  Accuracy:  {np.median(all_acc_chrom):.4f}" if all_acc_chrom else "N/A")
    print(f"  Precision: {np.median(all_prec_chrom):.4f}" if all_prec_chrom else "N/A")
    print(f"  Recall:    {np.median(all_rec_chrom):.4f}" if all_rec_chrom else "N/A")
    print(f"  AUROC:     {np.median(all_auroc_chrom):.4f}" if all_auroc_chrom else "N/A")
    print("="*60)
    
    return all_f1_plasmid, all_f1_chrom