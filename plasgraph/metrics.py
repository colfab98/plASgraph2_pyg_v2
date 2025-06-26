# In plasgraph/metrics.py

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from collections import Counter

def calculate_and_print_metrics(outputs, data, masks_test, G, node_list, parameters):
    """
    Calculates and prints per-sample and aggregate metrics for the test set.
    
    MODIFIED:
    1. Includes all metrics from the original project (F1, Acc, Prec, Rec, AUROC).
    2. Excludes all metrics from samples with only one class in the ground truth.
    """
    node_to_idx_map = {node_id: i for i, node_id in enumerate(node_list)}
    sample_ids = sorted(list(set(G.nodes[node_id]["sample"] for node_id in node_list)))

    # ADDED: Lists for the additional metrics
    all_f1_plasmid_scores, all_acc_plasmid_scores, all_prec_plasmid_scores, all_rec_plasmid_scores, all_auroc_plasmid_scores = [], [], [], [], []
    all_f1_chromosome_scores, all_acc_chromosome_scores, all_prec_chromosome_scores, all_rec_chromosome_scores, all_auroc_chromosome_scores = [], [], [], [], []


    for sample_id in sample_ids:
        # --- (This part remains the same) ---
        current_sample_node_indices_global = [
            node_to_idx_map[node_id] 
            for node_id in node_list
            if G.nodes[node_id]["sample"] == sample_id
        ]

        # --- NEW: Calculate class proportions for the current sample ---
        current_sample_node_ids = [node_list[i] for i in current_sample_node_indices_global]
        
        # Count the text labels for the nodes in the current sample
        label_counts = Counter(G.nodes[node_id]["text_label"] for node_id in current_sample_node_ids)
        total_nodes_in_sample = len(current_sample_node_ids)

        # Format the counts into a readable string
        counts_str = (
            f"Counts (P: {label_counts.get('plasmid', 0)}, "
            f"C: {label_counts.get('chromosome', 0)}, "
            f"U: {label_counts.get('unlabeled', 0)}, "
            f"A: {label_counts.get('ambiguous', 0)})"
        )

        print(f"\n  Sample {sample_id} | Total Contigs: {total_nodes_in_sample} | {counts_str}")

        current_sample_node_indices_global = torch.tensor(current_sample_node_indices_global, dtype=torch.long)

        y_sample = data.y[current_sample_node_indices_global]
        outputs_sample = outputs[current_sample_node_indices_global]
        masks_sample = masks_test[current_sample_node_indices_global]

        active_mask_for_sample = masks_sample > 0
        if not torch.any(active_mask_for_sample):
            print(f"  Sample {sample_id} - No labeled contigs to evaluate. Skipping.")
            continue

        # --- Plasmid Evaluation ---
        labels_plasmid_sample_active = y_sample[active_mask_for_sample, 0]
        y_true_plasmid = labels_plasmid_sample_active.cpu().numpy()
        
        # --- Conditional Check for Plasmid Metrics ---
        if len(np.unique(y_true_plasmid)) < 2:
            print(f"  Sample {sample_id} - Test PLASMID   | SKIPPING (only one class present in ground truth)")
        else:
            probs_plasmid_sample_active = outputs_sample[active_mask_for_sample, 0]
            y_probs_plasmid = probs_plasmid_sample_active.cpu().numpy()
            sample_weight_plasmid = masks_sample[active_mask_for_sample].cpu().numpy()
            
            final_best_thresh_plasmid = parameters['plasmid_threshold']
            y_pred_plasmid = (y_probs_plasmid >= final_best_thresh_plasmid).astype(int)

            # CHANGED: Calculate all metrics
            f1_plasmid = f1_score(y_true_plasmid, y_pred_plasmid, sample_weight=sample_weight_plasmid, zero_division=0)
            acc_plasmid = accuracy_score(y_true_plasmid, y_pred_plasmid, sample_weight=sample_weight_plasmid)
            prec_plasmid = precision_score(y_true_plasmid, y_pred_plasmid, sample_weight=sample_weight_plasmid, zero_division=0)
            rec_plasmid = recall_score(y_true_plasmid, y_pred_plasmid, sample_weight=sample_weight_plasmid, zero_division=0)
            auroc_plasmid = roc_auc_score(y_true_plasmid, y_probs_plasmid, sample_weight=sample_weight_plasmid)
            
            # CHANGED: Append all scores
            all_f1_plasmid_scores.append(f1_plasmid)
            all_acc_plasmid_scores.append(acc_plasmid)
            all_prec_plasmid_scores.append(prec_plasmid)
            all_rec_plasmid_scores.append(rec_plasmid)
            all_auroc_plasmid_scores.append(auroc_plasmid)
            
            # CHANGED: Update print statement to be more verbose
            print(f"           Test PLASMID   | F1: {f1_plasmid:.4f} | Acc: {acc_plasmid:.4f} | Prec: {prec_plasmid:.4f} | Rec: {rec_plasmid:.4f} | AUROC: {auroc_plasmid:.4f}")

        # --- Chromosome Evaluation ---
        labels_chromosome_sample_active = y_sample[active_mask_for_sample, 1]
        y_true_chromosome = labels_chromosome_sample_active.cpu().numpy()
        
        # --- Conditional Check for Chromosome Metrics ---
        if len(np.unique(y_true_chromosome)) < 2:
            print(f"  Sample {sample_id} - Test CHROMOSOME| SKIPPING (only one class present in ground truth)")
        else:
            probs_chromosome_sample_active = outputs_sample[active_mask_for_sample, 1]
            y_probs_chromosome = probs_chromosome_sample_active.cpu().numpy()
            sample_weight_chromosome = masks_sample[active_mask_for_sample].cpu().numpy()
            
            final_best_thresh_chromosome = parameters['chromosome_threshold']
            y_pred_chromosome = (y_probs_chromosome >= final_best_thresh_chromosome).astype(int)

            # CHANGED: Calculate all metrics
            f1_chromosome = f1_score(y_true_chromosome, y_pred_chromosome, sample_weight=sample_weight_chromosome, zero_division=0)
            acc_chromosome = accuracy_score(y_true_chromosome, y_pred_chromosome, sample_weight=sample_weight_chromosome)
            prec_chromosome = precision_score(y_true_chromosome, y_pred_chromosome, sample_weight=sample_weight_chromosome, zero_division=0)
            rec_chromosome = recall_score(y_true_chromosome, y_pred_chromosome, sample_weight=sample_weight_chromosome, zero_division=0)
            auroc_chromosome = roc_auc_score(y_true_chromosome, y_probs_chromosome, sample_weight=sample_weight_chromosome)
            
            # CHANGED: Append all scores
            all_f1_chromosome_scores.append(f1_chromosome)
            all_acc_chromosome_scores.append(acc_chromosome)
            all_prec_chromosome_scores.append(prec_chromosome)
            all_rec_chromosome_scores.append(rec_chromosome)
            all_auroc_chromosome_scores.append(auroc_chromosome)

            # CHANGED: Update print statement to be more verbose
            print(f"           Test CHROMOSOME| F1: {f1_chromosome:.4f} | Acc: {acc_chromosome:.4f} | Prec: {prec_chromosome:.4f} | Rec: {rec_chromosome:.4f} | AUROC: {auroc_chromosome:.4f}")


    # --- Print Aggregate Metrics ---
    print("\n" + "="*60)
    print("📊 Aggregate Test Metrics (Median of valid samples)")
    
    # ADDED: Calculation for new aggregate metrics
    median_f1_plasmid = np.median(all_f1_plasmid_scores) if all_f1_plasmid_scores else np.nan
    median_acc_plasmid = np.median(all_acc_plasmid_scores) if all_acc_plasmid_scores else np.nan
    median_prec_plasmid = np.median(all_prec_plasmid_scores) if all_prec_plasmid_scores else np.nan
    median_rec_plasmid = np.median(all_rec_plasmid_scores) if all_rec_plasmid_scores else np.nan
    median_auroc_plasmid = np.median(all_auroc_plasmid_scores) if all_auroc_plasmid_scores else np.nan
    
    median_f1_chromosome = np.median(all_f1_chromosome_scores) if all_f1_chromosome_scores else np.nan
    median_acc_chromosome = np.median(all_acc_chromosome_scores) if all_acc_chromosome_scores else np.nan
    median_prec_chromosome = np.median(all_prec_chromosome_scores) if all_prec_chromosome_scores else np.nan
    median_rec_chromosome = np.median(all_rec_chromosome_scores) if all_rec_chromosome_scores else np.nan
    median_auroc_chromosome = np.median(all_auroc_chromosome_scores) if all_auroc_chromosome_scores else np.nan
    
    # CHANGED: More verbose final report
    print("\n--- Median PLASMID Metrics ---")
    print(f"  F1-Score: {median_f1_plasmid:.4f}")
    print(f"  Accuracy: {median_acc_plasmid:.4f}")
    print(f"  Precision: {median_prec_plasmid:.4f}")
    print(f"  Recall: {median_rec_plasmid:.4f}")
    print(f"  AUROC: {median_auroc_plasmid:.4f}")

    print("\n--- Median CHROMOSOME Metrics ---")
    print(f"  F1-Score: {median_f1_chromosome:.4f}")
    print(f"  Accuracy: {median_acc_chromosome:.4f}")
    print(f"  Precision: {median_prec_chromosome:.4f}")
    print(f"  Recall: {median_rec_chromosome:.4f}")
    print(f"  AUROC: {median_auroc_chromosome:.4f}")
    print("="*60)
    
    return all_f1_plasmid_scores, all_f1_chromosome_scores