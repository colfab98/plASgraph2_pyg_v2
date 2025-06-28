import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np


def fix_gradients(config_params, model: torch.nn.Module): 
    for param in model.parameters():
        if param.grad is not None:
            if torch.isnan(param.grad.data).any() or torch.isinf(param.grad.data).any():
                param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0)
    gradient_clipping_value = config_params._params['gradient_clipping']
    if gradient_clipping_value is not None and gradient_clipping_value > 0:
        torch.nn.utils.clip_grad_value_(model.parameters(), gradient_clipping_value)


def get_gradient_magnitudes(model: torch.nn.Module):
    grad_magnitudes_by_layer = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            # We want to group by layer, so let's try to infer the layer name.
            # This is a heuristic and might need adjustment based on model structure.
            # Example: 'preproc.weight', 'fc_input_1.weight', 'gcn_layers.0.weight'
            parts = name.split('.')
            if len(parts) > 1:
                layer_name_candidate = parts[0]
                if parts[0] == 'gcn_layers' or parts[0] == 'dense_layers':
                    # For ModuleList, include the index to distinguish layers
                    layer_name_candidate = f"{parts[0]}.{parts[1]}"
                elif parts[0] == 'final_fc1' or parts[0] == 'final_fc2':
                    layer_name_candidate = parts[0] # Group final FC layers separately
                else:
                    # For single layers like preproc, fc_input_1 etc.
                    layer_name_candidate = parts[0]
            else:
                layer_name_candidate = name # Should not happen often for complex models

            # Flatten the gradient tensor and get absolute values
            magnitudes = param.grad.abs().flatten().cpu().numpy()
            grad_magnitudes_by_layer.setdefault(layer_name_candidate, []).extend(magnitudes)
    return grad_magnitudes_by_layer

def plot_gradient_magnitudes(grad_data, epoch, log_dir, plot_frequency=10):
    if epoch % plot_frequency != 0:
        return

    if not grad_data: # No gradients collected
        return

    # Prepare data for plotting: list of (layer_name, magnitude) pairs
    plot_df_data = []
    for layer, magnitudes in grad_data.items():
        for mag in magnitudes:
            plot_df_data.append({'Layer': layer, 'Gradient Magnitude': mag})

    if not plot_df_data:
        return

    plot_df = pd.DataFrame(plot_df_data)

    # Sort layers for consistent plotting order
    # You might want a custom sorting order here to reflect network flow
    unique_layers = sorted(plot_df['Layer'].unique())
    plot_df['Layer'] = pd.Categorical(plot_df['Layer'], categories=unique_layers, ordered=True)
    plot_df = plot_df.sort_values('Layer')


    plt.figure(figsize=(15, 8))
    sns.violinplot(x='Layer', y='Gradient Magnitude', data=plot_df, inner='quartile', palette='viridis')
    plt.yscale('log') # Use log scale for y-axis due to potentially large range of magnitudes
    plt.title(f'Gradient Magnitude Distribution per Layer (Epoch {epoch})')
    plt.xlabel('Neural Network Layer')
    plt.ylabel('Log Gradient Magnitude')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plot_path = os.path.join(log_dir, f"gradient_magnitudes_epoch_{epoch}.png")
    plt.savefig(plot_path)
    plt.close()



def plot_f1_violin(plasmid_scores, chromosome_scores, output_dir):
    """Generates and saves a violin plot of F1 scores."""
    plt.figure(figsize=(10, 7))
    data_to_plot = [plasmid_scores, chromosome_scores]
    labels = ['Plasmid F1 Scores', 'Chromosome F1 Scores']
    
    # --- (Paste the entire matplotlib plotting block here) ---
    parts = plt.violinplot(data_to_plot, showmeans=False, showmedians=True, showextrema=True)
    for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor('skyblue' if labels[i].startswith('Plasmid') else 'lightcoral')
                pc.set_edgecolor('black')
                pc.set_alpha(0.8)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor('grey')
            vp.set_linewidth(1)
    parts['cmedians'].set_edgecolor('red')
    parts['cmedians'].set_linewidth(1.5)
    for i, d_list in enumerate(data_to_plot): 
            x_jitter = np.random.normal(loc=i + 1, scale=0.04, size=len(d_list))
            plt.scatter(x_jitter, d_list, alpha=0.4, s=20, color='dimgray', zorder=3) # zorder to plot dots on top
    plt.ylabel('F1 Score', fontsize=14)
    plt.title('Distribution of F1 Scores Across Samples', fontsize=16)
    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=0, ha='center', fontsize=12) # Use np.arange
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "f1_scores_violin_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\n✅ Evaluation plot saved to: {plot_path}")



"""Module for changing setting optimal threshold wrt F1 measure on validation set 
and applying them to new data"""

import numpy as np

import os
# import architecture

# ... (imports remain the same) ...
import torch # Add torch import
import torch.nn.functional as F # Add F if you need to manually apply sigmoid/softmax if model output is logits

# Note: The provided `architecture.py` now applies `self.output_activation` in `forward`.
# So, `model(graph)` will return probabilities directly.

def set_thresholds(model, data, masks_validate, parameters, log_dir=None):
    """Set optimal thresholds (plasmid, chromosome) by maximizing F1 on validation set,
    and store them in the parameters object. This version works with PyTorch models/data."""

    # Ensure model is in evaluation mode
    model.eval()
    
    # Move data to model's device (if not already there)
    device = next(model.parameters()).device
    data = data.to(device)

    with torch.no_grad():
        # Get raw predictions/probabilities from the model for the entire graph
        # model.forward() now applies the output activation, so `outputs` are already probabilities
        outputs = torch.sigmoid(model(data))

        
        # Extract true labels for validation nodes
        # Use data.y for true labels and masks_validate for selection
        
        # --- Process Plasmid ---
        # Select only validation set nodes for plasmid output
        labels_plasmid_val = data.y[masks_validate.bool(), 0]
        probs_plasmid_val = outputs[masks_validate.bool(), 0]

        # In your train9.py, `valid_eval_mask_plasmid = labels_plasmid_val_masked != -1` was used.
        # This typically means labels could be -1 (e.g., ignored or missing).
        # Assuming `data.y` is clean (0/1/other numerical labels), and `masks_validate` already
        # handles the train/val split (0 for train, >0 for val), then you just need to
        # filter out labels that are specifically meant to be ignored from F1 calculation.
        # If your labels are always 0 or 1 for actual classes, and 0 for unlabeled, and 1,1 for ambiguous,
        # then you primarily care about samples where the label is 0 or 1 for the *specific class*.
        # The `masks_validate` array should handle the actual samples to consider.
        
        # Filter for actual 0/1 true labels where relevant (excluding [0,0] from being target 0/1)
        # This part depends on how you want to treat the 0,0 labels (unlabeled) in F1 score.
        # If the `masks_validate` already sets their weight to 0, then `score_thresholds` should handle it.
        # The `score_thresholds` in `thresholds.py` explicitly uses `if weights[i] > 0:`
        # So, we just need to pass the y_true and y_pred for all relevant nodes from the validation set
        # with their corresponding masks_validate.
        
        # Convert to numpy arrays for sklearn's f1_score
        y_true_plasmid = labels_plasmid_val.cpu().numpy()
        y_probs_plasmid = probs_plasmid_val.cpu().numpy()
        sample_weight_plasmid = masks_validate[masks_validate.bool()].cpu().numpy() # weights for validation nodes

        # Call score_thresholds which is designed to work with y_true (0/1), y_pred (float), and weights
        # Note: score_thresholds in the original thresholds.py computes F1 iteratively.
        # It's not sklearn.f1_score. We need to check if it already uses sample_weight.
        # Checking thresholds.py: score_thresholds takes y_true, y_pred, weights and filters `if weights[i] > 0`
        # and then counts tp based on `if pairs[i][0] > 0.5`. This is compatible.
        
        plasmid_scores = score_thresholds(y_true_plasmid, y_probs_plasmid, sample_weight_plasmid)
        store_best(plasmid_scores, parameters, 'plasmid_threshold', log_dir)

        # --- Process Chromosome ---
        # Select only validation set nodes for chromosome output
        labels_chromosome_val = data.y[masks_validate.bool(), 1]
        probs_chromosome_val = outputs[masks_validate.bool(), 1]

        y_true_chromosome = labels_chromosome_val.cpu().numpy()
        y_probs_chromosome = probs_chromosome_val.cpu().numpy()
        sample_weight_chromosome = masks_validate[masks_validate.bool()].cpu().numpy()

        chromosome_scores = score_thresholds(y_true_chromosome, y_probs_chromosome, sample_weight_chromosome)
        store_best(chromosome_scores, parameters, 'chromosome_threshold', log_dir)



def apply_thresholds(y, parameters):
    """Apply thresholds during testing, return transformed scores so that 0.5 corresponds to threshold"""
    columns = []
    for (column_idx, which_parameter) in [(0, 'plasmid_threshold'), (1, 'chromosome_threshold')]:
        threshold = parameters[which_parameter]
        orig_column = y[:, column_idx]
        # apply the scaling function with different parameters for small and large numbers
        new_column = np.piecewise(
            orig_column,
            [orig_column < threshold, orig_column >= threshold],
            [lambda x : scale_number(x, 0, threshold, 0, 0.5), lambda x : scale_number(x, threshold, 1, 0.5, 1)]
        )
        columns.append(new_column)

    y_new = np.array(columns).transpose()
    return y_new

def scale_number(x, s1, e1, s2, e2):
    """Scale number x so that interval (s1,e1) is transformed to (s2, e2)"""

    factor = (e2 - s2) / (e1 - s1)
    return (x - s1) * factor + s2

def store_best(scores, parameters, which, log_dir):
    """store the optimal threshold for one output in parameter and if requested, print all thresholds to a log file"""
    # scores is a list of pairs threshold, F1 score
    if len(scores) > 0:
        # find index of maximum in scores[*][1]
        maxindex = max(range(len(scores)), key = lambda i : scores[i][1])
        # corrsponding item in scores[*][0] is the threshold
        threshold = scores[maxindex][0]
    else:
        # is input array empty, use default 0.5
        threshold = 0.5
    # store the found threshold
    parameters[which] = float(threshold)

    if log_dir is not None:
        # store thresholds and F1 scores
        filename = os.path.join(log_dir, which + ".csv")
        with open(filename, 'wt') as file:
            print(f"{which},f1", file=file)
            for x in scores:
                print(",".join(str(value) for value in x), file=file)

def score_thresholds(y_true, y_pred, weights):
    """Compute F1 score of all thresholds for one output (plasmid or chromosome)"""
    # compute vector weight and check that all are the same
    length = y_true.shape[0]
    assert tuple(y_true.shape) == (length,)
    assert tuple(y_pred.shape) == (length,)
    assert tuple(weights.shape) == (length,)
    # get data points with non-zero weight
    pairs = []
    for i in range(length):
        if weights[i] > 0:
            pairs.append((y_true[i], y_pred[i]))
    pairs.sort(key=lambda x : x[1], reverse=True)
    
    # count all positives in true labels
    pos = 0
    for pair in pairs:
        if pair[0] > 0.5:
            pos += 1

    scores = []
    tp = 0
    for i in range(len(pairs)):
        # increase true positives if true label is 
        if pairs[i][0] > 0.5:
            tp += 1
        if i > 0 and pairs[i][1] < pairs[i-1][1]:
            recall = tp / pos
            precision = tp / (i+1)
            if (precision + recall) == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            threshold = (pairs[i-1][1] + pairs[i][1]) / 2
            scores.append((threshold, f1))
    
    return scores



import itertools
from Bio.Seq import Seq
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def get_node_id(sample_id, contig_id):
    return f"{sample_id}:{contig_id}"

def label_to_pair(label):
    if label == "chromosome":
        return [0, 1]
    elif label == "plasmid":
        return [1, 0]
    elif label == "ambiguous":
        return [1, 1]
    elif (label == "unlabelled" or label == "no_label" 
          or label == "unlabeled" or label is None):
        return [0,0]
    else:
        raise AssertionError("bad label {label}")

def pair_to_label(pair):
    if pair == [0, 1]:
        return "chromosome"
    elif pair == [1, 0]:
        return "plasmid"
    elif pair == [1, 1]:
        return "ambiguous"
    elif pair == [0, 0]:
        return "unlabeled"
    else:
        raise AssertionError("bad pair {pair}")
        
def prepare_kmer_lists(kmer_length):

    # list of all possible DNA sequences of a specific length
    k_mers = ["".join(x) for x in itertools.product("ACGT", repeat=kmer_length)]

    # stores a "canonical" or representative k-mer for each k-mer and its reverse complement pair
    fwd_kmers = []

    fwd_kmer_set = set()
    rev_kmer_set = set()
    
    for k_mer in k_mers:
        if not ((k_mer in fwd_kmer_set) or (k_mer in rev_kmer_set)):
            fwd_kmers.append(k_mer)
            fwd_kmer_set.add(k_mer)
            rev_kmer_set.add(str(Seq(k_mer).reverse_complement()))

    return (k_mers, fwd_kmers)


def get_kmer_distribution(sequence, kmer_length=5, scale=False):
    assert kmer_length % 2 == 1
    (k_mers, fwd_kmers) = prepare_kmer_lists(kmer_length)
    
    # maps each k-mer from the full k_mers list to its count within the input sequence
    dict_kmer_count = {}

    # keys are k-mer strings (from k_mers) and values are floating-point numbers
    for k_mer in k_mers:
        dict_kmer_count[k_mer] = 0.01 # pseudocounts

    for i in range(len(sequence) + 1 - kmer_length):
        kmer = sequence[i : i + kmer_length]
        if kmer in dict_kmer_count:
            dict_kmer_count[kmer] += 1

    # represents the frequency (or proportion if scaled) of each canonical k-mer pair (k-mer + its reverse complement) in the sequence
    k_mer_counts = [
        dict_kmer_count[k_mer] + dict_kmer_count[str(Seq(k_mer).reverse_complement())]
        for k_mer in fwd_kmers
    ]

    if scale:
        ksum = sum(k_mer_counts)
        k_mer_counts = [ x / ksum for x in k_mer_counts ]

    return k_mer_counts


def get_gc_content(seq):
    number_gc = 0
    number_acgt = 0
    for base in seq:
        if base in "GC":
            number_gc += 1
        if base in "ACGT":
            number_acgt += 1

    if number_acgt > 0:
        gc_content = round(number_gc / number_acgt, 4)
    else:
        gc_content = 0.5
    return gc_content
