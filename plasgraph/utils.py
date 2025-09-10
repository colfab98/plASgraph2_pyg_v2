import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import torch.nn.functional as F 
import random



def set_all_seeds(seed):
    """
    Sets the random seeds for all relevant libraries to ensure reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # The following two lines are crucial for deterministic results on CUDA.
    # Note that they can negatively impact performance.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def fix_gradients(config_params, model: torch.nn.Module): 
    """
    Stabilizes training by handling non-finite gradients and applying gradient clipping.

    This function performs two main operations:
    1. It iterates through all model parameters and replaces any `NaN`
       or `inf` values in the gradients with 0.0. This prevents
       optimizer errors during unstable training phases.
    2. It applies gradient clipping by value, which clamps the gradients to a specified
       range to prevent the "exploding gradients" problem.
    """

    for param in model.parameters():
        if param.grad is not None:
            if torch.isnan(param.grad.data).any() or torch.isinf(param.grad.data).any():
                param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0)
    gradient_clipping_value = config_params._params['gradient_clipping']
    if gradient_clipping_value is not None and gradient_clipping_value > 0:
        torch.nn.utils.clip_grad_value_(model.parameters(), gradient_clipping_value)


def get_gradient_magnitudes(model: torch.nn.Module):
    """
    Calculates and groups the magnitudes of gradients for each layer in a model.

    This function is a diagnostic tool used to monitor the health of the training
    process. It iterates through all model parameters that have gradients, calculates
    the absolute value of each gradient, and groups these magnitudes by the layer
    they belong to. This is useful for identifying potential issues like vanishing
    or exploding gradients in specific parts of the network.
    """

    grad_magnitudes_by_layer = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            parts = name.split('.')
            if len(parts) > 1:
                layer_name_candidate = parts[0]
                if parts[0] == 'gcn_layers' or parts[0] == 'dense_layers':
                    layer_name_candidate = f"{parts[0]}.{parts[1]}"
                elif parts[0] == 'final_fc1' or parts[0] == 'final_fc2':
                    layer_name_candidate = parts[0]
                else:
                    layer_name_candidate = parts[0]
            else:
                layer_name_candidate = name 

            magnitudes = param.grad.abs().flatten().cpu().numpy()
            grad_magnitudes_by_layer.setdefault(layer_name_candidate, []).extend(magnitudes)
    return grad_magnitudes_by_layer

def plot_gradient_magnitudes(grad_data, epoch, log_dir, plot_frequency=10):
    """
    Generates and saves a violin plot of gradient magnitudes grouped by layer.
    """

    if epoch % plot_frequency != 0:
        return
    if not grad_data: 
        return
    plot_df_data = []
    for layer, magnitudes in grad_data.items():
        for mag in magnitudes:
            plot_df_data.append({'Layer': layer, 'Gradient Magnitude': mag})
    if not plot_df_data:
        return
    plot_df = pd.DataFrame(plot_df_data)
    unique_layers = sorted(plot_df['Layer'].unique())
    plot_df['Layer'] = pd.Categorical(plot_df['Layer'], categories=unique_layers, ordered=True)
    plot_df = plot_df.sort_values('Layer')

    plt.figure(figsize=(15, 8))
    sns.violinplot(x='Layer', y='Gradient Magnitude', data=plot_df, inner='quartile', hue='Layer', palette='viridis', legend=False)
    if plot_df['Gradient Magnitude'].max() > 0:
        plt.yscale('log')

    plt.title(f'Gradient Magnitude Distribution per Layer (Epoch {epoch})')
    plt.xlabel('Neural Network Layer')
    plt.ylabel('Log Gradient Magnitude')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plot_path = os.path.join(log_dir, f"gradient_magnitudes_epoch_{epoch}.png")
    plt.savefig(plot_path)
    plt.close()


def plot_f1_violin(plasmid_scores, chromosome_scores, output_dir):
    """
    Generates and saves a violin plot of F1 scores.
    """

    plt.figure(figsize=(10, 7))
    data_to_plot = [plasmid_scores, chromosome_scores]
    labels = ['Plasmid F1 Scores', 'Chromosome F1 Scores']
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
            plt.scatter(x_jitter, d_list, alpha=0.4, s=20, color='dimgray', zorder=3) 
    plt.ylabel('F1 Score', fontsize=14)
    plt.title('Distribution of F1 Scores Across Samples', fontsize=16)
    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=0, ha='center', fontsize=12) 
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "f1_scores_violin_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\n✅ Evaluation plot saved to: {plot_path}")



# def set_thresholds(model, data, masks_validate, parameters, log_dir=None):
#     """Set optimal thresholds (plasmid, chromosome) by maximizing F1 on validation set,
#     and store them in the parameters object. This version works with PyTorch models/data."""

#     # Ensure model is in evaluation mode
#     model.eval()
    
#     # Move data to model's device (if not already there)
#     device = next(model.parameters()).device
#     data = data.to(device)

#     with torch.no_grad():
#         outputs = torch.sigmoid(model(data))

#         labels_plasmid_val = data.y[masks_validate.bool(), 0]
#         probs_plasmid_val = outputs[masks_validate.bool(), 0]

#         y_true_plasmid = labels_plasmid_val.cpu().numpy()
#         y_probs_plasmid = probs_plasmid_val.cpu().numpy()
#         sample_weight_plasmid = masks_validate[masks_validate.bool()].cpu().numpy() # weights for validation nodes
        
#         plasmid_scores = score_thresholds(y_true_plasmid, y_probs_plasmid, sample_weight_plasmid)
#         store_best(plasmid_scores, parameters, 'plasmid_threshold', log_dir)

#         # --- Process Chromosome ---
#         # Select only validation set nodes for chromosome output
#         labels_chromosome_val = data.y[masks_validate.bool(), 1]
#         probs_chromosome_val = outputs[masks_validate.bool(), 1]

#         y_true_chromosome = labels_chromosome_val.cpu().numpy()
#         y_probs_chromosome = probs_chromosome_val.cpu().numpy()
#         sample_weight_chromosome = masks_validate[masks_validate.bool()].cpu().numpy()

#         chromosome_scores = score_thresholds(y_true_chromosome, y_probs_chromosome, sample_weight_chromosome)
#         store_best(chromosome_scores, parameters, 'chromosome_threshold', log_dir)

def set_thresholds_from_predictions(y_true, y_probs, parameters, log_dir=None):
    """
    Sets optimal thresholds by maximizing the F1 score based on pre-computed
    true labels and predicted probabilities from a validation set.

    This function orchestrates the process of finding the best decision threshold
    for both the plasmid and chromosome classes independently.
    """

    # --- process plasmid ---
    # isolate the true labels and predicted probabilities for the plasmid class
    y_true_plasmid = y_true[:, 0].cpu().numpy()
    y_probs_plasmid = y_probs[:, 0].cpu().numpy()
    sample_weight_plasmid = np.ones_like(y_true_plasmid)

    # calculate F1 scores for all possible thresholds for the plasmid class
    plasmid_scores = score_thresholds(y_true_plasmid, y_probs_plasmid, sample_weight_plasmid)
    # Find the best threshold and store it in the parameters object
    store_best(plasmid_scores, parameters, 'plasmid_threshold', log_dir)

    # --- process chromosome ---
    y_true_chromosome = y_true[:, 1].cpu().numpy()
    y_probs_chromosome = y_probs[:, 1].cpu().numpy()
    sample_weight_chromosome = np.ones_like(y_true_chromosome)

    chromosome_scores = score_thresholds(y_true_chromosome, y_probs_chromosome, sample_weight_chromosome)
    store_best(chromosome_scores, parameters, 'chromosome_threshold', log_dir)

def apply_thresholds(y, parameters):
    """
    Transforms raw model probabilities into final scores using custom thresholds.

    This function rescales the probabilities for each class (plasmid, chromosome)
    such that the custom-defined threshold becomes the new 0.5 decision boundary.
    This allows for a standard prediction rule (score > 0.5) to be used after
    the transformation.
    """

    columns = []
    # iterate through each output class: 0 for plasmid, 1 for chromosome
    for (column_idx, which_parameter) in [(0, 'plasmid_threshold'), (1, 'chromosome_threshold')]:
        # retrieve the optimal threshold for the current class
        threshold = parameters[which_parameter]
        # get the original probability predictions for this class
        orig_column = y[:, column_idx]
        
        # Apply a piecewise scaling function. Values below the threshold are mapped
        # to the [0, 0.5] range, and values at or above the threshold are mapped
        # to the [0.5, 1] range.
        new_column = np.piecewise(
            orig_column,
            [orig_column < threshold, orig_column >= threshold],
            [lambda x : scale_number(x, 0, threshold, 0, 0.5), lambda x : scale_number(x, threshold, 1, 0.5, 1)]
        )
        columns.append(new_column)

    # combine the newly scaled columns and return as a single numpy array
    y_new = np.array(columns).transpose()
    return y_new

def scale_number(x, s1, e1, s2, e2):
    """
    Linearly scales a number from one range to another.
    """

    factor = (e2 - s2) / (e1 - s1)
    return (x - s1) * factor + s2

def store_best(scores, parameters, which, log_dir):
    """
    Finds the best threshold from a list of (threshold, F1 score) pairs
    and stores it in the parameters object.
    """

    # scores is a list of (threshold, F1 score) pairs
    if len(scores) > 0:
        # find index of maximum in scores[*][1]
        maxindex = max(range(len(scores)), key = lambda i : scores[i][1])
        threshold = scores[maxindex][0]
    else:
        threshold = 0.5
    parameters[which] = float(threshold)

    if log_dir is not None:
        filename = os.path.join(log_dir, which + ".csv")
        with open(filename, 'wt') as file:
            print(f"{which},f1", file=file)
            for x in scores:
                print(",".join(str(value) for value in x), file=file)

def score_thresholds(y_true, y_pred, weights):
    """
    Computes F1 scores for all possible decision thresholds for a single class.

    This function efficiently calculates the F1 score at every unique probability
    value in the predictions, treating each as a potential threshold.
    """

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
