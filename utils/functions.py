### Importing Modules
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
import numpy as np
import pandas as pd
import pickle
from Bio import AlignIO
import math
import argparse
from argparse import ArgumentParser
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import pytorch_lightning as pl
from collections import OrderedDict
from torchtext import vocab
from pytorch_lightning.loggers import CSVLogger
import random
from random import choice
import pathlib
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pathlib import Path
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
# from Levenshtein import distance as levenshtein_distance


# Compute Log Probability
def compute_scores_from_batch(batch, model, max_batch_size=128):
    """
    Computes the cross-entropy loss scores for a given batch of sequences using a pre-trained neural network model.
    Args:
        batch (torch.Tensor): A LongTensor of shape (b, L) with
            b = batch size
            L = sequence length.
        model (torch.nn.Module): A pre-trained neural network model that takes in a batch of sequences and returns
            four tensors: z_mean, z_log_var, encoded, and decoded.
        max_batch_size (int): The maximum batch size to use during processing.
    Returns:
        scores (torch.Tensor): A 1D tensor of shape (b,) containing the cross-entropy loss scores for each sequence in
            the batch.
    """
    # Split the batch into smaller sub-batches if necessary
    if batch.size(0) > max_batch_size:
        scores_list = []
        for i in range(0, batch.size(0), max_batch_size):
            sub_batch = batch[i:i + max_batch_size]  # Select the sub-batch
            sub_scores = compute_scores_from_batch(sub_batch, model, max_batch_size)  # Recursive call for sub-batch
            scores_list.append(sub_scores)
        scores = torch.cat(scores_list)  # Concatenate all sub-batch scores
    else:
        with torch.no_grad():  # We do not want training to occur during scoring
            # Pass the batch through the model to get the z_mean, z_log_var, encoded, and decoded tensors
            z_mean, z_log_var, encoded, decoded = model(batch)

            # Use z_mean instead of encoded to remove stochastic reparameterization
            decoded_from_mean = model.decoder(z_mean)

            # Compute the cross-entropy loss scores for each sequence in the batch
            scores = F.cross_entropy(decoded_from_mean, batch, reduction='none').sum(dim=-1)

    return scores

def generate_single_point_mutants(sequence, AAs, aa2ind):
    mutant_tensors = []
    for i, aa in enumerate(sequence):
        for new_aa in AAs:
            if new_aa != aa:  # Skip the wild-type amino acid
                mutant = sequence[:i] + new_aa + sequence[i+1:]
                mutant_tensor = torch.tensor([aa2ind[a] for a in mutant], dtype=torch.long).unsqueeze(0)  # Add a batch dimension
                mutant_tensors.append(mutant_tensor)
    
    # Stack the list of tensors to create a batch
    mutant_batch = torch.cat(mutant_tensors, dim=0)
    return mutant_batch

def create_vae_single_mutant_heatmap(WT, AAs, mutant_scores_list, wt_score, output_path='VAE_single_mutant_scores', figsize=(34, 4)):
    """
    Plots a heatmap of VAE mutant scores relative to wild-type (WT) amino acid likelihood.
    
    Parameters:
    - WT: list of chars, wild-type sequence with gaps represented by '-'
    - AAs: list of chars, amino acids
    - mutant_scores_list: list of floats, VAE scores for each mutant
    - wt_score: float, reference score for WT
    - output_path: str, base file path for saving the heatmap (without extension)
    - figsize: tuple, dimensions of the heatmap figure
    """
    
    # Initialize 2D array for heatmap
    n_positions = len([aa for aa in WT if aa != '-'])  # Count non-gap positions
    n_AAs = len(AAs)
    vae_scores = np.zeros((n_AAs, n_positions))
    
    # Populate VAE scores relative to WT
    position_counter = 0
    score_counter = 0
    for pos in range(len(WT)):
        if WT[pos] != '-':
            for aa_index, aa in enumerate(AAs):
                if aa != WT[pos]:
                    score = mutant_scores_list[score_counter]
                    relative_score = score - wt_score  # Assuming wt_score is a scalar
                    vae_scores[aa_index, position_counter] = relative_score
                    score_counter += 1
            position_counter += 1
    
    # Collect WT amino acid coordinates
    wt_coordinates = []
    position_counter = 0
    for pos, wt_aa in enumerate(WT):
        if wt_aa != '-':
            aa_index = AAs.index(wt_aa)
            wt_coordinates.append((aa_index, position_counter))
            position_counter += 1
            
    # Custom colormap
    colors = [(0, 'blue'), (0.5, 'white'), (1, 'red')]
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors)
    
    # Heatmap range settings
    min_score, max_score = np.min(vae_scores), np.max(vae_scores)
    abs_max_score = max(abs(min_score), abs(max_score))
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    cax = sns.heatmap(
        vae_scores, cmap=custom_cmap, ax=ax,
        vmin=-abs_max_score, vmax=abs_max_score,
        cbar_kws={'label': 'Relative VAE Score'}
    )
    
    # Add WT amino acid markers
    for y, x in wt_coordinates:
        ax.scatter(x + 0.5, y + 0.5, color='black', s=15)
    
    # Titles and labels
    ax.set_title('VAE Predicted "Likelihood" of Single Mutants Functioning Similar to WT1.0', fontsize=18)
    ax.set_ylabel('Amino Acid', fontsize=14)
    ax.set_xlabel('Amino Acid Position', fontsize=14)
    ax.text(n_positions + 18, -2, 'Less Likely', ha='center', va='center', fontsize=12)
    ax.text(n_positions + 18, len(AAs) + 2, 'More Likely', ha='center', va='center', fontsize=12)
    ax.set_yticks(np.arange(len(AAs)) + 0.5)
    ax.set_yticklabels(AAs)
    xticks_positions = np.arange(0, n_positions)
    xticks_labels = [str(pos + 1) if (pos + 1) % 10 == 0 else '' for pos in xticks_positions]
    ax.set_xticks(xticks_positions + 0.5)
    ax.set_xticklabels(xticks_labels)
    
    plt.tight_layout()
    plt.savefig(f'{output_path}.png')
    plt.savefig(f'{output_path}.svg')

def frange_cycle_cosine(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    """
    Generates a cyclical annealing cosine curve for KL weight scheduling.
    
    Parameters:
    - start: float, starting value of the cycle
    - stop: float, ending value of the cycle
    - n_epoch: int, total number of epochs
    - n_cycle: int, number of cycles
    - ratio: float, ratio of the cycle length for the cosine curve
    
    Returns:
    - np.array: array of weights for each epoch
    """
    L = np.ones(n_epoch + 1)
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)
    
    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_epoch):
            L[int(i + c * period)] = 0.5 - 0.5 * math.cos(v * math.pi)
            v += step
            i += 1
    return L

def plot_conv_vae_loss_curves(metrics_path, msa_length, output_path='./figures/Best_ConVAE_Loss_Curves_Combined.svg', n_cycle=1, figsize=(12, 12)):
    """
    Plots training and validation loss curves for a ConvVAE, including cross-entropy loss,
    KL divergence loss, and combined loss, with cyclical annealing for KL weights.

    Parameters:
    - metrics_path: str, path to the CSV file containing training and validation metrics
    - msa_length: int, length of the multiple sequence alignment (MSA) for normalization
    - output_path: str, path to save the generated loss curves plot
    - n_cycle: int, number of cycles for cyclical annealing in KL weight calculation
    - figsize: tuple, dimensions of the plot figure
    """
    
    # Load metrics and separate training/validation loss
    pt_metrics = pd.read_csv(metrics_path)
    train = pt_metrics[~pt_metrics.train_ce_loss.isna()]
    val = pt_metrics[~pt_metrics.val_ce_loss.isna()]
    
    # Plot cross-entropy loss
    plt.figure(figsize=figsize)
    plt.subplot(3, 1, 1)
    plt.plot(train.epoch, train.train_ce_loss / msa_length, label='Train')
    plt.plot(val.epoch, val.val_ce_loss / msa_length, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.legend(loc='best')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Process KL divergence loss for plotting
    train_kl_divergence = pt_metrics['val_kl_divergence'].where(pt_metrics['val_ce_loss'].isna()).dropna()
    val_kl_divergence = pt_metrics['val_kl_divergence'].where(pt_metrics['val_ce_loss'].notna()).dropna()
    
    # Generate KL weights over epochs
    epochs = len(train.epoch)
    kl_weights = frange_cycle_cosine(0, 1, epochs, n_cycle)
    
    # Plot KL divergence loss and KL weight
    ax1 = plt.subplot(3, 1, 2)
    ax1.plot(train.epoch, train_kl_divergence / msa_length, label='Train')
    ax1.plot(val.epoch, val_kl_divergence / msa_length, label='Validation')
    ax1.plot(range(epochs), kl_weights[:epochs], label='Dkl Weight', linestyle='--', color='green')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Kullback–Leibler Divergence Loss / Weight')
    ax1.legend(loc='best')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot combined loss (cross-entropy + KL divergence)
    train_combined_loss = train.train_ce_loss + train_kl_divergence
    val_combined_loss = val.val_ce_loss + val_kl_divergence
    plt.subplot(3, 1, 3)
    plt.plot(train.epoch, train_combined_loss / msa_length, label='Training Combined Loss')
    plt.plot(val.epoch, val_combined_loss / msa_length, label='Validation Combined Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Combined Loss')
    plt.legend(loc='best')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save and display plot
    plt.tight_layout()
    plt.savefig(f'{output_path}.svg')
    plt.savefig(f'{output_path}.png')
    plt.show()

# Define the function to load splits
def load_splits(path):
    """Load the data splits from a file at the given path"""
    with open(path, 'rb') as f:
        train_idx, val_idx, test_idx = pickle.load(f)
    return train_idx, val_idx, test_idx

def indices_to_sequence(indices, ind2aa):
    """Converts a list of indices back to an amino acid string using ind2aa mapping."""
    return ''.join([ind2aa[idx] for idx in indices])

def hamming_dist(s1, s2):
    """Calculates the Hamming distance between two sequences"""
    return sum(1 for x, y in zip(s1, s2) if x != y and x != '-' and y != '-') # Quantify sequence similarity

def plot_heatmap_for_configuration(df, AAs, title, save_path, WT, data_type='mutations'):
    
    # Unzip sequences to align positions
    alignment = tuple(zip(*df.Sequence))
    
    # Count AAs
    if data_type == 'mutations':
        AA_count = np.array([[sum(1 for seq_at_pos in alignment[pos] if seq_at_pos == a and WT[pos] != a) for a in AAs] for pos in range(len(WT))])
    else:
        AA_count = np.array([[p.count(a) for a in AAs] for p in alignment]) # raw AA counts for MSA

    Magma_r = plt.cm.magma_r(np.linspace(0, 1, 256))
    Magma_r[0] = [0, 0, 0, 0.03]  # Set the first entry (corresponding to 0 value) to white
    # Magma_r[0] = [0.9, 0.9, 0.9, 1]  # Set the first entry (corresponding to 0 value) to grey
    cmap = LinearSegmentedColormap.from_list("Modified_Magma_r", Magma_r, N=256)

    # Plot the heatmap
    plt.figure(figsize=(len(WT)//3,6))
    heatmap = sns.heatmap(AA_count.T, cmap=cmap, square=True, linewidths=0.003, linecolor='0.7')
    cbar = heatmap.collections[0].colorbar
    if data_type == 'mutations':
        cbar.set_label('Count of Amino Acid Mutations', fontsize=16)
    else:
        cbar.set_label('Raw Count of Amino Acids', fontsize=16)
    cbar.ax.tick_params(labelsize=12)
    pos = cbar.ax.get_position()  # Get the original position
    cbar.ax.set_position([pos.x0 - 0.03, pos.y0, pos.width, pos.height])  # Shift the colorbar closer
    plt.yticks(np.arange(len(AAs)) + 0.5, AAs)
    plt.xlabel('Position', fontsize=18)
    plt.ylabel('Amino Acid', fontsize=18)
    plt.title(title)

    # Add black dots for WT sequence
    for pos, aa in enumerate(WT):
        if aa in AAs:  # Check if the AA is one of the considered AAs
            aa_index = AAs.index(aa)
            # Plot black dot; adjust dot size with 's' as needed
            plt.scatter(pos + 0.5, aa_index + 0.5, color='black', s=30)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='WT')]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Save the plot
    plt.savefig(save_path)
    plt.show()
    plt.close()

def convert_fasta_msa_to_dataframe(fasta_file):
    # Read sequences from FASTA file
    sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]
    # Convert to DataFrame
    msa_df = pd.DataFrame(sequences, columns=['Sequence'])
    return msa_df

def score_sequences_with_vae_mutant_marginal(df, WT, vae, aa2ind):
    """
    Scores sequences in a DataFrame using a single VAE model.
    
    Parameters:
    - df: pd.DataFrame with a column 'Sequence' containing protein sequences to be scored.
    - WT: str, wild-type sequence to use for reference.
    - vae: model, pre-trained VAE model.
    - aa2ind: function or mapping, converts amino acids to indices.
    
    Returns:
    - df: pd.DataFrame with an added column 'vae_score' containing scores for each sequence.
    """
    
    # Tokenize WT sequence and add a batch dimension
    tokenized_WT = torch.tensor(aa2ind(list(WT))).unsqueeze(0)

    # Initialize column for scores
    df['vae_mutant_marginal'] = 0.0

    # Score each sequence
    for j in range(len(df)):
        sequence = df['Sequence'].iloc[j]
        sequence_tensor = torch.tensor(aa2ind(list(sequence))).unsqueeze(0)  # Adds a batch dimension

        with torch.no_grad():
            # VAE score
            z_mean, z_log_var, encoded, decoded = vae(sequence_tensor)
            logits = vae.decoder(z_mean)
            WT_vae_score = F.cross_entropy(logits, tokenized_WT, reduction='none')
            mutant_vae_score = F.cross_entropy(logits, sequence_tensor, reduction='none')
            vae_score = -1*(mutant_vae_score - WT_vae_score).sum().item()
            df.at[j, 'vae_mutant_marginal'] = vae_score

        # Print progress every 1000 sequences
        if (j + 1) % 1000 == 0:
            print(f'{j + 1} sequences have been scored.')

    return df

def generate_protein_variants(WT, MSA_train, MSA_validation, MSA_test, model, slen, aa2ind, ind2aa, filepath, seed=88):
    """
    Generates shuffled variants of WT and random proteins, randomly samples columns of the MSA test set,
    and scores the MSA_train, MSA_validation, and MSA_test sets.

    Parameters:
    - WT: torch.Tensor, tensor representation of the wild-type sequence.
    - MSA_train: list of str, list of training MSA sequences.
    - MSA_validation: list of str, list of validation MSA sequences.
    - MSA_test: list of str, list of withheld MSA sequences.
    - model: Model to compute scores for MSA-related proteins.
    - slen: int, sequence length.
    - seed: int, random seed for reproducibility.

    Returns:
    - dict containing scores and string sequences for MSA shuffled variants, random protein sequences,
      MSA train, validation, and test sets, and a new MSA test set with randomly sampled columns.
    """

    # Set the seed for reproducibility
    torch.manual_seed(seed)

    # Convert WT to tensor if it's in list format
    WT_tensor = torch.tensor([aa2ind[aa] for aa in WT])

    # Score MSA_train, MSA_validation, and MSA_test sets
    MSA_train_tensors = torch.stack([torch.tensor([aa2ind[aa] for aa in seq]) for seq in MSA_train])
    MSA_train_scores = compute_scores_from_batch(MSA_train_tensors, model)
    MSA_validation_tensors = torch.stack([torch.tensor([aa2ind[aa] for aa in seq]) for seq in MSA_validation])
    MSA_validation_scores = compute_scores_from_batch(MSA_validation_tensors, model)
    MSA_test_tensors = torch.stack([torch.tensor([aa2ind[aa] for aa in seq]) for seq in MSA_test])
    MSA_test_scores = compute_scores_from_batch(MSA_test_tensors, model)

    # Generate shuffled variants of WT and compute scores
    MSA_shuffled_list = torch.zeros((len(MSA_test), slen), dtype=torch.long)
    MSA_shuffled_sequences = []
    for i in range(len(MSA_test)):
        shuffled_sequence = WT_tensor[torch.randperm(slen)]
        MSA_shuffled_list[i] = shuffled_sequence
        MSA_shuffled_sequences.append(indices_to_sequence(shuffled_sequence.tolist(), ind2aa))
    MSA_shuffled_scores = compute_scores_from_batch(MSA_shuffled_list, model)

    # Generate random protein sequences and compute scores
    MSA_random_prots = torch.randint(21, (len(MSA_test), slen), dtype=torch.long)
    MSA_random_sequences = [indices_to_sequence(prot.tolist(), ind2aa) for prot in MSA_random_prots]
    MSA_random_scores = compute_scores_from_batch(MSA_random_prots, model)

    # Build new sequences by randomly sampling an amino acid from each position across MSA_test
    sampled_MSA_test = []
    for _ in range(len(MSA_test)):
        sampled_sequence = ''.join([seq[torch.randint(len(MSA_test), (1,)).item()] for seq in zip(*MSA_test)])
        sampled_MSA_test.append(sampled_sequence)
    sampled_MSA_test_tensors = torch.stack([torch.tensor([aa2ind[aa] for aa in seq]) for seq in sampled_MSA_test])
    sampled_MSA_test_scores = compute_scores_from_batch(sampled_MSA_test_tensors, model)

    # Convert scores to numpy for external analysis
    MSA_train_scores_np = MSA_train_scores.numpy()
    MSA_validation_scores_np = MSA_validation_scores.numpy()
    MSA_test_scores_np = MSA_test_scores.numpy()
    MSA_shuffled_scores_np = MSA_shuffled_scores.numpy()
    MSA_random_scores_np = MSA_random_scores.numpy()
    sampled_MSA_test_scores_np = sampled_MSA_test_scores.numpy()

    # Save sequences as FASTA files
    def save_fasta(sequences, filename, description_prefix):
        fasta_path = Path(filepath) / filename
        with fasta_path.open('w') as f:
            for i, seq in enumerate(sequences):
                f.write(f">{description_prefix}_{i}\n{seq}\n")

    save_fasta(MSA_shuffled_sequences, "WT_shuffled.fasta", "MSA_shuffled")
    save_fasta(MSA_random_sequences, "Random_proteins.fasta", "MSA_random")
    save_fasta(sampled_MSA_test, "randomly_sampling_MSA_test_columns.fasta", "sampled_MSA_test")
    save_fasta(MSA_test, "MSA_test.fasta", "MSA_test")

    return {
        "MSA_train_scores": MSA_train_scores_np,
        "MSA_validation_scores": MSA_validation_scores_np,
        "MSA_test_scores": MSA_test_scores_np,
        "MSA_shuffled_scores": MSA_shuffled_scores_np,
        "MSA_shuffled_sequences": MSA_shuffled_sequences,
        "random_protein_scores_MSA": MSA_random_scores_np,
        "random_protein_sequences": MSA_random_sequences,
        "sampled_MSA_test_scores": sampled_MSA_test_scores_np,
        "sampled_MSA_test_sequences": sampled_MSA_test,
    }


def ESM2_mutant_marginal(model, tokenizer, sequence, WT):
    '''
    Masked marginal probability (1 forward pass per mutation per sequence)
    from https://proceedings.neurips.cc/paper_files/paper/2021/file/f51338d736f95dd42427296047067694-Supplemental.pdf
    
    Score sequences by masking every mutated position and computing the log odds ratio between the mutated and wild-type
    residues at each mutated position, assuming an additive model when a sequence contains multiple mutations
    '''
    # Tokenize WT and mutated sequence for ESM2
    WT_inputs = tokenizer(WT, return_tensors='pt', padding=True, truncation=True)
    inputs = tokenizer(sequence, return_tensors='pt', padding=True, truncation=True)

    # Determine mutated positions
    mutated_positions = [i for i, (wt, mt) in enumerate(zip(WT, sequence)) if wt != mt]

    # Get input_ids and prepare for masked operation
    input_ids = inputs['input_ids'].clone()
    scores = []

    with torch.no_grad():
        # Iterate only over mutated positions
        for index in mutated_positions:
            masked_input_ids = input_ids.clone()
            masked_index = index + 1  # Adjust index for tokenizer specifics (CLS token at the start)
            masked_input_ids[0, masked_index] = tokenizer.mask_token_id
            
            # Get model output for masked input
            outputs = model(masked_input_ids)
            logits = outputs.logits

            # Calculate log probabilities at the masked position
            log_probs = F.log_softmax(logits, dim=-1)

            # Get the log probabilities of the actual wildtype and mutant amino acids at this position
            wt_log_prob = log_probs[0, masked_index, WT_inputs['input_ids'][0, masked_index]]
            mutant_log_prob = log_probs[0, masked_index, input_ids[0, masked_index]]

            # Compute the score for this position (mutant - WT)
            score = (mutant_log_prob - wt_log_prob).item()
            scores.append(score)

    # Sum scores for all mutated positions
    ESM2_score = sum(scores)
    return ESM2_score

# # Function to calculate median logits and regression values
# def calculate_median_class_logits(logits_dict):
#     median_class_logits = []

#     for idx in logits_dict:
#         # Calculate the median of the class logits across models
#         median_logits = np.median(logits_dict[idx], axis=0)
#         median_class_logits.append(median_logits)

#     return np.array(median_class_logits)

# Function to calculate median regression values
def calculate_median_regression_values(reg_dict):
    median_reg_values = []

    for idx in reg_dict:
        # Calculate the median of the regression values across models
        median_reg = np.median(reg_dict[idx], axis=0)
        median_reg_values.append(np.squeeze(median_reg))  # Use np.squeeze to remove singleton dimensions

    return np.array(median_reg_values)

# Define the function to apply mutations to the wild-type sequence
def apply_mutations(WT, mutations):
    """Apply mutations to the wild-type sequence."""
    seq_list = list(WT)  # Convert WT to a list for mutability
    for mutation in mutations:
        original_aa, pos, new_aa = mutation[0], int(mutation[1:-1]), mutation[-1]
        seq_list[pos] = new_aa
    return ''.join(seq_list)

def find_all_files(base_path, num_mut):
    """
    Find all pickle files across all available num_steps and versions for the given num_mut.
    """
    steps_pattern = re.compile(rf'{num_mut}mut_all_models_(\d+)steps')
    files = []

    # Find all num_steps directories
    available_steps = [
        (int(re.search(steps_pattern, d).group(1)), d)
        for d in os.listdir(base_path)
        if re.search(steps_pattern, d)
    ]

    if not available_steps:
        raise ValueError(f"No valid num_steps options found for {num_mut} mutations!")

    for num_steps, steps_dir in available_steps:
        # Find all valid version files in the num_steps directory
        version_pattern_1 = re.compile(rf'best_Gre2_redesign_round_1_{num_mut}mut_start_pos0_all_models_v(\d+)\.pickle')
        version_pattern_2 = re.compile(rf'close_sequences_Gre2_redesign_round_1_{num_mut}mut_start_pos0_v(\d+)\.pickle')
        full_steps_dir = os.path.join(base_path, steps_dir)

        available_files_1 = [
            os.path.join(full_steps_dir, f)
            for f in os.listdir(full_steps_dir)
            if re.search(version_pattern_1, f)
        ]

        available_files_2 = [
            os.path.join(full_steps_dir, f)
            for f in os.listdir(full_steps_dir)
            if re.search(version_pattern_2, f)
        ]

        # Extend files list with both sets of available files
        files.extend(available_files_1)
        files.extend(available_files_2)

    return files

def find_all_files_with_F85L(base_path, num_mut):
    """
    Find all pickle files across all available num_steps and versions for the given num_mut.
    """
    steps_pattern = re.compile(rf'{num_mut}mut_all_models_(\d+)steps')
    files = []

    # Find all num_steps directories
    available_steps = [
        (int(re.search(steps_pattern, d).group(1)), d)
        for d in os.listdir(base_path)
        if re.search(steps_pattern, d)
    ]

    if not available_steps:
        raise ValueError(f"No valid num_steps options found for {num_mut} mutations!")

    for num_steps, steps_dir in available_steps:
        # Find all valid version files in the num_steps directory
        version_pattern_1 = re.compile(rf'best_Gre2_redesign_round_1_with_F85L_{num_mut}mut_start_pos0_all_models_assayW(\d+)\_cuda(\d+)\_v(\d+)\.pickle')
        version_pattern_2 = re.compile(rf'close_sequences_Gre2_redesign_round_1_with_F85L_{num_mut}mut_start_pos0_assayW(\d+)\_cuda(\d+)\_v(\d+)\.pickle')


        full_steps_dir = os.path.join(base_path, steps_dir)

        available_files_1 = [
            os.path.join(full_steps_dir, f)
            for f in os.listdir(full_steps_dir)
            if re.search(version_pattern_1, f)
        ]

        available_files_2 = [
            os.path.join(full_steps_dir, f)
            for f in os.listdir(full_steps_dir)
            if re.search(version_pattern_2, f)
        ]

        # Extend files list with both sets of available files
        files.extend(available_files_1)
        files.extend(available_files_2)

    return files

def load_data_and_scores(files, WT, ESM2_rxn_rate_1=None, VAE=None):
    """
    Load and process data from a list of files.
    """
    results = []
    for filepath in files:
        try:
            with open(filepath, 'rb') as f:
                # Load pickle data
                data = pickle.load(f)  # Assumes data is a tuple

                # process close sequence files
                if isinstance(data, list) and all(isinstance(row, tuple) and len(row) == 5 for row in data):
                    for row in data:  # Process each row
                        mutant, dist, ESM2_norm, SolubleMPNN_norm, VAE_norm = row
                        mutations = list(mutant)
                        num_muts = len(mutations)
                        dist = dist
                        ESM2_rxn_rate_1_score = ESM2_norm
                        SolubleMPNN_score = SolubleMPNN_norm
                        VAE_score = VAE_norm

                        if any(v in filepath for v in ['v120']):
                            seq = apply_mutations(WT, mutations)
                        elif any(v in filepath for v in ['v1000', 'v1001', 'v1002', 'v2000', 'v2001', 'v2002']):
                            mutations.append('F84L')
                            seq = apply_mutations(WT, mutations)
                        else:
                            if num_muts == 2:
                                if any(v in filepath for v in ['v20', 'v21', 'v50', 'v51', 'v52', 'v53', 'v54']): # added F86L by accident and rejected during simulated annealing
                                    seq = apply_mutations(WT, mutations)
                                else:
                                    mutations.append('C85L') # added C85L by accident and accepted during simulated annealing
                                    seq = apply_mutations(WT, mutations)
                            else:
                                seq = apply_mutations(WT, mutations) # add mutations to Gre2
            
                        # Append results
                        results.append({
                            "Sequence": seq,
                            "Mutations": mutations,
                            "Mutation_Type": num_muts,
                            "Distance": dist,
                            "ESM2_Score": ESM2_rxn_rate_1_score,
                            "SolubleMPNN_Score": SolubleMPNN_score,
                            "VAE_Score": VAE_score,
                            "File": filepath})     
                
                else:
                    if len(data) == 5:
                        best_mut = data

                        # Define AA sequence
                        mutations = list(best_mut[0])
                        num_muts = len(mutations)
                        dist = best_mut[1]
                        ESM2_rxn_rate_1_score = best_mut[2]
                        SolubleMPNN_score = best_mut[3]
                        VAE_score = best_mut[4]
        
                        if any(v in filepath for v in ['v120']):
                            seq = apply_mutations(WT, mutations)
                        elif any(v in filepath for v in ['v1000', 'v1001', 'v1002', 'v2000', 'v2001', 'v2002']):
                            mutations.append('F84L')
                            seq = apply_mutations(WT, mutations)
                        else:
                            if num_muts == 2:
                                if any(v in filepath for v in ['v20', 'v21', 'v50', 'v51', 'v52', 'v53', 'v54']): # added F86L by accident and rejected during simulated annealing
                                    seq = apply_mutations(WT, mutations)
                                else:
                                    mutations.append('C85L') # added C85L by accident and accepted during simulated annealing
                                    seq = apply_mutations(WT, mutations)
                            else:
                                seq = apply_mutations(WT, mutations) # add mutations to Gre2
        
                        # Append results
                        results.append({
                            "Sequence": seq,
                            "Mutations": mutations,
                            "Mutation_Type": num_muts,
                            "Distance": dist,
                            "ESM2_Score": ESM2_rxn_rate_1_score,
                            "SolubleMPNN_Score": SolubleMPNN_score,
                            "VAE_Score": VAE_score,
                            "File": filepath
                        })


                    elif len(data[0]) == 5:
                        best_mut, _, _, _, _ = data
        
                        # Define AA sequence
                        mutations = list(best_mut[0])
                        num_muts = len(mutations)
                        dist = best_mut[1]
                        ESM2_rxn_rate_1_score = best_mut[2]
                        SolubleMPNN_score = best_mut[3]
                        VAE_score = best_mut[4]
        
                        if any(v in filepath for v in ['v120']):
                            seq = apply_mutations(WT, mutations)
                        elif any(v in filepath for v in ['v1000', 'v1001', 'v1002', 'v2000', 'v2001', 'v2002']):
                            mutations.append('F84L')
                            seq = apply_mutations(WT, mutations)
                        else:
                            if num_muts == 2:
                                if any(v in filepath for v in ['v20', 'v21', 'v50', 'v51', 'v52', 'v53', 'v54']): # added F86L by accident and rejected during simulated annealing
                                    seq = apply_mutations(WT, mutations)
                                else:
                                    mutations.append('C85L') # added C85L by accident and accepted during simulated annealing
                                    seq = apply_mutations(WT, mutations)
                            else:
                                seq = apply_mutations(WT, mutations) # add mutations to Gre2
        
                        # Append results
                        results.append({
                            "Sequence": seq,
                            "Mutations": mutations,
                            "Mutation_Type": num_muts,
                            "Distance": dist,
                            "ESM2_Score": ESM2_rxn_rate_1_score,
                            "SolubleMPNN_Score": SolubleMPNN_score,
                            "VAE_Score": VAE_score,
                            "File": filepath
                        })
                    
                    elif len(data[0]) == 2:
                        best_mut = data
        
                        # Define AA sequence
                        mutations = list(best_mut[0])
                        num_muts = len(mutations)
                        dist = best_mut[1]
                        ESM2_rxn_rate_1_score = best_mut[2]
                        SolubleMPNN_score = best_mut[3]
                        VAE_score = best_mut[4]
        
                        if any(v in filepath for v in ['v120']):
                            seq = apply_mutations(WT, mutations)
                        elif any(v in filepath for v in ['v1000', 'v1001', 'v1002', 'v2000', 'v2001', 'v2002']):
                            mutations.append('F84L')
                            seq = apply_mutations(WT, mutations)
                        else:
                            if num_muts == 2:
                                if any(v in filepath for v in ['v20', 'v21', 'v50', 'v51', 'v52', 'v53', 'v54']): # added F86L by accident and rejected during simulated annealing
                                    seq = apply_mutations(WT, mutations)
                                else:
                                    mutations.append('C85L') # added C85L by accident and accepted during simulated annealing
                                    seq = apply_mutations(WT, mutations)
                            else:
                                seq = apply_mutations(WT, mutations) # add mutations to Gre2
        
                        # Append results
                        results.append({
                            "Sequence": seq,
                            "Mutations": mutations,
                            "Mutation_Type": num_muts,
                            "Distance": dist,
                            "ESM2_Score": ESM2_rxn_rate_1_score,
                            "SolubleMPNN_Score": SolubleMPNN_score,
                            "VAE_Score": VAE_score,
                            "File": filepath
                        })

                    else:
                        print('Error in logic handling for simulated annealing results. Check length of data[0]')


        except FileNotFoundError:
            print(f"File not found: {filepath}")
        except Exception as e:
            print(f"Error loading file {filepath}: {e}")
    
    return pd.DataFrame(results)

def generate_random_variants(sequence, mutations):
    variant = list(sequence)
    valid_positions = [i for i, char in enumerate(sequence) if char != '-']
    positions = random.sample(valid_positions, mutations)
    for pos in positions:
        amino_acid = random.choice('ACDEFGHIKLMNPQRSTVWY')
        while variant[pos] == amino_acid:
            amino_acid = random.choice('ACDEFGHIKLMNPQRSTVWY')
        variant[pos] = amino_acid
    return ''.join(variant)

def cluster_sequences_kmeans(dataframe, num_clusters=75, seed=3):
    """
    Performs K-Means clustering on sequences and assigns cluster labels.

    Parameters:
    - dataframe (pd.DataFrame): DataFrame containing sequences.
    - num_clusters (int): Number of clusters to create.
    - seed (int): Random seed for reproducibility.

    Returns:
    - pd.DataFrame: DataFrame with assigned cluster labels.
    """
    random.seed(seed)
    np.random.seed(seed)

    categories = np.array(list('ACDEFGHIKLMNPQRSTVWY-'))  # Define amino acid categories
    sequences = dataframe['Sequence'].values

    # One-hot encoding for sequences
    encoder = OneHotEncoder(categories=[categories], sparse_output=False, dtype=int)
    encoded_sequences = [encoder.fit_transform(np.array(list(seq)).reshape(-1, 1)).flatten() for seq in sequences]
    encoded_sequences = np.array(encoded_sequences)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=seed)
    kmeans.fit(encoded_sequences)

    # Assign cluster labels
    dataframe = dataframe.copy()  # Ensure we don't modify the original DataFrame
    dataframe['Kmeans_Cluster'] = kmeans.predict(encoded_sequences)

    return dataframe

# Function to compute pairwise Hamming distance matrix
def compute_hamming_distance_matrix(sequences):
    """Compute pairwise Hamming distance matrix (normalized)."""
    n = len(sequences)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = sum(a != b for a, b in zip(sequences[i], sequences[j])) / len(sequences[i])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    return distance_matrix

def cluster_all_mutation_types(filtered_results, WT, num_clusters_samp=75, min_samples=2, seed=3):
    """
    Clusters all mutation types together using DBSCAN and selects representative sequences.

    Parameters:
    - filtered_results (pd.DataFrame): DataFrame containing sequences and mutation types.
    - WT (str): Wild-type sequence.
    - min_samples (int): Minimum samples for DBSCAN clustering.
    - seed (int): Random seed for reproducibility.

    Returns:
    - pd.DataFrame: Selected sequences from different clusters.
    """
    np.random.seed(seed)
    
    # Extract sequences
    sequences = filtered_results["Sequence"].tolist()
    
    if len(sequences) < 2:
        return pd.DataFrame(columns=filtered_results.columns)  # Return empty DataFrame if not enough sequences
    
    # Compute the Hamming distance matrix for all sequences
    distance_matrix = compute_hamming_distance_matrix(sequences)

    # Compute an overall eps value based on mutation type statistics
    median_mutation_type = filtered_results["Mutation_Type"].median()
    print('median_mutation_type: ', median_mutation_type)
    
    eps_value = median_mutation_type / len(WT) - (0.02) / (median_mutation_type * len(WT))
    print("Overall eps_value:", eps_value)

    # Perform DBSCAN clustering
    clustering = DBSCAN(metric="precomputed", eps=eps_value, min_samples=min_samples).fit(distance_matrix)

    # Assign cluster labels to the entire dataset
    filtered_results = filtered_results.assign(DBSCAN_Cluster=clustering.labels_)

    # Get unique clusters and randomly select up to 25 clusters (excluding noise, i.e., -1)
    unique_clusters = np.unique(filtered_results["DBSCAN_Cluster"])
    unique_clusters = unique_clusters[unique_clusters != -1]  # Exclude noise points (-1)

    selected_clusters = np.random.choice(unique_clusters, size=min(num_clusters_samp, len(unique_clusters)), replace=False)
    print("Selected clusters:", selected_clusters)

    selected_sequences = []

    for cluster in selected_clusters:
        cluster_data = filtered_results[filtered_results["DBSCAN_Cluster"] == cluster]
        if not cluster_data.empty:
            selected_sequences.append(cluster_data.sample(1))  # Randomly select one sequence per cluster

    return pd.concat(selected_sequences, ignore_index=True), filtered_results



