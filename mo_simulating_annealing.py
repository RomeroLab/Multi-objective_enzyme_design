#!/usr/bin/env python
# coding: utf-8

# import packages
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import pytorch_lightning as pl
from collections import OrderedDict
from torchtext import vocab
import matplotlib.pyplot as plt
import os
import random
import pickle
import csv
import subprocess
import importlib.util
from pathlib import Path
from coral_pytorch.dataset import corn_label_from_logits
from transformers import AutoModelForMaskedLM, AutoTokenizer

# import models
from models.ConvVAE import (get_msa_from_fasta, ProtDataModule, ConvVAE)
from models.partial_esm2_ft_w_MLP import (finetuning_ESM2_with_mse_loss)

# import helper functions
from utils.functions import (compute_scores_from_batch, convert_fasta_msa_to_dataframe, score_sequences_with_vae_mutant_marginal)
from utils.running_solublempnn import (load_fasta_with_names, write_fasta, parse_fasta, load_npz_scores)
from utils.simulated_annealing_utils import (get_non_gap_indices, generate_all_point_mutants, mut2seq, find_top_n_mutations,
    generate_random_mut_non_gap_indices, SA_optimizer, seq2fitness_handler)

# Resolve all repository paths relative to this script. The helper utilities
# invoke protein_mpnn_run.py with repository-relative paths, so use the
# repository root as the working directory regardless of where this script is
# launched from.
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data" / "finetuned_esm2"
MODELS_DIR = REPO_ROOT / "models"
STRUCTURES_DIR = REPO_ROOT / "structures"
SEQS_TO_SCORE_DIR = REPO_ROOT / "seqs_to_score"
OUTPUTS_DIR = REPO_ROOT / "outputs"
os.chdir(REPO_ROOT)

# basic parameters
AAs = 'ACDEFGHIKLMNPQRSTVWY-' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
# WT = "MSVFVSGANGFIAQHIVDLLLKEDYKVIGSARSQEKAENLTEAFGNNPKFSMEVVPDISKLDAFDHVFQKHGKDIKIVLHTASPFCFDITDSERDLLIPAVNGVKGILHSIKKYAADSVERVVLTSSYAAVFDMAKENDKSLTFNEESWNPATWESCQSDPVNAYCGSKKFAEKAAWEFLEENRDSVKFELTAVNPVYVFGPQMFDKDVKKHLNTSCELVNSLMHLSPEDKIPELFGGYIDVRDVAKAHLVAFQKRETIGQRLIVSEARFTMQDVLDILNEDFPVLKGNIPVGKPGSGATHNTLGATLDNKKSKKLLGFKFRNLKETIDDTASQILKFEGRI" # GRE2
WT =   "MSVFVSGANGFIAQHIVDLLLKEDYKVIGSARSQEKAENLTEAFGNNPKFSMEVVPDISKLDAFDHVFQKHGKDIKIVLHTASPLCFDITDSERDLLIPAVNGVKGILHSIKKYAADSVERVVLTSSYAAVFDMAKENDKSLTFNEESWNPATWESCQSDPVNAYCGSKKFAEKAAWEFLEENRDSVKFELTAVNPVYVFGPQMFDKDVKKHLNTSCELVNSLMHLSPEDKIPELFGGYIDVRDVAKAHLVAFQKRETIGQRLIVSEARFTMQDVLDILNEDFPVLKGNIPVGKPGSGATHNTLGATLDNKKSKKLLGFKFRNLKETIDDTASQILKFEGRI" # F85L
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap
ind2aa = {aa2ind[a]: a for a in AAs} # Create a reverse mapping for indices to amino acids
num_mut = 5 # ! # 3 or 5
assay_data_weight = 4 # ! # 1 or 4
num_SolubleMPNN_samples = 5
cuda_num = 3 # ! 0, 1, 2, 3

# simulated annealing parameters (single model or all models only)
use_ESM2 = True
use_VAE = True
use_SolubleMPNN = True
run_name = 'Gre2_redesign_round_1_with_F85L'
start_temp = -0.5
final_temp = -2.25
fixed_window_size = 0
WT_no_gaps = WT
non_gap_indices = get_non_gap_indices(WT)
mutating_window_size = len(WT_no_gaps) # size of window where amino acids can be altered
start_position = 0 # position window where amino acids can be altered begins
AAs_SA_options = 'ACDEFGHIKLMNPQRSTVWY'
AA_options = [tuple([AA for AA in AAs_SA_options]) for i in range(len(WT))]
AA_options[non_gap_indices[0]] = WT_no_gaps[0] # Keep start codon
AA_options[non_gap_indices[84]] = WT_no_gaps[84] # Keep F85L
seed = random.randint(0, 100000) # Set random seeds for reproducibility
random.seed(seed)
np.random.seed(seed)


def validate_required_paths():
    required_paths = []
    if use_VAE:
        required_paths.append(MODELS_DIR / "Best_ConvVAE.ckpt")
    if use_ESM2:
        required_paths.extend([
            DATA_DIR / "normalized_processed_BI_R1_dataset_w_quantiles.pkl",
            MODELS_DIR / "finetuned_ESM2_for_reaction_rate.pt",
        ])
    if use_SolubleMPNN:
        required_paths.extend([
            STRUCTURES_DIR / "4PVC.pdb1",
            MODELS_DIR / "solublempnn" / "soluble_model_weights" / "v_48_020.pt",
            REPO_ROOT / "protein_mpnn_run.py",
        ])

    missing = [path for path in required_paths if not path.is_file()]
    if use_SolubleMPNN and not (REPO_ROOT / "protein_mpnn_utils.py").is_file():
        if importlib.util.find_spec("protein_mpnn_utils") is None:
            missing.append(REPO_ROOT / "protein_mpnn_utils.py")
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required annealing files:\n{formatted}")


validate_required_paths()
SEQS_TO_SCORE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# VAE parameters
if use_VAE:
    batch_size = 16
    ks = 17
    nlatent = 64
    epochs = 1000
    learning_rate = 0.0001
    slen = len(WT)
    n_cycle = 1
    factor_2 = 16
    factor_3 = 1
    dim_4 = 400
    VAE = ConvVAE(slen, ks, nlatent, learning_rate, epochs, n_cycle, factor_2, factor_3, dim_4)
    checkpoint = torch.load(MODELS_DIR / 'Best_ConvVAE.ckpt', map_location=device)
    state_dict = checkpoint['state_dict']  # Extract only the state_dict
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    VAE.load_state_dict(state_dict)
    VAE.to(device)
else:
    VAE = None

# ESM2 parameters
if use_ESM2:
    data_filepath = DATA_DIR / 'normalized_processed_BI_R1_dataset_w_quantiles.pkl'
    label = 'Quantile_Rxn_Rate_at_0.05mgml' # ESM2 finetuned to predict reaction rate at 0.05 mg/ml
    version = 1
    filepath = f'finetuning_ESM2_with_{data_filepath.name}_for_{label}'
    checkpoint_path = MODELS_DIR / 'finetuned_ESM2_for_reaction_rate.pt'
    huggingface_identifier ='esm2_t33_650M_UR50D'
    ESM2 = AutoModelForMaskedLM.from_pretrained(f"facebook/{huggingface_identifier}")
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{huggingface_identifier}")
    model_identifier = huggingface_identifier
    token_format = 'ESM2'
    num_unfrozen_layers = 27
    num_layers_unfreeze_each_epoch = 15
    max_num_layers_unfreeze_each_epoch = 36
    epoch_threshold_to_unlock_ESM2 = 100
    epochs = 350
    warm_restart = 1
    use_scheduler = 1
    WD = 0.005
    grad_clip_threshold = 3.0
    lr_mult = 1
    lr_mult_factor = 1
    seed = 3
    learning_rate = 1e-6
    reinit_optimizer = 0
    using_EMA = 1
    decay = 0.8
    batch_size = 16
    embedding_run_name = 'all_tokens'
    slen = len(WT)
    num_reg_tasks = 1
    reg_run_name = 'mse'
    reg_weights = 1
    ESM2_rxn_rate_1 = finetuning_ESM2_with_mse_loss(ESM2, huggingface_identifier, tokenizer, num_unfrozen_layers, num_layers_unfreeze_each_epoch, max_num_layers_unfreeze_each_epoch,
                     epochs, batch_size, seed, embedding_run_name,
                     learning_rate, lr_mult, lr_mult_factor,
                     WD, reinit_optimizer, grad_clip_threshold, use_scheduler, warm_restart,
                     slen, reg_weights, num_reg_tasks, reg_run_name,
                     using_EMA, decay, epoch_threshold_to_unlock_ESM2, WT, str(data_filepath))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ESM2_rxn_rate_1.load_state_dict(checkpoint)
    ESM2_rxn_rate_1.to(device)
    ESM2_rxn_rate_1.eval()
else:
    ESM2_rxn_rate_1 = None

if use_SolubleMPNN:
    SolubleMPNN = True
else:
    SolubleMPNN = None

################################################ Running Simulated Annealing ################################################
if ESM2_rxn_rate_1 is not None and SolubleMPNN is not None and VAE is not None:
    models = 'all_models'
elif ESM2_rxn_rate_1 is not None:
    models = 'ESM2_rxn_rate_1_only'
elif SolubleMPNN is not None:
    models = 'SolubleMPNN_only'
elif VAE is not None:
    models = 'VAE_only'
print('Models:', models)

if num_mut == 3:
    ESM2_rxn_rate_1_Max=1.468309 # ! update
    ESM2_rxn_rate_1_Min=0.304152     # ! update
    SolubleMPNN_Max=-1.289022   # ! update
    SolubleMPNN_Min=-1.338782    # ! update
    VAE_Max=-305.792816 # ! update
    VAE_Min=-358.017273 # ! update
    start_mut = None # None: WT
    mut_rate = 1
    num_trials = 10 # ! update
    nsteps = 30000 # ! update
if num_mut == 5:
    ESM2_rxn_rate_1_Max=1.468309 # ! update
    ESM2_rxn_rate_1_Min=0.304152     # ! update
    SolubleMPNN_Max=-1.289022   # ! update
    SolubleMPNN_Min=-1.338782    # ! update
    VAE_Max=-305.792816 # ! update
    VAE_Min=-358.017273 # ! update
    start_mut = None # None: WT
    mut_rate = 2
    num_trials = 10 # ! update
    nsteps = 35000 # ! update

# create directories to save results
run_dir = REPO_ROOT / run_name
dir_path = run_dir / f'{num_mut}mut_{models}_{nsteps}steps'
dir_path.mkdir(parents=True, exist_ok=True)

# Saving parameters
params_str = f"""################################################
Simulated Annealing Parameters
################################################
non_gap_indices = {non_gap_indices}
WT_no_gaps = '{WT_no_gaps}'
start_mut = '{start_mut}'
nsteps = {nsteps}
num_trials = {num_trials}
num_mut = {num_mut}
mut_rate = {mut_rate}
start_temp = {start_temp}
final_temp = {final_temp}
run_name = '{run_name}'
fixed_window_size = {fixed_window_size}
mutating_window_size = {mutating_window_size}
start_position = {start_position}
seed = {seed}
################################################
Simulated Annealing Parameters
################################################
"""

# save parameters text file
file_path = dir_path / f"parameters_{num_mut}mut_assayW{assay_data_weight}_cuda{cuda_num}.txt"
with open(file_path, "w") as file:
    file.write(params_str)
print(f"Parameters saved to {file_path}")

# Running Simulated annealing
for i in range(num_trials):
    # Set the file names with version numbers
    best_mutant_file = dir_path / f"best_{run_name}_{num_mut}mut_start_pos{start_position}_{models}_assayW{assay_data_weight}_cuda{cuda_num}_v{i}.pickle"
    trajectory_file = dir_path / f"traj_{run_name}_{num_mut}mut_start_pos{start_position}_{models}_assayW{assay_data_weight}_cuda{cuda_num}_v{i}.png"
    csv_filename = dir_path / f"fitness_trajectory_{run_name}_{num_mut}mut_start_pos{start_position}_{models}_assayW{assay_data_weight}_cuda{cuda_num}_v{i}.csv"

    # Create an instance of seq_fitness class with WT
    path_to_fasta = SEQS_TO_SCORE_DIR / f"seq_{num_mut}_assayW{assay_data_weight}_cuda{cuda_num}_v{i}.fasta"
    with open(path_to_fasta, "w") as fasta_file:
                    fasta_file.write(">seq\n")
                    fasta_file.write(WT + "\n")

    seq_fitness = seq2fitness_handler(WT,
                                      num_mut,
                                      assay_data_weight,
                                      num_SolubleMPNN_samples,
                                      ESM2_rxn_rate_1, ESM2_rxn_rate_1_Max, ESM2_rxn_rate_1_Min,
                                      SolubleMPNN, SolubleMPNN_Max, SolubleMPNN_Min,
                                      VAE, VAE_Max, VAE_Min,
                                      cuda_num,
                                      i)

    # Create an instance of SA_optimizer class for the current mutant
    sa_optimizer = SA_optimizer(seq_fitness.seq2fitness,
                                 WT,
                                 AA_options,
                                 num_mut=num_mut,
                                 mutating_window_size=mutating_window_size,
                                 mut_rate=mut_rate,
                                 nsteps=nsteps,
                                 cool_sched='log',
                                 non_gap_indices=non_gap_indices,
                                 start_temp=start_temp,
                                 final_temp=final_temp)

    # Optimize the mutant and store the best mutant and its fitness in a pickle file
    best_mut = sa_optimizer.optimize(start_mut)
    with open(best_mutant_file, 'wb') as f:
        pickle.dump((best_mut), f)

    # Save sequences along trajectory
    close_sequences_file = dir_path / (
        f"close_sequences_{run_name}_{num_mut}mut_start_pos{start_position}_"
        f"{models}_assayW{assay_data_weight}_cuda{cuda_num}_v{i}.pickle"
    )
    with open(close_sequences_file, "wb") as f:
        pickle.dump(sa_optimizer.close_sequences, f)

    # Save fitness trajectory in a CSV file
    with open(csv_filename, mode='w') as csv_file:
        fieldnames = ['Step', 'Fitness']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
        writer.writeheader()
        for step, (_, fitness) in enumerate(sa_optimizer.fitness_trajectory):
            writer.writerow({'Step': step, 'Fitness': float(fitness)})

    # Save Plotted Trajectory
    sa_optimizer.plot_trajectory(savefig_name=str(trajectory_file))


