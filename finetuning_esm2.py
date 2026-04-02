#!/usr/bin/env python
# coding: utf-8

# import packages
import os
import random
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from scipy.stats import spearmanr
from sklearn import metrics
from torchtext import vocab
from transformers import AutoModelForMaskedLM, AutoTokenizer

from models.ESM2_w_regression_MLP_head import ProtDataModule, finetuning_ESM2_with_mse_loss
from utils.functions import calculate_median_regression_values

# load preprocessed Gre2 data
data_path = Path("./data/finetuned_esm2/normalized_processed_BI_R1_dataset_w_quantiles")
if data_path.with_suffix(".pkl").exists():
    df = pd.read_pickle(data_path.with_suffix(".pkl"))
elif data_path.with_suffix(".csv").exists():
    df = pd.read_csv(data_path.with_suffix(".csv"))
else:
    raise FileNotFoundError("Neither .pkl nor .csv file found for dataset")
df = df.fillna(-1) # Fill NaN values with -1, I will mask these values later

# Set up Amino Acid Dictionary of Indices
AAs = 'ACDEFGHIKLMNPQRSTVWY-' # setup torchtext vocab to map AAs to indices
WT = "MSVFVSGANGFIAQHIVDLLLKEDYKVIGSARSQEKAENLTEAFGNNPKFSMEVVPDISKLDAFDHVFQKHGKDIKIVLHTASPFCFDITDSERDLLIPAVNGVKGILHSIKKYAADSVERVVLTSSYAAVFDMAKENDKSLTFNEESWNPATWESCQSDPVNAYCGSKKFAEKAAWEFLEENRDSVKFELTAVNPVYVFGPQMFDKDVKKHLNTSCELVNSLMHLSPEDKIPELFGGYIDVRDVAKAHLVAFQKRETIGQRLIVSEARFTMQDVLDILNEDFPVLKGNIPVGKPGSGATHNTLGATLDNKKSKKLLGFKFRNLKETIDDTASQILKFEGRI" # GRE2
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap
ind2aa = {aa2ind[a]: a for a in AAs} # Create a reverse mapping for indices to amino acids

######################################## Hyperparameters that can be altered ########################################
# ESM2 selection
huggingface_identifier ='esm2_t33_650M_UR50D' # esm2_t6_8M_UR50D # esm2_t12_35M_UR50D # esm2_t30_150M_UR50D # esm2_t33_650M_UR50D
ESM2 = AutoModelForMaskedLM.from_pretrained(f"facebook/{huggingface_identifier}")
tokenizer = AutoTokenizer.from_pretrained(f"facebook/{huggingface_identifier}")
model_identifier = huggingface_identifier
token_format = 'ESM2'
splits_path = './data/finetuned_esm2/normalized_processed_BI_R1_dataset_w_quantiles_data_splits.pkl'

# Model training hyperparameters
num_unfrozen_layers = 27
num_layers_unfreeze_each_epoch = 15
max_num_layers_unfreeze_each_epoch = 36

# Learning hyperparameters
epochs = 2 # rounds of training
warm_restart = 1 # with warm restart
use_scheduler = 1 # with scheduler
WD = 0.005
grad_clip_threshold = 3.0
lr_mult = 1
lr_mult_factor = 1
seed = 3
learning_rate = 1e-5
reinit_optimizer = 0
using_EMA = 1
decay = 0.8

# GPU hyperparameters
cls_token_only = 1
if cls_token_only == 1:
    batch_size = 20 # typically powers of 2: 32, 64, 128, 256, ...
else:
    batch_size = 20 # typically powers of 2: 32, 64, 128, 256, ...

# Data hyperparameters
slen = len(WT) # length of protein
num_reg_tasks = 1
reg_weights = 1
filepath = 'finetuning_ESM2'

# Determine if we're running on a GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Determine if we're running on a GPU
if device == "cuda":
    # Make models reproducible on GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # Set the PYTHONHASHSEED environment variable to the chosen seed to make hash-based operations predictable
    np.random.seed(seed) # Set NumPy's random seed to ensure reproducibility of operations using NumPy's random number generator
    random.seed(seed) # Set Python's built-in random module's seed to ensure reproducibility of random operations using Python's random functions
    np.random.seed(seed)
    torch.manual_seed(seed) # Set the seed for generating random numbers in PyTorch to ensure reproducibility on the CPU
    torch.cuda.manual_seed(seed) # Set the seed for generating random numbers in PyTorch to ensure reproducibility on the GPU
    torch.cuda.manual_seed_all(seed) # Ensure reproducibility for all GPUs by setting the seed for generating random numbers for all CUDA devices
    torch.backends.cudnn.deterministic = True # Force cuDNN to use only deterministic convolutional algorithms (can slow down computations but guarantees reproducibility)
    torch.backends.cudnn.benchmark = False # Prevent cuDnn from using any algorithms that are nondeterministic
    torch.set_float32_matmul_precision('medium')
    print('Training model on GPU')
else:
    # fix random seeds for reproducibility on CPU
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    print('Training model on CPU')

# feintuning esm2
if splits_path.exists():
    dm = ProtDataModule(df, batch_size, str(splits_path), token_format, seed)
else:
    dm = ProtDataModule(df, batch_size)
    dm.save_splits(str(splits_path))

model = finetuning_ESM2_with_mse_loss(ESM2, huggingface_identifier, tokenizer, num_unfrozen_layers, num_layers_unfreeze_each_epoch, max_num_layers_unfreeze_each_epoch,
                 epochs, batch_size, seed, cls_token_only,
                 learning_rate, lr_mult, lr_mult_factor,
                 WD, reinit_optimizer, grad_clip_threshold, use_scheduler, warm_restart,
                 slen, reg_weights, num_reg_tasks,
                 using_EMA, decay)

checkpoint_callback = ModelCheckpoint(
        dirpath=f"./logs/{filepath}/",
        filename=f"{filepath}",
        monitor="train_reg_loss",
        mode="min",
        save_top_k=1)
early_stopping = EarlyStopping(monitor="train_reg_loss", patience=1000, mode="min")
logger = CSVLogger('logs', name=f"{filepath}") # logger is a class instance that stores performance data to a csv after each epoch

# Dynamically set up Trainer based on available device
trainer = pl.Trainer(
    logger=logger,
    max_epochs=epochs,
    callbacks=[checkpoint_callback, early_stopping],
    enable_progress_bar=True,
    accelerator=device,  # Automatically chooses between "cpu" and "gpu"
    devices=1 if device == "cuda" else None,  # Use 1 GPU if available, else default to CPU
    deterministic=True  # Ensure reproducibility
)
trainer.fit(model, dm)

# Save the model
non_ema_path = f'./logs/{filepath}/version_{logger.version}/multitask_ESM2.pt'
ema_path = f'./logs/{filepath}/version_{logger.version}/multitask_ESM2_w_EMA.pt'
model.save_model(non_ema_path, ema_path)

# make learning curves
version = logger.version  # Replace `logger.version` with the specific version number if needed
train_losses = []
val_losses = []

# Load the metrics for the specified version
try:
    # Read metrics.csv for the specified version
    pt_metrics = pd.read_csv(f'./logs/{filepath}/version_{version}/metrics.csv')
    
    # Extract training and validation losses
    train = pt_metrics[~pt_metrics.train_reg_loss.isna()]
    val = pt_metrics[~pt_metrics.val_reg_loss.isna()]
    train_losses = train.train_reg_loss.values
    val_losses = val.val_reg_loss.values
except FileNotFoundError:
    print(f"Metrics file for version {version} not found.")
    train_losses = []
    val_losses = []

# Check if losses are available
if len(train_losses) > 0 and len(val_losses) > 0:
    # Ensure losses have the same length by padding if necessary
    max_length = max(len(train_losses), len(val_losses))
    train_losses = np.pad(train_losses, (0, max_length - len(train_losses)), 'constant', constant_values=np.nan)
    val_losses = np.pad(val_losses, (0, max_length - len(val_losses)), 'constant', constant_values=np.nan)

    # Compute epochs
    epochs = np.arange(1, max_length + 1)

    # Plot the loss curves
    plt.plot(epochs, train_losses, label='training loss')
    plt.plot(epochs, val_losses, label='validation loss')
    plt.title('Loss vs. Epoch')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # Save the loss curves
    file_path_svg = os.path.join(f'./logs/{filepath}/version_{version}', 'Loss_Curves.svg')
    plt.savefig(file_path_svg)
    file_path_png = os.path.join(f'./logs/{filepath}/version_{version}', 'Loss_Curves.png')
    plt.savefig(file_path_png)
    print(f"Loss curves saved to {file_path_svg} and {file_path_png}")
else:
    print("No loss data found for this model version.")

checkpoint_path = non_ema_path

# Initialize dictionaries to store logits and regression predictions for train and validation sets
reg_values_train = {idx: [] for idx in dm.train_idx}
reg_values_val = {idx: [] for idx in dm.val_idx}

# Load the saved model checkpoint
model = finetuning_ESM2_with_mse_loss(ESM2, huggingface_identifier, tokenizer, num_unfrozen_layers, num_layers_unfreeze_each_epoch, max_num_layers_unfreeze_each_epoch,
                 epochs, batch_size, seed, cls_token_only,
                 learning_rate, lr_mult, lr_mult_factor,
                 WD, reinit_optimizer, grad_clip_threshold, use_scheduler, warm_restart,
                 slen, reg_weights, num_reg_tasks,
                 using_EMA, decay)

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)
model.eval()

# Split data into train and validation sets
train_df, val_df = df.iloc[dm.train_idx], df.iloc[dm.val_idx]

# Predict and store logits and regression values for train and validation sets
for data_frame, reg_values_store in zip(
    [train_df, val_df],
    [reg_values_train, reg_values_val]
):
    for idx, seq in zip(data_frame.index, data_frame['Sequence']):
        reg_pred = model.predict(seq)
        reg_values_store[idx] = reg_pred.astype(float)  # Store regression predictions

# Calculate median regression values for train and validation sets
median_reg_train = calculate_median_regression_values(reg_values_train)
median_reg_val = calculate_median_regression_values(reg_values_val)

# Prepare actual values for plotting
X_reg_train = train_df['Quantile Rxn Rate at 0.05mgml'].values
X_reg_val = val_df['Quantile Rxn Rate at 0.05mgml'].values

# Create regression scatter plots
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

label = 'Quantile Rxn Rate at 0.05mgml'

# Plotting actual vs. predicted for training and validation sets
ax.scatter(X_reg_train, median_reg_train, color='blue', s=5, label="Train")
ax.scatter(X_reg_val, median_reg_val, color='orange', s=5, label="Validation")
ax.plot([X_reg_train.min(), X_reg_train.max()], 
        [X_reg_train.min(), X_reg_train.max()], color='black', linestyle='--', linewidth=0.5)
ax.set_xlabel(f"{label} (Actual)", fontsize=10)
ax.set_ylabel("Predicted Value", fontsize=10)
ax.legend()
ax.set_title(f"Predicted vs. Actual {label}")

# Calculate and annotate metrics on validation set
mse = metrics.mean_squared_error(X_reg_val, median_reg_val)
r = np.corrcoef(X_reg_val, median_reg_val)[0][1]
rho, _ = spearmanr(X_reg_val, median_reg_val)
ax.text(0.35, 0.95, f"MSE = {mse:.2f}", fontsize=10, transform=ax.transAxes)
ax.text(0.35, 0.9, f"R = {r:.2f}", fontsize=10, transform=ax.transAxes)
ax.text(0.35, 0.85, f"Rho = {rho:.2f}", fontsize=10, transform=ax.transAxes)

# Save the regression plot
os.makedirs(f'./logs/{filepath}', exist_ok=True)
fig.savefig(f'./logs/{filepath}/version_{version}/{filepath}_regression_predictions.png')
fig.savefig(f'./logs/{filepath}/version_{version}/{filepath}_regression_predictions.svg')
plt.show()

checkpoint_path = ema_path

# Initialize dictionaries to store logits and regression predictions for train and validation sets
reg_values_train = {idx: [] for idx in dm.train_idx}
reg_values_val = {idx: [] for idx in dm.val_idx}

# Load the saved model checkpoint
model = finetuning_ESM2_with_mse_loss(ESM2, huggingface_identifier, tokenizer, num_unfrozen_layers, num_layers_unfreeze_each_epoch, max_num_layers_unfreeze_each_epoch,
                 epochs, batch_size, seed, cls_token_only,
                 learning_rate, lr_mult, lr_mult_factor,
                 WD, reinit_optimizer, grad_clip_threshold, use_scheduler, warm_restart,
                 slen, reg_weights, num_reg_tasks,
                 using_EMA, decay)

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)
model.eval()

# Split data into train and validation sets
train_df, val_df = df.iloc[dm.train_idx], df.iloc[dm.val_idx]

# Predict and store logits and regression values for train and validation sets
for data_frame, reg_values_store in zip(
    [train_df, val_df],
    [reg_values_train, reg_values_val]
):
    for idx, seq in zip(data_frame.index, data_frame['Sequence']):
        reg_pred = model.predict(seq)
        reg_values_store[idx] = reg_pred.astype(float)  # Store regression predictions

# Calculate median regression values for train and validation sets
median_reg_train = calculate_median_regression_values(reg_values_train)
median_reg_val = calculate_median_regression_values(reg_values_val)

# Prepare actual values for plotting
X_reg_train = train_df['Quantile Rxn Rate at 0.05mgml'].values
X_reg_val = val_df['Quantile Rxn Rate at 0.05mgml'].values

# Create regression scatter plots
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

label = 'Quantile Rxn Rate at 0.05mgml'

# Plotting actual vs. predicted for training and validation sets
ax.scatter(X_reg_train, median_reg_train, color='blue', s=5, label="Train")
ax.scatter(X_reg_val, median_reg_val, color='orange', s=5, label="Validation")
ax.plot([X_reg_train.min(), X_reg_train.max()], 
        [X_reg_train.min(), X_reg_train.max()], color='black', linestyle='--', linewidth=0.5)
ax.set_xlabel(f"{label} (Actual)", fontsize=10)
ax.set_ylabel("Predicted Value", fontsize=10)
ax.legend()
ax.set_title(f"Predicted vs. Actual {label}")

# Calculate and annotate metrics on validation set
mse = metrics.mean_squared_error(X_reg_val, median_reg_val)
r = np.corrcoef(X_reg_val, median_reg_val)[0][1]
rho, _ = spearmanr(X_reg_val, median_reg_val)
ax.text(0.35, 0.95, f"MSE = {mse:.2f}", fontsize=10, transform=ax.transAxes)
ax.text(0.35, 0.9, f"R = {r:.2f}", fontsize=10, transform=ax.transAxes)
ax.text(0.35, 0.85, f"Rho = {rho:.2f}", fontsize=10, transform=ax.transAxes)

# Save the regression plot
os.makedirs(f'./logs/{filepath}', exist_ok=True)
fig.savefig(f'./logs/{filepath}/version_{version}/{filepath}_regression_predictions_w_EMA_model.png')
fig.savefig(f'./logs/{filepath}/version_{version}/{filepath}_regression_predictions_w_EMA_model.svg')
plt.show()

