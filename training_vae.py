#!/usr/bin/env python
# coding: utf-8

# import packges
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torchtext import vocab

from models.convVAE import get_msa_from_fasta, ProtDataModule, ConvVAE

torch.set_num_threads(16)

# Set up Amino Acid Dictionary of Indices
WT = "MSVFVSGANGFIAQHIVDLLLKEDYKVIGSARSQEKAENLTEAFGNNPKFSMEVVPDISKLDAFDHVFQKHGKDIKIVLHTASPFCFDITDSERDLLIPAVNGVKGILHSIKKYAADSVERVVLTSSYAAVFDMAKENDKSLTFNEESWNPATWESCQSDPVNAYCGSKKFAEKAAWEFLEENRDSVKFELTAVNPVYVFGPQMFDKDVKKHLNTSCELVNSLMHLSPEDKIPELFGGYIDVRDVAKAHLVAFQKRETIGQRLIVSEARFTMQDVLDILNEDFPVLKGNIPVGKPGSGATHNTLGATLDNKKSKKLLGFKFRNLKETIDDTASQILKFEGRI" # GRE2

# Load Data for Model Training, Validation, and Testing
MSA = get_msa_from_fasta('./data/vae/gre2_msa.fasta')
weights = (np.load('./data/vae/gre2_msa_weights.npy'))
splits_path = Path('./data/vae/gre2_msa_vae_data_splits.pkl')

# vae training parameters
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

early_stopping = EarlyStopping(monitor='val_ce_loss', patience=100, mode='min')  # Define the early stopping callback
checkpoint_callback = ModelCheckpoint(
    dirpath='',
    filename='Best_ConvVAE',
    monitor='val_ce_loss',
    mode='min',
    save_top_k=3)  # Save only the best model based on the validation loss

if splits_path.exists():
    dm = ProtDataModule(MSA, batch_size, weights, str(splits_path))
else:
    dm = ProtDataModule(MSA, batch_size, weights)
    dm.save_splits(str(splits_path))
model = ConvVAE(slen, ks, nlatent, learning_rate, epochs, n_cycle, factor_2, factor_3, dim_4)
logger_name = 'Best_ConvVAE'
logger = CSVLogger('logs',name=logger_name, version=None)
trainer = pl.Trainer(logger=logger,max_epochs=epochs, callbacks=[early_stopping, checkpoint_callback], enable_progress_bar=False)
trainer.fit(model,dm)

# Save metrics
pt_metrics = pd.read_csv('logs/Best_ConvVAE/version_0/metrics.csv')
metrics_file_name = 'metrics_Best_ConvVAE.csv'
pt_metrics.to_csv(metrics_file_name, index=False)

