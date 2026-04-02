#!/usr/bin/env python
# coding: utf-8

# import packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from collections import OrderedDict
from torchtext import vocab # This package can give problems sometimes, it may be necessary to downgrade to a specific version
from pytorch_lightning.loggers import CSVLogger
from random import choice
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn import metrics
import torchmetrics
import enum
import argparse
from argparse import ArgumentParser
import os
import pickle
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import csv
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch_ema import ExponentialMovingAverage

AAs = 'ACDEFGHIKLMNPQRSTVWY-' # setup torchtext vocab to map AAs to indices
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap
ind2aa = {aa2ind[a]: a for a in AAs} # Create a reverse mapping for indices to amino acids

# SeqFcnDataset is a data handling class.
class SeqFcnDataset(torch.utils.data.Dataset):
    """A custom PyTorch dataset for protein sequence-function data"""

    def __init__(self, data_frame, label_index, token_format='ESM2'):
        self.data_df = data_frame
        self.label_index = label_index
        self.token_format = token_format
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

    def __getitem__(self, idx):
        if self.token_format == 'aa2ind':
            sequence = torch.tensor(aa2ind(list(self.data_df.Sequence.iloc[idx])), device=self.device) # Extract sequence at index idx
        elif self.token_format == 'ESM2':
            sequence = self.data_df.Sequence.iloc[idx]  # Directly get the sequence string for ESM2 tokenizer

        # I convert all labels, including -1 values, to torch tensors
        reg_labels = torch.tensor(self.data_df.iloc[idx, self.label_index].tolist(), device=self.device).float() # Extract labels for sequence at index idx and convert to a list
        
        return sequence, reg_labels

    def __len__(self):
        return len(self.data_df)

# ProtDataModule splits the data into three different datasets.
class ProtDataModule(pl.LightningDataModule):
    """A PyTorch Lightning Data Module to handle data splitting"""

    def __init__(self, data_frame, label_index, batch_size, splits_path=None, token_format='aa2ind', seed=0, emb_precomputed=False):
        # Call the __init__ method of the parent class
        super().__init__()

        # Store the batch size
        self.data_df = data_frame
        self.label_index = label_index
        self.batch_size = batch_size
        self.token_format = token_format
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if splits_path is not None:
            train_indices, val_indices, test_indices = self.load_splits(splits_path)
            # print(test_indices)
            
            # Shuffle the indices to ensure that the data from each cluster is mixed. Do I want this?
            random.shuffle(train_indices)
            random.shuffle(val_indices)
            random.shuffle(test_indices)
            
            # Store the indices for the training, validation, and test sets
            self.train_idx = train_indices
            self.val_idx = val_indices
            self.test_idx = test_indices
                
        else:
            # Initialize empty lists to hold the indices for the training, validation, and test sets
            train_indices = []
            val_indices = []
            test_indices = []  # Empty test set
            
            # Calculate the target validation set size (5% of the total data)
            target_val_size = int(len(self.data_df) * 0.075)
            
            # Sort the DataFrame by 'Position_Count' in descending order
            sorted_df = self.data_df.sort_values(by='Position_Count', ascending=False)
            
            # Get the top positions by Position_Count up to the target validation size
            top_positions = sorted_df['Position'].value_counts().index[:target_val_size]
            top_df = sorted_df[sorted_df['Position'].isin(top_positions)]
            
            # Randomly sample from the top positions for the validation set
            sampled_val_df = top_df.drop_duplicates(subset=['Position']).sample(n=target_val_size, random_state=5)
            print(sampled_val_df)
            
            # Track unique positions for validation
            val_indices = list(sampled_val_df.index)
            selected_positions = set(sampled_val_df['Position'])
            
            # Add remaining indices to the training set, excluding the validation set
            for idx, row in sorted_df.iterrows():
                if idx not in val_indices:
                    train_indices.append(idx)
            
            # Shuffle the indices to ensure randomness within each set
            random.shuffle(train_indices)
            random.shuffle(val_indices)
            
            # Store the indices for the training, validation, and test sets
            self.train_idx = train_indices  # Training data used during model training
            self.val_idx = val_indices      # Validation data used to evaluate the model after epochs
            self.test_idx = test_indices    # Empty testing data
            
            # Verification
            print("Training set size:", len(self.train_idx))
            print("Validation set size:", len(self.val_idx))
            print("Test set size:", len(self.test_idx))

    # Assigns train, validation and test datasets for use in dataloaders.
    def setup(self, stage=None):
        
        # Assign train/validation datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            train_data_frame = self.data_df.iloc[list(self.train_idx)]
            self.train_ds = SeqFcnDataset(train_data_frame, self.label_index, self.token_format)
            val_data_frame = self.data_df.iloc[list(self.val_idx)]
            self.val_ds = SeqFcnDataset(val_data_frame, self.label_index, self.token_format)
                    
        # Assigns test dataset for use in dataloader
        if stage == 'test' or stage is None:
            test_data_frame = self.data_df.iloc[list(self.test_idx)]
            self.test_ds = SeqFcnDataset(test_data_frame, self.label_index, self.token_format)

    def seed_worker(worker_id, worker_info):
	    worker_seed = torch.initial_seed() % 2**32  # Compute a seed for the worker based on the initial seed of the torch Generator
	    np.random.seed(worker_seed)  # Set NumPy's random seed based on the worker seed
	    random.seed(worker_seed)  # Set Python's built-in random module's seed
            
    #The DataLoader object is created using the train_ds/val_ds/test_ds objects with the batch size set during initialization of the class and shuffle=True.
    def train_dataloader(self):
        # Determine if we're running on a GPU
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            generator = torch.Generator()  # Create a new torch Generator
            generator.manual_seed(self.seed)  # Manually seed the generator with the predefined seed from the class
            # Create and return a DataLoader configured for training
            print('Sending data to GPU')
            return data_utils.DataLoader(
                self.train_ds,  # The dataset to load, in this case, the training dataset
                batch_size=self.batch_size,  # The number of samples in each batch to load
                shuffle=True,  # Enable shuffling to randomize the order of data before each epoch
                worker_init_fn=self.seed_worker,  # Function to initialize each worker's seed to ensure reproducibility across runs
                generator=generator,  # Specify the generator used for random number generation in shuffling
                # num_workers=32,  # The number of subprocesses to use for data loading. More workers can increase the speed of data loading
                # pin_memory=True  # Pins memory, allowing faster and more efficient transfer of data from host to GPU when training on GPUs
            )
        else:
            print('Sending data to CPU')
            return data_utils.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
    
    
    def val_dataloader(self):
        # Determine if we're running on a GPU
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            generator = torch.Generator()  # Create a new torch Generator
            generator.manual_seed(self.seed)  # Manually seed the generator with the predefined seed from the class
            # Create and return a DataLoader configured for training
            # print('Sending data to GPU')
            return data_utils.DataLoader(
                self.val_ds,  # The dataset to load, in this case, the training dataset
                batch_size=self.batch_size,  # The number of samples in each batch to load
                shuffle=True,  # Enable shuffling to randomize the order of data before each epoch
                worker_init_fn=self.seed_worker,  # Function to initialize each worker's seed to ensure reproducibility across runs
                generator=generator,  # Specify the generator used for random number generation in shuffling
                # num_workers=32,  # The number of subprocesses to use for data loading. More workers can increase the speed of data loading
                # pin_memory=True  # Pins memory, allowing faster and more efficient transfer of data from host to GPU when training on GPUs
            )
        else:
            # print('Sending data to CPU')
            return data_utils.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True)
    
    
    def test_dataloader(self):
        # Determine if we're running on a GPU
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            generator = torch.Generator()  # Create a new torch Generator
            generator.manual_seed(self.seed)  # Manually seed the generator with the predefined seed from the class
            # Create and return a DataLoader configured for training
            # print('Sending data to GPU')
            return data_utils.DataLoader(
                self.test_ds,  # The dataset to load, in this case, the training dataset
                batch_size=self.batch_size,  # The number of samples in each batch to load
                shuffle=True,  # Enable shuffling to randomize the order of data before each epoch
                worker_init_fn=self.seed_worker,  # Function to initialize each worker's seed to ensure reproducibility across runs
                generator=generator,  # Specify the generator used for random number generation in shuffling
                # num_workers=32,  # The number of subprocesses to use for data loading. More workers can increase the speed of data loading
                # pin_memory=True  # Pins memory, allowing faster and more efficient transfer of data from host to GPU when training on GPUs
            )
        else:
            # print('Sending data to CPU')
            return data_utils.DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=True)
    
    def save_splits(self, path):
        """Save the data splits to a file at the given path"""
        with open(path, 'wb') as f:
            pickle.dump((self.train_idx, self.val_idx, self.test_idx), f)

    def load_splits(self, path):
        """Load the data splits from a file at the given path"""
        with open(path, 'rb') as f:
            self.train_idx, self.val_idx, self.test_idx = pickle.load(f)
            
            train_indices = self.train_idx
            val_indices = self.val_idx
            test_indices = self.test_idx
            
        return train_indices, val_indices, test_indices

# PTLModule is the actual neural network. Model architecture can be altered here.
class finetuning_ESM2_with_mse_loss(pl.LightningModule):
    """PyTorch Lightning Module that defines model and training"""
      
    # define network
    def __init__(self,
                 ESM2, huggingface_identifier, tokenizer, num_unfrozen_layers, num_layers_unfreeze_each_epoch, max_num_layers_unfreeze_each_epoch,
                 epochs, batch_size, seed, embedding_type,
                 learning_rate, lr_mult, lr_mult_factor,
                 WD, grad_clip_threshold, use_scheduler, warm_restart,
                 slen, reg_weights, num_reg_tasks, reg_type,
                 using_EMA, decay,
                 epoch_threshold_to_unlock_ESM2,
                 WT,
                 data_filepath
                ):
        super().__init__()

        # print("Initializing module...")

        # hyperparameters for training
        self.epochs = epochs
        self.epoch_threshold_to_unlock_ESM2 = epoch_threshold_to_unlock_ESM2
        self.batch_size = batch_size
        self.seed = seed
        self.embedding_type = embedding_type
        
        # models hyperparameters
        self.ESM2_wo_lmhead = ESM2
        self.huggingface_identifier = huggingface_identifier
        self.tokenizer = tokenizer
        self.num_unfrozen_layers = num_unfrozen_layers
        self.num_layers_unfreeze_each_epoch = num_layers_unfreeze_each_epoch
        self.max_num_layers_unfreeze_each_epoch = max_num_layers_unfreeze_each_epoch
        
        # learning rate hyperparameters
        self.learning_rate = learning_rate
        self.learning_rate_0 = learning_rate
        self.lr_mult = lr_mult
        self.lr_mult_factor = lr_mult_factor

        # optimizer hyperparameters
        self.WD = WD
        self.grad_clip_threshold = grad_clip_threshold

        # data hyperparameters
        self.WT = WT
        self.slen = slen # synthetic query sequence length (128 a.a.)
        self.register_buffer('reg_weights', torch.tensor(reg_weights, dtype=torch.float))
        self.num_reg_tasks = num_reg_tasks
        self.reg_type = reg_type

        # Classification MLP
        if self.embedding_type in ('all_tokens', 'precomputed_all_tokens'):
            self.input_size = ESM2.config.hidden_size * (len(self.WT)+2) # Remember to add for the cls and eos tokens
        else:
            self.input_size = ESM2.config.hidden_size

        # define MLP when finetuning base esm2 with regression head
        if self.embedding_type in ('all_tokens', 'cls_token_only', 'mean_pooling'):
            hidden_layer_size = 400
            self.regression_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.input_size, hidden_layer_size),
                    nn.ReLU(),
                    nn.Dropout(p=0.1),
                    nn.Linear(hidden_layer_size, 1)  # Single output for each regression task
                ) for _ in range(self.num_reg_tasks)
            ])

            # Set up parameters with progressively scaled learning rates for ESM2, constant for FCNN
            selected_layers = [name for name, param in self.ESM2_wo_lmhead.named_parameters() if "lm_head" not in name and "contact_head" not in name]
            trainable_layers = selected_layers[-self.num_unfrozen_layers:]  # Select only the final num_unfrozen_layers
            self.esm2_params = []
            current_lr = self.learning_rate
            
            # Define the same learning rate for all FCNN layers
            fcnn_layers = list(self.regression_mlps)
            for fcnn_layer in fcnn_layers:
                # print(f"Adding FCNN layer {fcnn_layer} to optimizer with lr: {current_lr}")
                self.esm2_params.append({'params': list(fcnn_layer.parameters()), 'lr': current_lr})
            
            # Apply learning rate schedule for the last num_unfrozen_layers in ESM2 in reverse order
            for name in reversed(trainable_layers):  # Reverse order of trainable layers to start schedule from the end
                # print(f"Adding trainable layer {name} to optimizer with lr: {current_lr}")
                layer_params = {
                    'params': [p for n, p in self.ESM2_wo_lmhead.named_parameters() if n == name and p.requires_grad],
                    'lr': current_lr
                }
                self.esm2_params.append(layer_params)
                current_lr *= self.lr_mult  # Progressively scale learning rate for each layer
            # print('Initial self.esm2_params', self.esm2_params)

            if self.embedding_type in ('all_tokens', 'cls_token_only', 'mean_pooling'):
                self.using_EMA = using_EMA
                if self.using_EMA == 1:
                    self.decay = decay
                    self.ema = ExponentialMovingAverage(self.ESM2_wo_lmhead.parameters(), decay=self.decay)
        
        # define simple linear regression with L1 norm
        elif self.embedding_type.startswith("precomputed"):
            self.l1_lambda = 0.001
            self.simple_linear_head = nn.Linear(self.input_size, 1)
        
        else:
            raise ValueError(f'Error in self.embedding_type: {self.embedding_type}')

        # parameters for custom training
        self.stop_training_status = False
        self.automatic_optimization = False
        self.use_scheduler = use_scheduler
        self.warm_restart = warm_restart
        if self.embedding_type in ('all_tokens', 'cls_token_only', 'mean_pooling'):
            if self.use_scheduler == 1:
                optimizers_config = self.configure_optimizers()
                self.optimizer = optimizers_config["optimizer"]
                self.scheduler = optimizers_config["lr_scheduler"]
        else:
            self.optimizer = self.configure_optimizers()

        self.save_hyperparameters(ignore=["ESM2","tokenizer"])

    def forward(self, x):
        
        if self.embedding_type in ('all_tokens', 'cls_token_only', 'mean_pooling'):
            # Step 1: Get embeddings from the ESM2 model
            outputs = self.ESM2_wo_lmhead(x, output_hidden_states=True, return_dict=True)

            last_hidden_states = outputs.hidden_states[-1]
            # print(f"last_hidden_states shape: {last_hidden_states.shape}")
            # print(f"last_hidden_states: {last_hidden_states}")

            # Step 2: Extract the <CLS> token embedding from the last hidden layer (https://www.nature.com/articles/s41467-024-51844-2)
            if self.embedding_type == 'cls_token_only':
                # Use the cls token for embedding
                embedding = last_hidden_states[:, 0, :]
            elif self.embedding_type == 'mean_pooling':
                # Use mean of all tokens for embedding
                embedding = torch.mean(last_hidden_states, dim=1)  # [batch_size, hidden_size]
            elif self.embedding_type == 'all_tokens':
                # Flatten all tokens into a single vector
                batch_size = last_hidden_states.shape[0]
                embedding = last_hidden_states.view(batch_size, -1)  # [batch_size, hidden_size * sequence_length]

            # Regression MLP
            reg_logits = torch.cat([mlp(embedding) for mlp in self.regression_mlps], dim=1)

        elif self.embedding_type.startswith("precomputed"):
            reg_logits = self.simple_linear_head(x)

        # Debugging outputs for logits
        # print(f"Regression logits shape: {reg_logits.shape}")
        return reg_logits


    def training_step(self, batch, batch_idx):
        # Load data and generate logits
        if self.embedding_type in ('all_tokens', 'cls_token_only', 'mean_pooling'):
            sequence, reg_labels = batch
            tokens = self.tokenizer(sequence, return_tensors='pt', padding=True).input_ids.to(self.device, non_blocking=True) # Convert sequence to tokens for ESM2
            reg_labels = reg_labels.to(self.device)
            reg_logits = self(tokens)

        # precomputed embeddings
        else:
            seq_embeddings, reg_labels = batch
            seq_embeddings = seq_embeddings.to(self.device)
            reg_labels = reg_labels.to(self.device)
            reg_logits = self(seq_embeddings)
            l1 = sum(p.abs().sum() for p in self.simple_linear_head.parameters())

        if self.reg_type == 'mse':
            # Calculate regression losses
            reg_mask = (reg_labels != -1).float() # create a mask tensor where 1 indicates a valid label and 0 indicates an invalid label 
            reg_loss = nn.MSELoss(reduction='none')(reg_logits, reg_labels.unsqueeze(-1)) # Calculate MSE
            reg_loss = torch.mul(reg_loss, self.reg_weights) # Use label weights tensor to weight the MSE loss
            reg_observations = torch.sum(reg_mask) # Calculate Number of Observed Labels
            reg_loss = torch.mul(reg_loss, reg_mask) # apply the mask to MSE to eliminate MSE's Calculated with missing labels
            reg_loss = torch.div(torch.sum(torch.nan_to_num(reg_loss, nan=0.0, posinf=0.0, neginf=0.0)), reg_observations) # computes the average loss over the observed labels, ignoring any invalid labels
            # The torch.nan_to_num() function replaces NaN, positive infinity, and negative infinity values in the loss tensor with 0.0
            # The torch.sum() function sums the resulting tensor
            # torch.div() divides the sum by the number of observed labels (reg_observations) to obtain the mean loss per label.
            if reg_observations == 0.0: # This might occur with small batch sizes
                reg_loss = 0.0

        if self.embedding_type.startswith("precomputed"):
            reg_loss = reg_loss + self.l1_lambda * l1
        
        self.log("train_reg_loss", reg_loss, prog_bar=True, logger=True, on_step = False, on_epoch=True) # reports MSE loss
        self.train_reg_loss = reg_loss.item()

        # Backpropagation
        self.optimizer.zero_grad()
        reg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ESM2_wo_lmhead.parameters(), self.grad_clip_threshold)
        self.optimizer.step()
        
        if self.embedding_type in ('all_tokens', 'cls_token_only', 'mean_pooling'):
            if self.using_EMA == 1:
            	self.ema.to(self.device)
            	self.ema.update()
            	self.ema.to('cpu')
        
        return reg_loss

    def validation_step(self, batch, batch_idx):
        # Load data and generate logits
        if self.embedding_type in ('all_tokens', 'cls_token_only', 'mean_pooling'):
            sequence, reg_labels = batch
            tokens = self.tokenizer(sequence, return_tensors='pt', padding=True).input_ids.to(self.device) # Convert sequence to tokens for ESM2
            reg_labels = reg_labels.to(self.device)
            reg_logits = self(tokens)

        # precomputed embeddings
        else:
            seq_embeddings, reg_labels = batch
            seq_embeddings = seq_embeddings.to(self.device)
            reg_labels = reg_labels.to(self.device)
            reg_logits = self(seq_embeddings)
            l1 = sum(p.abs().sum() for p in self.simple_linear_head.parameters())

        if self.reg_type == 'mse':
            # Calculate regression losses
            reg_mask = (reg_labels != -1).float() # create a mask tensor where 1 indicates a valid label and 0 indicates an invalid label 
            reg_loss = nn.MSELoss(reduction='none')(reg_logits, reg_labels.unsqueeze(-1)) # Calculate MSE
            reg_loss = torch.mul(reg_loss, self.reg_weights) # Use label weights tensor to weight the MSE loss
            reg_observations = torch.sum(reg_mask) # Calculate Number of Observed Labels
            reg_loss = torch.mul(reg_loss, reg_mask) # apply the mask to MSE to eliminate MSE's Calculated with missing labels
            reg_loss = torch.div(torch.sum(torch.nan_to_num(reg_loss, nan=0.0, posinf=0.0, neginf=0.0)), reg_observations) # computes the average loss over the observed labels, ignoring any invalid labels
            # The torch.nan_to_num() function replaces NaN, positive infinity, and negative infinity values in the loss tensor with 0.0
            # The torch.sum() function sums the resulting tensor
            # torch.div() divides the sum by the number of observed labels (reg_observations) to obtain the mean loss per label.
            if reg_observations == 0.0: # This might occur with small batch sizes
                reg_loss = 0.0

        if self.embedding_type.startswith("precomputed"):
            reg_loss = reg_loss + self.l1_lambda * l1

        self.log("val_reg_loss", reg_loss, prog_bar=True, logger=True, on_step = False, on_epoch=True) # reports MSE loss
        self.val_reg_loss = reg_loss.item()
        
        return reg_loss

    def test_step(self, batch):
        # Load data and generate logits
        if self.embedding_type in ('all_tokens', 'cls_token_only', 'mean_pooling'):
            sequence, reg_labels = batch
            tokens = self.tokenizer(sequence, return_tensors='pt', padding=True).input_ids.to(self.device) # Convert sequence to tokens for ESM2
            if torch.cuda.is_available():
                tokens = tokens.to(self.device)
            reg_logits = self(tokens)

        elif self.embedding_type.startswith("precomputed"):
            seq_embeddings, reg_labels = batch
            reg_logits = self(seq_embeddings)

    def configure_optimizers(self):
        if self.embedding_type in ('all_tokens', 'cls_token_only', 'mean_pooling'):
            optimizer = torch.optim.Adam(self.esm2_params, weight_decay=self.WD)
            if self.use_scheduler == 1:
                if self.warm_restart == 1:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
                    return {"optimizer": optimizer,
                            "lr_scheduler": {"scheduler": scheduler}}
                
                else:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
                    return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}

        elif self.embedding_type.startswith("precomputed"):
            optimizer = torch.optim.AdamW(self.simple_linear_head.parameters(), weight_decay=self.WD)
        
        return optimizer

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        """ This function manually steps the scheduler. """
        scheduler['scheduler'].step()

    def on_train_epoch_end(self):
        if self.embedding_type in ('all_tokens', 'cls_token_only', 'mean_pooling'):
            if self.current_epoch > 0:
                
                if self.current_epoch % self.epoch_threshold_to_unlock_ESM2 == 0:
                    """ Occurs at the end of each epoch """
                    self.learning_rate = self.learning_rate_0
                    self.num_unfrozen_layers = min(self.max_num_layers_unfreeze_each_epoch,self.num_unfrozen_layers+self.num_layers_unfreeze_each_epoch)
                    print(f"Updated number of unfrozen layers: {self.num_unfrozen_layers}")

                    # Collect all currently optimized parameters to avoid duplication
                    current_params = set()
                    for group in self.optimizer.param_groups:
                        current_params.update(set(group['params']))

                    # Set up parameters with progressively scaled learning rates for ESM2, constant for FCNN
                    selected_layers = [name for name, param in self.ESM2_wo_lmhead.named_parameters() if "lm_head" not in name and "contact_head" not in name]
                    trainable_layers = selected_layers[-self.num_unfrozen_layers:]  # Select only the final num_unfrozen_layers
                    self.esm2_params = []
                    current_lr = self.learning_rate
                    
                    # Define the same learning rate for all FCNN layers
                    fcnn_layers = list(self.regression_mlps)
                    for fcnn_layer in fcnn_layers:
                        # print(f"Adding FCNN layer {fcnn_layer} to optimizer with lr: {current_lr}")
                        self.esm2_params.append({'params': list(fcnn_layer.parameters()), 'lr': current_lr})
                    
                    # Apply learning rate schedule for the last num_unfrozen_layers in ESM2 in reverse order
                    for name in reversed(trainable_layers):  # Reverse order of trainable layers to start schedule from the end
                        # print(f"Adding trainable layer {name} to optimizer with lr: {current_lr}")
                        layer_params = {
                            'params': [p for n, p in self.ESM2_wo_lmhead.named_parameters() if n == name and p.requires_grad],
                            'lr': current_lr
                        }
                        self.esm2_params.append(layer_params)
                        current_lr *= self.lr_mult  # Progressively scale learning rate for each layer
                    # print(self.esm2_params)

            # Report gradient max norm
            max_norm = 0
            for name, parameters in self.ESM2_wo_lmhead.named_parameters():
                if parameters.requires_grad:
                    param_norm = torch.norm(parameters.grad).item() if parameters.grad is not None else 0
                    max_norm = max(max_norm, param_norm)
            self.log('max_norm', max_norm, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
    
    def predict(self, sequence, embedding_type):
        """
        Predict the class logits and regression values for a single sequence.
        Args:
            sequence (str): Input sequence.
        Returns:
            class_logits (numpy.ndarray): Predicted classification logits.
            reg_logits (numpy.ndarray): Predicted regression values.
        """
        if embedding_type in ('all_tokens', 'cls_token_only', 'mean_pooling'):
            # Tokenize the sequence
            tokens = self.tokenizer(sequence, return_tensors='pt', padding=True).input_ids # Convert sequence to tokens
            if torch.cuda.is_available():
                tokens = tokens.to(self.device)
        
        else:
            tokens = torch.as_tensor(sequence) # already sequence embeddings

        # Forward pass through the model
        with torch.no_grad():
            reg_logits = self(tokens)

        # Convert outputs to numpy arrays for storage
        return reg_logits.cpu().numpy()

    def save_model(self, save_path, ema_save_path, checkpoint_path = None):
        """
        Save two versions of the model's state dictionary:
        1. Non-EMA applied parameters.
        2. EMA-applied parameters for ESM2_wo_lmhead.
        """
        if checkpoint_path is None: 
            # Save non-EMA version of the state_dict
            try:
                torch.save(self.state_dict(), save_path)
                print(f"Non-EMA model saved to {save_path}")
            except Exception as e:
                print(f"An error occurred while saving the non-EMA model: {e}") 
            # Save EMA-applied version of the state_dict
            if ema_save_path:
                self.ema.to(self.device)  # Ensure EMA is on the same device
                try:
                    # Store the original ESM2_wo_lmhead parameters
                    self.ema.store(self.ESM2_wo_lmhead.parameters())    
                    # Apply EMA weights to ESM2_wo_lmhead
                    self.ema.copy_to(self.ESM2_wo_lmhead.parameters())  
                    # Save the state_dict with EMA weights applied
                    torch.save(self.state_dict(), ema_save_path)
                    print(f"EMA model saved to {ema_save_path}")    
                    # Restore the original ESM2_wo_lmhead parameters
                    self.ema.restore(self.ESM2_wo_lmhead.parameters())
                except Exception as e:
                    print(f"An error occurred while saving the EMA model: {e}")
                finally:
                	self.ema.to('cpu')
        else:
            torch.save(self.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

# CreiLOVFcnDataset is a data handling class.
class CreiLOVFcnDataset(torch.utils.data.Dataset):
    """A custom PyTorch dataset for protein sequence-function data"""

    def __init__(self, data_frame, token_format='ESM2', emb_precomputed = False, embedding_type = None):
        self.data_df = data_frame
        self.token_format = token_format
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emb_precomputed = emb_precomputed
        self.embedding_type = embedding_type

    def __getitem__(self, idx):
        if self.emb_precomputed:
            if self.embedding_type == 'precomputed_mean_pooling':
                sequence_representation = torch.as_tensor(self.data_df.mean_pooled.iloc[idx])
            elif self.embedding_type == 'precomputed_cls_token_only':
                sequence_representation = torch.as_tensor(self.data_df.cls_token.iloc[idx])
            elif self.embedding_type == 'precomputed_all_tokens':
                sequence_representation = torch.as_tensor(self.data_df.all_tokens.iloc[idx])
        
        else:
            if self.token_format == 'aa2ind':
                sequence_representation = torch.tensor(aa2ind(list(self.data_df.Sequence.iloc[idx]))) # Extract sequence at index idx
            elif self.token_format == 'ESM2':
                sequence_representation = self.data_df.Sequence.iloc[idx]  # Directly get the sequence string for ESM2 tokenizer

        # I convert all labels, including -1 values, to torch tensors
        reg_labels = torch.tensor(self.data_df.log_mean.iloc[idx].tolist()).float() # Extract labels for sequence at index idx and convert to a list
        
        return sequence_representation, reg_labels

    def __len__(self):
        return len(self.data_df)

# ProtDataModule splits the data into three different datasets.
class CreiLOVDataModule(pl.LightningDataModule):
    """A PyTorch Lightning Data Module to handle data splitting"""

    def __init__(self, data_frame, batch_size, splits_path=None, token_format='ESM2', seed=0, test_set_path = None, emb_precomputed=False, embedding_type = None):
        # Call the __init__ method of the parent class
        super().__init__()

        # Store the batch size
        self.batch_size = batch_size
        self.data_df = data_frame
        self.token_format = token_format
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emb_precomputed = emb_precomputed
        self.embedding_type = embedding_type

        if test_set_path is not None:
            # Load test set from external file
            self.test_df = pd.read_pickle(test_set_path)
        else:
            self.test_df = None
        
        if splits_path is not None:
            train_indices, val_indices = self.load_splits(splits_path)
            
            # Shuffle the indices to ensure that the data from each cluster is mixed. Do I want this?
            random.shuffle(train_indices)
            random.shuffle(val_indices)

            # Store the indices for the training, validation, and test sets
            self.train_idx = train_indices
            self.val_idx = val_indices

            # Verification
            print("Training set size:", len(self.train_idx))
            print("Validation set size:", len(self.val_idx))
            print("Test set size:", len(self.test_df))

        else:
            # Calculate split sizes
            total_samples = len(self.data_df)
            train_size = int(0.9 * total_samples)
            val_size = int(0.1 * total_samples)
            
            # Shuffle the DataFrame to ensure randomness
            shuffled_df = self.data_df.sample(frac=1, random_state=5)

            # Split the indices for training, validation, and testing
            train_indices = list(shuffled_df.index[:train_size])
            val_indices = list(shuffled_df.index[train_size:train_size + val_size])

            # Store the indices for the training, validation, and test sets
            self.train_idx = train_indices
            self.val_idx = val_indices
            self.test_idx = []
            
            # Verification
            print("Training set size:", len(self.train_idx))
            print("Validation set size:", len(self.val_idx))
            print("Test set size:", len(self.test_df))

    # Assigns train, validation and test datasets for use in dataloaders.
    def setup(self, stage=None):
        
        # Assign train/validation datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            train_data_frame = self.data_df.iloc[list(self.train_idx)]
            self.train_ds = CreiLOVFcnDataset(train_data_frame, self.token_format, self.emb_precomputed, self.embedding_type)
            val_data_frame = self.data_df.iloc[list(self.val_idx)]
            self.val_ds = CreiLOVFcnDataset(val_data_frame, self.token_format, self.emb_precomputed, self.embedding_type)
                    
        # Assigns test dataset for use in dataloader
        if stage == 'test' or stage is None:
            if self.test_df is None:
                raise RuntimeError("test_set_path was not provided, but test_dataloader was requested.")
            self.test_ds = CreiLOVFcnDataset(self.test_df, self.token_format, self.emb_precomputed, self.embedding_type)

    def seed_worker(worker_id, worker_info):
        worker_seed = torch.initial_seed() % 2**32  # Compute a seed for the worker based on the initial seed of the torch Generator
        np.random.seed(worker_seed)  # Set NumPy's random seed based on the worker seed
        random.seed(worker_seed)  # Set Python's built-in random module's seed
            
    #The DataLoader object is created using the train_ds/val_ds/test_ds objects with the batch size set during initialization of the class and shuffle=True.
    def train_dataloader(self):
        # Determine if we're running on a GPU
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            generator = torch.Generator()  # Create a new torch Generator
            generator.manual_seed(self.seed)  # Manually seed the generator with the predefined seed from the class
            # Create and return a DataLoader configured for training
            print('Sending data to GPU')
            return data_utils.DataLoader(
                self.train_ds,  # The dataset to load, in this case, the training dataset
                batch_size=self.batch_size,  # The number of samples in each batch to load
                shuffle=True,  # Enable shuffling to randomize the order of data before each epoch
                worker_init_fn=self.seed_worker,  # Function to initialize each worker's seed to ensure reproducibility across runs
                generator=generator,  # Specify the generator used for random number generation in shuffling
                num_workers=max(4, os.cpu_count() // 2),  # The number of subprocesses to use for data loading. More workers can increase the speed of data loading
                pin_memory=True  # Pins memory, allowing faster and more efficient transfer of data from host to GPU when training on GPUs
            )
        else:
            print('Sending data to CPU')
            return data_utils.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
    
    
    def val_dataloader(self):
        # Determine if we're running on a GPU
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            generator = torch.Generator()  # Create a new torch Generator
            generator.manual_seed(self.seed)  # Manually seed the generator with the predefined seed from the class
            # Create and return a DataLoader configured for training
            # print('Sending data to GPU')
            return data_utils.DataLoader(
                self.val_ds,  # The dataset to load, in this case, the training dataset
                batch_size=self.batch_size,  # The number of samples in each batch to load
                shuffle=False,
                worker_init_fn=self.seed_worker,  # Function to initialize each worker's seed to ensure reproducibility across runs
                generator=generator,  # Specify the generator used for random number generation in shuffling
                num_workers=max(4, os.cpu_count() // 2),  # The number of subprocesses to use for data loading. More workers can increase the speed of data loading
                pin_memory=True  # Pins memory, allowing faster and more efficient transfer of data from host to GPU when training on GPUs
            )
        else:
            # print('Sending data to CPU')
            return data_utils.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True)
    
    
    def test_dataloader(self):
        # Determine if we're running on a GPU
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            generator = torch.Generator()  # Create a new torch Generator
            generator.manual_seed(self.seed)  # Manually seed the generator with the predefined seed from the class
            # Create and return a DataLoader configured for training
            # print('Sending data to GPU')
            return data_utils.DataLoader(
                self.test_ds,  # The dataset to load, in this case, the training dataset
                batch_size=self.batch_size,  # The number of samples in each batch to load
                shuffle=False,
                worker_init_fn=self.seed_worker,  # Function to initialize each worker's seed to ensure reproducibility across runs
                generator=generator,  # Specify the generator used for random number generation in shuffling
                num_workers=max(4, os.cpu_count() // 2),  # The number of subprocesses to use for data loading. More workers can increase the speed of data loading
                pin_memory=True  # Pins memory, allowing faster and more efficient transfer of data from host to GPU when training on GPUs
            )
        else:
            # print('Sending data to CPU')
            return data_utils.DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=True)
    
    def save_splits(self, path):
        """Save the data splits to a file at the given path"""
        with open(path, 'wb') as f:
            pickle.dump((self.train_idx, self.val_idx), f)

    def load_splits(self, path):
        """Load the data splits from a file at the given path"""
        with open(path, 'rb') as f:
            self.train_idx, self.val_idx = pickle.load(f)
            
            train_indices = self.train_idx
            val_indices = self.val_idx
            
        return train_indices, val_indices

# GeneralFcnDataset is a data handling class.
class GeneralFcnDataset(torch.utils.data.Dataset):
    """A custom PyTorch dataset for protein sequence-function data"""

    def __init__(self, data_frame, token_format='ESM2', emb_precomputed = False, embedding_type = None):
        self.data_df = data_frame
        self.token_format = token_format
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emb_precomputed = emb_precomputed
        self.embedding_type = embedding_type

    def __getitem__(self, idx):
        if self.emb_precomputed:
            if self.embedding_type == 'precomputed_mean_pooling':
                sequence_representation = torch.as_tensor(self.data_df.mean_pooled.iloc[idx])
            elif self.embedding_type == 'precomputed_cls_token_only':
                sequence_representation = torch.as_tensor(self.data_df.cls_token.iloc[idx])
            elif self.embedding_type == 'precomputed_all_tokens':
                sequence_representation = torch.as_tensor(self.data_df.all_tokens.iloc[idx])
        
        else:
            if self.token_format == 'aa2ind':
                sequence_representation = torch.tensor(aa2ind(list(self.data_df.sequence.iloc[idx]))) # Extract sequence at index idx
            elif self.token_format == 'ESM2':
                sequence_representation = self.data_df.sequence.iloc[idx]  # Directly get the sequence string for ESM2 tokenizer

        # I convert all labels, including -1 values, to torch tensors
        reg_labels = torch.tensor(self.data_df.functional_score.iloc[idx].tolist()).float() # Extract labels for sequence at index idx and convert to a list
        
        return sequence_representation, reg_labels

    def __len__(self):
        return len(self.data_df)

# ProtDataModule splits the data into three different datasets.
class GeneralDataModule(pl.LightningDataModule):
    """A PyTorch Lightning Data Module to handle data splitting"""

    def __init__(self, data_frame, batch_size, splits_path=None, token_format='ESM2', seed=0, test_set_path = None, emb_precomputed = False, embedding_type = None):
        # Call the __init__ method of the parent class
        super().__init__()

        # Store the batch size
        self.batch_size = batch_size
        self.data_df = data_frame
        self.token_format = token_format
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emb_precomputed = emb_precomputed
        self.embedding_type = embedding_type

        if test_set_path is not None:
            # Load test set from external file
            self.test_df = pd.read_pickle(test_set_path)
        else:
            self.test_df = None
        
        if splits_path is not None:
            train_indices, val_indices = self.load_splits(splits_path)
            
            # Shuffle the indices to ensure that the data from each cluster is mixed. Do I want this?
            random.shuffle(train_indices)
            random.shuffle(val_indices)

            # Store the indices for the training, validation, and test sets
            self.train_idx = train_indices
            self.val_idx = val_indices

            # Verification
            print("Training set size:", len(self.train_idx))
            print("Validation set size:", len(self.val_idx))
            print("Test set size:", len(self.test_df))

        else:
            # Calculate split sizes
            total_samples = len(self.data_df)
            train_size = int(0.9 * total_samples)
            val_size = int(0.1 * total_samples)
            
            # Shuffle the DataFrame to ensure randomness
            shuffled_df = self.data_df.sample(frac=1, random_state=5)
            
            # Extract the shuffled index (from original df)
            shuffled_idx = shuffled_df.index.to_list()

            # Split on shuffled indices
            self.train_idx = shuffled_idx[:train_size]
            self.val_idx   = shuffled_idx[train_size:]

            # Verification
            print("Training set size:", len(self.train_idx))
            print("Validation set size:", len(self.val_idx))
            print("Test set size:", len(self.test_df))

    # Assigns train, validation and test datasets for use in dataloaders.
    def setup(self, stage=None):
        
        # Assign train/validation datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            train_data_frame = self.data_df.iloc[list(self.train_idx)]
            self.train_ds = GeneralFcnDataset(train_data_frame, self.token_format, self.emb_precomputed, self.embedding_type)
            val_data_frame = self.data_df.iloc[list(self.val_idx)]
            self.val_ds = GeneralFcnDataset(val_data_frame, self.token_format, self.emb_precomputed, self.embedding_type)
                    
        # Assigns test dataset for use in dataloader
        if stage == 'test' or stage is None:
            if self.test_df is None:
                raise RuntimeError("test_set_path was not provided, but test_dataloader was requested.")
            self.test_ds = GeneralFcnDataset(self.test_df, self.token_format, self.emb_precomputed, self.embedding_type)

    def seed_worker(worker_id, worker_info):
        worker_seed = torch.initial_seed() % 2**32  # Compute a seed for the worker based on the initial seed of the torch Generator
        np.random.seed(worker_seed)  # Set NumPy's random seed based on the worker seed
        random.seed(worker_seed)  # Set Python's built-in random module's seed
            
    #The DataLoader object is created using the train_ds/val_ds/test_ds objects with the batch size set during initialization of the class and shuffle=True.
    def train_dataloader(self):
        # Determine if we're running on a GPU
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            generator = torch.Generator()  # Create a new torch Generator
            generator.manual_seed(self.seed)  # Manually seed the generator with the predefined seed from the class
            # Create and return a DataLoader configured for training
            print('Sending data to GPU')
            return data_utils.DataLoader(
                self.train_ds,  # The dataset to load, in this case, the training dataset
                batch_size=self.batch_size,  # The number of samples in each batch to load
                shuffle=True,  # Enable shuffling to randomize the order of data before each epoch
                worker_init_fn=self.seed_worker,  # Function to initialize each worker's seed to ensure reproducibility across runs
                generator=generator,  # Specify the generator used for random number generation in shuffling
                num_workers=max(4, os.cpu_count() // 2),  # The number of subprocesses to use for data loading. More workers can increase the speed of data loading
                pin_memory=True  # Pins memory, allowing faster and more efficient transfer of data from host to GPU when training on GPUs
            )
        else:
            print('Sending data to CPU')
            return data_utils.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
    
    
    def val_dataloader(self):
        # Determine if we're running on a GPU
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            generator = torch.Generator()  # Create a new torch Generator
            generator.manual_seed(self.seed)  # Manually seed the generator with the predefined seed from the class
            # Create and return a DataLoader configured for training
            # print('Sending data to GPU')
            return data_utils.DataLoader(
                self.val_ds,  # The dataset to load, in this case, the training dataset
                batch_size=self.batch_size,  # The number of samples in each batch to load
                shuffle=False,
                worker_init_fn=self.seed_worker,  # Function to initialize each worker's seed to ensure reproducibility across runs
                generator=generator,  # Specify the generator used for random number generation in shuffling
                num_workers=max(4, os.cpu_count() // 2),  # The number of subprocesses to use for data loading. More workers can increase the speed of data loading
                pin_memory=True  # Pins memory, allowing faster and more efficient transfer of data from host to GPU when training on GPUs
            )
        else:
            # print('Sending data to CPU')
            return data_utils.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True)
    
    
    def test_dataloader(self):
        # Determine if we're running on a GPU
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            generator = torch.Generator()  # Create a new torch Generator
            generator.manual_seed(self.seed)  # Manually seed the generator with the predefined seed from the class
            # Create and return a DataLoader configured for training
            # print('Sending data to GPU')
            return data_utils.DataLoader(
                self.test_ds,  # The dataset to load, in this case, the training dataset
                batch_size=self.batch_size,  # The number of samples in each batch to load
                shuffle=False,
                worker_init_fn=self.seed_worker,  # Function to initialize each worker's seed to ensure reproducibility across runs
                generator=generator,  # Specify the generator used for random number generation in shuffling
                num_workers=max(4, os.cpu_count() // 2),  # The number of subprocesses to use for data loading. More workers can increase the speed of data loading
                pin_memory=True  # Pins memory, allowing faster and more efficient transfer of data from host to GPU when training on GPUs
            )
        else:
            # print('Sending data to CPU')
            return data_utils.DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=True)
    
    def save_splits(self, path):
        """Save the data splits to a file at the given path"""
        with open(path, 'wb') as f:
            pickle.dump((self.train_idx, self.val_idx), f)

    def load_splits(self, path):
        """Load the data splits from a file at the given path"""
        with open(path, 'rb') as f:
            self.train_idx, self.val_idx = pickle.load(f)
            
            train_indices = self.train_idx
            val_indices = self.val_idx
            
        return train_indices, val_indices








