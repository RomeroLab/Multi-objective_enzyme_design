from __future__ import annotations

import os
import pickle
import random
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModel, AutoTokenizer

class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ESM2LoRAResidueRegressor(pl.LightningModule):
    """LoRA fine-tuning of ESM-2 using residue-level final-layer embeddings."""

    def __init__(
        self,
        huggingface_identifier: str,
        tokenizer: AutoTokenizer,
        wt_sequence: str,
        num_lora_layers: int = 27,
        lora_learning_rate: float = 1e-6,
        regression_head_learning_rate: float = 1e-5,
        bottleneck: int = 640,
        hidden: int = 64,
        weight_decay: float = 1e-5,
        l1_lambda: float = 1e-7,
        lora_r: int = 4,
        lora_alpha: int = 1,
        lora_dropout: float = 0.10,
        head_dropout: float = 0.10,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])
        self.tokenizer = tokenizer
        self.wt_sequence = wt_sequence
        self.max_length = len(wt_sequence) + 2
        self.lora_learning_rate = lora_learning_rate
        self.regression_head_learning_rate = regression_head_learning_rate
        self.weight_decay = weight_decay
        self.l1_lambda = l1_lambda

        cfg = AutoConfig.from_pretrained(f"facebook/{huggingface_identifier}")
        if hasattr(cfg, "add_pooling_layer"):
            cfg.add_pooling_layer = False
        base = AutoModel.from_pretrained(f"facebook/{huggingface_identifier}", config=cfg)

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            bias="none",
            target_modules=[
                "attention.self.query",
                "attention.self.key",
                "attention.self.value",
                "attention.output.dense",
            ],
        )
        self.esm = get_peft_model(base, lora_cfg)
        self.enable_lora_last_n_layers(self.esm, int(num_lora_layers))

        hidden_size = self.esm.config.hidden_size
        self.proj = nn.Linear(hidden_size, int(bottleneck))
        self.mlp_head = MLPHead(
            in_dim=self.max_length * int(bottleneck),
            hidden=int(hidden),
            dropout=float(head_dropout),
        )

    def enable_lora_last_n_layers(self, peft_model: nn.Module, num_lora_layers: int) -> None:
        for param in peft_model.parameters():
            param.requires_grad = False

        if hasattr(peft_model, "encoder") and hasattr(peft_model.encoder, "layer"):
            blocks = peft_model.encoder.layer
            layer_prefix = "encoder.layer."
        elif hasattr(peft_model, "esm") and hasattr(peft_model.esm, "encoder"):
            blocks = peft_model.esm.encoder.layer
            layer_prefix = "esm.encoder.layer."
        else:
            raise RuntimeError("Could not locate ESM-2 encoder blocks.")

        total_layers = len(blocks)
        n_active = max(0, min(num_lora_layers, total_layers))

        for name, param in peft_model.named_parameters():
            if "lora_" in name:
                param.requires_grad = False

        for layer_idx in range(total_layers - n_active, total_layers):
            needle = f"{layer_prefix}{layer_idx}."
            for name, param in peft_model.named_parameters():
                if "lora_" in name and needle in name:
                    param.requires_grad = True

        print(f"Enabled LoRA adapters in {n_active}/{total_layers} ESM-2 transformer blocks.")

    def _tokenize(self, sequences: Sequence[str]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            list(sequences),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return {key: value.to(self.device, non_blocking=True) for key, value in encoded.items()}

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last = out.hidden_states[-1]
        if last.size(1) != self.max_length:
            raise RuntimeError(f"Expected token length {self.max_length}, got {last.size(1)}.")
        x = self.proj(last).flatten(1)
        return self.mlp_head(x)

    def _regularization(self) -> torch.Tensor:
        params: Iterable[torch.Tensor] = list(self.proj.parameters()) + list(self.mlp_head.parameters())
        return sum(param.abs().sum() for param in params)

    def training_step(self, batch, batch_idx):
        sequences, y = batch
        toks = self._tokenize(sequences)
        y = y.to(self.device, non_blocking=True).float().view(-1)
        preds = self(**toks)
        loss = F.mse_loss(preds, y)
        if self.l1_lambda > 0:
            loss = loss + self.l1_lambda * self._regularization()
        self.log("train_reg_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences, y = batch
        toks = self._tokenize(sequences)
        y = y.to(self.device, non_blocking=True).float().view(-1)
        preds = self(**toks)
        loss = F.mse_loss(preds, y)
        if self.l1_lambda > 0:
            loss = loss + self.l1_lambda * self._regularization()
        self.log("val_reg_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        lora_decay, lora_no_decay, head_decay, head_no_decay = [], [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_lora = "lora_" in name
            target = (lora_decay, lora_no_decay) if is_lora else (head_decay, head_no_decay)
            target[0 if param.ndim > 1 else 1].append(param)

        groups = [
            {"params": lora_decay, "lr": self.lora_learning_rate, "weight_decay": self.weight_decay},
            {"params": lora_no_decay, "lr": self.lora_learning_rate, "weight_decay": 0.0},
            {"params": head_decay, "lr": self.regression_head_learning_rate, "weight_decay": self.weight_decay},
            {"params": head_no_decay, "lr": self.regression_head_learning_rate, "weight_decay": 0.0},
        ]
        groups = [group for group in groups if group["params"]]
        return torch.optim.AdamW(groups, betas=(0.9, 0.999))

    @torch.no_grad()
    def predict(self, sequences: Sequence[str] | str) -> np.ndarray:
        if isinstance(sequences, str):
            sequences = [sequences]
        self.eval()
        toks = self._tokenize(sequences)
        preds = self(**toks)
        return preds.detach().cpu().numpy()


class SequenceFunctionDataset(torch.utils.data.Dataset):
    def __init__(self, data_frame: pd.DataFrame):
        self.df = data_frame.reset_index(drop=False).rename(columns={"index": "source_index"})
        if "Sequence" in self.df.columns:
            self.seq_col = "Sequence"
        elif "sequence" in self.df.columns:
            self.seq_col = "sequence"
        else:
            raise KeyError("Expected a 'Sequence' or 'sequence' column.")

        if "log_mean" in self.df.columns:
            self.label_col = "log_mean"
        elif "functional_score" in self.df.columns:
            self.label_col = "functional_score"
        else:
            raise KeyError("Expected a 'log_mean' or 'functional_score' label column.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        return row[self.seq_col], torch.tensor(row[self.label_col], dtype=torch.float32)


class SequenceFunctionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_frame: pd.DataFrame,
        batch_size: int,
        splits_path: Optional[str] = None,
        test_set_path: Optional[str] = None,
        seed: int = 0,
        num_workers: Optional[int] = None,
    ):
        super().__init__()
        self.df = data_frame.reset_index(drop=True)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.num_workers = min(8, os.cpu_count() or 1) if num_workers is None else int(num_workers)
        self.test_df = pd.read_pickle(test_set_path).reset_index(drop=True) if test_set_path else None

        if splits_path and os.path.exists(splits_path):
            self.train_idx, self.val_idx = self._load_splits(splits_path)
        else:
            self.train_idx, self.val_idx = self._make_random_split()

        print(f"Train: {len(self.train_idx)} | Val: {len(self.val_idx)} | Test: {0 if self.test_df is None else len(self.test_df)}")

    def _load_splits(self, path: str):
        with open(path, "rb") as handle:
            splits = pickle.load(handle)
        if len(splits) < 2:
            raise ValueError(f"Split file must contain train and validation indices: {path}")
        train_idx, val_idx = list(splits[0]), list(splits[1])
        random.Random(self.seed).shuffle(train_idx)
        return train_idx, val_idx

    def _make_random_split(self):
        idx = list(range(len(self.df)))
        random.Random(self.seed).shuffle(idx)
        n_train = int(0.9 * len(idx))
        return idx[:n_train], idx[n_train:]

    @staticmethod
    def seed_worker(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_ds = SequenceFunctionDataset(self.df.iloc[self.train_idx])
            self.val_ds = SequenceFunctionDataset(self.df.iloc[self.val_idx])
        if stage in (None, "test", "predict") and self.test_df is not None:
            self.test_ds = SequenceFunctionDataset(self.test_df)

    def _loader(self, dataset, shuffle: bool):
        generator = torch.Generator().manual_seed(self.seed)
        return data_utils.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=self.seed_worker,
            generator=generator,
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self):
        if self.test_df is None:
            raise RuntimeError("No test set was provided.")
        return self._loader(self.test_ds, shuffle=False)


