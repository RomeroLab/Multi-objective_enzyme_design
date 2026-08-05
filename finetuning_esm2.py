#!/usr/bin/env python
"""Partially fine-tune ESM-2 to predict Gre2 initial reaction rate."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from transformers import AutoModelForMaskedLM, AutoTokenizer

from models.partial_esm2_ft_w_MLP import (ProtDataModule, finetuning_ESM2_with_mse_loss)


GRE2_WT = ("MSVFVSGANGFIAQHIVDLLLKEDYKVIGSARSQEKAENLTEAFGNNPKFSMEVVPDISKLDAFDHVFQKHGKDIKIVLHTASPFCFDITDSERDLLIPAVNGVKGILHSIKKYAADSVERVVLTSSYAAVFDMAKENDKSLTFNEESWNPATWESCQSDPVNAYCGSKKFAEKAAWEFLEENRDSVKFELTAVNPVYVFGPQMFDKDVKKHLNTSCELVNSLMHLSPEDKIPELFGGYIDVRDVAKAHLVAFQKRETIGQRLIVSEARFTMQDVLDILNEDFPVLKGNIPVGKPGSGATHNTLGATLDNKKSKKLLGFKFRNLKETIDDTASQILKFEGRI")
LABEL_COLUMN = "Quantile Rxn Rate at 0.05mgml"
MODEL_ID = "esm2_t33_650M_UR50D"
TOKEN_FORMAT = "ESM2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Partially fine-tune ESM-2 650M on the Gre2 dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "data/finetuned_esm2/"
            "normalized_processed_BI_R1_dataset_w_quantiles.pkl"
        ),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path(
            "data/finetuned_esm2/"
            "normalized_processed_BI_R1_dataset_w_quantiles_data_splits.pkl"
        ),
    )
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/finetuned_ESM2_for_reaction_rate.pt"),
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")
    if path.suffix == ".pkl":
        df = pd.read_pickle(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("--data must be a .pkl or .csv file")

    missing = {"Sequence", LABEL_COLUMN}.difference(df.columns)
    if missing:
        raise KeyError(f"Training dataset is missing columns: {sorted(missing)}")
    return df.fillna(-1)


def build_datamodule(
    df: pd.DataFrame,
    batch_size: int,
    splits_path: Path,
    seed: int,
) -> ProtDataModule:
    if not splits_path.exists():
        raise FileNotFoundError(f"Deposited split file not found: {splits_path}")

    # Repository signature:
    # ProtDataModule(data_frame, batch_size, token_format, splits_path, seed)
    # The previous script reversed token_format and splits_path, so it tried to
    # open "ESM2" as the split file.
    return ProtDataModule(
        df,
        batch_size,
        TOKEN_FORMAT,
        str(splits_path),
        seed,
    )


def build_model(
    esm2: torch.nn.Module,
    tokenizer: AutoTokenizer,
    epochs: int,
    batch_size: int,
    seed: int,
) -> finetuning_ESM2_with_mse_loss:
    return finetuning_ESM2_with_mse_loss(
        esm2,
        MODEL_ID,
        tokenizer,
        27,       # num_unfrozen_layers
        15,       # num_layers_unfreeze_each_epoch
        36,       # max_num_layers_unfreeze_each_epoch
        epochs,
        batch_size,
        seed,
        1,        # cls_token_only
        1e-5,     # learning_rate
        1,        # lr_mult
        1,        # lr_mult_factor
        0.005,    # weight decay
        0,        # reinitialize optimizer
        3.0,      # gradient clipping threshold
        1,        # use scheduler
        1,        # use warm restart
        len(GRE2_WT),
        1,        # regression-task weight
        1,        # number of regression tasks
        1,        # use EMA
        0.8,      # EMA decay
    )

def predict(model: finetuning_ESM2_with_mse_loss, sequences: pd.Series) -> np.ndarray:
    values = []
    for sequence in sequences:
        prediction = np.asarray(model.predict(sequence), dtype=float).reshape(-1)
        values.append(float(np.median(prediction)))
    return np.asarray(values)

def save_evaluation(
    model: finetuning_ESM2_with_mse_loss,
    df: pd.DataFrame,
    dm: ProtDataModule,
    output_dir: Path,
) -> None:
    rows = []
    metric_rows = []
    for split, indices in (("train", dm.train_idx), ("validation", dm.val_idx)):
        split_df = df.iloc[indices]
        actual = split_df[LABEL_COLUMN].to_numpy(dtype=float)
        predicted = predict(model, split_df["Sequence"])
        rows.append(
            pd.DataFrame(
                {
                    "split": split,
                    "source_index": split_df.index,
                    "Sequence": split_df["Sequence"].to_numpy(),
                    "actual": actual,
                    "predicted": predicted,
                }
            )
        )
        metric_rows.append(
            {
                "split": split,
                "n": len(split_df),
                "mse": float(mean_squared_error(actual, predicted)),
                "pearson_r": float(np.corrcoef(actual, predicted)[0, 1]),
                "spearman_rho": float(spearmanr(actual, predicted).statistic),
            }
        )

    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(
        output_dir / "prediction_metrics.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for split, split_df in predictions.groupby("split"):
        ax.scatter(
            split_df["actual"],
            split_df["predicted"],
            s=14,
            alpha=0.75,
            label=split,
        )
    low = float(predictions[["actual", "predicted"]].min().min())
    high = float(predictions[["actual", "predicted"]].max().max())
    ax.plot([low, high], [low, high], "k--", linewidth=0.8)
    ax.set_xlabel(f"Measured {LABEL_COLUMN}")
    ax.set_ylabel(f"Predicted {LABEL_COLUMN}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "predictions.png", dpi=300)
    plt.close(fig)


def save_loss_curve(output_dir: Path) -> None:
    metrics_path = output_dir / "metrics.csv"
    if not metrics_path.exists():
        return
    metrics = pd.read_csv(metrics_path)
    train = metrics.loc[metrics["train_reg_loss"].notna(), "train_reg_loss"]
    validation = metrics.loc[metrics["val_reg_loss"].notna(), "val_reg_loss"]
    if train.empty or validation.empty:
        return

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(np.arange(1, len(train) + 1), train.to_numpy(), label="train")
    ax.plot(
        np.arange(1, len(validation) + 1),
        validation.to_numpy(),
        label="validation",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "loss.png", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this script on a GPU node.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    df = load_dataframe(args.data)
    dm = build_datamodule(df, args.batch_size, args.splits, args.seed)
    esm2 = AutoModelForMaskedLM.from_pretrained(f"facebook/{MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{MODEL_ID}")
    model = build_model(esm2, tokenizer, args.epochs, args.batch_size, args.seed)

    logger = CSVLogger(save_dir=str(args.log_dir), name="esm2")
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[
            EarlyStopping(
                monitor="val_reg_loss",
                patience=1000,
                mode="min",
            )
        ],
        accelerator="gpu",
        devices=1,
        deterministic=True,
        enable_checkpointing=False,
        enable_progress_bar=True,
        log_every_n_steps=1,
    )
    trainer.fit(model, dm)
    logger.finalize("success")

    ema = getattr(model, "ema", None)
    if ema is None:
        raise AttributeError(
            "The model has no 'ema' attribute; confirm that using_EMA=1."
        )

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(logger.log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Apply EMA parameters for saving and evaluation. No non-EMA model or
    # Lightning checkpoint is written.
    with ema.average_parameters():
        torch.save(model.state_dict(), args.model_output)
        model.eval()
        save_evaluation(model, df, dm, run_dir)

    save_loss_curve(run_dir)
    print(f"Saved EMA model: {args.model_output}")
    print(f"Logs and evaluation outputs: {run_dir}")


if __name__ == "__main__":
    main()

