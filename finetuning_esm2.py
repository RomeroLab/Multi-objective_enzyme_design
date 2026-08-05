#!/usr/bin/env python
"""Partially fine-tune ESM-2 on Gre2 using hyperparameters from YAML."""

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
import yaml
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from transformers import AutoModelForMaskedLM, AutoTokenizer

from models.partial_esm2_ft_w_MLP import (
    ProtDataModule,
    finetuning_ESM2_with_mse_loss,
)


LABEL_COLUMN = "Quantile Rxn Rate at 0.05mgml"
TOKEN_FORMAT = "ESM2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Partially fine-tune ESM-2 to predict Gre2 initial reaction rate."
    )
    parser.add_argument("--hparams", type=Path, default=Path("./models/training_finetuned_ESM2_for_reaction_rate/hparams.yaml"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/finetuned_esm2"),
    )
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/finetuned_ESM2_for_reaction_rate.pt"),
    )
    return parser.parse_args()


def load_hparams(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Hyperparameter file not found: {path}")
    with path.open() as handle:
        hparams = yaml.safe_load(handle)

    required = {
        "WD",
        "WT",
        "batch_size",
        "data_filepath",
        "decay",
        "embedding_type",
        "epoch_threshold_to_unlock_ESM2",
        "epochs",
        "grad_clip_threshold",
        "huggingface_identifier",
        "learning_rate",
        "lr_mult",
        "lr_mult_factor",
        "max_num_layers_unfreeze_each_epoch",
        "num_layers_unfreeze_each_epoch",
        "num_reg_tasks",
        "num_unfrozen_layers",
        "reg_weights",
        "seed",
        "slen",
        "use_scheduler",
        "using_EMA",
        "warm_restart",
    }
    missing = required.difference(hparams)
    if missing:
        raise KeyError(f"hparams.yaml is missing keys: {sorted(missing)}")
    if hparams["embedding_type"] not in {
        "all_tokens",
        "cls_token_only",
        "mean_pooling",
    }:
        raise ValueError(
            "EMA fine-tuning requires all_tokens, cls_token_only, or mean_pooling"
        )
    if int(hparams["using_EMA"]) != 1:
        raise ValueError("using_EMA must equal 1 because only an EMA model is saved")
    if int(hparams["slen"]) != len(hparams["WT"]):
        raise ValueError("slen does not match the WT sequence length")
    return hparams


def resolve_paths(args: argparse.Namespace, hparams: dict) -> tuple[Path, Path]:
    data_path = args.data or args.data_dir / hparams["data_filepath"]
    splits_path = args.splits or data_path.with_name(
        f"{data_path.stem}_data_splits.pkl"
    )
    return data_path, splits_path


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
        raise ValueError("Training data must be a .pkl or .csv file")

    missing = {"Sequence", LABEL_COLUMN}.difference(df.columns)
    if missing:
        raise KeyError(f"Training dataset is missing columns: {sorted(missing)}")
    return df.fillna(-1)


def build_datamodule(
    df: pd.DataFrame,
    hparams: dict,
    splits_path: Path,
) -> ProtDataModule:
    if not splits_path.exists():
        raise FileNotFoundError(f"Deposited split file not found: {splits_path}")
    return ProtDataModule(
        data_frame=df,
        label_index=df.columns.get_loc(LABEL_COLUMN),
        batch_size=int(hparams["batch_size"]),
        splits_path=str(splits_path),
        token_format=TOKEN_FORMAT,
        seed=int(hparams["seed"]),
    )


def build_model(
    esm2: torch.nn.Module,
    tokenizer: AutoTokenizer,
    hparams: dict,
    data_path: Path,
) -> finetuning_ESM2_with_mse_loss:
    # Keywords intentionally mirror the current model class signature.
    return finetuning_ESM2_with_mse_loss(
        ESM2=esm2,
        huggingface_identifier=hparams["huggingface_identifier"],
        tokenizer=tokenizer,
        num_unfrozen_layers=int(hparams["num_unfrozen_layers"]),
        num_layers_unfreeze_each_epoch=int(
            hparams["num_layers_unfreeze_each_epoch"]
        ),
        max_num_layers_unfreeze_each_epoch=int(
            hparams["max_num_layers_unfreeze_each_epoch"]
        ),
        epochs=int(hparams["epochs"]),
        batch_size=int(hparams["batch_size"]),
        seed=int(hparams["seed"]),
        embedding_type=hparams["embedding_type"],
        learning_rate=float(hparams["learning_rate"]),
        lr_mult=float(hparams["lr_mult"]),
        lr_mult_factor=float(hparams["lr_mult_factor"]),
        WD=float(hparams["WD"]),
        grad_clip_threshold=float(hparams["grad_clip_threshold"]),
        use_scheduler=int(hparams["use_scheduler"]),
        warm_restart=int(hparams["warm_restart"]),
        slen=int(hparams["slen"]),
        reg_weights=hparams["reg_weights"],
        num_reg_tasks=int(hparams["num_reg_tasks"]),
        reg_type="mse",
        using_EMA=int(hparams["using_EMA"]),
        decay=float(hparams["decay"]),
        epoch_threshold_to_unlock_ESM2=int(
            hparams["epoch_threshold_to_unlock_ESM2"]
        ),
        WT=hparams["WT"],
        data_filepath=str(data_path),
    )


def predict(
    model: finetuning_ESM2_with_mse_loss,
    sequences: pd.Series,
    embedding_type: str,
) -> np.ndarray:
    predictions = []
    for sequence in sequences:
        value = np.asarray(
            model.predict(sequence, embedding_type),
            dtype=float,
        ).reshape(-1)
        predictions.append(float(np.median(value)))
    return np.asarray(predictions)


def save_evaluation(
    model: finetuning_ESM2_with_mse_loss,
    df: pd.DataFrame,
    dm: ProtDataModule,
    embedding_type: str,
    output_dir: Path,
) -> None:
    rows = []
    metric_rows = []
    for split, indices in (("train", dm.train_idx), ("validation", dm.val_idx)):
        split_df = df.iloc[indices]
        actual = split_df[LABEL_COLUMN].to_numpy(dtype=float)
        predicted = predict(model, split_df["Sequence"], embedding_type)
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
    hparams = load_hparams(args.hparams)
    data_path, splits_path = resolve_paths(args, hparams)
    set_seed(int(hparams["seed"]))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this script on a GPU node.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Hyperparameters: {args.hparams}")
    print(f"Training data: {data_path}")
    print(f"Data splits: {splits_path}")

    df = load_dataframe(data_path)
    dm = build_datamodule(df, hparams, splits_path)
    model_id = hparams["huggingface_identifier"]
    esm2 = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_id}")
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_id}")
    model = build_model(esm2, tokenizer, hparams, data_path)

    logger = CSVLogger(save_dir=str(args.log_dir), name="esm2")
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=int(hparams["epochs"]),
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
        raise AttributeError("EMA was not initialized by the fine-tuning model")

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(logger.log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # The model's EMA tracks ESM-2 parameters. Apply them temporarily while
    # saving and evaluating; the regression head remains at its trained values.
    ema.to(model.device)
    ema.store(model.ESM2_wo_lmhead.parameters())
    try:
        ema.copy_to(model.ESM2_wo_lmhead.parameters())
        torch.save(model.state_dict(), args.model_output)
        model.eval()
        save_evaluation(
            model,
            df,
            dm,
            hparams["embedding_type"],
            run_dir,
        )
    finally:
        ema.restore(model.ESM2_wo_lmhead.parameters())
        ema.to("cpu")

    save_loss_curve(run_dir)
    print(f"Saved EMA model: {args.model_output}")
    print(f"Logs and evaluation outputs: {run_dir}")


if __name__ == "__main__":
    main()

