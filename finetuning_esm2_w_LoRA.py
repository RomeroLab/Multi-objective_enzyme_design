#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from scipy.stats import spearmanr
from sklearn import metrics
from transformers import AutoTokenizer

from models.esm2_w_lora_w_MLP import ESM2LoRAResidueRegressor, SequenceFunctionDataModule

CREILOV_WT = (
    "MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRK"
    "DGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA"
)

PROTEIN_NAME = "CreiLOV"
EMBEDDING_TYPE = "all_tokens"
TRAINING_SEED = 3

DEFAULT_DATASETS = [
    "subset_single_CreiLOV_mutants",
    "subset_double_CreiLOV_mutants",
    "subset_triple_CreiLOV_mutants",
    "subset_quadruple_CreiLOV_mutants",
    "subset_five_CreiLOV_mutants",
    "dataset_df_for_0thru5_CreiLOV_mutants",
    # download additional datasets from https://huggingface.co/datasets/RomeroLab-Duke/protein-fitness-datasets-for-benchmarking-ft-esm2-strategies
]

DEFAULT_DATA_DIR = Path("data/finetuned_esm2")

# Dataset-specific training schedules from
# Finetuning_ESM2_w_lora_and_linear_head.py. This reproduction is restricted
# to CreiLOV, all-token embeddings, and seed 3. All other model and optimizer
# hyperparameters are shared between the subset and 0thru5 runs.
SUBSET_TRAINING_HPARAMS = {
    "epochs": 2000,
    "patience": 400,
    "batch_size": 16,
}

ZERO_TO_FIVE_TRAINING_HPARAMS = {
    "epochs": 2000,
    "patience": 500,
    "batch_size": 16,
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune ESM-2 with LoRA on all-token CreiLOV representations."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/lora"))
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--test-set", default="test_CreiLOV_mutants.csv")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=5,
        help="Seed used once to create the fixed 90/10 train/validation split.",
    )
    parser.add_argument("--model", default="esm2_t33_650M_UR50D")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the dataset-specific number of training epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Override the dataset-specific early-stopping patience.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Override the dataset-specific batch size.",
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-lora-layers", type=int, default=27)
    parser.add_argument("--lora-lr", type=float, default=1e-6)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.005)
    parser.add_argument("--l1-lambda", type=float, default=1e-7)
    parser.add_argument("--bottleneck", type=int, default=640)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=1)
    parser.add_argument("--lora-dropout", type=float, default=0.10)
    parser.add_argument("--head-dropout", type=float, default=0.10)
    parser.add_argument("--label-col", default="log_mean")
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


def safe_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(spearmanr(x, y).statistic)
    return pearson, spearman


def score_arrays(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pearson, spearman = safe_corr(x, y)
    return {
        "n": int(len(x)),
        "mse": float(metrics.mean_squared_error(x, y)) if len(x) else np.nan,
        "r": pearson,
        "rho": spearman,
    }


def detect_sequence_col(df: pd.DataFrame) -> str:
    if "Sequence" in df.columns:
        return "Sequence"
    if "sequence" in df.columns:
        return "sequence"
    raise KeyError("Expected a 'Sequence' or 'sequence' column.")


def resolve_table_path(data_dir: Path, name: str) -> Path:
    """Resolve an explicit filename or a dataset stem within data_dir."""
    supplied = Path(name)
    candidates = [supplied] if supplied.suffix else [
        supplied.with_suffix(".csv"),
        supplied.with_suffix(".pkl"),
    ]
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else data_dir / candidate
        if path.exists():
            return path
    checked = ", ".join(str(data_dir / candidate) for candidate in candidates)
    raise FileNotFoundError(f"Dataset not found. Checked: {checked}")


def load_table(path: Path, label_col: str) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    sequence_col = detect_sequence_col(df)
    if label_col not in df.columns:
        raise KeyError(f"{path} does not contain label column {label_col!r}.")
    if df[sequence_col].isna().any() or df[label_col].isna().any():
        raise ValueError(
            f"{path} contains missing values in {sequence_col!r} or {label_col!r}."
        )
    return df.reset_index(drop=True)


def resolve_training_hparams(
    args: argparse.Namespace,
    dataset_path: Path,
) -> dict[str, int | str]:
    """Return the deposited training schedule for the selected dataset."""
    dataset_name = dataset_path.stem.casefold()
    if "0thru5_creilov_mutants" in dataset_name:
        hparams = ZERO_TO_FIVE_TRAINING_HPARAMS.copy()
        dataset_regime = "0thru5"
    elif dataset_name.startswith("subset_") and "creilov_mutants" in dataset_name:
        hparams = SUBSET_TRAINING_HPARAMS.copy()
        dataset_regime = "subset"
    else:
        raise ValueError(
            "Could not determine the CreiLOV training regime from "
            f"{dataset_path.name!r}. Expected a subset_*_CreiLOV_mutants or "
            "dataset_df_for_0thru5_CreiLOV_mutants dataset."
        )

    for key, override in (
        ("epochs", args.epochs),
        ("patience", args.patience),
        ("batch_size", args.batch_size),
    ):
        if override is not None:
            if override <= 0:
                raise ValueError(f"--{key.replace('_', '-')} must be positive.")
            hparams[key] = int(override)

    hparams["dataset_regime"] = dataset_regime
    return hparams


def save_run_hparams(
    log_dir: Path,
    args: argparse.Namespace,
    dataset_path: Path,
    seed: int,
    training_hparams: dict[str, int | str],
) -> None:
    """Record the complete effective configuration for reproducibility."""
    hparams = {
        "dataset": str(dataset_path),
        "dataset_regime": training_hparams["dataset_regime"],
        "protein": PROTEIN_NAME,
        "embedding_type": EMBEDDING_TYPE,
        "seed": seed,
        "split_seed": args.split_seed,
        "model": args.model,
        "epochs": training_hparams["epochs"],
        "patience": training_hparams["patience"],
        "batch_size": training_hparams["batch_size"],
        "num_lora_layers": args.num_lora_layers,
        "lora_learning_rate": args.lora_lr,
        "regression_head_learning_rate": args.head_lr,
        "weight_decay": args.weight_decay,
        "l1_lambda": args.l1_lambda,
        "bottleneck": args.bottleneck,
        "hidden": args.hidden,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "head_dropout": args.head_dropout,
        "label_column": args.label_col,
    }
    with (log_dir / "effective_hparams.json").open("w") as handle:
        json.dump(hparams, handle, indent=2)


def prepare_data_module_files(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
    output_dir: Path,
    split_seed: int,
) -> tuple[Path, Path]:
    """Create stable split indices and a pickle required by the data module."""
    cache_dir = output_dir / "prepared_splits"
    cache_dir.mkdir(parents=True, exist_ok=True)

    splits_path = cache_dir / f"{dataset_name}_splits.pkl"
    if not splits_path.exists():
        shuffled_indices = (
            df.sample(frac=1, random_state=split_seed).index.to_list()
        )
        train_size = int(0.9 * len(shuffled_indices))
        if train_size == 0 or train_size == len(shuffled_indices):
            raise ValueError(
                f"{dataset_name} needs at least two rows for a train/validation split."
            )
        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[train_size:]
        with splits_path.open("wb") as handle:
            pickle.dump((train_indices, val_indices), handle)

    test_pickle_path = cache_dir / "test_CreiLOV_mutants.pkl"
    test_df.to_pickle(test_pickle_path)
    return splits_path, test_pickle_path


def predict_dataframe(
    model: ESM2LoRAResidueRegressor,
    df: pd.DataFrame,
    label_col: str,
    batch_size: int,
    split: str,
) -> pd.DataFrame:
    seq_col = detect_sequence_col(df)
    rows = []
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch = df.iloc[start:end]
        preds = model.predict(batch[seq_col].tolist()).reshape(-1)
        for source_index, (_, row), pred in zip(batch.index, batch.iterrows(), preds):
            rows.append(
                {
                    "split": split,
                    "source_index": source_index,
                    "Sequence": row[seq_col],
                    "actual": row[label_col],
                    "predicted": float(pred),
                    "MutationCount": row.get("MutationCount", np.nan),
                }
            )
    return pd.DataFrame(rows)


def save_loss_curve(log_dir: Path) -> None:
    metrics_path = log_dir / "metrics.csv"
    if not metrics_path.exists():
        return
    df = pd.read_csv(metrics_path)
    train = df.loc[df["train_reg_loss"].notna(), "train_reg_loss"].to_numpy()
    val = df.loc[df["val_reg_loss"].notna(), "val_reg_loss"].to_numpy()
    if len(train) == 0 or len(val) == 0:
        return
    n = max(len(train), len(val))
    train = np.pad(train, (0, n - len(train)), constant_values=np.nan)
    val = np.pad(val, (0, n - len(val)), constant_values=np.nan)
    epochs = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(epochs, train, label="Train")
    ax.plot(epochs, val, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(log_dir / "loss_curve.png", dpi=300)
    fig.savefig(log_dir / "loss_curve.svg")
    plt.close(fig)


def save_prediction_plot(pred_df: pd.DataFrame, log_dir: Path, label_col: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for split in ["train", "val", "test"]:
        sub = pred_df[pred_df["split"] == split]
        if not sub.empty:
            ax.scatter(sub["actual"], sub["predicted"], s=8, alpha=0.7, label=split)
    finite = pred_df[["actual", "predicted"]].replace([np.inf, -np.inf], np.nan).dropna()
    if not finite.empty:
        low = float(finite.min().min())
        high = float(finite.max().max())
        ax.plot([low, high], [low, high], linestyle="--", linewidth=0.8)
    ax.set_xlabel(f"Measured {label_col}")
    ax.set_ylabel(f"Predicted {label_col}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(log_dir / "predicted_vs_measured.png", dpi=300)
    fig.savefig(log_dir / "predicted_vs_measured.svg")
    plt.close(fig)


def summarize_metrics(pred_df: pd.DataFrame, log_dir: Path) -> None:
    rows = []
    for split, sub in pred_df.groupby("split"):
        xy = sub[["actual", "predicted"]].dropna()
        row = {"split": split}
        row.update(score_arrays(xy["actual"].to_numpy(), xy["predicted"].to_numpy()))
        rows.append(row)
    pd.DataFrame(rows).to_csv(log_dir / "metrics_by_split.csv", index=False)

    test = pred_df[pred_df["split"] == "test"].dropna(subset=["MutationCount"])
    if test.empty:
        return
    rows = []
    for mut_count, sub in test.groupby("MutationCount"):
        xy = sub[["actual", "predicted"]].dropna()
        row = {"MutationCount": mut_count}
        row.update(score_arrays(xy["actual"].to_numpy(), xy["predicted"].to_numpy()))
        rows.append(row)
    pd.DataFrame(rows).sort_values("MutationCount").to_csv(log_dir / "test_metrics_by_mutationcount.csv", index=False)


def run_one_dataset(args: argparse.Namespace, dataset_name: str, seed: int, tokenizer: AutoTokenizer) -> None:
    set_seed(seed)
    data_path = resolve_table_path(args.data_dir, dataset_name)
    training_hparams = resolve_training_hparams(args, data_path)
    batch_size = training_hparams["batch_size"]
    test_path = resolve_table_path(args.data_dir, args.test_set)
    df = load_table(data_path, args.label_col)
    test_df = load_table(test_path, args.label_col)
    splits_path, test_pickle_path = prepare_data_module_files(
        df=df,
        test_df=test_df,
        dataset_name=data_path.stem,
        output_dir=args.output_dir,
        split_seed=args.split_seed,
    )

    dm = SequenceFunctionDataModule(
        data_frame=df,
        batch_size=batch_size,
        splits_path=str(splits_path),
        test_set_path=str(test_pickle_path),
        seed=seed,
        num_workers=args.num_workers,
    )

    model = ESM2LoRAResidueRegressor(
        huggingface_identifier=args.model,
        tokenizer=tokenizer,
        wt_sequence=CREILOV_WT,
        num_lora_layers=args.num_lora_layers,
        lora_learning_rate=args.lora_lr,
        regression_head_learning_rate=args.head_lr,
        bottleneck=args.bottleneck,
        hidden=args.hidden,
        weight_decay=args.weight_decay,
        l1_lambda=args.l1_lambda,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        head_dropout=args.head_dropout,
    )

    run_name = f"lora_all_tokens_esm2_{data_path.stem}_seed_{seed}"
    logger = CSVLogger(save_dir=str(args.output_dir), name=run_name)
    log_dir = Path(logger.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    save_run_hparams(log_dir, args, data_path, seed, training_hparams)
    checkpoint = ModelCheckpoint(
        dirpath=log_dir,
        monitor="val_reg_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stop = EarlyStopping(
        monitor="val_reg_loss",
        mode="min",
        patience=training_hparams["patience"],
    )

    print(
        f"{data_path.name}: regime={training_hparams['dataset_regime']}, "
        f"epochs={training_hparams['epochs']}, "
        f"patience={training_hparams['patience']}, batch_size={batch_size}"
    )

    trainer = pl.Trainer(
        max_epochs=training_hparams["epochs"],
        logger=logger,
        callbacks=[checkpoint, early_stop],
        accelerator="auto",
        devices="auto",
        deterministic=True,
        enable_progress_bar=True,
        log_every_n_steps=1,
    )
    trainer.fit(model, dm)
    logger.finalize("success")

    best_path = checkpoint.best_model_path or checkpoint.last_model_path
    if best_path:
        model = ESM2LoRAResidueRegressor.load_from_checkpoint(best_path, tokenizer=tokenizer)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    train_df = df.iloc[dm.train_idx].copy()
    val_df = df.iloc[dm.val_idx].copy()
    pred_df = pd.concat(
        [
            predict_dataframe(model, train_df, args.label_col, batch_size, "train"),
            predict_dataframe(model, val_df, args.label_col, batch_size, "val"),
            predict_dataframe(model, test_df, args.label_col, batch_size, "test"),
        ],
        ignore_index=True,
    )
    pred_df.to_csv(log_dir / "predictions.csv", index=False)
    summarize_metrics(pred_df, log_dir)
    save_loss_curve(log_dir)
    save_prediction_plot(pred_df, log_dir, args.label_col)
    print(f"Finished {dataset_name}, seed {seed}. Outputs: {log_dir}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{args.model}")
    for dataset_name in args.datasets:
        run_one_dataset(args, dataset_name, TRAINING_SEED, tokenizer)


if __name__ == "__main__":
    main()
