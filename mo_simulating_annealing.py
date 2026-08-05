#!/usr/bin/env python
"""Calibrate model-score bounds and run Gre2 multi-objective annealing."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data" / "finetuned_esm2"
MODELS_DIR = REPO_ROOT / "models"
STRUCTURES_DIR = REPO_ROOT / "structures"
SEQS_TO_SCORE_DIR = REPO_ROOT / "seqs_to_score"
SOLUBLEMPNN_OUTPUT_DIR = REPO_ROOT / "outputs"

F85L_PARENT = (
    "MSVFVSGANGFIAQHIVDLLLKEDYKVIGSARSQEKAENLTEAFGNNPKFSMEVVPDISKLDAFDHVFQKHGKDI"
    "KIVLHTASPLCFDITDSERDLLIPAVNGVKGILHSIKKYAADSVERVVLTSSYAAVFDMAKENDKSLTFNEESWNPA"
    "TWESCQSDPVNAYCGSKKFAEKAAWEFLEENRDSVKFELTAVNPVYVFGPQMFDKDVKKHLNTSCELVNSLMHLSP"
    "EDKIPELFGGYIDVRDVAKAHLVAFQKRETIGQRLIVSEARFTMQDVLDILNEDFPVLKGNIPVGKPGSGATHNTLG"
    "ATLDNKKSKKLLGFKFRNLKETIDDTASQILKFEGRI"
)

OBJECTIVES = ("esm2", "vae", "solublempnn")
DEFAULT_STEPS = {3: 30000, 5: 35000}
DEFAULT_MUTATION_RATES = {3: 1, 5: 2}


@dataclass
class LoadedModels:
    esm2: Any
    vae: Any
    solublempnn: bool


class ESM2InferenceAdapter:
    """Expose the one-argument prediction API expected by the SA helper."""

    def __init__(self, model: Any, embedding_type: str = "all_tokens") -> None:
        self.model = model
        self.embedding_type = embedding_type

    def predict(self, sequences):
        return self.model.predict(sequences, self.embedding_type)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate single-objective normalization bounds or run "
            "multi-objective simulated annealing for Gre2 F85L."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("calibrate", "build-bounds", "optimize"),
        default="optimize",
        help=(
            "calibrate optimizes one raw model score; build-bounds combines "
            "the three calibration summaries; optimize runs all objectives."
        ),
    )
    parser.add_argument(
        "--objective",
        choices=OBJECTIVES,
        help="Single objective to optimize in calibrate mode.",
    )
    parser.add_argument(
        "--num-mut",
        type=int,
        choices=(3, 5),
        default=5,
        help="Additional mutations beyond the F85L parent.",
    )
    parser.add_argument(
        "--assay-data-weight",
        type=float,
        default=1.0,
        help="Weight on the normalized ESM-2 reaction-rate objective.",
    )
    parser.add_argument("--num-solublempnn-samples", type=int, default=5)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument(
        "--nsteps",
        type=int,
        default=None,
        help="Defaults to 30,000 for 3 mutations and 35,000 for 5 mutations.",
    )
    parser.add_argument("--start-temp", type=float, default=-0.5)
    parser.add_argument("--final-temp", type=float, default=-2.25)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument(
        "--cuda-label",
        default=None,
        help=(
            "Label used in temporary SolubleMPNN paths. This does not select "
            "a GPU; use CUDA_VISIBLE_DEVICES for that."
        ),
    )
    parser.add_argument(
        "--bounds-file",
        type=Path,
        default=None,
        help="Normalization-bounds JSON. A num-mut-specific default is used.",
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=Path("normalization_calibration"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Gre2_redesign_round_1_with_F85L"),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use 1 trial, 10 steps, and 1 SolubleMPNN sample.",
    )
    return parser.parse_args()


def resolve_cli(args: argparse.Namespace) -> argparse.Namespace:
    if args.mode == "calibrate" and args.objective is None:
        raise ValueError("--objective is required when --mode calibrate is used.")
    if args.mode != "calibrate" and args.objective is not None:
        raise ValueError("--objective is only valid with --mode calibrate.")

    args.nsteps = args.nsteps or DEFAULT_STEPS[args.num_mut]
    if args.smoke_test:
        args.num_trials = 1
        args.nsteps = 10
        args.num_solublempnn_samples = 1

    for name in ("num_trials", "nsteps", "num_solublempnn_samples"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.assay_data_weight <= 0:
        raise ValueError("--assay-data-weight must be positive.")

    if args.cuda_label is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        args.cuda_label = visible.split(",")[0]

    if not args.calibration_root.is_absolute():
        args.calibration_root = REPO_ROOT / args.calibration_root
    if not args.output_root.is_absolute():
        args.output_root = REPO_ROOT / args.output_root
    if args.bounds_file is None:
        args.bounds_file = (
            REPO_ROOT / "normalization_bounds" / f"{args.num_mut}mut.json"
        )
    elif not args.bounds_file.is_absolute():
        args.bounds_file = REPO_ROOT / args.bounds_file
    return args


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_required_paths() -> None:
    required = [
        DATA_DIR / "normalized_processed_BI_R1_dataset_w_quantiles.pkl",
        MODELS_DIR / "Best_ConvVAE.ckpt",
        MODELS_DIR / "finetuned_ESM2_for_reaction_rate.pt",
        MODELS_DIR / "solublempnn" / "soluble_model_weights" / "v_48_020.pt",
        STRUCTURES_DIR / "4PVC.pdb1",
        REPO_ROOT / "protein_mpnn_run.py",
    ]
    missing = [path for path in required if not path.is_file()]
    if not (REPO_ROOT / "protein_mpnn_utils.py").is_file():
        if importlib.util.find_spec("protein_mpnn_utils") is None:
            missing.append(REPO_ROOT / "protein_mpnn_utils.py")
    if missing:
        paths = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required annealing files:\n{paths}")


def load_vae(device):
    import torch

    from models.ConvVAE import ConvVAE

    model = ConvVAE(
        len(F85L_PARENT),
        17,
        64,
        0.0001,
        1000,
        1,
        16,
        1,
        400,
    )
    checkpoint = torch.load(MODELS_DIR / "Best_ConvVAE.ckpt", map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {key.replace("model.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def load_esm2(device) -> ESM2InferenceAdapter:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    from models.partial_esm2_ft_w_MLP import finetuning_ESM2_with_mse_loss

    identifier = "esm2_t33_650M_UR50D"
    backbone = AutoModelForMaskedLM.from_pretrained(f"facebook/{identifier}")
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{identifier}")
    data_path = DATA_DIR / "normalized_processed_BI_R1_dataset_w_quantiles.pkl"

    model = finetuning_ESM2_with_mse_loss(
        ESM2=backbone,
        huggingface_identifier=identifier,
        tokenizer=tokenizer,
        num_unfrozen_layers=27,
        num_layers_unfreeze_each_epoch=15,
        max_num_layers_unfreeze_each_epoch=36,
        epochs=2000,
        batch_size=16,
        seed=3,
        embedding_type="all_tokens",
        learning_rate=1e-6,
        lr_mult=1,
        lr_mult_factor=1,
        WD=0.005,
        grad_clip_threshold=3.0,
        use_scheduler=1,
        warm_restart=1,
        slen=len(F85L_PARENT),
        reg_weights=1,
        num_reg_tasks=1,
        reg_type="mse",
        using_EMA=1,
        decay=0.8,
        epoch_threshold_to_unlock_ESM2=100,
        WT=F85L_PARENT,
        data_filepath=str(data_path),
    )
    checkpoint = torch.load(
        MODELS_DIR / "finetuned_ESM2_for_reaction_rate.pt",
        map_location=device,
    )
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return ESM2InferenceAdapter(model)


def load_models() -> LoadedModels:
    import torch

    validate_required_paths()
    SEQS_TO_SCORE_DIR.mkdir(parents=True, exist_ok=True)
    SOLUBLEMPNN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    return LoadedModels(
        esm2=load_esm2(device),
        vae=load_vae(device),
        solublempnn=True,
    )


def make_score_handler(
    args: argparse.Namespace,
    models: LoadedModels,
    trial: int,
    objective: str | None = None,
    bounds: dict[str, dict[str, float]] | None = None,
):
    from utils.simulated_annealing_utils import seq2fitness_handler

    esm2 = models.esm2 if objective in (None, "esm2") else None
    vae = models.vae if objective in (None, "vae") else None
    solublempnn = models.solublempnn if objective in (None, "solublempnn") else None

    return seq2fitness_handler(
        F85L_PARENT,
        args.num_mut,
        args.assay_data_weight,
        args.num_solublempnn_samples,
        esm2,
        None if bounds is None else bounds["esm2"]["max"],
        None if bounds is None else bounds["esm2"]["min"],
        solublempnn,
        None if bounds is None else bounds["solublempnn"]["max"],
        None if bounds is None else bounds["solublempnn"]["min"],
        vae,
        None if bounds is None else bounds["vae"]["max"],
        None if bounds is None else bounds["vae"]["min"],
        args.cuda_label,
        trial,
    )


def run_optimizer(args: argparse.Namespace, score_function):
    from utils.simulated_annealing_utils import (
        SA_optimizer,
        get_non_gap_indices,
    )

    non_gap_indices = get_non_gap_indices(F85L_PARENT)
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_options = [tuple(amino_acids) for _ in F85L_PARENT]
    aa_options[non_gap_indices[0]] = F85L_PARENT[0]
    aa_options[non_gap_indices[84]] = F85L_PARENT[84]

    optimizer = SA_optimizer(
        score_function,
        F85L_PARENT,
        aa_options,
        num_mut=args.num_mut,
        mutating_window_size=len(F85L_PARENT),
        mut_rate=DEFAULT_MUTATION_RATES[args.num_mut],
        nsteps=args.nsteps,
        cool_sched="log",
        non_gap_indices=non_gap_indices,
        start_temp=args.start_temp,
        final_temp=args.final_temp,
    )
    return optimizer, optimizer.optimize(start_mut=None)


def mutations_to_sequence(mutations) -> str:
    from utils.simulated_annealing_utils import mut2seq

    return mut2seq(F85L_PARENT, mutations)


def raw_scores(
    args: argparse.Namespace,
    models: LoadedModels,
    sequence: str,
    trial: int,
) -> dict[str, float]:
    handler = make_score_handler(args, models, trial, bounds=None)
    _, esm2, solublempnn, vae = handler.seq2fitness(sequence)
    return {
        "esm2": float(esm2),
        "vae": float(vae),
        "solublempnn": float(solublempnn),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def save_trajectory(output_dir: Path, trial: int, optimizer) -> None:
    csv_path = output_dir / f"trial_{trial:03d}_trajectory.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("step", "best_fitness", "current_fitness"),
        )
        writer.writeheader()
        for step, (best_fitness, current_fitness) in enumerate(
            optimizer.fitness_trajectory
        ):
            writer.writerow(
                {
                    "step": step,
                    "best_fitness": float(best_fitness),
                    "current_fitness": float(current_fitness),
                }
            )
    optimizer.plot_trajectory(
        savefig_name=str(output_dir / f"trial_{trial:03d}_trajectory.png")
    )


def calibration_dir(args: argparse.Namespace, objective: str) -> Path:
    return (
        args.calibration_root
        / f"{args.num_mut}mut"
        / objective
        / f"{args.nsteps}steps"
    )


def run_calibration(args: argparse.Namespace) -> None:
    models = load_models()
    output_dir = calibration_dir(args, args.objective)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = []

    for trial in range(args.num_trials):
        trial_seed = args.seed + trial
        set_seed(trial_seed)
        target_handler = make_score_handler(
            args,
            models,
            trial,
            objective=args.objective,
            bounds=None,
        )
        optimizer, best = run_optimizer(args, target_handler.seq2fitness)
        mutations = list(best[0])
        sequence = mutations_to_sequence(mutations)
        scores = raw_scores(args, models, sequence, trial)
        record = {
            "trial": trial,
            "seed": trial_seed,
            "mutations": mutations,
            "sequence": sequence,
            "optimized_score": float(best[1]),
            "raw_scores": scores,
        }
        trials.append(record)
        write_json(output_dir / f"trial_{trial:03d}_best.json", record)
        save_trajectory(output_dir, trial, optimizer)

    best_trial = max(trials, key=lambda row: row["optimized_score"])
    summary = {
        "mode": "calibrate",
        "objective": args.objective,
        "num_mut": args.num_mut,
        "nsteps": args.nsteps,
        "num_trials": args.num_trials,
        "num_solublempnn_samples": args.num_solublempnn_samples,
        "best_trial": best_trial,
        "trials": trials,
    }
    write_json(output_dir / "summary.json", summary)
    print(f"Calibration summary: {output_dir / 'summary.json'}")


def run_build_bounds(args: argparse.Namespace) -> None:
    representatives = {}
    source_summaries = {}
    for objective in OBJECTIVES:
        path = calibration_dir(args, objective) / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Run --mode calibrate --objective {objective} "
                "with the same --num-mut and --nsteps first."
            )
        with path.open() as handle:
            summary = json.load(handle)
        representatives[objective] = summary["best_trial"]["raw_scores"]
        source_summaries[objective] = str(path)

    bounds = {}
    for score_name in OBJECTIVES:
        maximum = float(representatives[score_name][score_name])
        cross_scores = [
            float(representatives[objective][score_name])
            for objective in OBJECTIVES
            if objective != score_name
        ]
        minimum = min(cross_scores)
        if maximum <= minimum:
            raise ValueError(
                f"Invalid {score_name} bounds: max={maximum} <= min={minimum}. "
                "Inspect the calibration summaries or run more trials."
            )
        bounds[score_name] = {"min": minimum, "max": maximum}

    payload = {
        "num_mut": args.num_mut,
        "nsteps": args.nsteps,
        "method": (
            "For each model, max is its score on the representative obtained "
            "by optimizing that model; min is its lowest score among the two "
            "representatives obtained by optimizing the other models."
        ),
        "bounds": bounds,
        "representative_raw_scores": representatives,
        "source_summaries": source_summaries,
    }
    write_json(args.bounds_file, payload)
    print(f"Normalization bounds: {args.bounds_file}")


def load_bounds(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    if not args.bounds_file.is_file():
        raise FileNotFoundError(
            f"Normalization bounds not found: {args.bounds_file}\n"
            "Run all three calibration objectives followed by --mode "
            "build-bounds for this mutation count."
        )
    with args.bounds_file.open() as handle:
        payload = json.load(handle)
    if int(payload["num_mut"]) != args.num_mut:
        raise ValueError(
            f"Bounds file is for num_mut={payload['num_mut']}, not {args.num_mut}."
        )
    bounds = payload["bounds"]
    for objective in OBJECTIVES:
        if objective not in bounds or not {"min", "max"}.issubset(bounds[objective]):
            raise ValueError(f"Bounds file is missing {objective} min/max values.")
        if float(bounds[objective]["max"]) <= float(bounds[objective]["min"]):
            raise ValueError(f"Invalid min/max ordering for {objective}.")
    return bounds


def run_multi_objective(args: argparse.Namespace) -> None:
    bounds = load_bounds(args)
    models = load_models()
    output_dir = (
        args.output_root
        / f"{args.num_mut}mut_all_models_{args.nsteps}steps"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "parameters.json",
        {
            "mode": "optimize",
            "num_mut": args.num_mut,
            "assay_data_weight": args.assay_data_weight,
            "num_solublempnn_samples": args.num_solublempnn_samples,
            "num_trials": args.num_trials,
            "nsteps": args.nsteps,
            "start_temp": args.start_temp,
            "final_temp": args.final_temp,
            "seed": args.seed,
            "bounds_file": str(args.bounds_file),
            "bounds": bounds,
        },
    )

    trials = []
    for trial in range(args.num_trials):
        trial_seed = args.seed + trial
        set_seed(trial_seed)
        handler = make_score_handler(args, models, trial, bounds=bounds)
        optimizer, best = run_optimizer(args, handler.seq2fitness)
        mutations = list(best[0])
        sequence = mutations_to_sequence(mutations)
        candidate_raw_scores = raw_scores(args, models, sequence, trial)
        normalized_scores = {
            objective: (
                candidate_raw_scores[objective] - bounds[objective]["min"]
            )
            / (bounds[objective]["max"] - bounds[objective]["min"])
            for objective in OBJECTIVES
        }
        record = {
            "trial": trial,
            "seed": trial_seed,
            "mutations": mutations,
            "sequence": sequence,
            "fitness": float(best[1]),
            "normalized_scores": normalized_scores,
            "raw_scores": candidate_raw_scores,
        }
        trials.append(record)
        write_json(output_dir / f"trial_{trial:03d}_best.json", record)
        with (output_dir / f"trial_{trial:03d}_best.pickle").open("wb") as handle:
            pickle.dump(best, handle)
        with (output_dir / f"trial_{trial:03d}_close_sequences.pickle").open(
            "wb"
        ) as handle:
            pickle.dump(optimizer.close_sequences, handle)
        save_trajectory(output_dir, trial, optimizer)

    best_trial = max(trials, key=lambda row: row["fitness"])
    write_json(
        output_dir / "summary.json",
        {
            "mode": "optimize",
            "num_mut": args.num_mut,
            "best_trial": best_trial,
            "trials": trials,
        },
    )
    print(f"Design summary: {output_dir / 'summary.json'}")


def main() -> None:
    os.chdir(REPO_ROOT)
    args = resolve_cli(parse_args())
    print(
        f"mode={args.mode} num_mut={args.num_mut} nsteps={args.nsteps} "
        f"num_trials={args.num_trials} assay_data_weight={args.assay_data_weight}"
    )
    if args.mode == "calibrate":
        run_calibration(args)
    elif args.mode == "build-bounds":
        run_build_bounds(args)
    else:
        run_multi_objective(args)


if __name__ == "__main__":
    main()
