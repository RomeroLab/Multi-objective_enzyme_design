# Multi-objective enzyme design

This repository contains the data, trained models, and scripts used for the machine learning-guided multi-objective engineering of the ketoreductase Gre2. The design framework integrates three complementary model objectives:

1. predicted initial reaction rate from a supervised ESM-2 activity model;
2. evolutionary likelihood from a variational autoencoder (VAE) trained on natural Gre2 homologs; and
3. structural compatibility and solubility from SolubleMPNN scoring against the Gre2 structure.

Together, these models combine functional measurements with complementary evolutionary and structural information. Multi-objective simulated annealing searches for variants close to the normalized utopia point while retaining the experimentally validated F85L parent mutation.

All commands below should be run from the repository root unless otherwise specified.

---

## Fine-tuning ESM-2 to predict Gre2 initial reaction rate

The Gre2 activity predictor uses residue-level ESM-2 representations, partial fine-tuning of the ESM-2 650M model, and a multilayer perceptron regression head. It is trained on the experimentally characterized Gre2 single-mutant dataset to predict initial reaction rate from sequence.

### 1. Clone the repository

```bash
git clone https://github.com/RomeroLab/Multi-objective_enzyme_design.git
cd Multi-objective_enzyme_design
```

### 2. Create the Conda environment

From the repository root:

```bash
conda env create -f envs/finetune_esm2.yml
conda activate gre2-esm2-finetuning
```

Note: The environment uses a CUDA 12.1 build of PyTorch and requires a compatible NVIDIA driver.

### 3. Run ESM-2 paritial fine-tuning on a GPU

Example using the first visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u finetuning_esm2.py \
  > ./logs/finetuning_esm2.out 2>&1 &
```

Monitor training:

```bash
tail -f ./logs/finetuning_esm2.out
```

This configuration uses the ESM-2 650M backbone and partially fine-tunes selected pretrained parameters together with the MLP regression head. If GPU memory is insufficient, reduce `batch_size` in `finetuning_esm2.py`.

### 4. Locate the fine-tuned model

Training outputs are written under:

```text
logs/finetuning_ESM2/version_<N>/
```

### 5. Optional LoRA benchmark

`finetuning_esm2_w_LoRA.py` implements the LoRA-based parameter-efficient fine-tuning strategy used in the external DMS benchmarks. It is separate from the partially fine-tuned Gre2 activity model used for sequence design and requires the corresponding CreiLOV pickle datasets and `models/esm2_w_lora_w_MLP.py`.

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u finetuning_esm2_w_LoRA.py \
  --data-dir data/finetuned_esm2 \
  --output-dir logs/lora \
  > ./logs/finetuning_esm2_lora.out 2>&1 &
```

Monitor the run:

```bash
tail -f ./logs/finetuning_esm2_lora.out
```

---

## Running multi-objective simulated annealing

The workflow co-optimizes predicted initial reaction rate from fine-tuned ESM-2, VAE evolutionary likelihood, and SolubleMPNN structural compatibility/solubility. Designs are generated from the F85L parent. The N-terminal methionine and F85L are fixed, so `--num-mut 3` and `--num-mut 5` produce variants with four or six mutations relative to Gre2.

The raw model scores have different scales. Before multi-objective design, optimize each model independently to estimate mutation-count-specific normalization endpoints. The normalized multi-objective fitness is proximity to the utopia point `(1, 1, 1)`.

### 1. Create the Conda environment

From the repository root:

```bash
conda env create -f envs/multi_objective_simulated_annealing.yml
conda activate gre2-multi-objective-sa
```

### 2. Inspect the command-line interface

```bash
python mo_simulating_annealing.py --help
```

The script provides three modes:

```text
calibrate     Optimize one raw objective and cross-score its best sequence
build-bounds  Combine the three calibration summaries into a bounds file
optimize      Run normalized multi-objective simulated annealing
```

### 3. Obtain normalization bounds

Calibrate ESM-2, the VAE, and SolubleMPNN separately for each mutation count. For three additional mutations:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u mo_simulating_annealing.py \
  --mode calibrate --objective esm2 --num-mut 3 > ./logs/esm2_only.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u mo_simulating_annealing.py \
  --mode calibrate --objective vae --num-mut 3 > ./logs/vae_only.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u mo_simulating_annealing.py \
  --mode calibrate --objective solublempnn --num-mut 3 > ./logs/solublempnn_only.out &

python mo_simulating_annealing.py --mode build-bounds --num-mut 3
```

Each calibration trial optimizes one raw objective and then scores its best sequence with all three models. The representative for an objective is the highest-scoring candidate across its independent trials. For each model, the bounds use:

```text
Max = its score for the representative obtained by optimizing that model
Min = its lowest score among representatives obtained by optimizing the other models
```

Outputs are written to:

```text
normalization_calibration/<NUM_MUT>mut/<OBJECTIVE>/<NSTEPS>steps/summary.json
normalization_bounds/<NUM_MUT>mut.json
```

Do not reuse three-mutation bounds for five-mutation designs or vice versa.

### 4. Run a functional smoke test

`--smoke-test` uses one trial, 10 steps, and one SolubleMPNN sample. These bounds validate the installation only and are not scientifically meaningful.

```bash
for objective in esm2 vae solublempnn
do
  CUDA_VISIBLE_DEVICES=0 python -u mo_simulating_annealing.py \
    --mode calibrate \
    --objective "$objective" \
    --num-mut 5 \
    --smoke-test
done

python mo_simulating_annealing.py \
  --mode build-bounds \
  --num-mut 5 \
  --smoke-test

CUDA_VISIBLE_DEVICES=0 python -u mo_simulating_annealing.py \
  --mode optimize \
  --num-mut 5 \
  --smoke-test
```

### 5. Run multi-objective optimization

After creating production bounds for the selected mutation count:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u mo_simulating_annealing.py \
  --mode optimize \
  --num-mut 5 \
  > multi_objective_simulated_annealing.out 2>&1 &
```

The reaction-rate weight defaults to 1. To intentionally change it, pass an explicit value such as `--assay-data-weight 4`.

Monitor the run:

```bash
tail -f multi_objective_simulated_annealing.out
```

Stop monitoring without terminating the run by pressing `Ctrl-C`. The script creates `seqs_to_score/` and `outputs/` automatically.

### 6. Locate the design results

The default five-additional-mutation outputs are written under:

```text
Gre2_redesign_round_1_with_F85L/5mut_all_models_35000steps/
```

Each directory contains:

```text
parameters.json
summary.json
trial_<N>_best.json
trial_<N>_best.pickle
trial_<N>_close_sequences.pickle
trial_<N>_trajectory.csv
trial_<N>_trajectory.png
```

Temporary FASTA files are written to `seqs_to_score/`, and SolubleMPNN scoring outputs are written to `outputs/`.

Because annealing is stochastic, use multiple trials for production calibration and design. Runtime scales approximately with:

```text
number of trials x number of annealing steps x model-scoring cost
```

SolubleMPNN calibration is the most expensive because it invokes SolubleMPNN at every annealing step. ESM-2- and VAE-only calibration invoke SolubleMPNN only when cross-scoring each trial's best sequence.

---

## Software requirements

The fine-tuning and simulated-annealing workflows use the separate Conda environments supplied under `envs/`. Both workflows were designed for an NVIDIA GPU. Runtime depends on GPU hardware, model-loading time, batch size, the number of annealing steps, and the number of independent trials.
