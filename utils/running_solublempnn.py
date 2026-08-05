import random
import numpy as np
import torch
import os

# Load sequences from a FASTA file
def load_fasta_with_names(file_path):
    """Load sequences and their names from a FASTA file."""
    sequences = []
    names = []
    with open(file_path, "r") as file:
        current_seq = []
        current_name = None
        for line in file:
            if line.startswith(">"):
                if current_seq:
                    sequences.append(''.join(current_seq))
                    current_seq = []
                current_name = line.strip()[1:]
                names.append(current_name)
            else:
                current_seq.append(line.strip())
        if current_seq:
            sequences.append(''.join(current_seq))
    return names, sequences


def write_fasta(sequences, output_path, header_prefix, WT):
    """
    Write sequences to a FASTA file, replacing '-' with the corresponding WT amino acid.

    Parameters:
    - sequences: list of tuples, each containing a name and a sequence.
    - output_path: str, path to the output FASTA file.
    - header_prefix: str, prefix for FASTA headers.
    - WT: str, wild-type amino acid sequence.
    """
    with open(output_path, "w") as file:
        for idx, (name, seq) in enumerate(sequences):
            # Replace '-' with the corresponding WT amino acid
            clean_seq = ''.join(WT[i] if aa == '-' else aa for i, aa in enumerate(seq))
            file.write(f">{header_prefix}_{name}\n")
            file.write(f"{clean_seq}\n")

def write_fasta_random_aa(sequences, output_path):
    """
    Write sequences to a FASTA file, replacing '-' with random amino acids.

    Parameters:
    - sequences: list of tuples, each containing a name and a sequence.
    - output_path: str, path to the output FASTA file.
    - header_prefix: str, prefix for FASTA headers.
    """
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"  # Set of valid amino acids

    with open(output_path, "w") as file:
        for idx, (name, seq) in enumerate(sequences):
            # Replace '-' with a random amino acid
            clean_seq = ''.join(random.choice(amino_acids) if aa == '-' else aa for aa in seq)
            file.write(f">{name}\n")
            file.write(f"{clean_seq}\n")

# Parse fasta file to get sequence names
def parse_fasta(file_path):
    """Parse a fasta file to extract sequence names."""
    names = []
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith(">"):
                names.append(line.strip()[1:])  # Remove ">" and strip whitespace
    return names

def get_chains_from_pdb(pdb_path):
    """
    Extract unique chain identifiers from a PDB file.
    """
    chains = set()
    with open(pdb_path, "r") as file:
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain_id = line[21]  # Chain ID is in column 22 (index 21)
                chains.add(chain_id.strip())
    return sorted(chains)

# Function to load and compute the mean of scores from .npz files
def load_npz_scores(num_sequences, output_dir, base_name="4PVC._fasta_"):
    """Load mean scores from all .npz files for the given number of sequences."""
    mean_scores = []
    for i in range(1, num_sequences + 1):
        npz_file = os.path.join(f"{output_dir}/score_only", f"{base_name}{i}.npz")
        # print(f"Loading: {npz_file}")
        if os.path.exists(npz_file):
            data = np.load(npz_file)
            if "score" in data:
                score_mean = np.mean(data["score"])  # Compute mean of the scores
                mean_scores.append(score_mean)
                # print(f"Mean score for {npz_file}: {score_mean}")
            else:
                print(f"'score' key not found in {npz_file}")
                mean_scores.append(None)
        else:
            print(f"File not found: {npz_file}")
            mean_scores.append(None)  # Append None if file is missing
    return mean_scores