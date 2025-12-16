import os
import json
import numpy as np
import pandas as pd
from collections import Counter

DEFAULT_VOCAB = {
    "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6, "/": 7, ".": 8,
    "1": 9, "0": 10, "3": 11, "2": 12, "5": 13, "4": 14, "7": 15, "6": 16,
    "9": 17, "8": 18, "=": 19, "A": 20, "@": 21, "C": 22, "B": 23, "E": 24,
    "D": 25, "G": 26, "F": 27, "I": 28, "H": 29, "K": 30, "M": 31, "L": 32,
    "O": 33, "N": 34, "P": 35, "S": 36, "R": 37, "U": 38, "T": 39, "W": 40,
    "V": 41, "Y": 42, "[": 43, "Z": 44, "]": 45, "\\": 46, "a": 47, "c": 48,
    "b": 49, "e": 50, "d": 51, "g": 52, "f": 53, "i": 54, "h": 55, "m": 56,
    "l": 57, "o": 58, "n": 59, "s": 60, "r": 61, "u": 62, "t": 63, "y": 64
}

def scan_dataset(dataset_name, root_path="data"):
    print(f"Scanning dataset: {dataset_name}...")
    all_smiles = []
    ligands_path = os.path.join(root_path, dataset_name, "ligands.txt")
    if os.path.exists(ligands_path):
        import json
        with open(ligands_path, 'r') as f:
            ligands = json.load(f)
            all_smiles = list(ligands.values())
    else:
        data_path = os.path.join(root_path, dataset_name, "data.txt")
        if not os.path.exists(data_path):
            data_path = os.path.join(root_path, dataset_name, "data.csv")

        if os.path.exists(data_path):
            df = pd.read_csv(data_path, sep='\t', header=None)
            all_smiles = df[0].astype(str).tolist()

    if not all_smiles:
        print(f"Warning: No SMILES found for {dataset_name}")
        return set(), 0
    unique_chars = set()
    max_len = 0
    lengths = []
    for smi in all_smiles:
        if len(smi) > max_len:
            max_len = len(smi)
        lengths.append(len(smi))
        unique_chars.update(set(smi))
    print(f"  - Max length: {max_len}")
    print(f"  - Avg length: {np.mean(lengths):.2f}")
    print(f"  - Unique chars: {len(unique_chars)}")

    return unique_chars, int(np.percentile(lengths, 95))

def build_vocab(datasets, root_path="data"):
    global_chars = set(DEFAULT_VOCAB.keys())
    suggested_max_len = 0
    for ds in datasets:
        chars, length_95 = scan_dataset(ds, root_path)
        global_chars.update(chars)
        suggested_max_len = max(suggested_max_len, length_95)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, char in enumerate(sorted(list(global_chars))):
        vocab[char] = i + 2
    save_path = os.path.join(root_path, "vocab_1d.json")
    with open(save_path, 'w') as f:
        json.dump({"vocab": vocab, "max_len": suggested_max_len}, f, indent=4)
    print(f"\nSaved vocab to {save_path}")
    print(f"Vocab Size: {len(vocab)}")
    print(f"Suggested Max Len: {suggested_max_len}")

if __name__ == "__main__":
    datasets_to_scan = ['DAVIS', 'KIBA', 'BINDINGDB', 'BIOSNAP']
    valid_datasets = [d for d in datasets_to_scan if os.path.exists(os.path.join("data", d))]
    build_vocab(valid_datasets)