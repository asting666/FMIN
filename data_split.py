import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from collections import OrderedDict

DATA_ROOT = "data"
def save_fold(data, fold_idx, split_type, setting, dataset_name):
    save_path = os.path.join(DATA_ROOT, dataset_name, "data_folds", setting)
    os.makedirs(save_path, exist_ok=True)
    filename = f"{split_type}_fold_{fold_idx}.csv"
    data.to_csv(os.path.join(save_path, filename), index=False)

def process_dta_data(dataset_name):
    dataset_path = os.path.join(DATA_ROOT, dataset_name)
    print(f"Loading DTA data from {dataset_path}...")
    with open(os.path.join(dataset_path, "ligands.txt"), 'r') as f:
        ligands = json.load(f, object_pairs_hook=OrderedDict)
    with open(os.path.join(dataset_path, "proteins.txt"), 'r') as f:
        proteins = json.load(f, object_pairs_hook=OrderedDict)
    y_path = os.path.join(dataset_path, "Y")
    try:
        Y = pickle.load(open(y_path, "rb"), encoding='latin1')
    except:
        if os.path.exists(y_path + ".pkl"):
            Y = pickle.load(open(y_path + ".pkl", "rb"), encoding='latin1')
        else:
            Y = np.loadtxt(os.path.join(dataset_path, "Y.txt"))
    drug_ids = list(ligands.keys())
    prot_ids = list(proteins.keys())
    Y = np.asarray(Y)
    rows, cols = np.where(np.isnan(Y) == False)
    data_list = []
    for r, c in zip(rows, cols):
        d_id = drug_ids[r]
        p_id = prot_ids[c]
        val = Y[r, c]
        if 'DAVIS' in dataset_name:
            val = -np.log10(val / 1e9)
        data_list.append([d_id, p_id, ligands[d_id], proteins[p_id], val])
    return pd.DataFrame(data_list, columns=['drug_id', 'prot_id', 'smiles', 'sequence', 'label'])

def process_dti_data(dataset_name):
    dataset_path = os.path.join(DATA_ROOT, dataset_name)
    print(f"Loading DTI data from {dataset_path}...")
    file_path = os.path.join(dataset_path, "data.txt")
    if not os.path.exists(file_path):
        file_path = os.path.join(dataset_path, "data.csv")
    df = pd.read_csv(file_path, sep='\t', header=None, names=['smiles', 'sequence', 'label'])
    # Cold Start
    df['drug_id'] = pd.factorize(df['smiles'])[0]
    df['drug_id'] = df['drug_id'].apply(lambda x: f"drug_{x}")
    df['prot_id'] = pd.factorize(df['sequence'])[0]
    df['prot_id'] = df['prot_id'].apply(lambda x: f"prot_{x}")
    return df[['drug_id', 'prot_id', 'smiles', 'sequence', 'label']]

def split_dataset(task, dataset_name, n_splits=5):
    if task == 'dta':
        data = process_dta_data(dataset_name)
    else:
        data = process_dti_data(dataset_name)
    X = data.index.to_numpy()
    unique_drugs = data['drug_id'].unique()
    unique_prots = data['prot_id'].unique()
    # 1. Warm Start
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        save_fold(data.iloc[train_idx], fold, 'train', 'warm_start', dataset_name)
        save_fold(data.iloc[test_idx], fold, 'test', 'warm_start', dataset_name)
    # 2. Cold Start Drug
    kf_drug = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for fold, (train_d_idx, test_d_idx) in enumerate(kf_drug.split(unique_drugs)):
        test_drugs = unique_drugs[test_d_idx]
        train_mask = ~data['drug_id'].isin(test_drugs)
        test_mask = data['drug_id'].isin(test_drugs)
        save_fold(data[train_mask], fold, 'train', 'drug_coldstart', dataset_name)
        save_fold(data[test_mask], fold, 'test', 'drug_coldstart', dataset_name)
    # 3. Cold Start Protein
    kf_prot = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for fold, (train_p_idx, test_p_idx) in enumerate(kf_prot.split(unique_prots)):
        test_prots = unique_prots[test_p_idx]
        train_mask = ~data['prot_id'].isin(test_prots)
        test_mask = data['prot_id'].isin(test_prots)
        save_fold(data[train_mask], fold, 'train', 'protein_coldstart', dataset_name)
        save_fold(data[test_mask], fold, 'test', 'protein_coldstart', dataset_name)
    print(f"Dataset {dataset_name} split completed.\n")

if __name__ == "__main__":
    for db in ['DAVIS', 'KIBA']:
        if os.path.exists(os.path.join(DATA_ROOT, db)):
            split_dataset('dta', db)

    for db in ['BINDINGDB', 'BIOSNAP']:
        if os.path.exists(os.path.join(DATA_ROOT, db)):
            split_dataset('dti', db)