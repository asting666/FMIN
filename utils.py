import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from scipy import stats
from math import sqrt
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from drug_preprocess_2d import smile_to_graph

def get_metrics_dti(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = (y_score >= threshold).astype(int)
    try:
        auroc = metrics.roc_auc_score(y_true, y_score)
    except ValueError:
        auroc = 0.0
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
    aupr = metrics.auc(recall, precision)
    f1 = metrics.f1_score(y_true, y_pred)
    return {"AUROC": auroc, "AUPR": aupr, "F1": f1}


def get_metrics_dta(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mse_val = metrics.mean_squared_error(y_true, y_pred)
    rmse_val = sqrt(mse_val)
    ci_val = ci(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    return {"MSE": mse_val, "RMSE": rmse_val, "CI": ci_val, "R2": r2}


def ci(y: np.ndarray, f: np.ndarray) -> float:
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    return S / z if z != 0 else 0.0


class FMINDataset(Dataset):
    def __init__(self, csv_path, transform=None, pre_transform=None):
        super(FMINDataset, self).__init__(None, transform, pre_transform)
        self.df = pd.read_csv(csv_path)
        print("Processing graphs...")
        unique_smiles = self.df['smiles'].unique()
        self.smile_graph = {}
        for smile in unique_smiles:
            g = smile_to_graph(smile)
            if g is not None:
                self.smile_graph[smile] = g
            else:
                print(f"Warning: Invalid SMILES {smile}")
                pass
    def len(self):
        return len(self.df)

    def get(self, idx):
        row = self.df.iloc[idx]
        smile = row['smiles']
        sequence = row['sequence']
        label = row['label']

        if smile in self.smile_graph:
            data = self.smile_graph[smile].clone()
        else:
            raise ValueError(f"Graph not found for {smile}")

        data.y = torch.tensor([label], dtype=torch.float32)

        return data



def get_dataloader(csv_path, batch_size, shuffle=True, num_workers=4):
    """
    PyG DataLoader
    """
    dataset = FMINDataset(csv_path)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader

class ProteinGraphDataset(Dataset):
    def __init__(self, root, data_df, transform=None, pre_transform=None):
        self.data_df = data_df
        self.root = root
        super(ProteinGraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def len(self):
        return len(self.data_df)

    def get(self, idx):
        row = self.data_df.iloc[idx]
        pdb_id = row['PROTEIN_ID']
        label = row['label']
        pt_path = os.path.join(self.root, 'processed', 'pocket_graph', f'{pdb_id}.pt')

        if os.path.exists(pt_path):
            data = torch.load(pt_path)
        else:
            raise FileNotFoundError(f"Graph file not found: {pt_path}")
        data.y = torch.tensor([label], dtype=torch.float)

        return data