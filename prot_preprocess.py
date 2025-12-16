import os
import math
import warnings
import timeit
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import pairwise_distances
from torch_geometric.data import Data
from itertools import permutations

warnings.filterwarnings("ignore")
SEQ_VOCAB = "ABCDEFGHIKLMNOPQRSTUVWXYZ"

SEQ_DICT = {v: (i + 1) for i, v in enumerate(SEQ_VOCAB)}
MAX_SEQ_LEN = 1000

res_dict = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
            'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']
pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}
res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36, 'K': 2.18,
                 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21, 'T': 2.09, 'V': 2.32,
                 'W': 2.83, 'Y': 2.32}
res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17, 'I': 9.60, 'K': 8.95,
                 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13, 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62,
                 'W': 9.39, 'Y': 9.62}
res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00, 'I': 0.00, 'K': 10.53,
                 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00, 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00,
                 'W': 0.00, 'Y': 0.00}
res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59, 'I': 6.02, 'K': 9.74,
                'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65, 'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96,
                'W': 5.89, 'Y': 5.96}
res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100, 'K': -37,
                             'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7, 'T': 13, 'V': 79,
                             'W': 84, 'Y': 49}
res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99, 'K': -23, 'L': 97,
                             'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5, 'T': 13, 'V': 76, 'W': 97,
                             'Y': 63}
def dic_normalize(dic):
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic
res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)
def seq_to_indices(seq):
    indices = np.zeros(MAX_SEQ_LEN, dtype=np.int64)
    seq = seq[:MAX_SEQ_LEN]
    for i, ch in enumerate(seq):
        if ch in SEQ_DICT:
            indices[i] = SEQ_DICT[ch]
    return indices
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0,
                     1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    return np.array(res_property1 + res_property2)
def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)


def cos_sim(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return vec1.dot(vec2) / (norm1 * norm2)


def cal_angle(point_a, point_b, point_c):
    a_x, b_x, c_x = point_a[0], point_b[0], point_c[0]
    a_y, b_y, c_y = point_a[1], point_b[1], point_c[1]
    a_z, b_z, c_z = point_a[2], point_b[2], point_c[2]

    x1, y1, z1 = (a_x - b_x), (a_y - b_y), (a_z - b_z)
    x2, y2, z2 = (c_x - b_x), (c_y - b_y), (c_z - b_z)

    dot_prod = x1 * x2 + y1 * y2 + z1 * z2
    mag1 = math.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2)
    mag2 = math.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)

    if mag1 == 0 or mag2 == 0: return 0.0
    cos_b = max(min(dot_prod / (mag1 * mag2), 1.0), -1.0)
    return cos_b


def Get_Ca_Coords(path, check_length=5000):
    with open(path, mode="r") as file:
        lines = file.readlines()
    out = []
    flag = 0
    for line in lines:
        if line.startswith('ATOM') and line.split()[2] == 'CA':
            parts = line.split()
            if len(parts) >= 9:
                resn = parts[3]
                try:
                    x = float(parts[6])
                    y = float(parts[7])
                    z = float(parts[8])
                    out.append([resn, x, y, z])
                except ValueError:
                    continue
            flag += 1
            if flag >= check_length: break
    return pd.DataFrame(out, columns=['res_name', 'x', 'y', 'z'])


def protein_to_graph(pdb_path, cutoff=8.0):
    try:
        df = Get_Ca_Coords(pdb_path)
    except Exception as e:
        print(f"[Error] Failed to read {pdb_path}: {e}")
        return None
    if len(df) == 0: return None
    coords = df[['x', 'y', 'z']].values
    residues = df['res_name'].values
    seq_1letter = [res_dict.get(r, 'X') for r in residues]
    node_feats_np = seq_feature(seq_1letter)
    x = torch.tensor(node_feats_np, dtype=torch.float)
    dist_mat = pairwise_distances(coords)
    num_nodes = len(coords)
    edge_index = []
    edge_attr = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dist = dist_mat[i, j]
                if dist < cutoff:
                    edge_index.append([i, j])
                    sim_ij = cos_sim(node_feats_np[i], node_feats_np[j])
                    dis_feat = 1.0 if dist <= 1.0 else 1.0 / dist
                    angle_feat = cal_angle(coords[i], [0, 0, 0], coords[j])
                    edge_attr.append([sim_ij, dis_feat, angle_feat])
    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
if __name__ == '__main__':
    dataset = 'Davis'
    base_path = f'data/{dataset}'
    processed_dir = os.path.join(base_path, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    print(f"=== Processing Dataset: {dataset} ===")
    pdb_dir = os.path.join(base_path, 'PDB_AF2')
    output_graph_dir = os.path.join(processed_dir, 'pocket_graph')
    os.makedirs(output_graph_dir, exist_ok=True)
    if os.path.exists(pdb_dir):
        pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
        print(f"\n[Task A] Found {len(pdb_files)} PDB files. Constructing graphs...")
        start_time = timeit.default_timer()
        for idx, pdb_file in enumerate(pdb_files):
            pdb_id = pdb_file.split('.')[0]
            save_path = os.path.join(output_graph_dir, f"{pdb_id}.pt")
            if not os.path.exists(save_path):
                full_pdb_path = os.path.join(pdb_dir, pdb_file)
                graph_data = protein_to_graph(full_pdb_path)
                if graph_data is not None:
                    torch.save(graph_data, save_path)
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(pdb_files)} graphs...")

        print(f"[Task A] Done. Time elapsed: {timeit.default_timer() - start_time:.2f}s")
    else:
        print(f"\n[Task A] Skipped: PDB directory not found at {pdb_dir}")
    csv_path = os.path.join(base_path, 'train.csv')
    if os.path.exists(csv_path):
        print(f"\n[Task B] Processing 1D sequences from {csv_path}...")
        try:
            df = pd.read_csv(csv_path)
            if 'target_sequence' in df.columns:
                all_sequences = df['target_sequence'].unique()
                print(f"  Found {len(all_sequences)} unique sequences.")
                processed_prot_dict = {}
                for seq in all_sequences:
                    processed_prot_dict[seq] = torch.tensor(seq_to_indices(seq), dtype=torch.long)
                save_file = os.path.join(processed_dir, 'protein_indices_dict.pt')
                torch.save(processed_prot_dict, save_file)
                print(f"[Task B] Saved sequence dictionary to {save_file}")
            else:
                print("  [Warning] Column 'target_sequence' not found in CSV.")
        except Exception as e:
            print(f"  [Error] Failed to process CSV: {e}")
    else:
        print(f"\n[Task B] Skipped: CSV file not found at {csv_path}")
    print("\nAll preprocessing finished!")