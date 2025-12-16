import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import numpy as np


class SmilesTokenizer:
    def __init__(self, vocab_path="data/vocab_1d.json", max_len=100):
        if not os.path.exists(vocab_path):
            print(f"[Warning] Vocab file not found at {vocab_path}. Using default dummy vocab.")
            self.vocab = {"<PAD>": 0, "<UNK>": 1, "C": 2, "N": 3, "O": 4}  # 仅作防止报错的示例
        else:
            with open(vocab_path, 'r') as f:
                data = json.load(f)
                self.vocab = data['vocab']
        self.max_len = max_len if max_len else 100

        self.pad_idx = self.vocab.get("<PAD>", 0)
        self.unk_idx = self.vocab.get("<UNK>", 1)

    def encode(self, smiles_list):
        batch_indices = []
        for smi in smiles_list:
            indices = []
            for char in smi:
                indices.append(self.vocab.get(char, self.unk_idx))

            # Padding / Truncation
            if len(indices) < self.max_len:
                indices += [self.pad_idx] * (self.max_len - len(indices))
            else:
                indices = indices[:self.max_len]

            batch_indices.append(indices)

        return torch.tensor(batch_indices, dtype=torch.long)

    def get_vocab_size(self):
        return len(self.vocab)

class Drug1DEncoder(nn.Module):
    def __init__(self, args):
        super(Drug1DEncoder, self).__init__()
        self.tokenizer = SmilesTokenizer(max_len=100)
        vocab_size = self.tokenizer.get_vocab_size()
        embed_dim = 128
        kernel_sizes = [3, 5, 7]
        num_filters = 32
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)  # Global Max Pooling
            ) for k in kernel_sizes
        ])

        self.fc = nn.Sequential(
            nn.Linear(num_filters * len(kernel_sizes), args.latent_dim),
            nn.ReLU()
        )

    def forward(self, smiles_list):
        device = next(self.parameters()).device
        if isinstance(smiles_list, list):
            x = self.tokenizer.encode(smiles_list).to(device)
        else:
            x = smiles_list.to(device)
        emb = self.embedding(x).permute(0, 2, 1)
        conv_outs = [conv(emb).squeeze(-1) for conv in self.convs]
        feat = torch.cat(conv_outs, dim=1)
        out = self.fc(feat)
        return out

class ProteinCNN(nn.Module):
    def __init__(self, args):
        super(ProteinCNN, self).__init__()
        vocab_size = 26
        embed_dim = 128
        n_filters = 32
        kernel_size = 8
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=kernel_size)
        self.conv3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 3, kernel_size=kernel_size)
        self.fc1 = nn.Linear(n_filters * 3, 1024)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(1024, args.latent_dim)
    def forward(self, x):
        embedded = self.embedding(x)
        x = embedded.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.adaptive_max_pool1d(x, 1)
        x = x.squeeze(2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        return out