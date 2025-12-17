import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GINEConv, global_add_pool


class DrugGIN(nn.Module):
    def __init__(self, num_features_xd=78, hidden_dim=32, output_dim=128, dropout=0.2):
        super(DrugGIN, self).__init__()
        nn1 = nn.Sequential(
            nn.Linear(num_features_xd, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = GINConv(nn1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        nn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv2 = GINConv(nn2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 1024)
        self.fc2 = nn.Linear(1024, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, data):
        # data.x: [num_nodes, 78]
        # data.edge_index: [2, num_edges]
        # data.batch: [num_nodes]
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Layer 1
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)

        # Layer 2
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = global_add_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)  # [batch_size, output_dim]

        return out


class ProteinGINE(nn.Module):
    def __init__(self, node_input_dim=33, edge_input_dim=3, hidden_dim=64, output_dim=128, num_layers=3):
        super(ProteinGINE, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.edge_encoders = nn.ModuleList()
        for i in range(num_layers):
            in_dim = node_input_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.edge_encoders.append(nn.Linear(edge_input_dim, in_dim))
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, output_dim)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = x.float()
        edge_attr = edge_attr.float()
        for i, conv in enumerate(self.convs):
            edge_emb = self.edge_encoders[i](edge_attr)
            x = conv(x, edge_index, edge_attr=edge_emb)
            x = self.bns[i](x)
            x = F.relu(x)
        x = global_add_pool(x, batch)
        out = self.fc(x)
        
        return out
