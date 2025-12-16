import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometryProjector(nn.Module):
    def __init__(self, input_dims, project_dim=128):
        super().__init__()
        self.projectors = nn.ModuleDict()
        for key, input_dim in input_dims.items():
            self.projectors[key] = nn.Linear(input_dim, project_dim, bias=False)
    def forward(self, embeddings_dict):
        projected_dict = {}
        for key, layer in self.projectors.items():
            if key in embeddings_dict:
                feat = embeddings_dict[key]
                feat = layer(feat)
                feat = F.normalize(feat, dim=-1)
                projected_dict[key] = feat
        return projected_dict

def compute_gram_volume_matrix(anchor, complements):
    batch_size = anchor.shape[0]
    all_modalities = [anchor] + complements
    num_modalities = len(all_modalities)
    matrix_blocks = [[None for _ in range(num_modalities)] for _ in range(num_modalities)]
    for r, mod_r in enumerate(all_modalities):
        for c, mod_c in enumerate(all_modalities):
            if r == 0:  # Row is Anchor
                matrix_blocks[r][c] = torch.matmul(mod_r, mod_c.T)
            elif c == 0:  # Col is Anchor
                matrix_blocks[r][c] = torch.matmul(anchor, mod_r.T).T
            else:  # Within-sample complements interaction
                dot_prod = torch.sum(mod_r * mod_c, dim=1)
                matrix_blocks[r][c] = dot_prod.unsqueeze(0).expand(batch_size, -1)
    gram_rows = [torch.stack(matrix_blocks[r], dim=-1) for r in range(num_modalities)]
    G = torch.stack(gram_rows, dim=-2)
    gram_det = torch.det(G.float())
    volume = torch.sqrt(torch.abs(gram_det) + 1e-6)
    return volume

def compute_group_loss(group_dict, anchor_key, temperature=0.1):
    if anchor_key not in group_dict:
        return torch.tensor(0.0).to(list(group_dict.values())[0].device)
    feat_anchor = group_dict[anchor_key]
    feat_complements = [v for k, v in group_dict.items() if k != anchor_key]
    if not feat_complements:
        return torch.tensor(0.0).to(feat_anchor.device)
    volume_matrix = compute_gram_volume_matrix(feat_anchor, feat_complements)
    logits = -volume_matrix / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)

def compute_alignment_loss(projected_dict):
    total_loss = 0.0
    drug_feats = {k: v for k, v in projected_dict.items() if 'drug' in k}
    if drug_feats:
        total_loss += compute_group_loss(drug_feats, anchor_key='drug_1d')
    prot_feats = {k: v for k, v in projected_dict.items() if 'prot' in k}
    if prot_feats:
        total_loss += compute_group_loss(prot_feats, anchor_key='prot_1d')
    return total_loss