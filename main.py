import torch
import torch.nn as nn
from embedding_1d import Drug1DEncoder, ProteinCNN as Prot1DEncoder
from models_embedding import DrugGIN as Drug2DEncoder, ProteinGINE as Prot2DEncoder
from model.embedding_3d import Drug3DEncoder, Prot3DEncoder
from model.FMINmodel import IterativeFusion

class FMIN_Full_Model(nn.Module):
    def __init__(self, args):
        super(FMIN_Full_Model, self).__init__()
        self.drug_enc_1d = Drug1DEncoder(args)
        self.prot_enc_1d = Prot1DEncoder(args)
        self.drug_enc_2d = Drug2DEncoder(
            num_features_xd=78,
            output_dim=args.latent_dim,
            dropout=args.dropout
        )
        self.prot_enc_2d = Prot2DEncoder(
            node_input_dim=33,
            edge_input_dim=3,
            output_dim=args.latent_dim,
            num_layers=3
        )

        self.drug_enc_3d = Drug3DEncoder(args)
        self.prot_enc_3d = Prot3DEncoder(args)

        self.fusion_module = IterativeFusion(args)
        fusion_dim = args.latent_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 1)
        )

    def forward(self, batch_data):

        drug_1d_input = batch_data['drug_1d']
        prot_1d_input = batch_data['prot_1d']
        drug_2d_graph = batch_data['drug_graph']
        prot_2d_graph = batch_data['prot_graph']
        drug_3d_input = batch_data['drug_3d']
        prot_3d_input = batch_data['prot_3d']
        drug_emb_1d = self.drug_enc_1d(drug_1d_input)
        prot_emb_1d = self.prot_enc_1d(prot_1d_input)
        drug_emb_2d = self.drug_enc_2d(drug_2d_graph)
        prot_emb_2d = self.prot_enc_2d(prot_2d_graph)
        drug_emb_3d = self.drug_enc_3d(drug_3d_input)
        prot_emb_3d = self.prot_enc_3d(prot_3d_input)
        embeddings_dict = {
            'drug_1D': drug_emb_1d, 'prot_1D': prot_emb_1d,
            'drug_2D': drug_emb_2d, 'prot_2D': prot_emb_2d,
            'drug_3D': drug_emb_3d, 'prot_3D': prot_emb_3d
        }

        drug_fused, prot_fused = self.fusion_module(
            drug_emb_1d, drug_emb_2d, drug_emb_3d,
            prot_emb_1d, prot_emb_2d, prot_emb_3d
        )
        combined_feat = torch.cat([drug_fused, prot_fused], dim=1)
        pred = self.mlp(combined_feat)


        return pred, embeddings_dict
