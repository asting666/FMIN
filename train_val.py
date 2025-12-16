import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_dataloader, get_metrics_dti, get_metrics_dta
from alignment import compute_alignment_loss
from model.fmin_full import FMIN_Full_Model


class FMIN_Trainer:
    def __init__(self, args, train_loader, val_loader):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {self.device}")
        self.model = FMIN_Full_Model(args).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        if args.task == 'dti':
            self.criterion_task = nn.BCEWithLogitsLoss()
        else:
            self.criterion_task = nn.MSELoss()

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_align_loss = 0

        for batch_data in self.train_loader:
            targets = batch_data['label'].to(self.device)
            self.optimizer.zero_grad()
            pred, embeddings_dict = self.model(batch_data)
            loss_task = self.criterion_task(pred, targets)
            loss_align = compute_alignment_loss(embeddings_dict)
            loss_total = loss_task + self.args.lambda_align * loss_align
            loss_total.backward()
            self.optimizer.step()
            total_loss += loss_total.item()
            total_align_loss += loss_align.item() if isinstance(loss_align, torch.Tensor) else 0
        return total_loss / len(self.train_loader), total_align_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        preds = []
        truths = []
        with torch.no_grad():
            for batch_data in self.val_loader:
                targets = batch_data['label'].to(self.device)
                pred, _ = self.model(batch_data)
                if self.args.task == 'dti':
                    pred = torch.sigmoid(pred)
                preds.extend(pred.cpu().numpy())
                truths.extend(targets.cpu().numpy())
        preds = np.array(preds).flatten()
        truths = np.array(truths).flatten()
        if self.args.task == 'dti':
            return get_metrics_dti(truths, preds)
        else:
            return get_metrics_dta(truths, preds)


def run_training(args):
    print(f"Start Training: Task={args.task}, Dataset={args.dataset}, Setting={args.setting}")
    data_root = os.path.join("data", args.dataset, "data_folds", args.setting)
    n_folds = 5
    metrics_log = []
    for fold in range(n_folds):
        print(f"\n====== Fold {fold} ======")
        train_csv = os.path.join(data_root, f"train_fold_{fold}.csv")
        test_csv = os.path.join(data_root, f"test_fold_{fold}.csv")
        train_loader = get_dataloader(train_csv, batch_size=args.batch_size, shuffle=True)
        val_loader = get_dataloader(test_csv, batch_size=args.batch_size, shuffle=False)
        trainer = FMIN_Trainer(args, train_loader, val_loader)
        for epoch in range(args.epochs):
            loss_train, loss_align = trainer.train_epoch()
            val_results = trainer.validate()
            print(f"Ep {epoch}: Loss={loss_train:.4f} (Align={loss_align:.4f}) | Val={val_results}")
        metrics_log.append(val_results)
    df = pd.DataFrame(metrics_log)
    print("\n====== Final Results (Mean ± Std) ======")
    print(df.mean())
    print(df.std())
    os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/{args.dataset}_{args.setting}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['dti', 'dta'])
    parser.add_argument('--dataset', type=str, required=True, choices=['DAVIS', 'KIBA', 'BINDINGDB', 'BIOSNAP'])
    parser.add_argument('--setting', type=str, default='warm_start')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_align', type=float, default=0.1, help='Weight for alignment loss')
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimension of fused representation')
    args = parser.parse_args()
    if not os.path.exists(os.path.join("data", args.dataset)):
        raise FileNotFoundError(f"Dataset data/{args.dataset} not found.")
    run_training(args)