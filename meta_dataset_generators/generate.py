import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from utils import get_datasets, get_hyperparameters
from mlp import build_mlp, extract_mlp_probe_features
from cnn import build_cnn, extract_cnn_probe_features
from rnn import build_rnn, extract_rnn_probe_features
from transformer import build_transformer, extract_transformer_probe_features


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Generate meta-dataset")
parser.add_argument("--model", type=str, choices=["mlp", "cnn", "rnn", "transformer"],)
parser.add_argument("--save_dir", type=str, default="./meta_datasets")

args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

DATASETS, TRANSFORMS = get_datasets(args.model)
LR_VALUES, BATCH_SIZES, EPS_VALUES, N_TRIALS, MAX_STEPS = get_hyperparameters(args.model)


model_builder = {
    "mlp": build_mlp,
    "cnn": build_cnn,
    "rnn": build_rnn,
    "transformer": build_transformer  # Placeholder for transformer model
}[args.model]


extract_probe_features = {
    "mlp": extract_mlp_probe_features,
    "cnn": extract_cnn_probe_features,
    "rnn": extract_rnn_probe_features,
    "transformer": extract_transformer_probe_features  # Placeholder for transformer model
}[args.model]


def measure_convergence(model: nn.Module, X: torch.Tensor, y: torch.Tensor, eps: float, lr: float, criterion):
    model.to(DEVICE).train()
    X, y = X.to(DEVICE), y.to(DEVICE)

    if args.model == "mlp":
        X = X.view(X.size(0), -1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    init_loss = None

    for epoch in range(1, MAX_STEPS+1):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        if epoch == 1:
            init_loss = loss.item()
        if loss.item() <= eps * init_loss:
            return epoch, True, init_loss, loss.item()
        loss.backward()
        optimizer.step()

    return MAX_STEPS, False, init_loss, loss.item()


def construct_meta_dataset():
    records = []
    for ds_name, ds_cls in DATASETS.items():
        print(f"[{args.model}] Building for dataset: {ds_name}")
        ds = ds_cls(root='./data', train=True, download=True,
                    transform=TRANSFORMS[ds_name])
        num_classes = len(ds.classes)
        input_shape = ds[0][0].shape
        total_N     = len(ds)
        criterion   = nn.CrossEntropyLoss()

        for lr in LR_VALUES:
            logLR = np.log(lr)
            for B in BATCH_SIZES:
                loader = DataLoader(ds, batch_size=B, shuffle=True)
                for eps in EPS_VALUES:
                    desc = f"{ds_name}-lr{lr}-B{B}-eps{eps}"
                    for _ in tqdm(range(N_TRIALS), desc=desc):
                        model = model_builder(input_shape, num_classes)
                        Xp, yp = next(iter(loader))
                        
                        logP, logB, logG2, logTau = extract_probe_features(
                            model, Xp, yp, criterion, DEVICE)
                        T_star, converged, init_loss, final_loss = measure_convergence(
                            model, Xp, yp, eps, lr, criterion)
                        records.append({
                            'dataset':       ds_name,
                            'learning_rate': lr,
                            'batch_size':    B,
                            'epsilon':       eps,
                            'logP':          logP,
                            'logB':          logB,
                            'logG2':         logG2,
                            'logTau':        logTau,
                            'logLR':         logLR,
                            'logN':          np.log(total_N),
                            'logInitLoss':   np.log(init_loss),
                            'logFinalLoss':  np.log(final_loss),
                            'T_star':        T_star,
                            'converged':     converged
                        })

    df = pd.DataFrame(records)
    df.to_csv(f"{args.save_dir}/meta_dataset_{args.model}.csv", index=False)
    print(f"Saved meta_dataset_{args.model}.csv in {args.save_dir} dir with {len(df)} entries.")
    

def measure_transformer_convergence(model, loader, eps, lr, criterion, max_steps=500):
    model.train()
    X0, y0 = next(iter(loader))
    X0, y0 = X0.to(DEVICE), y0.to(DEVICE)
    model.eval()
    with torch.no_grad():
        init_loss = criterion(model(X0), y0).item()
    thresh = eps * init_loss
    model.train()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    it = iter(loader)
    for step in range(1, max_steps + 1):
        try:
            Xb, yb = next(it)
        except StopIteration:
            it = iter(loader)
            Xb, yb = next(it)
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        if loss.item() <= thresh:
            return step, True, init_loss, loss.item()
        loss.backward()
        optimizer.step()
    return max_steps, False, init_loss, loss.item()


def construct_transformer_meta_dataset():
    records = []
    criterion = nn.CrossEntropyLoss()
    ARCHITECTURE = "deit_tiny_patch16_224"

    for ds_name, (ds_cls, ds_args) in DATASETS.items():
        print(f"Dataset: {ds_name}")
        full_ds = ds_cls(root='./data', download=True, transform=TRANSFORMS[ds_name], **ds_args)
        subset = Subset(full_ds, range(1000))  # Speed optimization
        num_classes = len(full_ds.classes) if hasattr(full_ds, 'classes') else 10
        total_N = len(subset)

        for lr in LR_VALUES:
            logLR = np.log(lr)
            for B in BATCH_SIZES:
                loader = DataLoader(subset, batch_size=B, shuffle=True)
                for eps in EPS_VALUES:
                    desc = f"{ds_name}-lr{lr}-B{B}-eps{eps}"
                    for _ in tqdm(range(N_TRIALS), desc=desc):
                        model = build_transformer(ARCHITECTURE, num_classes)
                        Xp, yp = next(iter(loader))
                        logP, logB, logG2, logTau = extract_transformer_probe_features(model, Xp, yp, criterion, DEVICE)
                        T_star, converged, init_loss, final_loss = measure_transformer_convergence(
                            model, loader, eps, lr, criterion, max_steps=MAX_STEPS)

                        records.append({
                            'dataset':       ds_name,
                            'transformer':   ARCHITECTURE,
                            'learning_rate': lr,
                            'batch_size':    B,
                            'epsilon':       eps,
                            'logP':          logP,
                            'logB':          logB,
                            'logG2':         logG2,
                            'logTau':        logTau,
                            'logLR':         logLR,
                            'logN':          np.log(total_N),
                            'logInitLoss':   np.log(init_loss),
                            'logFinalLoss':  np.log(final_loss),
                            'T_star':        T_star,
                            'converged':     converged
                        })

    df = pd.DataFrame(records)
    df.to_csv('meta_dataset_transformer.csv', index=False)
    print("Saved meta_dataset_transformer.csv with", len(df), "rows.")


if __name__ == '__main__':
    if args.model == "transformer":
        construct_transformer_meta_dataset()
    else:
        construct_meta_dataset()