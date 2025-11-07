# -*- coding: utf-8 -*-
"""
CAPE RNN Evaluation — MAE-only comparison by dataset
Outputs:
  - RNN_Ablation.csv  (pivot: dataset × feature_set -> MAE)

Alignment with meta-dataset:
  - Datasets: MNIST, FashionMNIST, CIFAR10, CIFAR100 (same normalization as MLP)
  - Probe batch == trial batch (logB = log(B))
  - logG2 and logTau are MEANS over the probe samples
  - Convergence: AdamW, soft cap, plateau early-exit
  - ONE global tqdm bar
"""

import warnings
warnings.filterwarnings('ignore')

import os
import random
import numpy as np
import pandas as pd
from itertools import product

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from tqdm.auto import tqdm

# -------------------
# Config
# -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True

# Datasets (aligned with RNN/MLP meta-dataset)
DATASETS = {
    'MNIST':        (datasets.MNIST,        {'train': True}),
    'FashionMNIST': (datasets.FashionMNIST, {'train': True}),
    'CIFAR10':      (datasets.CIFAR10,      {'train': True}),
    'CIFAR100':     (datasets.CIFAR100,     {'train': True}),
}
TRANSFORMS = {
    'MNIST': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    'FashionMNIST': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ]),
    'CIFAR10': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
    ]),
    'CIFAR100': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761))
    ]),
}

# Evaluation grid (match generator)
LR_VALUES     = [0.0005, 0.001, 0.002]
BATCH_SIZES   = [32, 64, 128]
EPS_VALUES    = [0.10, 0.15, 0.20]
N_EVAL_TRIALS = 100

# Convergence (aligned with generator)
SOFT_MAX_STEPS    = 5000
PLATEAU_PATIENCE  = 200
PLATEAU_MIN_DELTA = 1e-4

META_CSV = '../Meta Datasets/meta_dataset_rnn.csv'

# CAPE-only features
ALL_FEATURES = ['logP','logB','logG2','logTau','logLR','logN']

# Full feature-set ablations
FEATURE_SETS = {
    'P':        ['logP'],
    'B':        ['logB'],
    'G2':       ['logG2'],
    'Tau':      ['logTau'],
    'LR':       ['logLR'],
    'N':        ['logN'],
    'G2+Tau':   ['logG2','logTau'],
    'P+B':      ['logP','logB'],
    'LR+N':     ['logLR','logN'],
    'P+B+LR+N': ['logP','logB','logLR','logN'],
    'P+B+G2':   ['logP','logB','logG2'],
    'P+B+Tau':  ['logP','logB','logTau'],
    'FULL-noG2':  ['logP','logB','logTau','logLR','logN'],
    'FULL-noTau': ['logP','logB','logG2','logLR','logN'],
    'FULL-noP':   ['logB','logG2','logTau','logLR','logN'],
    'FULL-noB':   ['logP','logG2','logTau','logLR','logN'],
    'FULL-noLR':  ['logP','logB','logG2','logTau','logN'],
    'FULL-noN':   ['logP','logB','logG2','logTau'],
    'FULL':     ALL_FEATURES
}

# -------------------
# Research-style RNN (matches generator family)
# -------------------
class RNNResearch(nn.Module):
    def __init__(self, input_shape, num_classes,
                 cell_type='LSTM', hidden_size=256, num_layers=2, bidirectional=False):
        super().__init__()
        C, H, W = input_shape
        self.input_size = C * W
        Cell = nn.LSTM if cell_type == 'LSTM' else nn.GRU
        self.rnn = Cell(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape
        seq = x.permute(0, 2, 1, 3).contiguous().view(B, H, C * W)  # [B, H, C*W]
        out, _ = self.rnn(seq)
        last = out[:, -1, :]
        return self.fc(last)

def sample_rnn_cfg():
    return {
        'cell_type': random.choice(['LSTM', 'GRU']),
        'hidden_size': random.choice([128, 256, 512]),
        'num_layers': random.randint(1, 3),
        'bidirectional': random.choice([False, True]),
    }

# -------------------
# Probing — CAPE-only features (probe B == trial B; means across samples)
# -------------------
def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def extract_probe_features_dict(model, X, y, criterion) -> dict:
    model.to(DEVICE).train()
    X = X.to(DEVICE); y = y.to(DEVICE).long()

    Bp   = X.size(0)                     # USE FULL TRIAL BATCH
    logP = float(np.log(_count_params(model)))
    logB = float(np.log(max(Bp, 1)))

    params = [p for p in model.parameters() if p.requires_grad]
    g2_list, tau_list = [], []

    for i in range(Bp):
        xi = X[i:i+1]; yi = y[i:i+1]

        model.zero_grad(set_to_none=True)
        logits = model(xi)
        loss   = criterion(logits, yi)
        grads  = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False)
        gv     = torch.cat([g.reshape(-1) for g in grads if g is not None])
        g2_list.append((gv**2).sum().item())

        model.zero_grad(set_to_none=True)
        logits = model(xi)
        true_logit = logits.view(-1)[yi.item()]
        grads_f    = torch.autograd.grad(true_logit, params, retain_graph=False, create_graph=False)
        fv         = torch.cat([g.reshape(-1) for g in grads_f if g is not None])
        tau_list.append((fv**2).sum().item())

    logG2  = float(np.log(max(np.mean(g2_list), 1e-12)))
    logTau = float(np.log(max(np.mean(tau_list), 1e-12)))
    return {'logP': logP, 'logB': logB, 'logG2': logG2, 'logTau': logTau}

# -------------------
# Convergence measurement (align with generator)
# -------------------
def ensure_2d_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    return logits

def measure_convergence(model, X, y, eps, lr, criterion):
    model.to(DEVICE).train()
    X, y = X.to(DEVICE), y.to(DEVICE).long()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    losses, init_loss = [], None
    for t in range(1, SOFT_MAX_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = ensure_2d_logits(model(X))
        loss   = criterion(logits, y)
        cur = float(loss.item())
        if t == 1:
            init_loss = cur
        losses.append(cur)

        if cur <= eps * init_loss:
            return t

        loss.backward()
        optimizer.step()

        if t >= PLATEAU_PATIENCE:
            window = losses[-PLATEAU_PATIENCE:]
            rel_impr = (window[0] - window[-1]) / max(window[0], 1e-12)
            if rel_impr < PLATEAU_MIN_DELTA:
                return t  # plateau reached
    return SOFT_MAX_STEPS

# -------------------
# Meta-regressors (XGBoost) from meta CSV
# -------------------
assert os.path.exists(META_CSV), f"Missing {META_CSV}"
df_meta = pd.read_csv(META_CSV)
missing = set(ALL_FEATURES + ['T_star']) - set(df_meta.columns)
assert not missing, f"meta CSV missing columns: {missing}"

X_full = df_meta[ALL_FEATURES].values
y_full = df_meta['T_star'].values
X_tr, X_te, y_tr, y_te = train_test_split(X_full, y_full, test_size=0.25, random_state=SEED)

def fit_meta(feature_list):
    cols = [ALL_FEATURES.index(f) for f in feature_list]
    model = XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=SEED, n_jobs=-1
    )
    model.fit(X_tr[:, cols], y_tr)
    return model

# Train meta-models for all feature sets
meta_models = {name: fit_meta(flist) for name, flist in FEATURE_SETS.items()}

# -------------------
# Evaluation loop — MAE only (single global tqdm)
# -------------------
rows = []
feature_names = list(FEATURE_SETS.keys())
datasets_list = list(DATASETS.items())
total_combos = len(feature_names) * len(datasets_list) * len(LR_VALUES) * len(BATCH_SIZES) * len(EPS_VALUES)

with tqdm(total=total_combos, desc="Evaluating MAE across grids") as pbar:
    for set_name in feature_names:
        flist = FEATURE_SETS[set_name]
        model_reg = meta_models[set_name]

        for ds_name, (ds_cls, ds_args) in datasets_list:
            ds = ds_cls(root='./data', download=True,
                        transform=TRANSFORMS[ds_name], **ds_args)
            num_classes = len(getattr(ds, "classes", list(range(10))))
            input_shape = ds[0][0].shape
            total_N     = len(ds)
            logN        = float(np.log(total_N))
            criterion   = nn.CrossEntropyLoss()

            for lr, B, eps in product(LR_VALUES, BATCH_SIZES, EPS_VALUES):
                loader = DataLoader(ds, batch_size=B, shuffle=True, drop_last=True, num_workers=0)
                it_loader = iter(loader)  # persistent iterator for fresh batches

                y_preds, y_trues = [], []
                for _ in range(N_EVAL_TRIALS):
                    # randomized RNN (same family as generator)
                    cfg = sample_rnn_cfg()
                    net = RNNResearch(
                        input_shape=input_shape, num_classes=num_classes,
                        cell_type=cfg['cell_type'], hidden_size=cfg['hidden_size'],
                        num_layers=cfg['num_layers'], bidirectional=cfg['bidirectional']
                    ).to(DEVICE)

                    # fetch a NEW batch each trial
                    try:
                        Xb, yb = next(it_loader)
                    except StopIteration:
                        it_loader = iter(loader)
                        Xb, yb = next(it_loader)

                    # CAPE features on the SAME batch (probe B == trial B)
                    pdict = extract_probe_features_dict(net, Xb, yb, criterion)
                    feat = {
                        'logP':  pdict['logP'],
                        'logB':  pdict['logB'],   # varies with actual B (logB=log(B))
                        'logG2': pdict['logG2'],
                        'logTau':pdict['logTau'],
                        'logLR': float(np.log(lr)),
                        'logN':  logN
                    }
                    z = np.array([feat[f] for f in flist], dtype=float).reshape(1, -1)

                    T_pred = float(model_reg.predict(z)[0])
                    T_act  = measure_convergence(net, Xb, yb, eps, lr, criterion)

                    y_preds.append(T_pred)
                    y_trues.append(T_act)

                mae = mean_absolute_error(y_trues, y_preds)
                rows.append({
                    'feature_set': set_name,
                    'dataset':     ds_name,
                    'MAE':         float(mae)
                })
                pbar.update(1)

# -------------------
# Aggregate & save
# -------------------
df_all = pd.DataFrame(rows)
df_avg = df_all.groupby(['feature_set','dataset'], as_index=False)['MAE'].mean()

pivot = df_avg.pivot_table(index='dataset', columns='feature_set', values='MAE', aggfunc='mean')
pivot = pivot.reindex(sorted(pivot.columns), axis=1).round(2)
pivot.to_csv('RNN_Ablation.csv')

print("Done!")
