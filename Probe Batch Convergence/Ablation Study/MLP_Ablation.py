# -*- coding: utf-8 -*-
"""
CAPE MLP Evaluation — MAE-only comparison by dataset
Outputs:
  - MLP_Ablation.csv  (pivot: dataset × feature_set -> MAE)

Protocol alignment:
  - Probe batch == trial batch B (logB=log(B))
  - Same batch used for probing and convergence
  - BatchNorm & Dropout set to eval() only during single-sample probes
  - Each trial draws a fresh batch (persistent dataloader iterator)
"""

import warnings
warnings.filterwarnings('ignore')

import os
import random
import numpy as np
import pandas as pd
import contextlib

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

DATASETS = {
    'MNIST':        (datasets.MNIST,        {'train': True}),
    'FashionMNIST': (datasets.FashionMNIST, {'train': True}),
    'CIFAR10':      (datasets.CIFAR10,      {'train': True}),
    'CIFAR100':     (datasets.CIFAR100,     {'train': True})
}

# Match the meta-dataset normalization
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

# Evaluation grid
LR_VALUES     = [0.0005, 0.001, 0.002]
BATCH_SIZES   = [32, 64, 128]
EPS_VALUES    = [0.10, 0.15, 0.20]
N_EVAL_TRIALS = 100

# Convergence (aligned with meta generator)
SOFT_MAX_STEPS    = 5000
PLATEAU_PATIENCE  = 200
PLATEAU_MIN_DELTA = 1e-4

META_CSV = '../Meta Datasets/meta_dataset_mlp.csv'

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
# Research-style MLP (matches generator)
# -------------------
class MLPResearch(nn.Module):
    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dim: int, depth: int,
                 dropout_p: float = 0.2, use_bn: bool = True, activation: str = "gelu"):
        super().__init__()
        act_layer = nn.GELU() if activation == "gelu" else nn.ReLU()

        layers = [nn.Flatten()]
        dims = [input_dim] + [hidden_dim]*depth + [num_classes]
        for i in range(len(dims)-2):
            d_in, d_out = dims[i], dims[i+1]
            block = [nn.Linear(d_in, d_out)]
            if use_bn:
                block.append(nn.BatchNorm1d(d_out))
            block.append(act_layer)
            if dropout_p > 0.0:
                block.append(nn.Dropout(dropout_p))
            layers.extend(block)
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

        # Init: xavier for GELU; kaiming for ReLU
        def init_fn(m):
            if isinstance(m, nn.Linear):
                if activation == "gelu":
                    nn.init.xavier_normal_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        self.apply(init_fn)

    def forward(self, x):
        return self.net(x)

def choose_dims(ds_name: str):
    if ds_name in ("MNIST", "FashionMNIST"):
        hidden_choices = [128, 256, 512]
        depth_choices  = [2, 3]
    else:  # CIFAR10/100
        hidden_choices = [256, 512, 1024]
        depth_choices  = [2, 3, 4]
    return random.choice(hidden_choices), random.choice(depth_choices)

# -------------------
# Helpers (BN/Dropout-safe probing + logits handling)
# -------------------
def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def _ensure_2d_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits if logits.ndim == 2 else logits.view(1, -1)

@contextlib.contextmanager
def stabilize_probes(model: nn.Module):
    """Set BN/Dropout to eval() during probe to stabilize per-sample gradients."""
    changed = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            changed.append((m, m.training))
            m.eval()
    try:
        yield
    finally:
        for m, was_training in changed:
            m.train(was_training)

# -------------------
# Probing (CAPE features) — probe B == actual B
# -------------------
def extract_probe_features_dict(model, X, Y, criterion) -> dict:
    model.to(DEVICE).train()
    X = X.to(DEVICE); Y = Y.to(DEVICE).long()

    Bp   = int(X.size(0))                 # USE FULL TRIAL BATCH
    logP = float(np.log(_count_params(model)))
    logB = float(np.log(max(Bp, 1)))

    params = [p for p in model.parameters() if p.requires_grad]
    g2_list, tau_list = [], []
    with stabilize_probes(model):
        for i in range(Bp):
            xi = X[i:i+1]; yi = Y[i:i+1]

            model.zero_grad(set_to_none=True)
            logits = _ensure_2d_logits(model(xi))
            loss   = criterion(logits, yi)
            grads  = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False)
            gv     = torch.cat([g.reshape(-1) for g in grads if g is not None])
            g2_list.append((gv**2).sum().item())

            model.zero_grad(set_to_none=True)
            logits = _ensure_2d_logits(model(xi))
            true_logit = logits[0, yi[0].item()]
            grads_f    = torch.autograd.grad(true_logit, params, retain_graph=False, create_graph=False)
            fv         = torch.cat([g.reshape(-1) for g in grads_f if g is not None])
            tau_list.append((fv**2).sum().item())

    # Paper-correct: BOTH are means across the probe batch
    logG2  = float(np.log(max(np.mean(g2_list), 1e-12)))
    logTau = float(np.log(max(np.mean(tau_list), 1e-12)))  # <-- mean, not sum
    return {'logP': logP, 'logB': logB, 'logG2': logG2, 'logTau': logTau}

# -------------------
# Convergence measurement (align with generator)
# -------------------
def measure_convergence(model, X, Y, eps, lr, criterion):
    model.to(DEVICE).train()  # training uses normal train() (BN updates enabled)
    X = X.to(DEVICE); Y = Y.to(DEVICE).long()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    losses, init_loss = [], None
    for t in range(1, SOFT_MAX_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = _ensure_2d_logits(model(X))
        loss   = criterion(logits, Y)
        if t == 1:
            init_loss = float(loss.item())
        cur = float(loss.item())
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
        subsample=0.9, colsample_bytree=0.9, random_state=SEED
    )
    model.fit(X_tr[:, cols], y_tr)
    return model

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
            num_classes = len(ds.classes)
            input_dim   = int(np.prod(ds[0][0].shape))
            total_N     = len(ds)
            logN        = float(np.log(total_N))
            criterion   = nn.CrossEntropyLoss()

            for lr in LR_VALUES:
                for B in BATCH_SIZES:
                    loader = DataLoader(ds, batch_size=B, shuffle=True, drop_last=True, num_workers=0)
                    it_loader = iter(loader)  # persistent iterator for fresh batches
                    for eps in EPS_VALUES:
                        y_preds, y_trues = [], []
                        for _ in range(N_EVAL_TRIALS):
                            # fetch a NEW batch each trial
                            try:
                                Xb, Yb = next(it_loader)
                            except StopIteration:
                                it_loader = iter(loader)
                                Xb, Yb = next(it_loader)

                            hidden_dim, depth = choose_dims(ds_name)
                            net = MLPResearch(input_dim, num_classes, hidden_dim, depth,
                                              dropout_p=0.2, use_bn=True, activation="gelu")

                            # CAPE features on the SAME batch (probe B == trial B)
                            pdict = extract_probe_features_dict(net, Xb, Yb, criterion)
                            feat = {
                                'logP':  pdict['logP'],
                                'logB':  pdict['logB'],
                                'logG2': pdict['logG2'],
                                'logTau':pdict['logTau'],
                                'logLR': float(np.log(lr)),
                                'logN':  logN
                            }
                            z = np.array([feat[f] for f in flist], dtype=float).reshape(1, -1)

                            T_pred = float(model_reg.predict(z)[0])
                            T_act  = measure_convergence(net, Xb, Yb, eps, lr, criterion)

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
pivot.to_csv('MLP_Ablation.csv')

print("MLP Ablation Done!")
