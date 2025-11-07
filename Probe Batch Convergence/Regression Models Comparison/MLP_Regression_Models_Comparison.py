# -*- coding: utf-8 -*-
"""
Compare meta-regressors on CAPE features (MAE + r).

Protocol alignment:
  - Probe batch == trial batch B (logB=log(B))
  - τ uses MEAN across probe samples (paper Eq. 11 proxy)
  - Same batch used for probing and convergence
  - BatchNorm set to eval() during single-sample probes
  - Each trial draws a fresh batch (persistent dataloader iterator)

Saves:
  - MLP_Regression_Comparison.csv  (rows: overall, per-dataset, per-lr, per-batch)
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

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

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
    'CIFAR100':     (datasets.CIFAR100,     {'train': True}),
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

LR_VALUES     = [0.0005, 0.001, 0.002]
BATCH_SIZES   = [32, 64, 128]
EPS_VALUES    = [0.10, 0.15, 0.20]
N_EVAL_TRIALS = 100

SOFT_MAX_STEPS    = 5000
PLATEAU_PATIENCE  = 200
PLATEAU_MIN_DELTA = 1e-4

META_CSV = '../Meta Datasets/meta_dataset_mlp.csv'
FEATURES = ['logP','logB','logG2','logTau','logLR','logN']

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
    else:
        hidden_choices = [256, 512, 1024]
        depth_choices  = [2, 3, 4]
    return random.choice(hidden_choices), random.choice(depth_choices)

# -------------------
# CAPE probing (BN-safe) — probe batch == actual B
# -------------------
def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

@contextlib.contextmanager
def batchnorm_eval(model: nn.Module):
    bn_layers = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            bn_layers.append((m, m.training))
            m.eval()
    try:
        yield
    finally:
        for m, was_training in bn_layers:
            m.train(was_training)

def extract_probe_features(model, X, y, criterion):
    model.to(DEVICE).train()
    Bp = X.size(0)  # FULL trial batch
    Xp, yp = X.to(DEVICE), y.to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]

    logP = float(np.log(_count_params(model)))
    logB = float(np.log(max(Bp, 1)))
    g2_list, tau_list = [], []

    with batchnorm_eval(model):
        for i in range(Bp):
            xi = Xp[i:i+1]; yi = yp[i:i+1]

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

    # Paper-consistent: MEAN across probe samples for both g^2 and tau
    logG2  = float(np.log(max(np.mean(g2_list), 1e-12)))
    logTau = float(np.log(max(np.mean(tau_list), 1e-12)))
    return np.array([logP, logB, logG2, logTau], dtype=float)

# -------------------
# Convergence (aligned with generator)
# -------------------
def measure_convergence(model, X, y, eps, lr, criterion):
    model.to(DEVICE).train()
    X, y = X.to(DEVICE), y.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    losses, init_loss = [], None
    for t in range(1, SOFT_MAX_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss   = criterion(logits, y)
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
                return t
    return SOFT_MAX_STEPS

# -------------------
# Load meta-dataset & train regressors
# -------------------
assert os.path.exists(META_CSV), f"Missing {META_CSV}"
df_meta = pd.read_csv(META_CSV)
missing = set(FEATURES + ['T_star']) - set(df_meta.columns)
assert not missing, f"meta CSV missing columns: {missing}"

X_meta = df_meta[FEATURES].values
y_meta = df_meta['T_star'].values

meta_models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'RandomForest': RandomForestRegressor(n_estimators=300, max_depth=12, random_state=SEED, n_jobs=-1),
    'SVR': SVR(kernel='rbf', C=10.0, epsilon=0.1, gamma='scale'),
    'XGBoost': XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                            subsample=0.9, colsample_bytree=0.9, random_state=SEED, n_jobs=-1),
    'MLPRegressor': MLPRegressor(hidden_layer_sizes=(256, 128),
                                 activation='relu', solver='adam',
                                 alpha=1e-4, batch_size='auto',
                                 learning_rate='adaptive', learning_rate_init=1e-3,
                                 max_iter=500, random_state=SEED, early_stopping=True)
}

meta_models_fitted = {}
for name, model in meta_models.items():
    model.fit(X_meta, y_meta)
    meta_models_fitted[name] = model

# -------------------
# Evaluate (MAE + Pearson r)
# -------------------
records = []
combos = [(m, ds, lr, B, eps)
          for m in meta_models_fitted.keys()
          for ds in DATASETS.keys()
          for lr in LR_VALUES
          for B in BATCH_SIZES
          for eps in EPS_VALUES]

with tqdm(total=len(combos), desc="Evaluating regressors across grids") as pbar:
    for model_name, ds_name, lr, B, eps in combos:
        meta_model = meta_models_fitted[model_name]
        ds_cls, ds_args = DATASETS[ds_name]
        ds = ds_cls(root='./data', download=True, transform=TRANSFORMS[ds_name], **ds_args)

        num_classes = len(ds.classes)
        input_dim   = int(np.prod(ds[0][0].shape))
        total_N     = len(ds)
        logLR       = float(np.log(lr))
        logN        = float(np.log(total_N))
        criterion   = nn.CrossEntropyLoss()

        loader = DataLoader(ds, batch_size=B, shuffle=True, drop_last=True, num_workers=0)
        it_loader = iter(loader)

        y_preds, y_trues = [], []
        for _ in range(N_EVAL_TRIALS):
            # fresh batch each trial
            try:
                Xb, yb = next(it_loader)
            except StopIteration:
                it_loader = iter(loader)
                Xb, yb = next(it_loader)

            hidden_dim, depth = choose_dims(ds_name)
            model_mlp = MLPResearch(input_dim, num_classes, hidden_dim, depth,
                                    dropout_p=0.2, use_bn=True, activation="gelu")

            # Probe features on the SAME batch we use for convergence
            z0 = extract_probe_features(model_mlp, Xb, yb, criterion)  # [logP, logB, logG2, logTau]
            z  = np.concatenate([z0, [logLR, logN]], dtype=float).reshape(1, -1)

            T_pred = float(meta_model.predict(z)[0])
            T_act  = measure_convergence(model_mlp, Xb, yb, eps, lr, criterion)

            y_preds.append(T_pred)
            y_trues.append(T_act)

        mae  = float(mean_absolute_error(y_trues, y_preds))
        corr = 0.0 if (np.std(y_trues) == 0 or np.std(y_preds) == 0) else float(np.corrcoef(y_trues, y_preds)[0, 1])
        mean_t_actual = float(np.mean(y_trues))

        records.append({
            'model':        model_name,
            'dataset':      ds_name,
            'lr':           lr,
            'batch_size':   B,
            'epsilon':      eps,
            'MAE':          mae,
            'Pearson_r':    corr,
            'T_actual_avg': mean_t_actual
        })
        pbar.update(1)

df = pd.DataFrame(records)

# -------------------
# Summaries: overall + per-dataset + per-lr + per-batch
# -------------------
def agg(df_in, keys):
    return (df_in
            .groupby(keys, as_index=False)
            .agg({'MAE':'mean','Pearson_r':'mean','T_actual_avg':'mean'}))

summary_rows = []

# overall
overall = agg(df, ['model'])
overall.insert(1, 'Group', 'overall')
summary_rows.append(overall)

# per-dataset
by_ds = agg(df, ['model','dataset'])
by_ds['Group'] = 'dataset=' + by_ds['dataset'].astype(str)
by_ds = by_ds.drop(columns=['dataset'])
summary_rows.append(by_ds)

# per-lr
by_lr = agg(df, ['model','lr'])
by_lr['Group'] = 'lr=' + by_lr['lr'].astype(str)
by_lr = by_lr.drop(columns=['lr'])
summary_rows.append(by_lr)

# per-batch
by_B = agg(df, ['model','batch_size'])
by_B['Group'] = 'batch=' + by_B['batch_size'].astype(str)
by_B = by_B.drop(columns=['batch_size'])
summary_rows.append(by_B)

df_summary = pd.concat(summary_rows, axis=0, ignore_index=True)
df_summary = df_summary[['model','Group','MAE','Pearson_r','T_actual_avg']]

df_summary.to_csv("MLP_Regression_Comparison.csv", index=False)
print("Saved results to MLP_Regression_Comparison.csv")
