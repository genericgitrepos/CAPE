# -*- coding: utf-8 -*-
"""
Compare meta-regressors on CAPE features (MAE + r) for RNNs.
Aligned with the new RNN meta-dataset generator:
- Datasets: MNIST, FashionMNIST, CIFAR10, CIFAR100
- Normalization: same stats as the MLP pipeline
- Features: logP, logB, logG2 (mean), logTau (mean), logLR, logN
- Probing: probe batch == trial batch (logB = log(B)); BN-safe single-sample probes
- Convergence: probe-batch, AdamW, soft cap + plateau early-exit

Outputs:
  RNN_Regression_Comparison.csv
"""

import warnings
warnings.filterwarnings('ignore')

import os
import random
from typing import Tuple
import contextlib

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from sklearn.metrics import mean_absolute_error
from tqdm.auto import tqdm

# -------------------
# Config
# -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True

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

# Match the (new) RNN meta-dataset / MLP normalization
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

# Grids (match generator)
LR_VALUES     = [0.0005, 0.001, 0.002]
BATCH_SIZES   = [32, 64, 128]
EPS_VALUES    = [0.10, 0.15, 0.20]
N_EVAL_TRIALS = 100

# Convergence (match generator)
SOFT_MAX_STEPS    = 5000
PLATEAU_PATIENCE  = 200
PLATEAU_MIN_DELTA = 1e-4

META_CSV = '../Meta Datasets/meta_dataset_rnn.csv'
FEATURES = ['logP','logB','logG2','logTau','logLR','logN']

# -------------------
# RNN builder (same family as generator)
# -------------------
class RNNClassifier(nn.Module):
    """
    Sequence: scan image rows → [B, H, C*W]
    Cell: LSTM or GRU
    hidden: {128, 256, 512}, layers: {1, 2, 3}, optional bidirectional
    """
    def __init__(self, input_shape: Tuple[int,int,int], num_classes: int,
                 cell_type=None, hidden_size=None, num_layers=None, bidirectional=None):
        super().__init__()
        C, H, W = input_shape
        input_size = C * W

        self.cell_type     = cell_type if cell_type is not None else random.choice(['LSTM', 'GRU'])
        self.hidden_size   = hidden_size if hidden_size is not None else random.choice([128, 256, 512])
        self.num_layers    = num_layers if num_layers is not None else random.randint(1, 3)
        self.bidirectional = bidirectional if bidirectional is not None else random.choice([False, True])

        Cell = nn.LSTM if self.cell_type == 'LSTM' else nn.GRU
        self.rnn = Cell(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        self.fc = nn.Linear(self.hidden_size * (2 if self.bidirectional else 1), num_classes)

    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape
        seq = x.permute(0, 2, 1, 3).contiguous().view(B, H, C * W)  # [B, H, C*W]
        out, _ = self.rnn(seq)                                      # [B, H, hidden*(2 if bi)]
        last = out[:, -1, :]
        return self.fc(last)

def build_rnn(input_shape: tuple, num_classes: int) -> nn.Module:
    return RNNClassifier(input_shape, num_classes)

# -------------------
# CAPE probing (aligned)
# -------------------
def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@contextlib.contextmanager
def batchnorm_eval(model: nn.Module):
    """Keep BN stable during single-sample probes (rare in this RNN but future-proof)."""
    bns = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            bns.append((m, m.training))
            m.eval()
    try:
        yield
    finally:
        for m, was in bns:
            m.train(was)

def extract_probe_features(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion) -> np.ndarray:
    """
    Returns CAPE-only features: [logP, logB, logG2, logTau]
    Protocol:
      - Probe batch == trial batch (logB = log(B))
      - logG2 and logTau are MEANS across probe samples
    """
    model.to(DEVICE).train()
    X = X.to(DEVICE); y = y.to(DEVICE).long()

    Bp   = X.size(0)  # full trial batch
    logP = float(np.log(_count_params(model)))
    logB = float(np.log(max(Bp, 1)))

    params = [p for p in model.parameters() if p.requires_grad]
    g2_list, tau_list = [], []

    with batchnorm_eval(model):
        for i in range(Bp):
            xi = X[i:i+1]; yi = y[i:i+1]

            model.zero_grad(set_to_none=True)
            logits = model(xi)
            loss = criterion(logits, yi)
            grads = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False)
            gv = torch.cat([g.reshape(-1) for g in grads if g is not None])
            g2_list.append((gv**2).sum().item())

            model.zero_grad(set_to_none=True)
            logits = model(xi)
            true_logit = logits.view(-1)[yi.item()]
            grads_f = torch.autograd.grad(true_logit, params, retain_graph=False, create_graph=False)
            fv = torch.cat([g.reshape(-1) for g in grads_f if g is not None])
            tau_list.append((fv**2).sum().item())

    logG2  = float(np.log(max(np.mean(g2_list), 1e-12)))
    logTau = float(np.log(max(np.mean(tau_list), 1e-12)))
    return np.array([logP, logB, logG2, logTau], dtype=float)

# -------------------
# Convergence (aligned with generator)
# -------------------
def measure_convergence_adaptive(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    lr: float,
    criterion,
    soft_max_steps: int = SOFT_MAX_STEPS,
    plateau_patience: int = PLATEAU_PATIENCE,
    plateau_min_delta: float = PLATEAU_MIN_DELTA,
) -> int:
    """
    Train on the SAME probe batch until loss <= eps * init_loss, or plateau/no progress.
    Returns: T_star
    """
    model.to(DEVICE).train()
    X = X.to(DEVICE); y = y.to(DEVICE).long()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    losses = []
    init_loss = None

    with batchnorm_eval(model):
        for step in range(1, soft_max_steps + 1):
            opt.zero_grad(set_to_none=True)
            logits = model(X)
            loss = criterion(logits, y)
            cur = float(loss.item())
            if step == 1:
                init_loss = cur
            losses.append(cur)

            if cur <= eps * init_loss:
                return step

            loss.backward()
            opt.step()

            if step >= plateau_patience:
                window = losses[-plateau_patience:]
                rel_impr = (window[0] - window[-1]) / max(window[0], 1e-12)
                if rel_impr < plateau_min_delta:
                    return step

    return soft_max_steps

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
    'RandomForest': RandomForestRegressor(
        n_estimators=300, max_depth=12, random_state=SEED, n_jobs=-1),
    'SVR': SVR(kernel='rbf', C=10.0, epsilon=0.1, gamma='scale'),
}
if HAS_XGB:
    meta_models['XGBoost'] = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=SEED, n_jobs=-1)
# Light neural baseline on tabular features
meta_models['MLPRegressor'] = MLPRegressor(
    hidden_layer_sizes=(256, 128),
    activation='relu', solver='adam',
    alpha=1e-4, batch_size='auto',
    learning_rate='adaptive', learning_rate_init=1e-3,
    max_iter=500, random_state=SEED, early_stopping=True)

meta_models_fitted = {name: m.fit(X_meta, y_meta) for name, m in meta_models.items()}

# -------------------
# Evaluate (MAE + Pearson r)
# -------------------
records = []
combos = [(mname, ds_name, lr, B, eps)
          for mname in meta_models_fitted.keys()
          for ds_name in DATASETS.keys()
          for lr in LR_VALUES
          for B in BATCH_SIZES
          for eps in EPS_VALUES]

with tqdm(total=len(combos), desc="Evaluating RNN regressors across grids") as pbar:
    for model_name, ds_name, lr, B, eps in combos:
        reg = meta_models_fitted[model_name]
        ds_cls, ds_args = DATASETS[ds_name]
        ds = ds_cls(root='./data', download=True, transform=TRANSFORMS[ds_name], **ds_args)

        num_classes = len(getattr(ds, "classes", list(range(10))))
        input_shape = ds[0][0].shape
        total_N     = len(ds)
        logLR       = float(np.log(lr))
        logN        = float(np.log(total_N))
        criterion   = nn.CrossEntropyLoss()

        loader = DataLoader(ds, batch_size=B, shuffle=True, drop_last=True, num_workers=0)
        it_loader = iter(loader)  # persistent iterator for fresh batches

        y_preds, y_trues = [], []
        for _ in range(N_EVAL_TRIALS):
            # get a fresh batch from the persistent iterator
            try:
                Xb, yb = next(it_loader)
            except StopIteration:
                it_loader = iter(loader)
                Xb, yb = next(it_loader)

            # randomized RNN exactly like the generator
            model_rnn = build_rnn(input_shape, num_classes)

            # CAPE probe features on the SAME batch (probe B == trial B)
            z0 = extract_probe_features(model_rnn, Xb, yb, criterion)  # [logP, logB, logG2, logTau]
            z  = np.concatenate([z0, [logLR, logN]], dtype=float).reshape(1, -1)

            # predict & measure actual T*
            T_pred = float(reg.predict(z)[0])
            T_act  = measure_convergence_adaptive(model_rnn, Xb, yb, eps, lr, criterion)
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

# -------------------
# Summarize results
# -------------------
df = pd.DataFrame(records)
df_summary = df.groupby(['model', 'dataset'], as_index=False).agg({
    'MAE': 'mean',
    'Pearson_r': 'mean',
    'T_actual_avg': 'mean'
})

out_csv = "RNN_Regression_Comparison.csv"
df_summary.to_csv(out_csv, index=False)
print(f"Saved results to {out_csv}")
