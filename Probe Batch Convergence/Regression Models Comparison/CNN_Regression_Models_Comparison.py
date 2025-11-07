# -*- coding: utf-8 -*-
"""
CNN Evaluation — Compare meta-regressors on CAPE features (MAE + Pearson r)

Protocol alignment (matches CNN meta generator):
  - Probe batch == trial batch B (logB=log(B))
  - logG2/logTau are MEANS across probe samples (paper Eq. 10–11 proxies)
  - Same batch used for probing and convergence
  - BatchNorm set to eval() during single-sample probes
  - Each trial draws a fresh batch (persistent dataloader iterator)
  - Preprocess: MNIST/FashionMNIST -> Grayscale→3ch, Resize(32,32), ToTensor()
                CIFAR10/100        -> Resize(32,32), ToTensor()

Saves:
  - CNN_Regression_Comparison.csv  (rows: overall, per-dataset, per-lr, per-batch)
"""

import warnings
warnings.filterwarnings('ignore')

import os
import random
from itertools import product
from typing import Tuple, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

from tqdm.auto import tqdm

# -------------------
# Config
# -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

# Match the meta-dataset generator
DATASETS = ["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"]
RESIZE_TO = 32

LR_VALUES     = [0.0005, 0.001, 0.005]
BATCH_SIZES   = [32, 64, 128]
EPS_VALUES    = [0.10, 0.15, 0.20]
N_EVAL_TRIALS = 100

SOFT_MAX_STEPS    = 5000
PLATEAU_PATIENCE  = 200
PLATEAU_MIN_DELTA = 1e-4

META_CSV  = '../Meta Datasets/meta_dataset_cnn.csv'
FEATURES  = ['logP','logB','logG2','logTau','logLR','logN']
OUT_CSV   = 'CNN_Regression_Comparison.csv'

# -----------------------------
# Dataset utilities
# -----------------------------
def get_vision_dataset(name: str, resize_to: int) -> Tuple[Dataset, int, int]:
    tfms = []
    if name in ("MNIST", "FashionMNIST"):
        tfms += [transforms.Grayscale(num_output_channels=3)]
    tfms += [transforms.Resize((resize_to, resize_to)), transforms.ToTensor()]
    tfms = transforms.Compose(tfms)

    if name == "MNIST":
        ds = datasets.MNIST(root="./data", train=True, transform=tfms, download=True); num_classes = 10
    elif name == "FashionMNIST":
        ds = datasets.FashionMNIST(root="./data", train=True, transform=tfms, download=True); num_classes = 10
    elif name == "CIFAR10":
        ds = datasets.CIFAR10(root="./data", train=True, transform=tfms, download=True); num_classes = 10
    elif name == "CIFAR100":
        ds = datasets.CIFAR100(root="./data", train=True, transform=tfms, download=True); num_classes = 100
    else:
        raise ValueError(f"Unknown dataset {name}")
    return ds, 3, num_classes

# -----------------------------
# CNN families
# -----------------------------
def conv3x3(in_ch, out_ch, stride=1, bias=False):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=bias)

def conv1x1(in_ch, out_ch, stride=1, bias=False):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=bias)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch, stride=1)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(conv1x1(in_ch, out_ch, stride=stride), nn.BatchNorm2d(out_ch))
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(out + identity)

def make_resnet_stage(in_ch, out_ch, num_blocks, first_stride):
    layers = [BasicBlock(in_ch, out_ch, stride=first_stride)]
    for _ in range(1, num_blocks):
        layers.append(BasicBlock(out_ch, out_ch, stride=1))
    return nn.Sequential(*layers)

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden = int(round(in_ch * expand_ratio))
        self.use_res = (stride == 1 and in_ch == out_ch)
        layers = []
        if expand_ratio != 1:
            layers += [conv1x1(in_ch, hidden, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True)]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            conv1x1(hidden, out_ch, bias=False), nn.BatchNorm2d(out_ch)
        ]
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv(x)
        return x + out if self.use_res else out

def build_vgg_lite(input_shape, num_classes, ds_name):
    c, h, w = input_shape
    if ds_name in ('MNIST', 'FashionMNIST'):
        chans = [32, 64, 128]; use_pool = True
    else:
        chans = [64, 128, 256]; use_pool = False
    repeats_options = [[1,1,1], [2,1,1], [2,2,1]]
    repeats = random.choice(repeats_options)

    blocks, in_ch = [], c
    stage_strides = [1, 2, 2]
    for stage_idx, out_ch in enumerate(chans):
        s = stage_strides[stage_idx] if stage_idx > 0 else 1
        blocks += [conv3x3(in_ch, out_ch, stride=s, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        for _ in range(repeats[stage_idx]-1):
            blocks += [conv3x3(out_ch, out_ch, stride=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        if use_pool and stage_idx < len(chans)-1:
            blocks.append(nn.MaxPool2d(kernel_size=2))
        in_ch = out_ch
    features = nn.Sequential(*blocks)
    head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Dropout(p=random.choice([0.0, 0.1])),
                         nn.Linear(in_ch, num_classes))
    return nn.Sequential(features, head), {'cnn_family':'vgg_lite','depth_signature':repeats,'downsample_type':'pool' if use_pool else 'stride2_conv'}

def build_resnet_mini(input_shape, num_classes, ds_name):
    c, h, w = input_shape
    base = 32 if ds_name in ('MNIST','FashionMNIST') else 64
    chs = [base, base*2, base*4]
    repeats_options = [[1,1,1], [2,2,2]]
    repeats = random.choice(repeats_options)
    stem = nn.Sequential(conv3x3(c, chs[0], 1, False), nn.BatchNorm2d(chs[0]), nn.ReLU(inplace=True))
    stage1 = make_resnet_stage(chs[0], chs[0], repeats[0], 1)
    stage2 = make_resnet_stage(chs[0], chs[1], repeats[1], 2)
    stage3 = make_resnet_stage(chs[1], chs[2], repeats[2], 2)
    head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Dropout(p=random.choice([0.0, 0.1])),
                         nn.Linear(chs[2], num_classes))
    model = nn.Sequential(stem, stage1, stage2, stage3, head)
    return model, {'cnn_family':'resnet_mini','depth_signature':repeats,'downsample_type':'proj_block'}

def build_mobilenetv2_mini(input_shape, num_classes, ds_name):
    c, h, w = input_shape
    width_mult = random.choice([0.5, 0.75])
    def _c(ch): return max(8, int(ch * width_mult))
    if ds_name in ('MNIST','FashionMNIST'):
        cfg = [[1,16,1,1],[6,24,2,2],[6,32,2,2],[6,64,2,2]]; first_out = _c(16)
    else:
        cfg = [[1,16,1,1],[6,24,2,2],[6,32,3,2],[6,64,3,2]]; first_out = _c(32)
    layers = [conv3x3(c, first_out, 1, False), nn.BatchNorm2d(first_out), nn.ReLU6(inplace=True)]
    in_ch = first_out
    for t, c_out, n, s in cfg:
        outc = _c(c_out)
        for i in range(n):
            stride = s if i == 0 else 1
            layers.append(InvertedResidual(in_ch, outc, stride, t))
            in_ch = outc
    features = nn.Sequential(*layers)
    head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Dropout(p=random.choice([0.0, 0.1])),
                         nn.Linear(in_ch, num_classes))
    model = nn.Sequential(features, head)
    return model, {'cnn_family':'mobilenetv2_mini','depth_signature':'tiny_cfg','downsample_type':'stride2_dw'}

def build_random_cnn(input_shape: tuple, num_classes: int, ds_name: str):
    family = random.choices(['vgg_lite','resnet_mini','mobilenetv2_mini'],
                            weights=[0.5,0.35,0.15], k=1)[0]
    if family == 'vgg_lite':
        model, meta = build_vgg_lite(input_shape, num_classes, ds_name)
    elif family == 'resnet_mini':
        model, meta = build_resnet_mini(input_shape, num_classes, ds_name)
    else:
        model, meta = build_mobilenetv2_mini(input_shape, num_classes, ds_name)
    P = sum(p.numel() for p in model.parameters())
    if P > 2_000_000:  # keep it lightweight as in generator
        return build_random_cnn(input_shape, num_classes, ds_name)
    return model, meta

# -----------------------------
# CAPE probe (BN-safe, full batch) + convergence (AdamW, soft 5000, plateau)
# -----------------------------
def extract_probe_features(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion) -> Dict[str, float]:
    """
    - probe batch == trial batch (logB = log(B))
    - logG2, logTau are MEANS across samples
    """
    model.to(DEVICE).train()
    Xp, yp = X.to(DEVICE), y.to(DEVICE)

    logP = float(np.log(sum(p.numel() for p in model.parameters())))
    logB = float(np.log(max(int(Xp.size(0)), 1)))

    params = [p for p in model.parameters() if p.requires_grad]
    g2_list, tau_list = [], []

    # BN-stability: set BN to eval() during per-sample grads
    bn_layers, bn_states = [], []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            bn_layers.append(m); bn_states.append(m.training); m.eval()

    try:
        for i in range(Xp.size(0)):
            xi = Xp[i:i+1]; yi = yp[i:i+1]

            model.zero_grad(set_to_none=True)
            logits = model(xi)
            loss = criterion(logits, yi)
            grads = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False, allow_unused=False)
            gv = torch.cat([g.reshape(-1) for g in grads if g is not None])
            g2_list.append(float((gv**2).sum().item()))

            model.zero_grad(set_to_none=True)
            logits = model(xi)
            true_logit = logits.view(-1)[yi.item()]
            grads_f = torch.autograd.grad(true_logit, params, retain_graph=False, create_graph=False, allow_unused=False)
            fv = torch.cat([g.reshape(-1) for g in grads_f if g is not None])
            tau_list.append(float((fv**2).sum().item()))
    finally:
        for m, st in zip(bn_layers, bn_states):
            m.train(st)

    g2_mean  = float(np.mean(g2_list)) if len(g2_list)  else 1e-12
    tau_mean = float(np.mean(tau_list)) if len(tau_list) else 1e-12

    return {
        'logP':   logP,
        'logB':   logB,
        'logG2':  float(np.log(max(g2_mean,  1e-12))),
        'logTau': float(np.log(max(tau_mean, 1e-12))),
    }

def measure_convergence(model: nn.Module,
                        X: torch.Tensor,
                        y: torch.Tensor,
                        eps: float,
                        lr: float,
                        criterion: nn.Module):
    """
    Train on the SAME batch until loss <= eps * init_loss, or plateau/soft cap.
    Optimizer: AdamW(lr, weight_decay=1e-4)
    Returns: T_star
    """
    model.to(DEVICE).train()
    X = X.to(DEVICE); y = y.to(DEVICE).long()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    losses, init_loss = [], None

    for step in range(1, SOFT_MAX_STEPS + 1):
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

        if step >= PLATEAU_PATIENCE:
            window = losses[-PLATEAU_PATIENCE:]
            rel_impr = (window[0] - window[-1]) / max(window[0], 1e-12)
            if rel_impr < PLATEAU_MIN_DELTA:
                return step

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
          for ds in DATASETS
          for lr in LR_VALUES
          for B in BATCH_SIZES
          for eps in EPS_VALUES]

with tqdm(total=len(combos), desc="Evaluating CNN regressors across grids") as pbar:
    for model_name, ds_name, lr, B, eps in combos:
        meta_model = meta_models_fitted[model_name]
        ds, in_ch, num_classes = get_vision_dataset(ds_name, RESIZE_TO)
        total_N     = len(ds)
        logLR       = float(np.log(lr))
        logN        = float(np.log(total_N))
        criterion   = nn.CrossEntropyLoss()

        loader   = DataLoader(ds, batch_size=B, shuffle=True, drop_last=True, num_workers=0)
        it_loader = iter(loader)

        y_preds, y_trues = [], []
        for _ in range(N_EVAL_TRIALS):
            # fresh batch each trial
            try:
                Xb, yb = next(it_loader)
            except StopIteration:
                it_loader = iter(loader)
                Xb, yb = next(it_loader)

            input_shape = Xb.shape[1:]  # (3, 32, 32)
            model_cnn, _meta = build_random_cnn(input_shape, num_classes, ds_name)

            # Probe features on the SAME batch we use for convergence
            feats = extract_probe_features(model_cnn, Xb, yb, criterion)  # dict with logP/logB/logG2/logTau
            z = np.array([
                feats['logP'], feats['logB'], feats['logG2'], feats['logTau'],
                logLR, logN
            ], dtype=float).reshape(1, -1)

            T_pred = float(meta_model.predict(z)[0])
            T_act  = measure_convergence(model_cnn, Xb, yb, eps, lr, criterion)

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

df_summary.to_csv(OUT_CSV, index=False)
print(f"Saved results to {OUT_CSV}")
