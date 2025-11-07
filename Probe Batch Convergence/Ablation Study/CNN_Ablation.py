# -*- coding: utf-8 -*-
"""
CAPE CNN Evaluation — MAE-only comparison by dataset
Outputs:
  - CNN_Ablation.csv  (pivot: dataset × feature_set -> MAE)

Protocol alignment (matches CNN meta generator):
  - Datasets: MNIST/FashionMNIST -> Grayscale→3ch, Resize(32,32), ToTensor()
              CIFAR10/100        -> Resize(32,32), ToTensor()
  - Architectures: VGG-lite, ResNet-mini, MobileNetV2-mini (same sampler as generator)
  - Probe batch == trial batch (logB = log(B))
  - logG2 and logTau are MEANS over probe samples
  - BN set to eval() during per-sample probes (Dropout unchanged — matches generator)
  - Convergence: AdamW(lr, weight_decay=1e-4), SOFT_MAX_STEPS=5000, plateau early-exit
  - Each trial draws a fresh batch (persistent dataloader iterator)
"""

import warnings
warnings.filterwarnings('ignore')

import os
import math
import random
from itertools import product
from typing import Tuple, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from tqdm.auto import tqdm

# -------------------
# Config
# -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# Evaluation grid — match generator
LR_VALUES     = [0.0005, 0.001, 0.005]
BATCH_SIZES   = [32, 64, 128]
EPS_VALUES    = [0.10, 0.15, 0.20]
N_EVAL_TRIALS = 100

SOFT_MAX_STEPS    = 5000
PLATEAU_PATIENCE  = 200
PLATEAU_MIN_DELTA = 1e-4

# Datasets & transforms 
DATASETS  = ["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"]
RESIZE_TO = 32

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

# -------------------
# CNN building blocks (identical to generator)
# -------------------
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
    return nn.Sequential(features, head)

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
    return model

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
    return model

def build_random_cnn(input_shape: tuple, num_classes: int, ds_name: str):
    family = random.choices(['vgg_lite','resnet_mini','mobilenetv2_mini'],
                            weights=[0.5,0.35,0.15], k=1)[0]
    if family == 'vgg_lite':
        model = build_vgg_lite(input_shape, num_classes, ds_name)
    elif family == 'resnet_mini':
        model = build_resnet_mini(input_shape, num_classes, ds_name)
    else:
        model = build_mobilenetv2_mini(input_shape, num_classes, ds_name)
    P = sum(p.numel() for p in model.parameters())
    if P > 2_000_000:
        return build_random_cnn(input_shape, num_classes, ds_name)
    return model

# -------------------
# Helpers
# -------------------
def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def _ensure_2d_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits if logits.ndim == 2 else logits.view(logits.size(0), -1)

# -------------------
# Probing (CAPE features) — BN eval() during per-sample probe (Dropout untouched)
# -------------------
def extract_probe_features_dict(model: nn.Module, X: torch.Tensor, Y: torch.Tensor, criterion) -> dict:
    model.to(DEVICE).train()
    X = X.to(DEVICE); Y = Y.to(DEVICE).long()

    logP = float(np.log(_count_params(model)))
    Bp   = int(X.size(0))
    logB = float(np.log(max(Bp, 1)))

    params = [p for p in model.parameters() if p.requires_grad]
    g2_list, tau_list = [], []

    # BN to eval() during per-sample gradients (matches generator)
    bn_layers, bn_states = [], []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            bn_layers.append(m); bn_states.append(m.training); m.eval()
    try:
        for i in range(Bp):
            xi = X[i:i+1]; yi = Y[i:i+1]

            model.zero_grad(set_to_none=True)
            logits = _ensure_2d_logits(model(xi))
            loss   = criterion(logits, yi)
            grads  = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False, allow_unused=False)
            gv     = torch.cat([g.reshape(-1) for g in grads if g is not None])
            g2_list.append(float((gv**2).sum().item()))

            model.zero_grad(set_to_none=True)
            logits = _ensure_2d_logits(model(xi))
            true_logit = logits[0, yi[0].item()]
            grads_f = torch.autograd.grad(true_logit, params, retain_graph=False, create_graph=False, allow_unused=False)
            fv      = torch.cat([g.reshape(-1) for g in grads_f if g is not None])
            tau_list.append(float((fv**2).sum().item()))
    finally:
        for m, st in zip(bn_layers, bn_states):
            m.train(st)

    logG2  = float(np.log(max(np.mean(g2_list),  1e-12)))
    logTau = float(np.log(max(np.mean(tau_list), 1e-12)))
    return {'logP': logP, 'logB': logB, 'logG2': logG2, 'logTau': logTau}

# -------------------
# Convergence measurement (aligned with generator)
# -------------------
def measure_convergence(model: nn.Module,
                        X: torch.Tensor,
                        Y: torch.Tensor,
                        eps: float,
                        lr: float,
                        criterion: nn.Module) -> int:
    model.to(DEVICE).train()
    X = X.to(DEVICE); Y = Y.to(DEVICE).long()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    losses, init_loss = [], None

    for t in range(1, SOFT_MAX_STEPS + 1):
        opt.zero_grad(set_to_none=True)
        logits = _ensure_2d_logits(model(X))
        loss   = criterion(logits, Y)
        cur    = float(loss.item())
        if t == 1:
            init_loss = cur
        losses.append(cur)

        if cur <= eps * init_loss:
            return t

        loss.backward()
        opt.step()

        if t >= PLATEAU_PATIENCE:
            window = losses[-PLATEAU_PATIENCE:]
            rel_impr = (window[0] - window[-1]) / max(window[0], 1e-12)
            if rel_impr < PLATEAU_MIN_DELTA:
                return t  # plateau reached

    return SOFT_MAX_STEPS

# -------------------
# Meta-regressors (XGBoost) from meta CSV
# -------------------
META_CSV = '../Meta Datasets/meta_dataset_cnn.csv'
ALL_FEATURES = ['logP','logB','logG2','logTau','logLR','logN']

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
# Evaluation loop — MAE only
# -------------------
rows = []
feature_names = list(FEATURE_SETS.keys())
total_combos = len(feature_names) * len(DATASETS) * len(LR_VALUES) * len(BATCH_SIZES) * len(EPS_VALUES)

with tqdm(total=total_combos, desc="Evaluating CNN MAE across grids") as pbar:
    for set_name in feature_names:
        flist = FEATURE_SETS[set_name]
        reg = meta_models[set_name]

        for ds_name in DATASETS:
            ds, in_ch, num_classes = get_vision_dataset(ds_name, RESIZE_TO)
            total_N = len(ds)
            logN    = float(np.log(total_N))
            criterion = nn.CrossEntropyLoss()

            for lr in LR_VALUES:
                logLR = float(np.log(lr))
                for B in BATCH_SIZES:
                    loader = DataLoader(ds, batch_size=B, shuffle=True, drop_last=True, num_workers=0)
                    it_loader = iter(loader)
                    for eps in EPS_VALUES:
                        y_preds, y_trues = [], []
                        for _ in range(N_EVAL_TRIALS):
                            try:
                                Xb, Yb = next(it_loader)
                            except StopIteration:
                                it_loader = iter(loader); Xb, Yb = next(it_loader)

                            input_shape = Xb.shape[1:]  # (C,H,W) — 3×32×32 after loader
                            net = build_random_cnn(input_shape, num_classes, ds_name)

                            # Probe on the SAME batch
                            pdict = extract_probe_features_dict(net, Xb, Yb, criterion)
                            feat = {
                                'logP':  pdict['logP'],
                                'logB':  pdict['logB'],
                                'logG2': pdict['logG2'],
                                'logTau':pdict['logTau'],
                                'logLR': logLR,
                                'logN':  logN
                            }
                            z = np.array([feat[f] for f in flist], dtype=float).reshape(1, -1)

                            T_pred = float(reg.predict(z)[0])
                            T_act  = measure_convergence(net, Xb, Yb, eps, lr, criterion)

                            y_preds.append(T_pred)
                            y_trues.append(T_act)

                        mae = mean_absolute_error(y_trues, y_preds)
                        rows.append({'feature_set': set_name, 'dataset': ds_name, 'MAE': float(mae)})
                        pbar.update(1)

# -------------------
# Aggregate & save
# -------------------
df_all = pd.DataFrame(rows)
df_avg = df_all.groupby(['feature_set','dataset'], as_index=False)['MAE'].mean()

pivot = df_avg.pivot_table(index='dataset', columns='feature_set', values='MAE', aggfunc='mean')
pivot = pivot.reindex(sorted(pivot.columns), axis=1).round(2)
pivot.to_csv('CNN_Ablation.csv')

print("Done!")
