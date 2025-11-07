# -*- coding: utf-8 -*-
"""
CNN meta-dataset generator

Architectures (same sampler as eval):
  - VGG-lite        (3×3 Conv + BN + ReLU; optional MaxPool; dropout head)
  - ResNet-mini     (post-activation BasicBlock; projection downsample; dropout head)
  - MobileNetV2-mini(Inverted Residual; width_mult in {0.5, 0.75}; dropout head)

Preprocess (kept eval-style for compatibility):
  - MNIST/FashionMNIST -> Grayscale to 3ch, Resize(32,32), ToTensor()
  - CIFAR10/100        -> Resize(32,32), ToTensor()

CAPE probe semantics (as in your previous meta code):
  - probe batch == trial batch (no cap)          -> logB = log(B)
  - logG2, logTau are MEANS across probe samples

Convergence semantics (as in your previous meta code):
  - Optimizer: AdamW(lr, weight_decay=1e-4)
  - SOFT_MAX_STEPS=5000, plateau early-exit with patience/min_delta

Grids:
  - lr ∈ {5e-4, 1e-3, 5e-3}, B ∈ {50, 100}, eps ∈ {0.10, 0.15, 0.20}
  - N_TRIALS per (dataset, lr, B, eps)

Output:
  - meta_dataset_cnn.csv  (columns align with evaluation runner)
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

from tqdm.auto import tqdm

# -----------------------------
# Config
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# Grids (match eval LR/B/eps; trials adjustable)
LR_VALUES   = [0.0005, 0.001, 0.005]
BATCH_SIZES = [32, 64, 128]
EPS_VALUES  = [0.10, 0.15, 0.20]
N_TRIALS    = 100


SOFT_MAX_STEPS    = 5000
PLATEAU_PATIENCE  = 200
PLATEAU_MIN_DELTA = 1e-4

# Dataset choices + resize (eval-style)
DATASETS  = ["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"]
RESIZE_TO = 32

OUT_CSV = "meta_dataset_cnn.csv"

# -----------------------------
# Dataset utilities (eval-style)
# -----------------------------
def get_vision_dataset(name: str, resize_to: int) -> Tuple[Dataset, int, int, str]:
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
    return ds, 3, num_classes, name

# -----------------------------
# Lightweight CNN families (eval)
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
    if P > 2_000_000:
        return build_random_cnn(input_shape, num_classes, ds_name)
    return model, meta

# -----------------------------
# CAPE probe (BN-safe, full batch) + convergence (AdamW, soft 5000, plateau)
# -----------------------------
def extract_probe_features(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion) -> Dict[str, float]:
    """
    CAPE features with previous meta semantics:
      - probe batch == trial batch (logB = log(B))
      - logG2, logTau are MEANS across samples
    """
    model.to(DEVICE).train()
    Xp, yp = X.to(DEVICE), y.to(DEVICE)

    logP = float(np.log(sum(p.numel() for p in model.parameters())))
    logB = float(np.log(max(int(Xp.size(0)), 1)))

    params = [p for p in model.parameters() if p.requires_grad]
    g2_list, tau_list = [], []

    # BN-stability: put BN into eval during per-sample grads
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
                        criterion: nn.Module,
                        soft_max_steps: int = SOFT_MAX_STEPS,
                        plateau_patience: int = PLATEAU_PATIENCE,
                        plateau_min_delta: float = PLATEAU_MIN_DELTA):

    model.to(DEVICE).train()
    X = X.to(DEVICE); y = y.to(DEVICE).long()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    losses, init_loss, reason = [], None, ""

    for step in range(1, soft_max_steps + 1):
        opt.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        cur = float(loss.item())
        if step == 1:
            init_loss = cur
        losses.append(cur)

        if cur <= eps * init_loss:
            return step, True, init_loss, cur, "met_threshold"

        loss.backward()
        opt.step()

        if step >= plateau_patience:
            window = losses[-plateau_patience:]
            rel_impr = (window[0] - window[-1]) / max(window[0], 1e-12)
            if rel_impr < plateau_min_delta:
                reason = f"plateau_{plateau_patience}"
                break

    return step, False, init_loss, float(losses[-1]), (reason or "soft_max")

# -----------------------------
# Meta-dataset construction
# -----------------------------
def construct_meta_dataset():
    records = []

    for ds_name in DATASETS:
        vds, in_ch, num_classes, _ = get_vision_dataset(ds_name, RESIZE_TO)
        N_total = len(vds)
        logN = float(np.log(N_total))
        ce = nn.CrossEntropyLoss()

        vloader = DataLoader(vds, batch_size=max(BATCH_SIZES), shuffle=True, num_workers=0, drop_last=True)
        it_v = iter(vloader)

        cfgs = list(product(LR_VALUES, BATCH_SIZES, EPS_VALUES, range(N_TRIALS)))
        pbar = tqdm(cfgs, desc=f"[META] {ds_name}", unit="trial")

        for lr, B, eps, trial_idx in pbar:
            try:
                xb, yb = next(it_v)
            except StopIteration:
                it_v = iter(vloader); xb, yb = next(it_v)

            xb = xb[:B]; yb = yb[:B]
            input_shape = xb.shape[1:]  # (C,H,W) — always 3×32×32 after loader

            # Build model from the SAME sampler used by evaluation
            model, meta = build_random_cnn(input_shape, num_classes, ds_name)

            # CAPE probe (probe == B; logG2/logTau are MEANS)
            feats = extract_probe_features(model, xb, yb, ce)
            feats['logLR'] = float(np.log(lr))
            feats['logN']  = float(logN)

            # Convergence (AdamW, soft 5000 + plateau early-exit)
            T_star, converged, init_loss, final_loss, reason = measure_convergence(
                model, xb, yb, eps, lr, ce,
                soft_max_steps=SOFT_MAX_STEPS,
                plateau_patience=PLATEAU_PATIENCE,
                plateau_min_delta=PLATEAU_MIN_DELTA
            )

            records.append({
                'dataset'      : ds_name,
                'learning_rate': float(lr),
                'batch_size'   : int(B),
                'epsilon'      : float(eps),

                # CAPE features (previous semantics preserved)
                'logP'         : feats['logP'],
                'logB'         : feats['logB'],
                'logG2'        : feats['logG2'],
                'logTau'       : feats['logTau'],
                'logLR'        : feats['logLR'],
                'logN'         : feats['logN'],

                # Outcomes
                'T_star'       : int(T_star),
                'converged'    : bool(converged),
                'censored_reason': ("" if converged else reason),
                'logInitLoss'  : float(np.log(max(init_loss, 1e-12))),
                'logFinalLoss' : float(np.log(max(final_loss, 1e-12))),

                # Helpful tags for analysis
                'cnn_family'   : meta.get('cnn_family',''),
                'depth_signature': str(meta.get('depth_signature','')),
                'downsample_type': meta.get('downsample_type',''),
            })

    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV} with {len(df)} rows "
          f"({df['converged'].sum()} converged, {(~df['converged']).sum()} censored).")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    construct_meta_dataset()
