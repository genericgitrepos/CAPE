# -*- coding: utf-8 -*-
"""
RNN Meta-Dataset Generator (CAPE-only features)
Datasets: MNIST, FashionMNIST, CIFAR10, CIFAR100 (aligned with MLP)
Protocol:
  - Probe batch == trial batch (logB = log(B))
  - logG2/logTau computed as MEAN across probe samples
  - BN layers (if present) set to eval() during per-sample probes
  - Same batch used for probing and convergence
Outputs:
  - meta_dataset_rnn.csv
"""

import warnings
warnings.filterwarnings('ignore')

import os
import math
import random
from typing import Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import contextlib

# ------------------- Config -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# Datasets (aligned with MLP)
DATASETS = {
    'MNIST':        datasets.MNIST,
    'FashionMNIST': datasets.FashionMNIST,
    'CIFAR10':      datasets.CIFAR10,
    'CIFAR100':     datasets.CIFAR100,
}

# Match the MLP normalization
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

# Grids (aligned with MLP generator)
LR_VALUES    = [0.0005, 0.001, 0.002]
BATCH_SIZES  = [32, 64, 128]
EPS_VALUES   = [0.10, 0.15, 0.20]
N_TRIALS     = 100

# Convergence (match MLP generator)
SOFT_MAX_STEPS     = 5000
PLATEAU_PATIENCE   = 200
PLATEAU_MIN_DELTA  = 1e-4

OUT_CSV = "meta_dataset_rnn.csv"


# ------------------- Small/medium RNN classifier -------------------
class RNNClassifier(nn.Module):
    """
    Scan image rows as a sequence:
      sequence length = H, feature size per step = C*W
    Cell: LSTM or GRU, hidden {128,256,512}, layers {1,2,3}, optional bidirectional
    """
    def __init__(self, input_shape: Tuple[int,int,int], num_classes: int,
                 cell_type: str = None, hidden_size: int = None,
                 num_layers: int = None, bidirectional: bool = None):
        super().__init__()
        C, H, W = input_shape
        input_size = C * W

        # sample config if not provided
        self.cell_type    = cell_type    if cell_type    is not None else random.choice(['LSTM', 'GRU'])
        self.hidden_size  = hidden_size  if hidden_size  is not None else random.choice([128, 256, 512])
        self.num_layers   = num_layers   if num_layers   is not None else random.randint(1, 3)
        self.bidirectional= bidirectional if bidirectional is not None else random.choice([False, True])

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


# ------------------- Helpers -------------------
def ensure_2d_logits(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 1:
        return t.unsqueeze(0)
    return t

def steps_per_epoch_from_NB(N: int, B: int) -> int:
    return int(math.ceil(N / float(B)))

def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@contextlib.contextmanager
def batchnorm_eval(model: nn.Module):
    """Temporarily set BN layers to eval() for safe single-sample gradient probes (rare in RNNs, but future-proof)."""
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


# ------------------- CAPE features (probe B == trial B) -------------------
def extract_probe_features(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion) -> Tuple[float, float, float, float]:
    """
    Returns CAPE-only features: logP, logB, logG2, logTau
    - Probe batch uses the FULL trial batch (Bp = B)
    - logG2, logTau use MEAN across the probe samples
    """
    model.to(DEVICE).train()
    X = X.to(DEVICE); y = y.to(DEVICE)

    Bp   = X.size(0)  # full trial batch
    logP = float(np.log(_count_params(model)))
    logB = float(np.log(max(Bp, 1)))

    params = [p for p in model.parameters() if p.requires_grad]
    g2_list, tau_list = [], []

    with batchnorm_eval(model):
        for i in range(Bp):
            xi = X[i:i+1]; yi = y[i:i+1]

            model.zero_grad(set_to_none=True)
            logits = ensure_2d_logits(model(xi))
            loss   = criterion(logits, yi)
            grads  = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False)
            gv     = torch.cat([g.reshape(-1) for g in grads if g is not None])
            g2_list.append((gv**2).sum().item())

            model.zero_grad(set_to_none=True)
            logits = ensure_2d_logits(model(xi))
            true_logit = logits[0, int(yi.item())]
            grads_f    = torch.autograd.grad(true_logit, params, retain_graph=False, create_graph=False)
            fv         = torch.cat([g.reshape(-1) for g in grads_f if g is not None])
            tau_list.append((fv**2).sum().item())

    # Means across probe samples (aligned with your latest MLP setting)
    logG2  = float(np.log(max(np.mean(g2_list), 1e-12)))
    logTau = float(np.log(max(np.mean(tau_list), 1e-12)))
    return logP, logB, logG2, logTau


# ------------------- Convergence (adaptive) -------------------
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
):
    """
    Train on the SAME probe batch until loss <= eps * init_loss, or plateau/no progress.
    Returns: (T_star, converged, init_loss, final_loss, reason)
    """
    model.to(DEVICE).train()
    X, y = X.to(DEVICE), y.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    losses = []
    init_loss = None
    reason = ""

    with batchnorm_eval(model):
        for step in range(1, soft_max_steps + 1):
            opt.zero_grad(set_to_none=True)
            logits = ensure_2d_logits(model(X))
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


# ------------------- Meta-dataset construction -------------------
def construct_meta_dataset():
    records = []

    total = len(DATASETS) * len(LR_VALUES) * len(BATCH_SIZES) * len(EPS_VALUES) * N_TRIALS
    pbar = tqdm(total=total, desc="Building RNN meta dataset (CAPE-only)")

    for ds_name, ds_cls in DATASETS.items():
        # Dataset + loader
        ds = ds_cls(root='./data', train=True, download=True, transform=TRANSFORMS[ds_name])
        num_classes = len(getattr(ds, "classes", list(range(10))))
        input_shape = ds[0][0].shape
        total_N = len(ds)

        base_loader = DataLoader(ds, batch_size=max(BATCH_SIZES), shuffle=True, num_workers=0, drop_last=True)
        it_base = iter(base_loader)
        crit = nn.CrossEntropyLoss()

        for lr in LR_VALUES:
            logLR = float(np.log(lr))
            for B in BATCH_SIZES:
                for eps in EPS_VALUES:
                    for _ in range(N_TRIALS):
                        # fresh batch
                        try:
                            Xb, yb = next(it_base)
                        except StopIteration:
                            it_base = iter(base_loader)
                            Xb, yb = next(it_base)

                        # cut to requested B (keep [C,H,W] shape)
                        n = min(B, Xb.size(0))
                        Xb = Xb[:n]; yb = yb[:n]

                        # fresh model
                        model = RNNClassifier(input_shape, num_classes)

                        # CAPE probe (on SAME batch)
                        logP, logB, logG2, logTau = extract_probe_features(model, Xb, yb, crit)

                        # Adaptive convergence on the SAME batch
                        T_star, converged, init_loss, final_loss, reason = measure_convergence_adaptive(
                            model, Xb, yb, eps=eps, lr=lr, criterion=crit,
                            soft_max_steps=SOFT_MAX_STEPS,
                            plateau_patience=PLATEAU_PATIENCE,
                            plateau_min_delta=PLATEAU_MIN_DELTA
                        )

                        records.append({
                            'dataset'      : ds_name,
                            'learning_rate': float(lr),
                            'batch_size'   : int(n),
                            'epsilon'      : float(eps),

                            # CAPE-only features (aligned)
                            'logP'         : float(logP),
                            'logB'         : float(logB),
                            'logG2'        : float(logG2),
                            'logTau'       : float(logTau),
                            'logLR'        : float(logLR),
                            'logN'         : float(np.log(total_N)),

                            # outcomes
                            'T_star'       : int(T_star),
                            'converged'    : bool(converged),
                            'censored_reason': ("" if converged else reason),
                            'logInitLoss'  : float(np.log(max(init_loss, 1e-12))),
                            'logFinalLoss' : float(np.log(max(final_loss, 1e-12))),
                        })

                        pbar.update(1)

    pbar.close()
    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV} with {len(df)} rows "
          f"({df['converged'].sum()} converged, {(~df['converged']).sum()} censored).")


if __name__ == '__main__':
    construct_meta_dataset()
