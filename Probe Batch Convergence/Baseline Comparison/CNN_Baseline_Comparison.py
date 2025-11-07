# -*- coding: utf-8 -*-
"""
CNN evaluation (CAPE vs. NTK/Scaling/Static/Curve), aligned with the meta-dataset generator.

Key alignments:
- Architectures: VGG-lite, ResNet-mini, MobileNetV2-mini (same sampler/configs as generator)
- Preprocessing: MNIST/FashionMNIST -> Gray→3ch, Resize(32); CIFAR10/100 -> Resize(32); ToTensor() only
- Grids: lr ∈ {5e-4, 1e-3, 5e-3}, B ∈ {32, 64, 128}, eps ∈ {0.10, 0.15, 0.20}
- Probe semantics: CAPE uses FULL trial batch (logB=log(B)); logG2/logTau are MEANS across probe samples
- Convergence: AdamW on the SAME probe batch, soft cap 5000 + plateau early-exit
- Calibration: **Isotonic in STEP-SPACE** (per family×dataset with GLOBAL fallback)

Outputs:
- CNN_Baselines_Comparison.csv  (overall + per-dataset + per-lr + per-batch; calibrated in step-space)
"""

import warnings
warnings.filterwarnings('ignore')

import os
import math
import random
from itertools import product
from typing import Tuple, Dict, List
import contextlib
import copy
import hashlib

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from tqdm.auto import tqdm

# ---------- Optional regressors ----------
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from sklearn.ensemble import RandomForestRegressor  # used if XGBoost unavailable
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score

# ============================================================
# Config
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# Eval grids (match meta)
LR_VALUES          = [0.0005, 0.001, 0.005]
BATCH_SIZES        = [32, 64, 128]
EPS_VALUES         = [0.10, 0.15, 0.20]
N_TRIALS           = 100  # per (dataset, lr, B, eps)

# Convergence settings (match generator)
SOFT_MAX_STEPS     = 5000
PLATEAU_PATIENCE   = 200
PLATEAU_MIN_DELTA  = 1e-4

# Probing (CAPE) & NTK
PROBE_BATCH_NTK    = 32   # NTK uses a fresh mini-batch; CAPE uses FULL trial batch
ALPHA_NTK          = 1.0  # scale for NTK rate proxy

# Early-run steps for curve extrapolation (LCE)
EARLY_STEPS_CURVE  = 60

# Path to meta dataset (adjust if needed)
META_DATASET_PATH  = "../Meta Datasets/meta_dataset_cnn.csv"
CAL_SUMMARY_CSV    = "CNN_Baselines_Comparison.csv"

EVAL_DATASETS      = ["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"]

# ============================================================
# Utils
# ============================================================
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def ensure_2d_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return logits.unsqueeze(0)
    if logits.ndim > 2:
        return logits.view(logits.size(0), -1)
    return logits

@contextlib.contextmanager
def stabilize_probes(model: nn.Module):
    """
    Temporarily set BN to eval() for stable per-sample probing.
    IMPORTANT: Dropout remains ON to match meta semantics.
    """
    changed = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            changed.append((m, m.training))
            m.eval()
    try:
        yield
    finally:
        for m, was in changed:
            m.train(was)

def seed_from_cfg(*parts: str) -> int:
    """Stable 32-bit seed from string parts."""
    s = "|".join(map(str, parts))
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # 32-bit

class RNGScope:
    """Context to set python/numpy/torch RNGs deterministically."""
    def __init__(self, seed: int):
        self.seed = int(seed)
        self.cuda = (DEVICE.type == "cuda")
        self.state_py = None
        self.state_np = None
        self.state_th = None
        self.state_cu = None
    def __enter__(self):
        self.state_py = random.getstate()
        self.state_np = np.random.get_state()
        self.state_th = torch.random.get_rng_state()
        if self.cuda:
            self.state_cu = torch.cuda.get_rng_state()
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.cuda:
            torch.cuda.manual_seed_all(self.seed)
        return self
    def __exit__(self, *exc):
        random.setstate(self.state_py)
        np.random.set_state(self.state_np)
        torch.random.set_rng_state(self.state_th)
        if self.cuda and self.state_cu is not None:
            torch.cuda.set_rng_state(self.state_cu)

# ============================================================
# CNN families (MATCH the meta construction)
# ============================================================
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

def build_cnn_deterministic(input_shape: tuple, num_classes: int, ds_name: str,
                            lr: float, B: int, eps: float, trial_idx: int):
    """
    Deterministic version of the random builder:
      - seeds RNGs from (ds, lr, B, eps, trial)
      - samples family with fixed weights; internal repeats/width/dropout under the seed
    """
    seed = seed_from_cfg(ds_name, f"{lr:.8f}", B, f"{eps:.4f}", trial_idx)
    with RNGScope(seed):
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
            # If it happens, re-sample deterministically by nudging seed
            return build_cnn_deterministic(input_shape, num_classes, ds_name, lr, B, eps, trial_idx+12345)
        return model, meta

# ============================================================
# Probing features (CAPE-only; FULL batch; BN-safe, Dropout ON)
# ============================================================
def extract_probe_features_generic(
    model: nn.Module,
    batch_inputs: torch.Tensor,
    batch_targets: torch.Tensor,
    criterion: nn.Module,
):
    model.to(DEVICE).train()
    xb = batch_inputs.to(DEVICE)
    yb = batch_targets.to(DEVICE).long()

    Bp = xb.shape[0]                     # USE FULL TRIAL BATCH
    logP = float(np.log(count_params(model)))
    logB = float(np.log(max(Bp, 1)))

    params = [p for p in model.parameters() if p.requires_grad]
    g2_list, tau_list = [], []

    # Only BN eval here; Dropout stays on
    with stabilize_probes(model):
        for i in range(Bp):
            xi = xb[i:i+1]; yi = yb[i:i+1]

            model.zero_grad(set_to_none=True)
            logits_i = ensure_2d_logits(model(xi))
            loss_i   = criterion(logits_i, yi)
            grads    = torch.autograd.grad(loss_i, params, retain_graph=True, create_graph=False)
            gv       = torch.cat([g.reshape(-1) for g in grads if g is not None])
            g2_list.append((gv**2).sum().item())

            model.zero_grad(set_to_none=True)
            logits_i  = ensure_2d_logits(model(xi))
            true_logit= logits_i[0, yi[0].item()]
            grads_f   = torch.autograd.grad(true_logit, params, retain_graph=False, create_graph=False)
            fv        = torch.cat([g.reshape(-1) for g in grads_f if g is not None])
            tau_list.append((fv**2).sum().item())

    # MEANS across probe batch
    logG2  = float(np.log(max(np.mean(g2_list), 1e-12)))
    logTau = float(np.log(max(np.mean(tau_list), 1e-12)))
    return logP, logB, logG2, logTau

# ============================================================
# NTKL
# ============================================================
def _grad_vec_for_true_logit(model, xi, yi, params):
    model.zero_grad(set_to_none=True)
    xi = xi.unsqueeze(0).to(DEVICE)
    logits = ensure_2d_logits(model(xi))
    true_logit = logits[0, int(yi)]
    grads = torch.autograd.grad(true_logit, params, retain_graph=False, create_graph=False, allow_unused=False)
    g = torch.cat([p.contiguous().view(-1) for p in grads])
    return g.detach().cpu()

def ntk_rate_from_batch(model, Xb, yb):
    model = model.to(DEVICE).train()
    params = [p for p in model.parameters() if p.requires_grad]
    B = Xb.size(0)

    G = []
    # BN-only stabilize; Dropout stays on
    with stabilize_probes(model):
        for i in range(B):
            g = _grad_vec_for_true_logit(model, Xb[i], yb[i], params)
            G.append(g.numpy())
    G = np.stack(G, axis=0)    # [B, P]
    K = G @ G.T                # [B, B]
    evals = np.linalg.eigvalsh((K + K.T) * 0.5)
    evals = np.clip(evals, 1e-12, None)
    return float(np.mean(evals))  # simple rate proxy

def predict_Tstar_ntk_fresh(model, fresh_X, fresh_y, eps: float) -> float:
    lam = ntk_rate_from_batch(model, fresh_X, fresh_y)
    k_hat = max(1e-8, ALPHA_NTK * lam)
    T = math.log(1.0/float(eps)) / k_hat
    return float(min(SOFT_MAX_STEPS, max(1.0, T)))

# ============================================================
# CAPE meta-regressor (features -> log T*)
# ============================================================

try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

class CAPERegressor:
    def __init__(self):
        self.model = None
        self.feature_cols = ['logP','logB','logG2','logTau','logLR','logN']

    def fit(self, df: pd.DataFrame):
        X = df[self.feature_cols].values
        y = np.log(np.clip(df['T_star'].values.astype(float), 1.0, None))

        # simple shuffle split to avoid leakage
        idx = np.arange(len(df))
        np.random.default_rng(SEED).shuffle(idx)
        k = int(len(df) * 0.2)  # 20% validation
        val_idx, tr_idx = idx[:k], idx[k:]

        if HAS_XGB:
            # Put eval_metric in the constructor (works for both 1.x and 2.x)
            self.model = XGBRegressor(
                max_depth=5, n_estimators=2000, learning_rate=0.02,
                subsample=0.9, colsample_bytree=0.9,
                reg_alpha=0.0, reg_lambda=1.0,
                random_state=SEED, n_jobs=-1, tree_method="hist",
                eval_metric="rmse"
            )
            # Try classic early stopping; if not supported, fall back to callbacks or plain fit
            try:
                self.model.fit(
                    X[tr_idx], y[tr_idx],
                    eval_set=[(X[val_idx], y[val_idx])],
                    verbose=False,
                    early_stopping_rounds=50
                )
            except TypeError:
                # XGBoost >= 2.0 without early_stopping_rounds in fit
                try:
                    from xgboost import callback
                    self.model.fit(
                        X[tr_idx], y[tr_idx],
                        eval_set=[(X[val_idx], y[val_idx])],
                        callbacks=[callback.EarlyStopping(rounds=50, metric_name='rmse', data_name='validation_0')],
                        verbose=False
                    )
                except Exception:
                    # Last resort: fit without validation
                    self.model.fit(X[tr_idx], y[tr_idx])
        else:
            # Fallback if XGBoost isn't available
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(
                n_estimators=800, max_depth=None, min_samples_leaf=2,
                random_state=SEED, n_jobs=-1
            )
            self.model.fit(X[tr_idx], y[tr_idx])

    def predict_T(self, feats: Dict[str, float]) -> float:
        x = np.array([[feats[c] for c in self.feature_cols]], dtype=np.float32)
        yhat_log = float(self.model.predict(x)[0])
        return float(np.exp(yhat_log))

# ============================================================
# SL
# ============================================================
def fit_scaling_law(meta_df: pd.DataFrame):
    eps = np.clip(meta_df['epsilon'].values.astype(float), 1e-12, None)
    T   = np.clip(meta_df['T_star'].values.astype(float), 1.0, None)
    x = np.log(1.0 / eps).reshape(-1)
    y = np.log(T).reshape(-1)
    X = np.stack([x, np.ones_like(x)], axis=1)
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]  # slope, intercept
    return float(a), float(b)

def predict_steps_scaling(a: float, b: float, eps: float) -> float:
    y = a * math.log(1.0/float(eps)) + b
    T = math.exp(y)
    return float(min(SOFT_MAX_STEPS, max(1.0, T)))

def fit_static_linear(meta_df: pd.DataFrame):
    FEATURES = ['logP', 'logB', 'logLR', 'logN']  # static only (no logG2/logTau)
    X = meta_df[FEATURES].values
    y = np.log(np.clip(meta_df['T_star'].values.astype(float), 1.0, None))
    reg = LinearRegression().fit(X, y)
    return reg, FEATURES

def predict_steps_static(reg: LinearRegression, feats: Dict[str, float], feat_names: List[str]) -> float:
    x = np.array([[feats[n] for n in feat_names]], dtype=float)
    yhat_log = float(reg.predict(x)[0])
    T = math.exp(yhat_log)
    return float(min(SOFT_MAX_STEPS, max(1.0, T)))

# ============================================================
# LCE
# ============================================================
def predict_Tstar_curveexp(model_ctor, ctor_kwargs, xb, yb, eps: float, lr: float,
                           criterion: nn.Module, early_steps: int = EARLY_STEPS_CURVE) -> float:
    model = model_ctor(**ctor_kwargs).to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    xb = xb.to(DEVICE); yb = yb.long().to(DEVICE)

    losses = []
    # Train with Dropout ON (no BN/Dropout stabilizer here)
    for t in range(1, early_steps + 1):
        opt.zero_grad(set_to_none=True)
        logits = ensure_2d_logits(model(xb))
        loss = criterion(logits, yb)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    L0 = max(losses[0], 1e-12)
    xs = np.arange(1, len(losses) + 1, dtype=float)
    ys = np.log(np.clip(np.array(losses, dtype=float) / L0, 1e-12, None))
    X = np.stack([xs, np.ones_like(xs)], axis=1)
    b, a = np.linalg.lstsq(X, ys, rcond=None)[0]  # slope b ~ -k
    k = max(1e-8, -float(b))
    T = math.log(1.0/float(eps)) / k
    return float(min(SOFT_MAX_STEPS, max(1.0, T)))

# ============================================================
# Datasets (eval-style preprocessing)
# ============================================================
def get_cnn_dataset(name: str) -> Tuple[Dataset, int, Tuple[int,int,int]]:
    tfms = []
    if name in ("MNIST", "FashionMNIST"):
        tfms += [transforms.Grayscale(num_output_channels=3)]
    tfms += [transforms.Resize((32, 32)), transforms.ToTensor()]
    tfms = transforms.Compose(tfms)

    if name == 'MNIST':
        ds = datasets.MNIST(root="./data", transform=tfms, download=True, train=True); num_classes = 10
    elif name == 'FashionMNIST':
        ds = datasets.FashionMNIST(root="./data", transform=tfms, download=True, train=True); num_classes = 10
    elif name == 'CIFAR10':
        ds = datasets.CIFAR10(root="./data", transform=tfms, download=True, train=True); num_classes = 10
    elif name == 'CIFAR100':
        ds = datasets.CIFAR100(root="./data", transform=tfms, download=True, train=True); num_classes = 100
    else:
        raise ValueError(f"Unknown dataset {name}")

    input_shape = ds[0][0].shape  # (3, 32, 32)
    return ds, num_classes, input_shape

# ============================================================
# Convergence on probe batch (AdamW + plateau) -- ground truth
# ============================================================
def measure_convergence_generic(
    model: nn.Module,
    batch_inputs: torch.Tensor,
    batch_targets: torch.Tensor,
    eps: float,
    lr: float,
    criterion: nn.Module,
    soft_max_steps: int = SOFT_MAX_STEPS,
    plateau_patience: int = PLATEAU_PATIENCE,
    plateau_min_delta: float = PLATEAU_MIN_DELTA,
):
    model.to(DEVICE).train()
    xb = batch_inputs.to(DEVICE); yb = batch_targets.to(DEVICE).long()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    losses = []
    init_loss = None

    # Train with Dropout ON; BN normal training mode
    for step in range(1, soft_max_steps + 1):
        opt.zero_grad(set_to_none=True)
        logits = ensure_2d_logits(model(xb))
        loss = criterion(logits, yb)
        if step == 1:
            init_loss = float(loss.item())
        cur = float(loss.item())
        losses.append(cur)

        if cur <= eps * init_loss:
            return step, True, init_loss, cur

        loss.backward()
        opt.step()

        if step >= plateau_patience:
            window = losses[-plateau_patience:]
            rel_impr = (window[0] - window[-1]) / max(window[0], 1e-12)
            if rel_impr < plateau_min_delta:
                return step, False, init_loss, cur

    return soft_max_steps, False, init_loss, float(losses[-1])

# ============================================================
# Isotonic calibration in STEP-SPACE (per family×dataset with fallback)
# ============================================================
def fit_isotonic_grouped_steps(df_in: pd.DataFrame, pred_col: str,
                               group_col: str = "fam_ds",
                               min_group_points: int = 20,
                               calib_frac: float = 0.15,
                               seed: int = SEED):
    """
    Calibrate raw predicted steps to true steps using isotonic regression.
    Returns:
      - calibrated predictions array (steps)
      - dict of fitted (group -> IsotonicRegression) and 'GLOBAL' fallback
      - dict of (group -> (#points used))
    """
    rng = np.random.default_rng(seed)
    n = len(df_in)
    idx = np.arange(n); rng.shuffle(idx)
    k = max(5, int(n * calib_frac))
    calib_idx = idx[:k]

    y_true = df_in["T_star"].values.astype(float)
    y_pred = df_in[pred_col].values.astype(float)
    groups = df_in[group_col].astype(str).values

    # Global fallback
    iso_global = IsotonicRegression(out_of_bounds="clip")
    iso_global.fit(y_pred[calib_idx], y_true[calib_idx])

    models = {"GLOBAL": iso_global}

    # Per group
    for g in np.unique(groups):
        sel_idx = np.where(groups == g)[0]
        calib_sel = np.intersect1d(sel_idx, calib_idx)
        if len(calib_sel) >= min_group_points:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(y_pred[calib_sel], y_true[calib_sel])
            models[g] = iso

    # Apply
    y_cal = np.empty_like(y_pred)
    for g in np.unique(groups):
        sel = (groups == g)
        iso = models.get(g, models["GLOBAL"])
        y_cal[sel] = iso.transform(y_pred[sel])

    T_cal = np.clip(y_cal, 1, SOFT_MAX_STEPS)
    return T_cal, models

# ============================================================
# Metrics (slim)
# ============================================================
def metrics_slim(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    mae   = float(np.mean(np.abs(y_pred - y_true)))
    rmse  = float(np.sqrt(np.mean((y_pred - y_true)**2)))
    medae = float(np.median(np.abs(y_pred - y_true)))
    r2    = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float('nan')
    avg_actual = float(np.mean(y_true))
    avg_pred   = float(np.mean(y_pred))
    acc = 1.0 - np.abs(y_pred - y_true) / np.maximum(y_true, 1.0)
    avg_acc = float(np.mean(np.clip(acc, 0.0, 1.0)))
    return {
        "MAE": mae, "RMSE": rmse, "MedianAE": medae, "R2": r2,
        "AvgActualSteps": avg_actual, "AvgPredSteps": avg_pred, "AvgAccuracy": avg_acc
    }

# ============================================================
# Runner
# ============================================================
def run_experiments():
    # --- Load meta dataset, fit CAPE & baselines ---
    if not os.path.exists(META_DATASET_PATH):
        raise FileNotFoundError(f"Meta dataset '{META_DATASET_PATH}' not found.")
    meta_df = pd.read_csv(META_DATASET_PATH)

    cape = CAPERegressor(); cape.fit(meta_df)
    sc_a, sc_b = fit_scaling_law(meta_df)
    static_reg, STATIC_FEATS = fit_static_linear(meta_df)

    # --- Build all configs across datasets ---
    cfgs = []
    ds_cache = {}
    for ds_name in EVAL_DATASETS:
        ds, num_classes, input_shape = get_cnn_dataset(ds_name)
        N_total = len(ds)
        base_loader = DataLoader(ds, batch_size=max(BATCH_SIZES), shuffle=True, num_workers=0, drop_last=True)
        it_ds = iter(base_loader)
        ds_cache[ds_name] = (ds, num_classes, input_shape, N_total, base_loader, it_ds)

        for lr, B, eps, trial_idx in product(LR_VALUES, BATCH_SIZES, EPS_VALUES, range(N_TRIALS)):
            cfgs.append((ds_name, lr, B, eps, trial_idx))

    rows = []
    with tqdm(total=len(cfgs), desc="CNN eval") as pbar:
        for ds_name, lr, B, eps, trial_idx in cfgs:
            ds, num_classes, input_shape, N_total, base_loader, it_ds = ds_cache[ds_name]

            # fresh batch
            try:
                xb, yb = next(it_ds)
            except StopIteration:
                it_ds = iter(base_loader)
                ds_cache[ds_name] = (ds, num_classes, input_shape, N_total, base_loader, it_ds)
                xb, yb = next(it_ds)
            xb = xb[:B]; yb = yb[:B]

            # DETERMINISTIC model + loss (mirrors meta distribution)
            model, meta = build_cnn_deterministic(input_shape, num_classes, ds_name, lr, B, eps, trial_idx)
            model = model.to(DEVICE).train()
            ce = nn.CrossEntropyLoss()

            # CAPE features (FULL batch; BN-only stabilize)
            logP, logB, logG2, logTau = extract_probe_features_generic(model, xb, yb, ce)
            feats = {
                'logP': logP,
                'logB': logB,
                'logG2': logG2,
                'logTau': logTau,
                'logLR': float(np.log(lr)),
                'logN': float(np.log(N_total)),
            }

            # Fresh batch for NTK baseline
            try:
                xb_ntk, yb_ntk = next(it_ds)
            except StopIteration:
                it_ds = iter(base_loader)
                ds_cache[ds_name] = (ds, num_classes, input_shape, N_total, base_loader, it_ds)
                xb_ntk, yb_ntk = next(it_ds)
            xb_ntk = xb_ntk[:min(PROBE_BATCH_NTK, B)]
            yb_ntk = yb_ntk[:min(PROBE_BATCH_NTK, B)]

            # Predictions (RAW, step-space)
            T_pred_cape    = float(np.clip(cape.predict_T(feats), 1, SOFT_MAX_STEPS))
            model_ntk      = copy.deepcopy(model).to(DEVICE).train()
            T_pred_ntk     = predict_Tstar_ntk_fresh(model_ntk, xb_ntk, yb_ntk, eps)
            T_pred_scaling = predict_steps_scaling(sc_a, sc_b, eps)
            static_feats   = {k: feats[k] for k in STATIC_FEATS}
            T_pred_static  = predict_steps_static(static_reg, static_feats, STATIC_FEATS)

            def cnn_ctor(input_shape=input_shape, num_classes=num_classes, ds_name=ds_name, lr=lr, B=B, eps=eps, trial_idx=trial_idx):
                def _ctor():
                    m, _ = build_cnn_deterministic(input_shape, num_classes, ds_name, lr, B, eps, trial_idx)
                    return m
                return _ctor
            T_pred_curve   = predict_Tstar_curveexp(cnn_ctor(), {}, xb, yb, eps, lr, ce, early_steps=EARLY_STEPS_CURVE)

            # Ground truth on the same batch (Dropout ON; no stabilizer)
            T_star, converged, init_loss, final_loss = measure_convergence_generic(
                model, xb, yb, eps=eps, lr=lr, criterion=ce,
                soft_max_steps=SOFT_MAX_STEPS,
                plateau_patience=PLATEAU_PATIENCE,
                plateau_min_delta=PLATEAU_MIN_DELTA
            )

            rows.append({
                "dataset": ds_name,
                "lr": lr, "batch": B, "eps": eps, "trial": trial_idx,
                "cnn_family": meta.get('cnn_family',''), "depth_signature": str(meta.get('depth_signature','')),
                "downsample_type": meta.get('downsample_type',''),
                "converged": bool(converged),
                "T_star": int(T_star),
                "cape_raw": T_pred_cape,
                "ntk_raw": T_pred_ntk,
                "scaling_raw": T_pred_scaling,
                "static_raw": T_pred_static,
                "curve_raw": T_pred_curve
            })
            pbar.update(1)

    df = pd.DataFrame(rows)

    # -------- Isotonic calibration in STEP-SPACE (per family × dataset) --------
    df["fam_ds"] = df["cnn_family"].astype(str) + "|" + df["dataset"].astype(str)
    pred_map = {
        "CAPE": "cape_raw",
        "NTK (linearized)": "ntk_raw",
        "ScalingLaw": "scaling_raw",
        "StaticLinear": "static_raw",
        "CurveExp": "curve_raw",
    }
    for _, col in pred_map.items():
        T_cal, _models = fit_isotonic_grouped_steps(df, col,
                                                    group_col="fam_ds",
                                                    min_group_points=20,
                                                    calib_frac=0.15,
                                                    seed=SEED)
        df[col.replace("_raw", "_cal")] = T_cal

    # -------- Summaries: overall + per-dataset + per-lr + per-batch --------
    def metrics_block(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mae  = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
        med  = float(np.median(np.abs(y_pred - y_true)))
        r2   = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float('nan')
        avg_actual = float(np.mean(y_true))
        avg_pred   = float(np.mean(y_pred))
        acc = 1.0 - np.abs(y_pred - y_true) / np.maximum(y_true, 1.0)
        avg_acc = float(np.mean(np.clip(acc, 0.0, 1.0)))
        return {"MAE": mae, "RMSE": rmse, "MedianAE": med,
                "R2": r2, "AvgActualSteps": avg_actual,
                "AvgPredSteps": avg_pred, "AvgAccuracy": avg_acc}

    predictors = {
        "CAPE": "cape_cal",
        "NTK (linearized)": "ntk_cal",
        "ScalingLaw": "scaling_cal",
        "StaticLinear": "static_cal",
        "CurveExp": "curve_cal",
    }

    records = []

    # Overall
    y_true_all = df["T_star"].values.astype(float)
    for name, col in predictors.items():
        y_pred_all = df[col].values.astype(float)
        m = metrics_block(y_true_all, y_pred_all)
        records.append({"Model": "CNN (det+iso, step-cal)", "Group": "overall", "Predictor": name, **m})

    # Per-dataset
    for ds_name in EVAL_DATASETS:
        sub = df[df["dataset"] == ds_name]
        if len(sub) == 0:
            continue
        y_true = sub["T_star"].values.astype(float)
        for name, col in predictors.items():
            y_pred = sub[col].values.astype(float)
            m = metrics_block(y_true, y_pred)
            records.append({"Model": "CNN (det+iso, step-cal)", "Group": f"dataset={ds_name}", "Predictor": name, **m})

    # Per-learning-rate
    for lr in LR_VALUES:
        sub = df[df["lr"] == lr]
        if len(sub) == 0:
            continue
        y_true = sub["T_star"].values.astype(float)
        for name, col in predictors.items():
            y_pred = sub[col].values.astype(float)
            m = metrics_block(y_true, y_pred)
            records.append({"Model": "CNN (det+iso, step-cal)", "Group": f"lr={lr}", "Predictor": name, **m})

    # Per-batch-size
    for B in BATCH_SIZES:
        sub = df[df["batch"] == B]
        if len(sub) == 0:
            continue
        y_true = sub["T_star"].values.astype(float)
        for name, col in predictors.items():
            y_pred = sub[col].values.astype(float)
            m = metrics_block(y_true, y_pred)
            records.append({"Model": "CNN (det+iso, step-cal)", "Group": f"batch={B}", "Predictor": name, **m})

    out = pd.DataFrame(records)
    out.to_csv(CAL_SUMMARY_CSV, index=False)
    print(f"\nSaved summaries to {CAL_SUMMARY_CSV}")
    print(out.head(12).to_string(index=False))

    return df, out

# ============================================================
# Main (Windows-safe)
# ============================================================
if __name__ == "__main__":
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    _ = run_experiments()
