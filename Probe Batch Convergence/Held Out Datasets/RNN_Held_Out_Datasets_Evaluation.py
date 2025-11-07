# -*- coding: utf-8 -*-
"""
Held-out Datasets Evaluation for RNN

- Unseen datasets: SVHN, IMDB
- Predictors: CAPE meta-regressor, NTK linearization, Scaling law, Static linear, Curve extrapolation
- Ground-truth: adaptive convergence on probe batch (same batch used for probing & training)
- Outputs:
    * Per-trial predictions + truth (DataFrame returned by run_experiments)
    * Calibrated summary CSV: RNN_Held_Out_Datasets_Results.csv
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
import re
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

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------- Optional IMDB loaders ----------
_HAS_HF = False
_HAS_TORCHTEXT = False
try:
    from datasets import load_dataset as hf_load_dataset
    _HAS_HF = True
except Exception:
    _HAS_HF = False
try:
    import torchtext
    from torchtext.datasets import IMDB as TT_IMDB
    _HAS_TORCHTEXT = True
except Exception:
    _HAS_TORCHTEXT = False

# ============================================================
# Config (aligned with RNN meta-dataset generator)
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# Eval grids
LR_VALUES          = [0.0005, 0.001, 0.002]
BATCH_SIZES        = [32, 64, 128]     # align with meta dataset
EPS_VALUES         = [0.10, 0.15, 0.20]
N_TRIALS           = 100

# Convergence settings (match generator)
SOFT_MAX_STEPS     = 5000
PLATEAU_PATIENCE   = 200
PLATEAU_MIN_DELTA  = 1e-4

# ---------------- Evaluation datasets ----------------
EVAL_DATASETS      = ["IMDB","SVHN"]

# Meta dataset path + output
META_DATASET_PATH  = "../Meta Datasets/meta_dataset_rnn.csv"
CAL_SUMMARY_CSV    = "RNN_Held_Out_Datasets_Results.csv"

# ============================================================
# Utils
# ============================================================
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def steps_per_epoch_from_NB(N: int, B: int) -> int:
    return int(math.ceil(N / float(B)))

def ensure_2d_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    return logits

def linear_calibration(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size < 2 or np.allclose(x, x.mean()):
        return 1.0, 0.0
    X = np.stack([x, np.ones_like(x)], axis=1)
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(a), float(b)

@contextlib.contextmanager
def batchnorm_eval(model: nn.Module):
    # Future-proof: if BN sneaks in anywhere, freeze it for per-sample probes
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

# ============================================================
# RNN classifiers
#   - VisionRNN: scans image rows (sequence length H, feature size C*W)
#   - TextRNN:   Embedding + LSTM/GRU over token indices (padded)
# ============================================================
class VisionRNN(nn.Module):
    def __init__(self, input_shape: Tuple[int,int,int], num_classes: int,
                 cell_type: str = None, hidden_size: int = None,
                 num_layers: int = None, bidirectional: bool = None):
        super().__init__()
        C, H, W = input_shape
        input_size = C * W

        self.cell_type     = cell_type     if cell_type     is not None else random.choice(['LSTM','GRU'])
        self.hidden_size   = hidden_size   if hidden_size   is not None else random.choice([128, 256, 512])
        self.num_layers    = num_layers    if num_layers    is not None else random.randint(1, 3)
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
        out, _ = self.rnn(seq)
        last = out[:, -1, :]
        return self.fc(last)

class TextRNN(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int,
                 embed_dim: int = None, hidden_size: int = None,
                 num_layers: int = None, bidirectional: bool = None,
                 cell_type: str = None, pad_idx: int = 0, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.pad_idx = int(pad_idx)

        self.embed_dim    = embed_dim    if embed_dim    is not None else random.choice([64, 128, 256])
        self.hidden_size  = hidden_size  if hidden_size  is not None else random.choice([128, 256, 512])
        self.num_layers   = num_layers   if num_layers   is not None else random.randint(1, 3)
        self.bidirectional= bidirectional if bidirectional is not None else random.choice([False, True])
        self.cell_type    = cell_type    if cell_type    is not None else random.choice(['LSTM','GRU'])

        self.emb = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.pad_idx)
        Cell = nn.LSTM if self.cell_type == 'LSTM' else nn.GRU
        self.rnn = Cell(
            input_size=self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout if self.num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(self.hidden_size * (2 if self.bidirectional else 1), num_classes)

        nn.init.normal_(self.emb.weight, std=0.02)
        if self.pad_idx is not None:
            with torch.no_grad():
                self.emb.weight[self.pad_idx].zero_()

    def forward(self, x):  # x: [B, T] int64
        emb = self.emb(x)   # [B, T, E]
        out, _ = self.rnn(emb)
        last = out[:, -1, :]    # last time-step
        return self.fc(last)

# ============================================================
# CAPE probing features (probe batch == trial batch)
# ============================================================
def extract_probe_features_generic(
    model: nn.Module,
    batch_inputs: torch.Tensor,
    batch_targets: torch.Tensor,
    criterion: nn.Module,
) -> Tuple[float, float, float, float]:
    model.to(DEVICE).train()
    xb = batch_inputs.to(DEVICE)
    yb = batch_targets.long().to(DEVICE)

    Bp = xb.shape[0]                     # USE FULL TRIAL BATCH
    logP = float(np.log(count_params(model)))
    logB = float(np.log(max(Bp, 1)))

    params = [p for p in model.parameters() if p.requires_grad]
    g2_list, tau_list = [], []

    with batchnorm_eval(model):
        for i in range(Bp):
            xi = xb[i:i+1]; yi = yb[i:i+1]

            model.zero_grad(set_to_none=True)
            logits_i = ensure_2d_logits(model(xi))
            loss_i = criterion(logits_i, yi)
            grads = torch.autograd.grad(loss_i, params, retain_graph=True, create_graph=False)
            gv = torch.cat([g.reshape(-1) for g in grads if g is not None])
            g2_list.append((gv**2).sum().item())

            model.zero_grad(set_to_none=True)
            logits_i = ensure_2d_logits(model(xi))
            true_logit = logits_i[0, yi[0].item()]
            grads_f = torch.autograd.grad(true_logit, params, retain_graph=False, create_graph=False)
            fv = torch.cat([g.reshape(-1) for g in grads_f if g is not None])
            tau_list.append((fv**2).sum().item())

    logG2  = float(np.log(max(np.mean(g2_list), 1e-12)))
    logTau = float(np.log(max(np.mean(tau_list), 1e-12)))
    return logP, logB, logG2, logTau

# ============================================================
# NTKL
# ============================================================
ALPHA_NTK = 1.0

def _grad_vec_for_true_logit(model, xi, yi, params):
    with batchnorm_eval(model):
        model.zero_grad(set_to_none=True)
        xi = xi.unsqueeze(0).to(DEVICE)         # [1, ...]
        logits = ensure_2d_logits(model(xi))    # [1, C]
        true_logit = logits[0, int(yi.item())]
        grads = torch.autograd.grad(true_logit, params,
                                    retain_graph=False, create_graph=False, allow_unused=False)
    g = torch.cat([p.contiguous().view(-1) for p in grads])
    return g.detach().cpu()

def ntk_rate_from_batch(model, Xb, yb):
    model = model.to(DEVICE).train()
    params = [p for p in model.parameters() if p.requires_grad]
    B = Xb.size(0)

    G = []
    with batchnorm_eval(model):
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
# CAPE meta-regressor (features -> T*)
# ============================================================
class CAPERegressor:
    def __init__(self):
        self.model = None
        self.feature_cols = ['logP','logB','logG2','logTau','logLR','logN']
    def fit(self, df: pd.DataFrame):
        X = df[self.feature_cols].values
        y = np.log(np.clip(df['T_star'].values.astype(float), 1.0, None))
        if HAS_XGB:
            self.model = XGBRegressor(
                max_depth=4, n_estimators=200, subsample=0.9, colsample_bytree=0.9,
                reg_alpha=0.0, reg_lambda=1.0, learning_rate=0.05, random_state=SEED
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=300, max_depth=12, random_state=SEED, n_jobs=-1
            )
        self.model.fit(X, y)
    def predict_T(self, feats: Dict[str, float]) -> float:
        x = np.array([[feats[c] for c in self.feature_cols]], dtype=np.float32)
        yhat_log = float(self.model.predict(x)[0])
        return float(np.exp(yhat_log))

# ============================================================
# SL
# ============================================================
def fit_scaling_law(meta_df: pd.DataFrame) -> Tuple[float, float]:
    eps = np.clip(meta_df['epsilon'].values.astype(float), 1e-12, None)
    T   = np.clip(meta_df['T_star'].values.astype(float), 1.0, None)
    x = np.log(1.0 / eps).reshape(-1)
    y = np.log(T).reshape(-1)
    X = np.stack([x, np.ones_like(x)], axis=1)
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(a), float(b)

def predict_steps_scaling(a: float, b: float, eps: float) -> float:
    y = a * math.log(1.0/float(eps)) + b
    T = math.exp(y)
    return float(min(SOFT_MAX_STEPS, max(1.0, T)))

# ============================================================
# SL
# ============================================================
def fit_static_linear(meta_df: pd.DataFrame):
    FEATURES = ['logP', 'logB', 'logLR', 'logN']  # static only
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
EARLY_STEPS_CURVE = 60

def predict_Tstar_curveexp(model_ctor, ctor_kwargs, xb, yb, eps: float, lr: float,
                           criterion: nn.Module, early_steps: int = EARLY_STEPS_CURVE) -> float:
    model = model_ctor(**ctor_kwargs).to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    xb = xb.to(DEVICE); yb = yb.long().to(DEVICE)

    losses = []
    with batchnorm_eval(model):
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
# DATASETS
# ============================================================
# ---- Vision transforms
TRANSFORMS_VISION = {
    'SVHN': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])
}

def load_svhn_train():
    ds = datasets.SVHN(root="./data", split="train", download=True,
                       transform=TRANSFORMS_VISION['SVHN'])
    num_classes = 10
    x0, _ = ds[0]                  # [3,32,32]
    input_shape = tuple(x0.shape)
    return ds, input_shape, num_classes

# ---- Text (IMDB) as token-index sequences (hash-based vocab)
TOKEN_PAT = re.compile(r"[A-Za-z']+")
def basic_tokenize(s: str) -> List[str]:
    return [t.lower() for t in TOKEN_PAT.findall(s)]

def hash_token_to_idx(tok: str, dim: int) -> int:
    h = int(hashlib.md5(tok.encode('utf-8')).hexdigest(), 16)
    return (h % (dim - 2)) + 2  # reserve 0:PAD, 1:UNK

class TokenSeqDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int],
                 vocab_size: int = 20000, max_len: int = 256):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels
        self.vocab_size = int(vocab_size)
        self.max_len = int(max_len)
        self.PAD = 0
        self.UNK = 1

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        s = self.texts[idx]
        y = int(self.labels[idx])
        toks = basic_tokenize(s)
        ids = []
        for t in toks:
            try:
                j = hash_token_to_idx(t, self.vocab_size)
            except Exception:
                j = self.UNK
            ids.append(j)
        if len(ids) == 0:
            ids = [self.UNK]
        # pad / trim to fixed length
        if len(ids) >= self.max_len:
            ids = ids[:self.max_len]
        else:
            ids = ids + [self.PAD] * (self.max_len - len(ids))
        x = torch.tensor(ids, dtype=torch.long)  # [T]
        return x, y

def load_imdb_train(vocab_size: int = 20000, max_len: int = 256):
    texts, labels = [], []
    if _HAS_HF:
        ds = hf_load_dataset("imdb", split="train")
        for ex in ds:
            texts.append(ex["text"]); labels.append(1 if int(ex["label"]) == 1 else 0)
    elif _HAS_TORCHTEXT:
        for label, text in TT_IMDB(root="./data"):
            y = 1 if label == 'pos' else 0
            texts.append(text); labels.append(y)
    else:
        raise ImportError("Need HuggingFace 'datasets' or 'torchtext' available to load IMDB.")
    ds = TokenSeqDataset(texts, labels, vocab_size=vocab_size, max_len=max_len)
    num_classes = 2
    input_shape = (ds.max_len,)  # sequence length only (int indices)
    return ds, input_shape, num_classes, ds.vocab_size, 0  # pad_idx=0

# Router returning: dataset, model_ctor, ctor_kwargs, num_classes
def get_rnn_dataset_and_model(name: str):
    if name == "SVHN":
        ds, input_shape, num_classes = load_svhn_train()
        def ctor():
            cell = random.choice(['LSTM','GRU'])
            hid  = random.choice([128, 256, 512])
            layers = random.randint(1, 3)
            bi = random.choice([False, True])
            return VisionRNN(input_shape=input_shape, num_classes=num_classes,
                             cell_type=cell, hidden_size=hid, num_layers=layers,
                             bidirectional=bi)
        return ds, num_classes, ctor, "vision"
    elif name == "IMDB":
        ds, input_shape, num_classes, vocab_size, pad_idx = load_imdb_train(vocab_size=20000, max_len=256)
        def ctor():
            cell = random.choice(['LSTM','GRU'])
            emb  = random.choice([64, 128, 256])
            hid  = random.choice([128, 256, 512])
            layers = random.randint(1, 3)
            bi = random.choice([False, True])
            return TextRNN(vocab_size=vocab_size, num_classes=num_classes,
                           embed_dim=emb, hidden_size=hid, num_layers=layers,
                           bidirectional=bi, cell_type=cell, pad_idx=pad_idx)
        return ds, num_classes, ctor, "text"
    else:
        raise ValueError(f"Unknown dataset {name}")

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
) -> Tuple[int, bool, float, float]:
    model.to(DEVICE).train()
    xb = batch_inputs.to(DEVICE); yb = batch_targets.long().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    losses = []
    init_loss = None

    with batchnorm_eval(model):
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
# Runner
# ============================================================
def run_experiments():
    # --- Load meta dataset, fit CAPE & published baselines ---
    if not os.path.exists(META_DATASET_PATH):
        raise FileNotFoundError(f"Meta dataset '{META_DATASET_PATH}' not found.")
    meta_df = pd.read_csv(META_DATASET_PATH)

    cape = CAPERegressor(); cape.fit(meta_df)
    sc_a, sc_b = fit_scaling_law(meta_df)
    static_reg, STATIC_FEATS = fit_static_linear(meta_df)

    all_rows = []

    for ds_name in EVAL_DATASETS:
        # --- Dataset / loader + ctor ---
        ds, num_classes, model_ctor, ds_kind = get_rnn_dataset_and_model(ds_name)
        N_total = len(ds)
        steps_per_eps = {B: steps_per_epoch_from_NB(N_total, B) for B in BATCH_SIZES}

        # batch size will be max(BATCH_SIZES) so we can slice per trial
        vloader = DataLoader(ds, batch_size=max(BATCH_SIZES), shuffle=True, num_workers=0, drop_last=True)
        it_v = iter(vloader)
        ce = nn.CrossEntropyLoss()

        cfgs = list(product(LR_VALUES, BATCH_SIZES, EPS_VALUES, range(N_TRIALS)))

        with tqdm(total=len(cfgs), desc=f"RNN eval on {ds_name} (CAPE+NTK+Scaling+Static+Curve)") as pbar:
            for lr, B, eps, trial_idx in cfgs:
                try:
                    xb, yb = next(it_v)
                except StopIteration:
                    it_v = iter(vloader); xb, yb = next(it_v)
                xb = xb[:B]; yb = yb[:B]

                # fresh model instance
                model = model_ctor().to(DEVICE).train()

                # CAPE features (probe batch == trial batch)
                logP, logB, logG2, logTau = extract_probe_features_generic(model, xb, yb, ce)
                feats = {
                    'logP': logP,
                    'logB': logB,
                    'logG2': logG2,
                    'logTau': logTau,
                    'logLR': float(np.log(lr)),
                    'logN': float(np.log(N_total)),
                }

                # --- Fresh batch for NTK baseline
                try:
                    xb_ntk, yb_ntk = next(it_v)
                except StopIteration:
                    it_v = iter(vloader); xb_ntk, yb_ntk = next(it_v)
                xb_ntk = xb_ntk[:B]; yb_ntk = yb_ntk[:B]

                # --- Predictions ---
                T_pred_cape = float(np.clip(cape.predict_T(feats), 1, SOFT_MAX_STEPS))

                model_ntk = copy.deepcopy(model).to(DEVICE).train()
                T_pred_ntk = predict_Tstar_ntk_fresh(model_ntk, xb_ntk, yb_ntk, eps)

                T_pred_scaling = predict_steps_scaling(sc_a, sc_b, eps)

                static_feats = {k: feats[k] for k in ['logP','logB','logLR','logN']}
                T_pred_static = predict_steps_static(static_reg, static_feats, STATIC_FEATS)

                # Curve extrapolation uses a *fresh* model (to avoid contamination)
                T_pred_curve = predict_Tstar_curveexp(model_ctor, {}, xb, yb, eps, lr, ce,
                                                      early_steps=EARLY_STEPS_CURVE)

                # Ground truth on the same batch (fresh model again)
                model_gt = model_ctor().to(DEVICE).train()
                T_star, converged, init_loss, final_loss = measure_convergence_generic(
                    model_gt, xb, yb, eps=eps, lr=lr, criterion=ce,
                    soft_max_steps=SOFT_MAX_STEPS,
                    plateau_patience=PLATEAU_PATIENCE,
                    plateau_min_delta=PLATEAU_MIN_DELTA
                )

                spe = steps_per_eps[B]
                all_rows.append({
                    "dataset": ds_name,
                    "lr": lr, "batch": B, "eps": eps, "trial": trial_idx,
                    "steps_per_epoch": spe, "converged": bool(converged),
                    "T_star": int(T_star),
                    "cape_raw": T_pred_cape,
                    "ntk_raw": T_pred_ntk,
                    "scaling_raw": T_pred_scaling,
                    "static_raw": T_pred_static,
                    "curve_raw": T_pred_curve
                })
                pbar.update(1)

    df = pd.DataFrame(all_rows)

    # -------- Epoch-space calibration (global) --------
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(df)); rng.shuffle(idx)
    k = max(5, int(len(df) * 0.10))  # 10% for simple linear calibration
    calib_idx = idx[:k]

    def calibrate_epochs(pred_steps_col: str, df_in: pd.DataFrame):
        e_true = df_in["T_star"].values / df_in["steps_per_epoch"].values
        e_pred = df_in[pred_steps_col].values / df_in["steps_per_epoch"].values
        a, b = linear_calibration(e_pred[calib_idx], e_true[calib_idx])
        e_cal = a * e_pred + b
        T_cal = np.clip(e_cal * df_in["steps_per_epoch"].values, 1, SOFT_MAX_STEPS)
        return T_cal, (a, b)

    df["cape_cal"],     ab_cape   = calibrate_epochs("cape_raw",    df)
    df["ntk_cal"],      ab_ntk    = calibrate_epochs("ntk_raw",     df)
    df["scaling_cal"],  ab_scal   = calibrate_epochs("scaling_raw", df)
    df["static_cal"],   ab_static = calibrate_epochs("static_raw",  df)
    df["curve_cal"],    ab_curve  = calibrate_epochs("curve_raw",   df)

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
        records.append({"Model": "RNN", "Group": "overall", "Predictor": name, **m})

    # Per-dataset
    for ds_name in EVAL_DATASETS:
        sub = df[df["dataset"] == ds_name]
        if len(sub) == 0:
            continue
        y_true = sub["T_star"].values.astype(float)
        for name, col in predictors.items():
            y_pred = sub[col].values.astype(float)
            m = metrics_block(y_true, y_pred)
            records.append({"Model": "RNN", "Group": f"dataset={ds_name}", "Predictor": name, **m})

    # Per-learning-rate
    for lr in LR_VALUES:
        sub = df[df["lr"] == lr]
        if len(sub) == 0:
            continue
        y_true = sub["T_star"].values.astype(float)
        for name, col in predictors.items():
            y_pred = sub[col].values.astype(float)
            m = metrics_block(y_true, y_pred)
            records.append({"Model": "RNN", "Group": f"lr={lr}", "Predictor": name, **m})

    # Per-batch-size
    for B in BATCH_SIZES:
        sub = df[df["batch"] == B]
        if len(sub) == 0:
            continue
        y_true = sub["T_star"].values.astype(float)
        for name, col in predictors.items():
            y_pred = sub[col].values.astype(float)
            m = metrics_block(y_true, y_pred)
            records.append({"Model": "RNN", "Group": f"batch={B}", "Predictor": name, **m})

    out = pd.DataFrame(records)
    out.to_csv(CAL_SUMMARY_CSV, index=False)
    print(f"\nSaved summaries to: {CAL_SUMMARY_CSV}")
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
