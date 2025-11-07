import warnings
warnings.filterwarnings('ignore')

import os
import math
import random
from itertools import product
from typing import Tuple, Dict, List, Optional
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

# --- Optional: HuggingFace / torchtext for text datasets ---
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

# ============================================================
# Config (aligned with MLP meta-dataset generator)
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
BATCH_SIZES        = [32, 64, 128]
EPS_VALUES         = [0.10, 0.15, 0.20]
N_TRIALS           = 100

# Convergence settings
SOFT_MAX_STEPS     = 5000
PLATEAU_PATIENCE   = 200
PLATEAU_MIN_DELTA  = 1e-4

# ---------------- Evaluation datasets ----------------
EVAL_DATASETS      = ["SVHN", "IMDB"]

# Meta dataset path + output
META_DATASET_PATH  = "../Meta Datasets/meta_dataset_mlp.csv"
CAL_SUMMARY_CSV    = "MLP_Held_Out_Datasets_Results.csv"

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
    bns = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            bns.append((m, m.training))
            m.eval()
    try:
        yield
    finally:
        for m, was in bns:
            m.train(was)

# ============================================================
# MLP (matches generator)
# ============================================================
class MLPResearch(nn.Module):
    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dim: int, depth: int,
                 dropout_p: float = 0.2, use_bn: bool = True, activation: str = "gelu"):
        super().__init__()
        act = nn.GELU() if activation == "gelu" else nn.ReLU()

        layers = [nn.Flatten()]
        dims = [input_dim] + [hidden_dim]*depth + [num_classes]
        for i in range(len(dims) - 2):
            d_in, d_out = dims[i], dims[i+1]
            block = [nn.Linear(d_in, d_out)]
            if use_bn:
                block.append(nn.BatchNorm1d(d_out))
            block.append(act)
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
    if ds_name in ("CelebA",):
        hidden_choices = [512, 1024, 1536]
        depth_choices  = [2, 3, 4]
    else:
        hidden_choices = [256, 512, 1024]
        depth_choices  = [2, 3]
    return random.choice(hidden_choices), random.choice(depth_choices)

# ============================================================
# Probing features (CAPE-only)
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

    Bp = xb.shape[0]
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
            fv = torch.cat([g.reshape(-1) for g in grads if g is not None])
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
        xi = xi.unsqueeze(0).to(DEVICE)
        logits = ensure_2d_logits(model(xi))
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

    G = np.stack(G, axis=0)
    K = G @ G.T
    evals = np.linalg.eigvalsh((K + K.T) * 0.5)
    evals = np.clip(evals, 1e-12, None)
    return float(np.mean(evals))

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
    FEATURES = ['logP', 'logB', 'logLR', 'logN']
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
    b, a = np.linalg.lstsq(X, ys, rcond=None)[0]
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
    ]),
    'CelebA': transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
}

def load_svhn_train() -> Tuple[Dataset, int, int]:
    ds = datasets.SVHN(root="./data", split="train", download=True,
                       transform=TRANSFORMS_VISION['SVHN'])
    num_classes = 10
    x0, _ = ds[0]
    input_dim = int(np.prod(x0.shape))
    return ds, input_dim, num_classes

class CelebASingleAttr(Dataset):
    def __init__(self, base: datasets.CelebA, attr_name: str = "Smiling"):
        self.base = base
        if hasattr(base, "attr_names") and attr_name in base.attr_names:
            self.attr_idx = base.attr_names.index(attr_name)
        else:
            self.attr_idx = 31
        self.num_classes = 2

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, attrs = self.base[idx]
        y = int((int(attrs[self.attr_idx]) + 1) // 2)
        return img, y

def load_celeba_train(attr_name: str = "Smiling") -> Tuple[Dataset, int, int]:
    base = datasets.CelebA(
        root="./data", split="train", download=True,
        transform=TRANSFORMS_VISION['CelebA'], target_type="attr"
    )
    ds = CelebASingleAttr(base, attr_name=attr_name)
    x0, _ = ds[0]
    input_dim = int(np.prod(x0.shape))
    num_classes = ds.num_classes
    return ds, input_dim, num_classes

# ---- Text (IMDB) via hashing bag-of-words
TOKEN_PAT = re.compile(r"[A-Za-z']+")

def basic_tokenize(s: str) -> List[str]:
    return [t.lower() for t in TOKEN_PAT.findall(s)]

def hash_token_to_idx(tok: str, dim: int) -> int:
    h = int(hashlib.md5(tok.encode('utf-8')).hexdigest(), 16)
    return h % dim

class HashedBoWDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], dim: int = 8192, binary: bool = False):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels
        self.dim = int(dim)
        self.binary = bool(binary)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        s = self.texts[idx]
        y = int(self.labels[idx])
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in basic_tokenize(s):
            j = hash_token_to_idx(tok, self.dim)
            if self.binary:
                vec[j] = 1.0
            else:
                vec[j] += 1.0
        norm = float(np.linalg.norm(vec)) + 1e-8
        vec /= norm
        x = torch.from_numpy(vec)
        return x, y

def load_imdb_train(dim: int = 8192) -> Tuple[Dataset, int, int]:
    texts, labels = [], []
    if _HAS_HF:
        ds = hf_load_dataset("imdb", split="train")
        for ex in ds:
            texts.append(ex["text"]); labels.append(1 if ex["label"] == 1 else 0)
    elif _HAS_TORCHTEXT:
        for label, text in TT_IMDB(root="./data"):
            y = 1 if label == 'pos' else 0
            texts.append(text); labels.append(y)
    else:
        raise ImportError("Neither HuggingFace 'datasets' nor 'torchtext' available to load IMDB.")
    ds = HashedBoWDataset(texts, labels, dim=dim, binary=False)
    num_classes = 2
    input_dim = dim
    return ds, input_dim, num_classes

# ============================================================
# Dataset router
# ============================================================
def get_mlp_dataset(name: str) -> Tuple[Dataset, int, int]:
    if name == "SVHN":
        return load_svhn_train()
    elif name == "CelebA":
        return load_celeba_train(attr_name="Smiling")
    elif name == "IMDB":
        return load_imdb_train(dim=8192)
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
# Runner (evaluates datasets; writes overall + per-dataset + per-lr + per-batch summaries)
# ============================================================
def run_experiments():
    if not os.path.exists(META_DATASET_PATH):
        raise FileNotFoundError(f"Meta dataset '{META_DATASET_PATH}' not found.")
    meta_df = pd.read_csv(META_DATASET_PATH)

    cape = CAPERegressor(); cape.fit(meta_df)
    sc_a, sc_b = fit_scaling_law(meta_df)
    static_reg, STATIC_FEATS = fit_static_linear(meta_df)

    all_rows = []

    for ds_name in EVAL_DATASETS:
        ds, input_dim, num_classes = get_mlp_dataset(ds_name)
        N_total = len(ds)
        steps_per_eps = {B: steps_per_epoch_from_NB(N_total, B) for B in BATCH_SIZES}

        vloader = DataLoader(ds, batch_size=max(BATCH_SIZES), shuffle=True, num_workers=0, drop_last=True)
        it_v = iter(vloader)
        ce = nn.CrossEntropyLoss()

        cfgs = list(product(LR_VALUES, BATCH_SIZES, EPS_VALUES, range(N_TRIALS)))

        with tqdm(total=len(cfgs), desc=f"MLP eval on {ds_name} (CAPE+NTK+Scaling+Static+Curve)") as pbar:
            for lr, B, eps, trial_idx in cfgs:
                try:
                    xb, yb = next(it_v)
                except StopIteration:
                    it_v = iter(vloader); xb, yb = next(it_v)
                xb = xb[:B]; yb = yb[:B]

                hidden_dim, depth = choose_dims(ds_name)
                model = MLPResearch(input_dim=input_dim, num_classes=num_classes,
                                    hidden_dim=hidden_dim, depth=depth,
                                    dropout_p=0.2, use_bn=True, activation="gelu").to(DEVICE).train()

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

                # Fresh batch for NTK baseline
                try:
                    xb_ntk, yb_ntk = next(it_v)
                except StopIteration:
                    it_v = iter(vloader); xb_ntk, yb_ntk = next(it_v)
                xb_ntk = xb_ntk[:B]; yb_ntk = yb_ntk[:B]

                # Predictions
                T_pred_cape = float(np.clip(cape.predict_T(feats), 1, SOFT_MAX_STEPS))

                model_ntk = copy.deepcopy(model).to(DEVICE).train()
                T_pred_ntk = predict_Tstar_ntk_fresh(model_ntk, xb_ntk, yb_ntk, eps)

                T_pred_scaling = predict_steps_scaling(sc_a, sc_b, eps)

                static_feats = {k: feats[k] for k in ['logP','logB','logLR','logN']}
                T_pred_static = predict_steps_static(static_reg, static_feats, STATIC_FEATS)

                model_ctor = MLPResearch
                ctor_kwargs = dict(input_dim=input_dim, num_classes=num_classes,
                                   hidden_dim=hidden_dim, depth=depth,
                                   dropout_p=0.2, use_bn=True, activation="gelu")
                T_pred_curve = predict_Tstar_curveexp(model_ctor, ctor_kwargs, xb, yb, eps, lr, ce,
                                                      early_steps=EARLY_STEPS_CURVE)

                # Ground truth on the same batch
                T_star, converged, init_loss, final_loss = measure_convergence_generic(
                    model, xb, yb, eps=eps, lr=lr, criterion=ce,
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
    k = max(5, int(len(df) * 0.10))
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
        records.append({"Model": "MLP", "Group": "overall", "Predictor": name, **m})

    # Per-dataset
    for ds_name in EVAL_DATASETS:
        sub = df[df["dataset"] == ds_name]
        if len(sub) == 0:
            continue
        y_true = sub["T_star"].values.astype(float)
        for name, col in predictors.items():
            y_pred = sub[col].values.astype(float)
            m = metrics_block(y_true, y_pred)
            records.append({"Model": "MLP", "Group": f"dataset={ds_name}", "Predictor": name, **m})

    # Per-learning-rate
    for lr in LR_VALUES:
        sub = df[df["lr"] == lr]
        if len(sub) == 0:
            continue
        y_true = sub["T_star"].values.astype(float)
        for name, col in predictors.items():
            y_pred = sub[col].values.astype(float)
            m = metrics_block(y_true, y_pred)
            records.append({"Model": "MLP", "Group": f"lr={lr}", "Predictor": name, **m})

    # Per-batch-size
    for B in BATCH_SIZES:
        sub = df[df["batch"] == B]
        if len(sub) == 0:
            continue
        y_true = sub["T_star"].values.astype(float)
        for name, col in predictors.items():
            y_pred = sub[col].values.astype(float)
            m = metrics_block(y_true, y_pred)
            records.append({"Model": "MLP", "Group": f"batch={B}", "Predictor": name, **m})

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
