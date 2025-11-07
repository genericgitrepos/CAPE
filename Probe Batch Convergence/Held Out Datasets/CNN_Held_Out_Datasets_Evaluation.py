# -*- coding: utf-8 -*-
"""
Held-out Datasets evaluation for CNNs (CAPE + NTK + Scaling + Static + CurveExp)

Datasets (match your MLP eval): SVHN, IMDB
- IMDB is hashed to a 3072-dim vector and reshaped to 3x32x32 for CNNs.

CNN sampler/semantics match your CNN meta generator:
  * VGG-lite / ResNet-mini / MobileNetV2-mini (<=~2M params; resample if exceeded)
  * CAPE: probe batch == trial batch (logB = log(B)); logG2/logTau are means
  * Convergence: AdamW, SOFT_MAX_STEPS=5000, plateau early-exit

Reads:  meta_dataset_cnn.csv
Writes: CNN_Held_Out_Datasets_Results.csv
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

# ============================================================
# Config (aligned with CNN meta generator)
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

LR_VALUES          = [0.0005, 0.001, 0.005]
BATCH_SIZES        = [32, 64, 128]
EPS_VALUES         = [0.10, 0.15, 0.20]
N_TRIALS           = 100

SOFT_MAX_STEPS     = 5000
PLATEAU_PATIENCE   = 200
PLATEAU_MIN_DELTA  = 1e-4

EVAL_DATASETS      = ["SVHN", "IMDB"]

META_DATASET_PATH  = "../Meta Datasets/meta_dataset_cnn.csv"
CAL_SUMMARY_CSV    = "CNN_Held_Out_Datasets_Results.csv"

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

@contextlib.contextmanager
def batchnorm_eval(model: nn.Module):
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
# CNN building blocks (match meta generator)
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

def build_random_cnn(input_shape: tuple, num_classes: int, ds_name: str):
    family = random.choices(['vgg_lite','resnet_mini','mobilenetv2_mini'],
                            weights=[0.5,0.35,0.15], k=1)[0]
    if family == 'vgg_lite':
        model, meta = build_vgg_lite(input_shape, num_classes, ds_name)
    elif family == 'resnet_mini':
        model, meta = build_resnet_mini(input_shape, num_classes, ds_name)
    else:
        model, meta = build_mobilenetv2_mini(input_shape, num_classes, ds_name)
    P = count_params(model)
    if P > 2_000_000:
        return build_random_cnn(input_shape, num_classes, ds_name)
    return model, meta

# ============================================================
# Probing features (CAPE-only)
# ============================================================
def extract_probe_features_cnn(
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    criterion: nn.Module,
) -> Dict[str, float]:
    model.to(DEVICE).train()
    xb = xb.to(DEVICE); yb = yb.long().to(DEVICE)

    logP = float(np.log(count_params(model)))
    logB = float(np.log(max(int(xb.size(0)), 1)))

    params = [p for p in model.parameters() if p.requires_grad]
    g2_list, tau_list = [], []

    with batchnorm_eval(model):
        for i in range(xb.size(0)):
            xi = xb[i:i+1]; yi = yb[i:i+1]

            model.zero_grad(set_to_none=True)
            logits = ensure_2d_logits(model(xi))
            loss = criterion(logits, yi)
            grads = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False, allow_unused=False)
            gv = torch.cat([g.reshape(-1) for g in grads if g is not None])
            g2_list.append(float((gv**2).sum().item()))

            model.zero_grad(set_to_none=True)
            logits = ensure_2d_logits(model(xi))
            true_logit = logits[0, yi.item()]
            grads_f = torch.autograd.grad(true_logit, params, retain_graph=False, create_graph=False, allow_unused=False)
            fv = torch.cat([g.reshape(-1) for g in grads_f if g is not None])
            tau_list.append(float((fv**2).sum().item()))

    logG2  = float(np.log(max(np.mean(g2_list), 1e-12)))
    logTau = float(np.log(max(np.mean(tau_list), 1e-12)))
    return {'logP': logP, 'logB': logB, 'logG2': logG2, 'logTau': logTau}

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
        grads = torch.autograd.grad(true_logit, params, retain_graph=False, create_graph=False, allow_unused=False)
    g = torch.cat([p.contiguous().view(-1) for p in grads])
    return g.detach().cpu()

def ntk_rate_from_batch(model, Xb, Yb):
    model = model.to(DEVICE).train()
    params = [p for p in model.parameters() if p.requires_grad]
    B = Xb.size(0)
    G = []
    with batchnorm_eval(model):
        for i in range(B):
            g = _grad_vec_for_true_logit(model, Xb[i], Yb[i], params)
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
def fit_scaling_law(meta_df: pd.DataFrame):
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
    b, a = np.linalg.lstsq(X, ys, rcond=None)[0]  # slope b ~ -k
    k = max(1e-8, -float(b))
    T = math.log(1.0/float(eps)) / k
    return float(min(SOFT_MAX_STEPS, max(1.0, T)))

# ============================================================
# DATASETS (SVHN, IMDB)
# ============================================================
# Vision transforms
TRANSFORMS_VISION = {
    'SVHN': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ]),
}

# ---- Text (IMDB) via hashing bag-of-words -> reshape to 3x32x32 for CNN
TOKEN_PAT = re.compile(r"[A-Za-z']+")

def basic_tokenize(s: str) -> List[str]:
    return [t.lower() for t in TOKEN_PAT.findall(s)]

def hash_token_to_idx(tok: str, dim: int) -> int:
    h = int(hashlib.md5(tok.encode('utf-8')).hexdigest(), 16)
    return h % dim

class HashedBoWDatasetCNN(Dataset):
    """
    Hash BoW to 3072 dims (=3*32*32); reshape to [3,32,32] for CNNs.
    Labels: 0/1 (binary sentiment).
    """
    def __init__(self, texts: List[str], labels: List[int], dim: int = 3072, binary: bool = False):
        assert dim == 3*32*32, "dim must be 3072 for 3x32x32 reshaping"
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
        x = torch.from_numpy(vec).view(3, 32, 32)   # [3,32,32]
        return x, y

# IMDB loaders via HF or torchtext
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

def load_svhn_train() -> Tuple[Dataset, int, int, str]:
    ds = datasets.SVHN(root="./data", split="train", download=True,
                       transform=TRANSFORMS_VISION['SVHN'])
    num_classes = 10
    return ds, 3, num_classes, "SVHN"

def load_imdb_train_cnn() -> Tuple[Dataset, int, int, str]:
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
        raise ImportError("Need HuggingFace 'datasets' or 'torchtext' to load IMDB.")
    ds = HashedBoWDatasetCNN(texts, labels, dim=3*32*32, binary=False)
    num_classes = 2
    return ds, 3, num_classes, "IMDB"

def get_cnn_dataset(name: str) -> Tuple[Dataset, int, int, str]:
    if name == "SVHN":
        return load_svhn_train()
    elif name == "IMDB":
        return load_imdb_train_cnn()
    else:
        raise ValueError(f"Unknown dataset {name}")

# ============================================================
# Convergence on probe batch (AdamW + plateau) -- ground truth
# ============================================================
def measure_convergence_cnn(
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    eps: float,
    lr: float,
    criterion: nn.Module,
    soft_max_steps: int = SOFT_MAX_STEPS,
    plateau_patience: int = PLATEAU_PATIENCE,
    plateau_min_delta: float = PLATEAU_MIN_DELTA,
):
    model.to(DEVICE).train()
    xb = xb.to(DEVICE); yb = yb.long().to(DEVICE)
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
    # Load meta dataset, fit CAPE & baselines
    if not os.path.exists(META_DATASET_PATH):
        raise FileNotFoundError(f"Meta dataset '{META_DATASET_PATH}' not found.")
    meta_df = pd.read_csv(META_DATASET_PATH)

    cape = CAPERegressor(); cape.fit(meta_df)
    sc_a, sc_b = fit_scaling_law(meta_df)
    static_reg, STATIC_FEATS = fit_static_linear(meta_df)

    all_rows = []

    for ds_name in EVAL_DATASETS:
        ds, in_ch, num_classes, ds_tag = get_cnn_dataset(ds_name)
        N_total = len(ds)
        steps_per_eps = {B: steps_per_epoch_from_NB(N_total, B) for B in BATCH_SIZES}

        vloader = DataLoader(ds, batch_size=max(BATCH_SIZES), shuffle=True, num_workers=0, drop_last=True)
        it_v = iter(vloader)
        ce = nn.CrossEntropyLoss()

        cfgs = list(product(LR_VALUES, BATCH_SIZES, EPS_VALUES, range(N_TRIALS)))

        with tqdm(total=len(cfgs), desc=f"CNN eval on {ds_name} (CAPE+NTK+Scaling+Static+Curve)") as pbar:
            for lr, B, eps, trial_idx in cfgs:
                try:
                    xb, yb = next(it_v)
                except StopIteration:
                    it_v = iter(vloader); xb, yb = next(it_v)
                xb = xb[:B]; yb = yb[:B]

                input_shape = xb.shape[1:]  # [C,H,W] (3,32,32)
                model, _meta = build_random_cnn(input_shape, num_classes, ds_tag)

                # CAPE features
                feats_probe = extract_probe_features_cnn(model, xb, yb, ce)
                feats = {
                    **feats_probe,
                    'logLR': float(np.log(lr)),
                    'logN' : float(np.log(N_total)),
                }

                # Fresh batch for NTK
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

                model_ctor = lambda **kw: build_random_cnn(**kw)[0]
                ctor_kwargs = dict(input_shape=input_shape, num_classes=num_classes, ds_name=ds_tag)
                T_pred_curve = predict_Tstar_curveexp(model_ctor, ctor_kwargs, xb, yb, eps, lr, ce,
                                                      early_steps=EARLY_STEPS_CURVE)

                # Ground truth on same batch
                T_star, converged, init_loss, final_loss = measure_convergence_cnn(
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

    # -------- Epoch-space calibration (global 10%) --------
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(df)); rng.shuffle(idx)
    k = max(5, int(len(df) * 0.10))
    calib_idx = idx[:k]

    def linear_calibration(x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.size < 2 or np.allclose(x, x.mean()):
            return 1.0, 0.0
        X = np.stack([x, np.ones_like(x)], axis=1)
        a, b = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(a), float(b)

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

    # -------- Summaries --------
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
        records.append({"Model": "CNN", "Group": "overall", "Predictor": name, **m})

    # Per-dataset
    for ds_name in EVAL_DATASETS:
        sub = df[df["dataset"] == ds_name]
        if len(sub) == 0:
            continue
        y_true = sub["T_star"].values.astype(float)
        for name, col in predictors.items():
            y_pred = sub[col].values.astype(float)
            m = metrics_block(y_true, y_pred)
            records.append({"Model": "CNN", "Group": f"dataset={ds_name}", "Predictor": name, **m})

    # Per-learning-rate
    for lr in LR_VALUES:
        sub = df[df["lr"] == lr]
        if len(sub) == 0:
            continue
        y_true = sub["T_star"].values.astype(float)
        for name, col in predictors.items():
            y_pred = sub[col].values.astype(float)
            m = metrics_block(y_true, y_pred)
            records.append({"Model": "CNN", "Group": f"lr={lr}", "Predictor": name, **m})

    # Per-batch-size
    for B in BATCH_SIZES:
        sub = df[df["batch"] == B]
        if len(sub) == 0:
            continue
        y_true = sub["T_star"].values.astype(float)
        for name, col in predictors.items():
            y_pred = sub[col].values.astype(float)
            m = metrics_block(y_true, y_pred)
            records.append({"Model": "CNN", "Group": f"batch={B}", "Predictor": name, **m})

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
