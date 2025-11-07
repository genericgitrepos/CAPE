# -*- coding: utf-8 -*-
"""
Compare meta-regressors on CAPE features (MAE + r) —

Protocol alignment:
  - Probe batch == trial batch B (logB=log(B))
  - τ uses MEAN across probe samples (paper Eq. 11 proxy)
  - Same batch used for probing and convergence
  - BatchNorm/Dropout set to eval() during single-sample probes
  - Each trial draws a fresh batch (persistent dataloader iterator)

Architectures (compact, CIFAR-friendly):
  - ViT-Tiny-32/4 (patch=4, CLS)
  - CCT-Small (conv tokenizer, mean token)
  - PiT-XS (hierarchical with token pooling)

Saves:
  - Transformer_Regression_Comparison.csv  (rows: overall, per-dataset, per-lr, per-batch)
"""

import warnings
warnings.filterwarnings('ignore')

import os
import random
import numpy as np
import pandas as pd
import contextlib
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Match the TRANSFORMER meta-dataset normalization (32x32, 3ch)
TRANSFORMS = {
    'MNIST': transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.expand(3, -1, -1)),  # 1->3ch
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
    ]),
    'FashionMNIST': transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.expand(3, -1, -1)),
        transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)),
    ]),
    'CIFAR10': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)),
    ]),
    'CIFAR100': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761)),
    ]),
}

LR_VALUES     = [0.0005, 0.001, 0.002]
BATCH_SIZES   = [32, 64, 128]
EPS_VALUES    = [0.10, 0.15, 0.20]
N_EVAL_TRIALS = 100

SOFT_MAX_STEPS    = 5000
PLATEAU_PATIENCE  = 200
PLATEAU_MIN_DELTA = 1e-4

META_CSV = '../Meta Datasets/meta_dataset_transformer.csv'
FEATURES = ['logP','logB','logG2','logTau','logLR','logN']

# -------------------
# Small Transformer family (same as generator)
# -------------------
class MLPHead(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)
        nn.init.xavier_normal_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=attn_drop)
        self.drop1 = nn.Dropout(proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLPHead(dim, mlp_ratio, drop=proj_drop)
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.drop1(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=192, patch=4, img_size=32):
        super().__init__()
        assert img_size % patch == 0
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="relu"); nn.init.zeros_(self.proj.bias)
    def forward(self, x):
        x = self.proj(x)                 # [B, D, 8, 8]
        x = x.flatten(2).transpose(1, 2) # [B, 64, D]
        return x

class ViTTiny32(nn.Module):
    def __init__(self, num_classes: int, embed_dim=192, depth=6, num_heads=3, mlp_ratio=2.0, patch=4, img_size=32, cls_token=True):
        super().__init__()
        self.cls_token = cls_token
        self.patch = PatchEmbed(3, embed_dim, patch=patch, img_size=img_size)
        n_tokens = (img_size // patch) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + (1 if cls_token else 0), embed_dim))
        if cls_token:
            self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        if cls_token: nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.xavier_normal_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self, x):
        x = self.patch(x)
        if self.cls_token:
            cls = self.cls.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = x + self.pos[:, :x.size(1), :]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        feat = x[:, 0] if self.cls_token else x.mean(dim=1)
        return self.head(feat)

class ConvTokenizer(nn.Module):
    def __init__(self, in_ch=3, embed_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),  # 32->16
            nn.BatchNorm2d(embed_dim), nn.GELU(),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    def forward(self, x):
        x = self.conv(x)                        # [B, D, 16, 16]
        x = x.flatten(2).transpose(1, 2)        # [B, 256, D]
        return x

class CCTSmall(nn.Module):
    def __init__(self, num_classes: int, embed_dim=128, depth=4, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.tok = ConvTokenizer(3, embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, 256, embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.xavier_normal_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self, x):
        x = self.tok(x)
        x = x + self.pos
        for blk in self.blocks: x = blk(x)
        x = self.norm(x)
        feat = x.mean(dim=1)
        return self.head(feat)

class TokenPool(nn.Module):
    def __init__(self, h, w):
        super().__init__(); self.h=h; self.w=w
    def forward(self, x):
        B,N,D = x.shape
        x = x.transpose(1,2).reshape(B,D,self.h,self.w)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)  # h,w -> h/2,w/2
        h2, w2 = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1,2)
        return x, h2, w2

class PiTXS(nn.Module):
    def __init__(self, num_classes: int, img_size=32, patch=4,
                 dims=(96, 192), heads=(2, 3), depths=(2, 2), mlp_ratio=2.0):
        super().__init__()
        self.patch = PatchEmbed(3, dims[0], patch=patch, img_size=img_size)  # 8x8 tokens
        self.pos1 = nn.Parameter(torch.zeros(1, 64, dims[0]))
        self.stage1 = nn.ModuleList([TransformerBlock(dims[0], heads[0], mlp_ratio) for _ in range(depths[0])])
        self.pool = TokenPool(8,8)
        self.proj = nn.Linear(dims[0], dims[1])
        self.pos2 = nn.Parameter(torch.zeros(1, 16, dims[1]))
        self.stage2 = nn.ModuleList([TransformerBlock(dims[1], heads[1], mlp_ratio) for _ in range(depths[1])])
        self.norm = nn.LayerNorm(dims[1])
        self.head = nn.Linear(dims[1], num_classes)
        nn.init.trunc_normal_(self.pos1, std=0.02); nn.init.trunc_normal_(self.pos2, std=0.02)
        nn.init.xavier_normal_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        nn.init.xavier_normal_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self, x):
        x = self.patch(x); x = x + self.pos1
        for blk in self.stage1: x = blk(x)
        x, h2, w2 = self.pool(x); x = self.proj(x); x = x + self.pos2
        for blk in self.stage2: x = blk(x)
        x = self.norm(x)
        feat = x.mean(dim=1)
        return self.head(feat)

def choose_transformer(ds_name: str, num_classes: int):
    choices = [
        ("CCT-Small",     lambda: CCTSmall(num_classes=num_classes, embed_dim=128, depth=4, num_heads=4, mlp_ratio=2.0)),
        ("ViT-Tiny-32/4", lambda: ViTTiny32(num_classes=num_classes, embed_dim=192, depth=6, num_heads=3, mlp_ratio=2.0, patch=4, img_size=32, cls_token=True)),
        ("PiT-XS",        lambda: PiTXS(num_classes=num_classes, img_size=32, patch=4, dims=(96, 192), heads=(2, 3), depths=(2, 2), mlp_ratio=2.0)),
    ]
    name, ctor = random.choice(choices)
    return name, ctor

# -------------------
# CAPE probing (BN/Dropout-safe)
# -------------------
def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def ensure_2d_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits if logits.ndim == 2 else logits.view(logits.size(0), -1)

@contextlib.contextmanager
def stabilize_eval(model: nn.Module):
    changed = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            changed.append((m, m.training)); m.eval()
    try:
        yield
    finally:
        for m, was in changed:
            m.train(was)

def extract_probe_features(model, X, y, criterion):
    model.to(DEVICE).train()
    Bp = X.size(0)  # FULL trial batch
    Xp, yp = X.to(DEVICE), y.long().to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]

    logP = float(np.log(max(_count_params(model), 1)))
    logB = float(np.log(max(Bp, 1)))
    g2_list, tau_list = [], []

    with stabilize_eval(model):
        for i in range(Bp):
            xi = Xp[i:i+1]; yi = yp[i:i+1]

            model.zero_grad(set_to_none=True)
            logits = ensure_2d_logits(model(xi))
            loss   = criterion(logits, yi)
            grads  = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False)
            gv     = torch.cat([g.reshape(-1) for g in grads if g is not None])
            g2_list.append((gv**2).sum().item())

            model.zero_grad(set_to_none=True)
            logits = ensure_2d_logits(model(xi))
            true_logit = logits[0, yi[0].item()]
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
    X, y = X.to(DEVICE), y.long().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    losses, init_loss = [], None
    with stabilize_eval(model):
        for t in range(1, SOFT_MAX_STEPS + 1):
            optimizer.zero_grad(set_to_none=True)
            logits = ensure_2d_logits(model(X))
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

with tqdm(total=len(combos), desc="Evaluating regressors across grids (Transformers)") as pbar:
    for model_name, ds_name, lr, B, eps in combos:
        meta_model = meta_models_fitted[model_name]
        ds_cls, ds_args = DATASETS[ds_name]
        ds = ds_cls(root='./data', download=True, transform=TRANSFORMS[ds_name], **ds_args)

        num_classes = 10 if ds_name in ("MNIST","FashionMNIST","CIFAR10") else 100
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

            # Sample a compact Transformer (same policy as generator)
            arch_name, ctor = choose_transformer(ds_name, num_classes)
            model_t = ctor()

            # Probe features on the SAME batch we use for convergence
            z0 = extract_probe_features(model_t, Xb, yb, criterion)  # [logP, logB, logG2, logTau]
            z  = np.concatenate([z0, [logLR, logN]], dtype=float).reshape(1, -1)

            T_pred = float(meta_model.predict(z)[0])
            T_act  = measure_convergence(model_t, Xb, yb, eps, lr, criterion)

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

df_summary.to_csv("Transformer_Regression_Comparison.csv", index=False)
print("Saved results to Transformer_Regression_Comparison.csv")
