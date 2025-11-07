# -*- coding: utf-8 -*-
"""
CAPE Transformer Evaluation — MAE-only comparison by dataset
Outputs:
  - Transformer_Ablation.csv  (pivot: dataset × feature_set -> MAE)

Protocol alignment:
  - Probe batch == trial batch B (logB=log(B))
  - Same batch used for probing and convergence
  - BatchNorm & Dropout set to eval() only during single-sample probes
  - Each trial draws a fresh batch (persistent dataloader iterator)
"""

import warnings
warnings.filterwarnings('ignore')

import os
import random
import numpy as np
import pandas as pd
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

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
    'CIFAR100':     (datasets.CIFAR100,     {'train': True})
}


TRANSFORMS = {
    'MNIST': transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.expand(3, -1, -1)),  # 1->3ch
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
    ]),
    'FashionMNIST': transforms.Compose([
        transforms.Resize((32, 32)),
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

# Evaluation grid
LR_VALUES     = [0.0005, 0.001, 0.002]
BATCH_SIZES   = [32, 64, 128]
EPS_VALUES    = [0.10, 0.15, 0.20]
N_EVAL_TRIALS = 100

# Convergence
SOFT_MAX_STEPS    = 5000
PLATEAU_PATIENCE  = 200
PLATEAU_MIN_DELTA = 1e-4

META_CSV = '../Meta Datasets/meta_dataset_transformer.csv'

# CAPE-only features
ALL_FEATURES = ['logP','logB','logG2','logTau','logLR','logN']

# Full feature-set ablations
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

# -------------------
# Compact Transformer family (same as generator)
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
        if cls_token: self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
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
        for blk in self.blocks: x = blk(x)
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
        x = self.tok(x); x = x + self.pos
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
        x, _, _ = self.pool(x); x = self.proj(x); x = x + self.pos2
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
# Helpers (BN/Dropout-safe probing + logits handling)
# -------------------
def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _ensure_2d_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits if logits.ndim == 2 else logits.view(logits.size(0), -1)

@contextlib.contextmanager
def stabilize_probes(model: nn.Module):
    """Set BN/Dropout to eval() during probe to stabilize per-sample gradients."""
    changed = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            changed.append((m, m.training))
            m.eval()
    try:
        yield
    finally:
        for m, was_training in changed:
            m.train(was_training)

# -------------------
# Probing (CAPE features) — probe B == actual B
# -------------------
def extract_probe_features_dict(model, X, Y, criterion) -> dict:
    model.to(DEVICE).train()
    X = X.to(DEVICE); Y = Y.to(DEVICE).long()

    Bp   = int(X.size(0))                 # USE FULL TRIAL BATCH
    logP = float(np.log(max(_count_params(model), 1)))
    logB = float(np.log(max(Bp, 1)))

    params = [p for p in model.parameters() if p.requires_grad]
    g2_list, tau_list = [], []
    with stabilize_probes(model):
        for i in range(Bp):
            xi = X[i:i+1]; yi = Y[i:i+1]

            model.zero_grad(set_to_none=True)
            logits = _ensure_2d_logits(model(xi))
            loss   = criterion(logits, yi)
            grads  = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False)
            gv     = torch.cat([g.reshape(-1) for g in grads if g is not None])
            g2_list.append((gv**2).sum().item())

            model.zero_grad(set_to_none=True)
            logits = _ensure_2d_logits(model(xi))
            true_logit = logits[0, yi[0].item()]
            grads_f    = torch.autograd.grad(true_logit, params, retain_graph=False, create_graph=False)
            fv         = torch.cat([g.reshape(-1) for g in grads_f if g is not None])
            tau_list.append((fv**2).sum().item())

    # Paper-correct: BOTH are means across the probe batch
    logG2  = float(np.log(max(np.mean(g2_list), 1e-12)))
    logTau = float(np.log(max(np.mean(tau_list), 1e-12)))
    return {'logP': logP, 'logB': logB, 'logG2': logG2, 'logTau': logTau}

# -------------------
# Convergence measurement (align with generator)
# -------------------
def measure_convergence(model, X, Y, eps, lr, criterion):
    model.to(DEVICE).train()  # training uses normal train() (BN updates enabled)
    X = X.to(DEVICE); Y = Y.to(DEVICE).long()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    losses, init_loss = [], None
    for t in range(1, SOFT_MAX_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = _ensure_2d_logits(model(X))
        loss   = criterion(logits, Y)
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
                return t  # plateau reached
    return SOFT_MAX_STEPS

# -------------------
# Meta-regressors (XGBoost) from Transformer meta CSV
# -------------------
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

# Train regressors for all feature sets
meta_models = {name: fit_meta(flist) for name, flist in FEATURE_SETS.items()}

# -------------------
# Evaluation loop — MAE only (single global tqdm)
# -------------------
rows = []
feature_names = list(FEATURE_SETS.keys())
datasets_list = list(DATASETS.items())
total_combos = len(feature_names) * len(datasets_list) * len(LR_VALUES) * len(BATCH_SIZES) * len(EPS_VALUES)

with tqdm(total=total_combos, desc="Evaluating MAE across grids (Transformers)") as pbar:
    for set_name in feature_names:
        flist = FEATURE_SETS[set_name]
        model_reg = meta_models[set_name]

        for ds_name, (ds_cls, ds_args) in datasets_list:
            ds = ds_cls(root='./data', download=True,
                        transform=TRANSFORMS[ds_name], **ds_args)
            num_classes = 10 if ds_name in ("MNIST","FashionMNIST","CIFAR10") else 100
            total_N     = len(ds)
            logN        = float(np.log(total_N))
            criterion   = nn.CrossEntropyLoss()

            for lr in LR_VALUES:
                for B in BATCH_SIZES:
                    loader = DataLoader(ds, batch_size=B, shuffle=True, drop_last=True, num_workers=0)
                    it_loader = iter(loader)  # persistent iterator for fresh batches
                    for eps in EPS_VALUES:
                        y_preds, y_trues = [], []
                        for _ in range(N_EVAL_TRIALS):
                            # fetch a NEW batch each trial
                            try:
                                Xb, Yb = next(it_loader)
                            except StopIteration:
                                it_loader = iter(loader)
                                Xb, Yb = next(it_loader)

                            # Sample a compact Transformer (same policy as generator)
                            arch_name, ctor = choose_transformer(ds_name, num_classes)
                            net = ctor()

                            # CAPE features on the SAME batch (probe B == trial B)
                            pdict = extract_probe_features_dict(net, Xb, Yb, criterion)
                            feat = {
                                'logP':  pdict['logP'],
                                'logB':  pdict['logB'],        # varies with actual B
                                'logG2': pdict['logG2'],
                                'logTau':pdict['logTau'],      # mean-based τ
                                'logLR': float(np.log(lr)),
                                'logN':  logN
                            }
                            z = np.array([feat[f] for f in flist], dtype=float).reshape(1, -1)

                            T_pred = float(model_reg.predict(z)[0])
                            T_act  = measure_convergence(net, Xb, Yb, eps, lr, criterion)

                            y_preds.append(T_pred)
                            y_trues.append(T_act)

                        mae = mean_absolute_error(y_trues, y_preds)
                        rows.append({
                            'feature_set': set_name,
                            'dataset':     ds_name,
                            'MAE':         float(mae)
                        })
                        pbar.update(1)

# -------------------
# Aggregate & save
# -------------------
df_all = pd.DataFrame(rows)
df_avg = df_all.groupby(['feature_set','dataset'], as_index=False)['MAE'].mean()

pivot = df_avg.pivot_table(index='dataset', columns='feature_set', values='MAE', aggfunc='mean')
pivot = pivot.reindex(sorted(pivot.columns), axis=1).round(2)
pivot.to_csv('Transformer_Ablation.csv')

print("Done!")
