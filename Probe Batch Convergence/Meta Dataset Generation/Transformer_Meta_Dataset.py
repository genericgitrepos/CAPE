# -*- coding: utf-8 -*-
"""
Transformer Meta-Dataset Generator (CAPE-only features)
Architectures (compact, research-based, CIFAR-friendly):
  - ViT-Tiny-32/4      (patch=4, 64 tokens)
  - CCT-Small          (conv tokenizer -> Transformer, mean token)
  - PiT-XS             (hierarchical; token pooling)
Datasets: MNIST, FashionMNIST, CIFAR10, CIFAR100 (all 32x32, 3ch)
Protocol:
  - Probe batch == trial batch (logB = log(B))
  - logG2/logTau are MEANS across probe samples
  - Same adaptive convergence as MLP
Output:
  - meta_dataset_transformer.csv
"""

import random
from typing import Tuple
import contextlib

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ------------------- Config -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

DATASETS = {
    'MNIST':        datasets.MNIST,
    'FashionMNIST': datasets.FashionMNIST,
    'CIFAR10':      datasets.CIFAR10,
    'CIFAR100':     datasets.CIFAR100,
}

# All models consume 3x32x32; duplicate grayscale to 3ch; resize MNIST/FashionMNIST to 32
TRANSFORMS = {
    'MNIST': transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.expand(3, -1, -1)),  # 1->3 channels
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
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]),
    'CIFAR100': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]),
}

# Grids (kept identical to your MLP setup)
LR_VALUES    = [0.0005, 0.001, 0.002]
BATCH_SIZES  = [32, 64, 128]
EPS_VALUES   = [0.10, 0.15, 0.20]
N_TRIALS     = 100   # per (dataset, lr, B, eps)

# Adaptive training
SOFT_MAX_STEPS     = 5000
PLATEAU_PATIENCE   = 200
PLATEAU_MIN_DELTA  = 1e-4

OUT_CSV = "meta_dataset_transformer.csv"

# ------------------- Building blocks -------------------
class MLPHead(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)
        self._init()

    def _init(self):
        nn.init.xavier_normal_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
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
        # Pre-Norm
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.drop1(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x

# ------------------- ViT-Tiny-32/4 -------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=192, patch=4, img_size=32):
        super().__init__()
        assert img_size % patch == 0
        self.grid = img_size // patch
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="relu"); nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        # x: [B,3,32,32] -> [B, embed, 8, 8] -> [B, 64, embed]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
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
        if cls_token:
            nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.xavier_normal_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.patch(x)                       # [B, N, D]
        if self.cls_token:
            cls = self.cls.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)      # [B, 1+N, D]
        x = x + self.pos[:, :x.size(1), :]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.cls_token:
            feat = x[:, 0]                      # CLS
        else:
            feat = x.mean(dim=1)                # mean token
        logits = self.head(feat)
        return logits

# ------------------- CCT-Small (conv tokenizer -> Transformer) -------------------
class ConvTokenizer(nn.Module):
    def __init__(self, in_ch=3, embed_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1, bias=False), # 32->16
            nn.BatchNorm2d(embed_dim), nn.GELU(),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        x = self.conv(x)              # [B, D, 16, 16]
        x = x.flatten(2).transpose(1, 2)   # [B, 256, D]
        return x

class CCTSmall(nn.Module):
    def __init__(self, num_classes: int, embed_dim=128, depth=4, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.tok = ConvTokenizer(3, embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, 256, embed_dim))  # 16*16
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.xavier_normal_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.tok(x)                   # [B, 256, D]
        x = x + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        feat = x.mean(dim=1)              # mean token pooling
        logits = self.head(feat)
        return logits

# ------------------- PiT-XS (hierarchical with token pooling) -------------------
class TokenPool(nn.Module):
    """Average pool tokens by reshaping to HxW and 2x2 pooling -> reduces sequence by 4x."""
    def __init__(self, img_tokens_h, img_tokens_w):
        super().__init__()
        self.h = img_tokens_h
        self.w = img_tokens_w

    def forward(self, x):
        # x: [B, N, D], with N = H*W
        B, N, D = x.shape
        x = x.transpose(1, 2).reshape(B, D, self.h, self.w)  # [B, D, H, W]
        x = F.avg_pool2d(x, kernel_size=2, stride=2)         # [B, D, H/2, W/2]
        h2, w2 = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)                     # [B, (H/2*W/2), D]
        return x, h2, w2

class PiTXS(nn.Module):
    """
    Stage1: 64 tokens (patch=4) -> depth2
    Pool -> 16 tokens -> depth2
    """
    def __init__(self, num_classes: int, img_size=32, patch=4,
                 dims=(96, 192), heads=(2, 3), depths=(2, 2), mlp_ratio=2.0):
        super().__init__()
        self.patch = PatchEmbed(3, dims[0], patch=patch, img_size=img_size)  # 8x8 tokens
        self.pos1 = nn.Parameter(torch.zeros(1, 64, dims[0]))
        self.stage1 = nn.ModuleList([TransformerBlock(dims[0], heads[0], mlp_ratio) for _ in range(depths[0])])

        self.pool = TokenPool(8, 8)   # 8x8 -> 4x4
        self.proj = nn.Linear(dims[0], dims[1])
        self.pos2 = nn.Parameter(torch.zeros(1, 16, dims[1]))
        self.stage2 = nn.ModuleList([TransformerBlock(dims[1], heads[1], mlp_ratio) for _ in range(depths[1])])

        self.norm = nn.LayerNorm(dims[1])
        self.head = nn.Linear(dims[1], num_classes)

        nn.init.trunc_normal_(self.pos1, std=0.02)
        nn.init.trunc_normal_(self.pos2, std=0.02)
        nn.init.xavier_normal_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        nn.init.xavier_normal_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.patch(x)                  # [B, 64, D1]
        x = x + self.pos1
        for blk in self.stage1:
            x = blk(x)
        x, h2, w2 = self.pool(x)           # [B, 16, D1]
        x = self.proj(x)                   # [B, 16, D2]
        x = x + self.pos2
        for blk in self.stage2:
            x = blk(x)
        x = self.norm(x)
        feat = x.mean(dim=1)
        logits = self.head(feat)
        return logits

# ------------------- Model chooser -------------------
def choose_transformer(ds_name: str, num_classes: int) -> Tuple[str, nn.Module]:
    """
    Randomly pick one of the compact Transformer families with stable hyperparams.
    """
    choices = []

    # Always include CCT-Small
    choices.append(("CCT-Small", CCTSmall(num_classes=num_classes, embed_dim=128, depth=4, num_heads=4, mlp_ratio=2.0)))

    # ViT-Tiny-32/4 (CLS)
    choices.append(("ViT-Tiny-32/4", ViTTiny32(num_classes=num_classes, embed_dim=192, depth=6, num_heads=3, mlp_ratio=2.0, patch=4, img_size=32, cls_token=True)))

    # PiT-XS
    choices.append(("PiT-XS", PiTXS(num_classes=num_classes, img_size=32, patch=4, dims=(96, 192), heads=(2, 3), depths=(2, 2), mlp_ratio=2.0)))

    name, model = random.choice(choices)
    return name, model

# ------------------- CAPE probing helpers -------------------
def _param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _ensure_2d_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits if logits.ndim == 2 else logits.view(logits.size(0), -1)

@contextlib.contextmanager
def stabilize_probes(model: nn.Module):
    """
    Temporarily set BN/Dropout to eval() for stable per-sample probing.
    (Covers 1D/2D BN and Dropout variants.)
    """
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

def extract_probe_features(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion) -> Tuple[float, float, float, float]:
    """
    CAPE features: logP, logB, logG2, logTau
      - logG2:  log(mean per-sample grad^2 of loss)
      - logTau: log(mean per-sample grad^2 of true logit)
    NOTE: Probe batch == trial batch B.
    """
    model.to(DEVICE).train()
    Xp, yp = X.to(DEVICE), y.to(DEVICE).long()
    Bp = int(Xp.size(0))
    params = [p for p in model.parameters() if p.requires_grad]

    g2_list, tau_list = [], []

    with stabilize_probes(model):
        for i in range(Bp):
            xi, yi = Xp[i:i+1], yp[i:i+1]

            model.zero_grad(set_to_none=True)
            logits_i = _ensure_2d_logits(model(xi))
            loss_i = criterion(logits_i, yi)
            grads = torch.autograd.grad(loss_i, params, retain_graph=True, create_graph=False)
            grad_vec = torch.cat([g.reshape(-1) for g in grads if g is not None])
            g2_list.append((grad_vec**2).sum().item())

            model.zero_grad(set_to_none=True)
            logits_i = _ensure_2d_logits(model(xi))
            true_logit = logits_i[0, yi[0].item()]
            grads_f = torch.autograd.grad(true_logit, params, retain_graph=False, create_graph=False)
            grad_f_vec = torch.cat([g.reshape(-1) for g in grads_f if g is not None])
            tau_list.append((grad_f_vec**2).sum().item())

    g2_mean  = float(np.mean(g2_list))
    tau_mean = float(np.mean(tau_list))

    logP   = float(np.log(max(_param_count(model), 1)))
    logB   = float(np.log(max(Bp, 1)))
    logG2  = float(np.log(max(g2_mean, 1e-12)))
    logTau = float(np.log(max(tau_mean, 1e-12)))
    return logP, logB, logG2, logTau

# ------------------- Convergence (adaptive) -------------------
def measure_convergence_adaptive(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                                 eps: float, lr: float, criterion,
                                 soft_max_steps: int = SOFT_MAX_STEPS,
                                 plateau_patience: int = PLATEAU_PATIENCE,
                                 plateau_min_delta: float = PLATEAU_MIN_DELTA):
    """
    Train on the SAME probe batch until loss <= eps * init_loss, or plateau/no progress.
    Returns: (T_star, converged, init_loss, final_loss, censored_reason)
    """
    model.to(DEVICE).train()
    X, y = X.to(DEVICE), y.to(DEVICE).long()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    losses = []
    init_loss = None
    reason = ""

    for step in range(1, soft_max_steps + 1):
        opt.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(_ensure_2d_logits(logits), y)
        if step == 1:
            init_loss = float(loss.item())
        current = float(loss.item())
        losses.append(current)

        if current <= eps * init_loss:
            return step, True, init_loss, current, "met_threshold"

        loss.backward()
        opt.step()

        if step >= plateau_patience:
            window = losses[-plateau_patience:]
            rel_impr = (window[0] - window[-1]) / max(window[0], 1e-12)
            if rel_impr < plateau_min_delta:
                reason = f"plateau_{plateau_patience}"
                break

    final_loss = float(losses[-1])
    return step, False, init_loss, final_loss, (reason or "soft_max")

# ------------------- Utility -------------------
def infer_num_classes(ds_name: str) -> int:
    if ds_name in ('MNIST','FashionMNIST','CIFAR10'):
        return 10
    if ds_name == 'CIFAR100':
        return 100
    raise ValueError(f"Unknown dataset: {ds_name}")

# ------------------- Meta-dataset construction (CAPE features only) -------------------
def construct_meta_dataset():
    records = []

    for ds_name, ds_cls in DATASETS.items():
        print(f"\nBuilding meta rows for: {ds_name}")
        ds = ds_cls(root='./data', train=True, download=True, transform=TRANSFORMS[ds_name])

        num_classes = infer_num_classes(ds_name)
        total_N = len(ds)
        logN = float(np.log(total_N))
        crit = nn.CrossEntropyLoss()

        base_loader = DataLoader(
            ds,
            batch_size=max(BATCH_SIZES),
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        it_base = iter(base_loader)

        grid = [(lr, B, eps, t)
                for lr in LR_VALUES
                for B in BATCH_SIZES
                for eps in EPS_VALUES
                for t in range(N_TRIALS)]

        for lr, B, eps, trial in tqdm(grid, desc=f"{ds_name} grid", unit="trial"):
            try:
                Xb, yb = next(it_base)
            except StopIteration:
                it_base = iter(base_loader)
                Xb, yb = next(it_base)

            # Use EXACT B for both probing and convergence (truncate if needed)
            n = min(B, Xb.size(0))
            Xb = Xb[:n]
            yb = yb[:n]

            # Choose a compact Transformer
            arch_name, model = choose_transformer(ds_name, num_classes)

            # CAPE features (probe batch == trial batch)
            logP, logB, logG2, logTau = extract_probe_features(model, Xb, yb, crit)
            logLR = float(np.log(lr))

            # Adaptive convergence on the SAME batch
            T_star, converged, init_loss, final_loss, reason = measure_convergence_adaptive(
                model, Xb, yb, eps=eps, lr=lr, criterion=crit,
                soft_max_steps=SOFT_MAX_STEPS,
                plateau_patience=PLATEAU_PATIENCE,
                plateau_min_delta=PLATEAU_MIN_DELTA
            )

            records.append({
                'dataset'         : ds_name,
                'architecture'    : arch_name,
                'learning_rate'   : float(lr),
                'batch_size'      : int(n),
                'epsilon'         : float(eps),

                # CAPE features only
                'logP'            : float(logP),
                'logB'            : float(logB),
                'logG2'           : float(logG2),
                'logTau'          : float(logTau),
                'logLR'           : float(logLR),
                'logN'            : float(logN),

                # outcomes
                'T_star'          : int(T_star),
                'converged'       : bool(converged),
                'censored_reason' : ("" if converged else reason),
                'logInitLoss'     : float(np.log(max(init_loss, 1e-12))),
                'logFinalLoss'    : float(np.log(max(final_loss, 1e-12))),
            })

    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV} with {len(df)} rows "
          f"({df['converged'].sum()} converged, {(~df['converged']).sum()} censored).")

if __name__ == '__main__':
    construct_meta_dataset()
