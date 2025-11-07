# MLP Meta Dataset - Meta Dataset Generation Code

import random
from typing import Tuple
import contextlib

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
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

# Standard normalization
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

# Grids (adjust as needed)
LR_VALUES    = [0.0005, 0.001, 0.002]
BATCH_SIZES  = [32, 64, 128]
EPS_VALUES   = [0.10, 0.15, 0.20]
N_TRIALS     = 100   # per (dataset, lr, B, eps)

# Adaptive training
SOFT_MAX_STEPS     = 5000
PLATEAU_PATIENCE   = 200
PLATEAU_MIN_DELTA  = 1e-4

OUT_CSV = "meta_dataset_mlp.csv"

# ------------------- Research-style Meta Dataset (lightweight) -------------------
class MLPResearch(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, depth: int,
                 dropout_p: float = 0.2, use_bn: bool = True, activation: str = "gelu"):
        super().__init__()
        act_layer = nn.GELU() if activation == "gelu" else nn.ReLU()

        layers = [nn.Flatten()]
        dims = [input_dim] + [hidden_dim]*depth + [num_classes]
        for i in range(len(dims)-2):
            d_in, d_out = dims[i], dims[i+1]
            block = [nn.Linear(d_in, d_out)]
            if use_bn:
                block.append(nn.BatchNorm1d(d_out))
            block.append(act_layer)
            if dropout_p > 0.0:
                block.append(nn.Dropout(dropout_p))
            layers.extend(block)
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

        # Init: xavier for GELU; kaiming for ReLU
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
    """Pick hidden/depth slightly bigger than toy, but still cheap."""
    if ds_name in ("MNIST", "FashionMNIST"):
        hidden_choices = [128, 256, 512]
        depth_choices  = [2, 3]
    else:  # CIFAR10/100: allow a wider/deeper option
        hidden_choices = [256, 512, 1024]
        depth_choices  = [2, 3, 4]
    return random.choice(hidden_choices), random.choice(depth_choices)

# ------------------- CAPE probing helpers -------------------
def _param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def _ensure_2d_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits if logits.ndim == 2 else logits.view(1, -1)

@contextlib.contextmanager
def stabilize_probes(model: nn.Module):
    """
    Temporarily set BN/Dropout to eval() for stable per-sample probing.
    """
    changed = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
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
    NOTE: Probe batch == actual trial batch B.
    """
    model.to(DEVICE).train()
    Xp, yp = X.to(DEVICE), y.to(DEVICE).long()
    Bp = int(Xp.size(0))                  # use full trial batch
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
    tau_mean = float(np.mean(tau_list))   # <-- mean, not sum

    logP   = float(np.log(_param_count(model)))
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
        input_shape = ds[0][0].shape
        input_dim = int(np.prod(input_shape))
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

            # Research-style (still light) Meta Dataset
            hidden_dim, depth = choose_dims(ds_name)
            model = MLPResearch(
                input_dim=input_dim, num_classes=num_classes,
                hidden_dim=hidden_dim, depth=depth,
                dropout_p=0.2, use_bn=True, activation="gelu"
            )

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
                'dataset'      : ds_name,
                'learning_rate': float(lr),
                'batch_size'   : int(n),
                'epsilon'      : float(eps),

                # CAPE features only
                'logP'         : float(logP),
                'logB'         : float(logB),     # varies with actual B now
                'logG2'        : float(logG2),
                'logTau'       : float(logTau),   # mean-based τ (paper-correct)
                'logLR'        : float(logLR),
                'logN'         : float(logN),

                # outcomes
                'T_star'       : int(T_star),
                'converged'    : bool(converged),
                'censored_reason': ("" if converged else reason),
                'logInitLoss'  : float(np.log(max(init_loss, 1e-12))),
                'logFinalLoss' : float(np.log(max(final_loss, 1e-12))),
            })

    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV} with {len(df)} rows "
          f"({df['converged'].sum()} converged, {(~df['converged']).sum()} censored).")

if __name__ == '__main__':
    construct_meta_dataset()
