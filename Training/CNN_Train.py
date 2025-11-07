import os, re, math, time, json, random, argparse, platform, warnings, zipfile, urllib.request
from typing import Tuple, Callable, Dict, Any, List

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms, models
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_SEED = 42

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

def gpu_info():
    if DEVICE.type == 'cuda':
        return torch.cuda.get_device_properties(0).name
    return 'CPU'

try:
    from torch.cuda.amp import autocast as _autocast_cuda
    from torch.cuda.amp import GradScaler as _GradScaler
    _USE_TORCH_AMP_ROOT = False
except Exception:
    from torch.amp import autocast as _autocast_root
    from torch.amp import GradScaler as _GradScaler
    _USE_TORCH_AMP_ROOT = True

from contextlib import nullcontext
def _amp_autocast(enabled: bool):
    if not enabled: return nullcontext()
    if _USE_TORCH_AMP_ROOT:
        return _autocast_root(device_type='cuda', dtype=torch.float16)
    else:
        return _autocast_cuda(dtype=torch.float16)


_TINY_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
_TINY_ZIP = "tiny-imagenet-200.zip"
_TINY_DIR = "tiny-imagenet-200"

def _download_tiny(root: str) -> str:
    root = os.path.abspath(root); os.makedirs(root, exist_ok=True)
    target = os.path.join(root, _TINY_DIR)
    if os.path.isdir(target) and os.path.isdir(os.path.join(target, "train")):
        return target
    zip_path = os.path.join(root, _TINY_ZIP)
    if not os.path.exists(zip_path):
        print("Downloading TinyImageNet-200 (~236MB)...")
        urllib.request.urlretrieve(_TINY_URL, zip_path)
    print("Extracting TinyImageNet-200...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    return target

class TinyImageNet200(Dataset):
    def __init__(self, root: str, split: str = "train", transform: Callable = None):
        assert split in ("train", "val")
        self.root = _download_tiny(root)
        self.split = split
        self.transform = transform
        with open(os.path.join(self.root, "wnids.txt"), "r") as f:
            self.wnids = [x.strip() for x in f if x.strip()]
        self.class_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}
        self.samples: List[Tuple[str, int]] = []
        if split == "train":
            tdir = os.path.join(self.root, "train")
            for wnid in self.wnids:
                imgd = os.path.join(tdir, wnid, "images")
                if not os.path.isdir(imgd): continue
                for fn in os.listdir(imgd):
                    if fn.lower().endswith((".jpeg", ".jpg", ".png")):
                        self.samples.append((os.path.join(imgd, fn), self.class_to_idx[wnid]))
        else:
            vdir = os.path.join(self.root, "val")
            ann = os.path.join(vdir, "val_annotations.txt")
            mapping = {}
            with open(ann, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        mapping[parts[0]] = parts[1]
            imgd = os.path.join(vdir, "images")
            for fn in os.listdir(imgd):
                if fn.lower().endswith((".jpeg", ".jpg", ".png")):
                    wnid = mapping.get(fn, None)
                    if wnid is None: continue
                    self.samples.append((os.path.join(imgd, fn), self.class_to_idx[wnid]))
        self.num_classes = 200
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx: int):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, target


def build_cifar10(root):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    ])
    ds = datasets.CIFAR10(root=root, train=True, download=True, transform=tfm)
    return ds, 10, (3,32,32), 'CIFAR10', len(ds)

def build_cifar100(root):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
    ])
    ds = datasets.CIFAR100(root=root, train=True, download=True, transform=tfm)
    return ds, 100, (3,32,32), 'CIFAR100', len(ds)

def build_tiny_imagenet(root):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4802,0.4481,0.3975),(0.2770,0.2691,0.2821))
    ])
    ds = TinyImageNet200(root=root, split="train", transform=tfm)
    return ds, 200, (3,64,64), 'TinyImageNet', len(ds)

def build_stl10(root):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4467,0.4398,0.4066),(0.2603,0.2566,0.2713))
    ])
    ds = datasets.STL10(root=root, split='train', download=True, transform=tfm)
    return ds, 10, (3,96,96), 'STL10', len(ds)

DATASET_BUILDERS = {
    'CIFAR10': build_cifar10,
    'CIFAR100': build_cifar100,
    'TinyImageNet': build_tiny_imagenet,
    'STL10': build_stl10,
}


def build_loader(ds, batch_size:int, shuffle:bool, num_workers:int=2) -> DataLoader:
    if platform.system() == "Windows":
        num_workers = 0
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=(DEVICE.type=='cuda'), persistent_workers=False
    )


def make_vgg16(num_classes:int) -> nn.Module:
    m = models.vgg16_bn(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m

def make_resnet50(num_classes:int) -> nn.Module:
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def make_densenet121(num_classes:int) -> nn.Module:
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m

def make_mobilenetv2(num_classes:int) -> nn.Module:
    m = models.mobilenet_v2(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m

def build_cnn(model_name:str, num_classes:int) -> nn.Module:
    n = model_name.lower()
    if n == 'vgg16':        return make_vgg16(num_classes)
    if n == 'resnet50':     return make_resnet50(num_classes)
    if n == 'densenet121':  return make_densenet121(num_classes)
    if n == 'mobilenetv2':  return make_mobilenetv2(num_classes)
    raise ValueError(f"Unknown CNN model: {model_name}")


def _safe_path_component(s: str, maxlen: int = 140) -> str:
    s = re.sub(r'[<>:\"/\\|?*\x00-\x1F]', '_', str(s))
    s = re.sub(r'\s+', '_', s.strip())
    return s[:maxlen]

def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def _initial_loss(model: nn.Module, criterion: nn.Module, xb: torch.Tensor, yb: torch.Tensor) -> float:
    model.eval()
    logits = model(xb)
    return float(criterion(logits, yb).item())

def extract_probe_features(model: nn.Module, criterion: nn.Module, xb: torch.Tensor, yb: torch.Tensor) -> Dict[str, float]:
    model.to(DEVICE)
    model.train(False)
    xb = xb.to(DEVICE, non_blocking=True)
    yb = yb.to(DEVICE, non_blocking=True).long()

    model.zero_grad(set_to_none=True)
    logits = model(xb)
    loss = criterion(logits, yb)
    grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad],
                                create_graph=False, retain_graph=False, allow_unused=True)
    flat_grads = [g.reshape(-1) for g in grads if g is not None]
    gcat = torch.cat(flat_grads) if len(flat_grads) else torch.zeros(1, device=DEVICE)
    grad_norm = torch.norm(gcat, p=2).item()
    ntk_trace_proxy = torch.sum(gcat * gcat).item()
    init_loss = loss.item()
    return {
        'gradient_norm_log10': float(np.log10(grad_norm + 1e-8)),
        'ntk_trace_proxy_log10': float(np.log10(ntk_trace_proxy + 1e-8)),
        'initial_loss_log10': float(np.log10(init_loss + 1e-8)),
    }


def _build_optimizer(params, name: str, lr: float, weight_decay: float):
    name = name.lower()
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, betas=(0.9,0.999), eps=1e-8, weight_decay=weight_decay)
    elif name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=False)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def train_until_plateau(model: nn.Module,
                        train_loader: DataLoader,
                        val_loader: DataLoader,
                        lr: float,
                        optimizer_name: str,
                        precision_bits: int,
                        max_epochs: int,
                        patience: int,
                        min_delta: float,
                        record_first_k: int = 20) -> Tuple[int, List[float]]:
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    opt = _build_optimizer(model.parameters(), optimizer_name, lr, weight_decay=1e-4)
    use_amp = (precision_bits == 16 and DEVICE.type == 'cuda')
    scaler = _GradScaler(enabled=use_amp)

    best_val = float('inf')
    no_improve = 0
    val_curve: List[float] = []

    for epoch in range(1, max_epochs + 1):
        model.train(True)
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True).long()
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with _amp_autocast(True):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()

        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True).long()
                logits = model(xb)
                vloss = criterion(logits, yb).item()
                val_loss += vloss
                n += 1
        val_loss /= max(1, n)
        if len(val_curve) < record_first_k:
            val_curve.append(float(val_loss))

        if val_loss < best_val - min_delta:
            best_val = val_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            return epoch, val_curve

    return max_epochs, val_curve


def main():
    p = argparse.ArgumentParser("CAPE++ CNN Meta-Gen (Validation Early-Stop, Zero-Shot Probe)")
    p.add_argument('--data-root', type=str, default='./data')
    p.add_argument('--logdir', type=str, default='./meta_logs')


    p.add_argument('--models', nargs='+', default=['resnet50'])

    p.add_argument('--datasets', nargs='+', default=['CIFAR10','CIFAR100','TinyImageNet','STL10'])

    p.add_argument('--batch-sizes', nargs='+', type=int, default=[32,64,128])
    p.add_argument('--lrs', nargs='+', type=float, default=[5e-4, 1e-3, 2e-3])
    p.add_argument('--optimizers', nargs='+', default=['SGD','AdamW'])
    p.add_argument('--precisions', nargs='+', type=int, default=[16], choices=[16,32])

    p.add_argument('--patience', type=int, default=7)
    p.add_argument('--min-delta', type=float, default=5e-4)
    p.add_argument('--val-fraction', type=float, default=0.2)
    p.add_argument('--record-first-k', type=int, default=20)

    p.add_argument('--num-workers', type=int, default=12)
    p.add_argument('--seed', type=int, default=DEFAULT_SEED)

    args = p.parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    print(f"[INFO] Meta logs will be saved to: {os.path.abspath(args.logdir)}")
    set_seed(args.seed)

    ALLOWED_BY_MODEL = {
        'resnet50':    ['TinyImageNet', 'CIFAR10'],
        'vgg16':       ['TinyImageNet', 'CIFAR100'],
        'mobilenetv2': ['CIFAR10', 'STL10'],
        'densenet121': ['CIFAR100', 'TinyImageNet'],
    }

    MAX_EPOCHS_BY_DS = {
        'CIFAR10': 200,
        'STL10': 200,
        'CIFAR100': 200,
        'TinyImageNet': 200,
    }

    ds_names = [d for d in args.datasets if d in DATASET_BUILDERS]
    built_ds: Dict[str, Any] = {}
    for ds_name in ds_names:
        ds, num_classes, input_shape, ds_label, total_N = DATASET_BUILDERS[ds_name](args.data_root)
        built_ds[ds_label] = (ds, num_classes, input_shape, total_N)

    gpu_name_str = gpu_info()
    rows: List[Dict[str, Any]] = []

    for model_name in args.models:
        allowed = [d for d in ALLOWED_BY_MODEL.get(model_name.lower(), []) if d in built_ds]
        if not allowed:
            print(f"[WARN] No allowed datasets available for model '{model_name}'. Skipping.")
            continue

        for ds_label in allowed:
            ds, num_classes, input_shape, total_N = built_ds[ds_label]
            in_ch, H, W = input_shape

            if ds_label == "TinyImageNet":
                train_ds = ds
                val_ds = TinyImageNet200(root=args.data_root, split="val",
                                         transform=ds.transform if hasattr(ds, 'transform') else None)
            else:
                n_total = len(ds)
                n_val = int(round(args.val_fraction * n_total))
                n_train = n_total - n_val
                train_ds, val_ds = random_split(
                    ds, [n_train, n_val],
                    generator=torch.Generator().manual_seed(args.seed)
                )

            probe_loader = build_loader(train_ds, batch_size=min(64, max(32, args.batch_sizes[0])),  # 32–64 recommended
                                        shuffle=True, num_workers=args.num_workers)
            probe_xb, probe_yb = next(iter(probe_loader))
            probe_xb = probe_xb.to(DEVICE); probe_yb = probe_yb.to(DEVICE).long()

            trials: List[Dict[str, Any]] = []
            for lr in args.lrs:
                for B in args.batch_sizes:
                    for optimizer_name in args.optimizers:
                        for precision in args.precisions:
                            trials.append({
                                'lr': lr, 'batch_size': B,
                                'optimizer_name': optimizer_name,
                                'precision': precision,
                            })

            for t in tqdm(trials, desc=f"{ds_label} — {model_name}", unit="trial"):
                lr = t['lr']; B = t['batch_size']
                optimizer_name = t['optimizer_name']
                precision = t['precision']
                use_amp = (precision == 16 and DEVICE.type == 'cuda')

                try:
                    model = build_cnn(model_name, num_classes)
                except Exception as e:
                    print(f"[ERROR] Model build failed ({model_name}) on {ds_label}: {e}")
                    continue

                P_params = param_count(model)
                criterion = nn.CrossEntropyLoss()

                try:
                    probe_feats = extract_probe_features(model, criterion, probe_xb, probe_yb)
                except RuntimeError as e:
                    print(f"[ERROR] Probe failed [{model_name}] on {ds_label}: {e}")
                    del model
                    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
                    continue

                features = {
                    'param_count_log10': float(np.log10(P_params + 1e-8)),
                    'learning_rate_log10': float(np.log10(lr + 1e-12)),
                    'batch_size_log10': float(np.log10(B + 1e-12)),
                    **probe_feats,
                }

                try:
                    train_loader = build_loader(train_ds, batch_size=B, shuffle=True,  num_workers=args.num_workers)
                    val_loader   = build_loader(val_ds,   batch_size=B, shuffle=False, num_workers=args.num_workers)
                except RuntimeError as e:
                    print(f"[ERROR] DataLoader failed (B={B}) on {ds_label}: {e}")
                    del model
                    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
                    continue

                max_epochs = int(MAX_EPOCHS_BY_DS.get(ds_label, 150))
                try:
                    t_conv, val_prefix = train_until_plateau(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        lr=lr,
                        optimizer_name=optimizer_name,
                        precision_bits=precision,
                        max_epochs=max_epochs,
                        patience=args.patience,
                        min_delta=args.min_delta,
                        record_first_k=args.record_first_k
                    )
                except RuntimeError as e:
                    print(f"[ERROR] Training failed [{model_name}] on {ds_label} (B={B}, lr={lr}, opt={optimizer_name}): {e}")
                    del model
                    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
                    continue

                t80 = int(max(1, math.ceil(0.8 * t_conv)))
                t90 = int(max(1, math.ceil(0.9 * t_conv)))

                rows.append({
                    'dataset': ds_label,
                    'num_classes': int(num_classes),
                    'dataset_size': int(total_N),
                    'model': model_name,
                    'architecture': 'CNN',

                    'precision': int(precision),
                    'optimizer': optimizer_name,
                    'learning_rate': float(lr),
                    'batch_size': int(B),

                    'param_count': int(P_params),

                    **features,

                    'T_conv': int(t_conv),
                    'T_80close': int(t80),
                    'T_90close': int(t90),

                    'patience_P': int(args.patience),
                    'min_delta': float(args.min_delta),
                    'max_epochs': int(max_epochs),
                    'val_loss_prefix': json.dumps([float(x) for x in val_prefix]),

                    'seed': int(args.seed),
                    'gpu_name': gpu_name_str if DEVICE.type == 'cuda' else 'CPU',
                })

                del model
                if DEVICE.type == 'cuda': torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    os.makedirs(args.logdir, exist_ok=True)
    out_path = os.path.join(os.path.abspath(args.logdir), "CNN_Convergence.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[OK] Saved CNN meta-dataset to:\n  {out_path}\nRows: {len(df)}")

if __name__ == '__main__':
    main()
