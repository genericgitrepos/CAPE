import os, re, math, json, random, argparse, platform, warnings
from typing import Dict, Any, List, Optional, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_SEED = 42

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def gpu_name():
    if DEVICE.type == 'cuda':
        return torch.cuda.get_device_properties(0).name
    return 'CPU'


BOS, EOS, PAD, UNK = "<bos>", "<eos>", "<pad>", "<unk>"
_word_re = re.compile(r"\w+([’']\w+)?", re.UNICODE)
def basic_split(text: str):
    return [m.group(0).lower() for m in _word_re.finditer(text or "")]

class Vocab:
    def __init__(self, stoi:Dict[str,int], unk_token:str=UNK):
        self.stoi = dict(stoi)
        self.itos = [None]*len(stoi)
        for k,v in self.stoi.items(): self.itos[v] = k
        self.unk = unk_token; self.pad = PAD
        self._unk_idx = self.stoi.get(unk_token, 0)
    def __len__(self): return len(self.stoi)
    def __getitem__(self, item:str) -> int:
        return self.stoi.get(item, self._unk_idx)

def build_vocab_from_texts(text_iter, max_tokens:int=30000, specials:List[str]=[UNK, PAD, BOS, EOS]) -> Vocab:
    freq: Dict[str,int] = {}
    for txt in text_iter:
        for tok in basic_split(txt):
            freq[tok] = freq.get(tok, 0) + 1
    stoi = {}
    for sp in specials:
        if sp not in stoi: stoi[sp] = len(stoi)
    for tok, _ in sorted(freq.items(), key=lambda kv: -kv[1]):
        if tok in stoi: continue
        if len(stoi) >= max_tokens: break
        stoi[tok] = len(stoi)
    return Vocab(stoi)

def _ensure_nonempty(ids: List[int], vocab: Vocab) -> List[int]:
    return ids if len(ids)>0 else [vocab[UNK]]


class _ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, pairs:List[Tuple[List[int],int]], vocab:Vocab, max_seq_len:int):
        self.data = pairs; self.vocab=vocab; self.max_seq_len=max_seq_len
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def _collate_cls(batch, pad_idx:int, max_seq_len:int):
    seqs, labels = zip(*batch)
    Ls = [min(len(s), max_seq_len) for s in seqs]
    lengths = torch.tensor(Ls, dtype=torch.long)
    X = torch.full((len(batch), max_seq_len), pad_idx, dtype=torch.long)
    for i, s in enumerate(seqs):
        s = s[:max_seq_len] if len(s)>0 else [pad_idx]
        L = min(len(s), max_seq_len)
        X[i,:L] = torch.tensor(s[:L], dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    return X, lengths, y

class _Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, pairs:List[Tuple[List[int],List[int]]], vocab:Vocab, max_seq_len:int):
        self.pairs = pairs; self.vocab=vocab; self.max_seq_len=max_seq_len
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]

def _collate_s2s(batch, pad_idx:int, max_seq_len:int):
    srcs, tgts = zip(*batch)
    len_src = torch.tensor([min(len(s), max_seq_len) for s in srcs], dtype=torch.long)
    len_tgt = torch.tensor([min(len(t), max_seq_len) for t in tgts], dtype=torch.long)
    Xs = torch.full((len(batch), max_seq_len), pad_idx, dtype=torch.long)
    Xt = torch.full((len(batch), max_seq_len), pad_idx, dtype=torch.long)
    for i,(s,t) in enumerate(zip(srcs,tgts)):
        s = s[:max_seq_len] if len(s)>0 else [pad_idx]
        t = t[:max_seq_len] if len(t)>0 else [pad_idx]
        Ls = min(len(s), max_seq_len); Lt = min(len(t), max_seq_len)
        Xs[i,:Ls] = torch.tensor(s[:Ls], dtype=torch.long)
        Xt[i,:Lt] = torch.tensor(t[:Lt], dtype=torch.long)
    return Xs, len_src, Xt, len_tgt


try:
    import datasets as hfds
    _HAS_HF = True
except Exception:
    _HAS_HF = False

def _hf_require():
    if not _HAS_HF:
        raise RuntimeError("Install 'datasets' to use HF loaders.")

def build_imdb(root, max_seq_len:int, vocab_size:int):
    _hf_require()
    ds = hfds.load_dataset("imdb", split="train", cache_dir=root)
    texts = (r["text"] for r in ds)
    vocab = build_vocab_from_texts(texts, max_tokens=vocab_size)
    pairs=[]
    for r in ds:
        ids = [vocab[t] for t in basic_split(r["text"])]
        ids = _ensure_nonempty(ids, vocab)[:max_seq_len]
        pairs.append((ids, int(r["label"])))
    dataset = _ClassificationDataset(pairs, vocab, max_seq_len)
    dataset.collate_fn = lambda b, pad_idx=vocab[PAD]: _collate_cls(b, pad_idx, max_seq_len)
    return dataset, 2, vocab, 'IMDB', len(dataset)

def build_agnews(root, max_seq_len:int, vocab_size:int):
    _hf_require()
    ds = hfds.load_dataset("ag_news", split="train", cache_dir=root)
    texts = (r["text"] for r in ds)
    vocab = build_vocab_from_texts(texts, max_tokens=vocab_size)
    pairs=[]
    for r in ds:
        ids = [vocab[t] for t in basic_split(r["text"])]
        ids = _ensure_nonempty(ids, vocab)[:max_seq_len]
        pairs.append((ids, int(r["label"])))
    dataset = _ClassificationDataset(pairs, vocab, max_seq_len)
    dataset.collate_fn = lambda b, pad_idx=vocab[PAD]: _collate_cls(b, pad_idx, max_seq_len)
    return dataset, 4, vocab, 'AG_NEWS', len(dataset)

def build_sst2(root, max_seq_len:int, vocab_size:int):
    _hf_require()
    ds = hfds.load_dataset("glue", "sst2", split="train", cache_dir=root)
    texts = (r["sentence"] for r in ds)
    vocab = build_vocab_from_texts(texts, max_tokens=vocab_size)
    pairs=[]
    for r in ds:
        ids = [vocab[t] for t in basic_split(r["sentence"])]
        ids = _ensure_nonempty(ids, vocab)[:max_seq_len]
        pairs.append((ids, int(r["label"])))
    dataset = _ClassificationDataset(pairs, vocab, max_seq_len)
    dataset.collate_fn = lambda b, pad_idx=vocab[PAD]: _collate_cls(b, pad_idx, max_seq_len)
    return dataset, 2, vocab, 'SST2', len(dataset)

def build_wmt14(root, max_seq_len:int, vocab_size:int):
    _hf_require()
    ds = hfds.load_dataset("wmt14", "de-en", split="train", cache_dir=root)
    src_texts = [ex["translation"]["de"] for ex in ds]
    tgt_texts = [ex["translation"]["en"] for ex in ds]
    vocab = build_vocab_from_texts(src_texts + tgt_texts, max_tokens=vocab_size)
    pairs=[]
    for ex in ds:
        de = [vocab[BOS]] + _ensure_nonempty([vocab[t] for t in basic_split(ex["translation"]["de"])], vocab) + [vocab[EOS]]
        en = [vocab[BOS]] + _ensure_nonempty([vocab[t] for t in basic_split(ex["translation"]["en"])], vocab) + [vocab[EOS]]
        pairs.append((de[:max_seq_len], en[:max_seq_len]))
    dataset = _Seq2SeqDataset(pairs, vocab, max_seq_len)
    dataset.collate_fn = lambda b, pad_idx=vocab[PAD]: _collate_s2s(b, pad_idx, max_seq_len)
    return dataset, vocab, vocab, 'WMT14_EN_DE', len(dataset)

def build_cnndm(root, max_seq_len:int, vocab_size:int):
    _hf_require()
    ds = hfds.load_dataset("cnn_dailymail", "3.0.0", split="train", cache_dir=root)
    src_texts = [ex.get("article") or ex.get("document") or "" for ex in ds]
    tgt_texts = [ex.get("highlights") or ex.get("summary") or "" for ex in ds]
    vocab = build_vocab_from_texts(src_texts + tgt_texts, max_tokens=vocab_size)
    pairs=[]
    for ex in ds:
        src = (ex.get("article") or ex.get("document") or "")
        tgt = (ex.get("highlights") or ex.get("summary") or "")
        x = [vocab[BOS]] + _ensure_nonempty([vocab[t] for t in basic_split(src)], vocab) + [vocab[EOS]]
        y = [vocab[BOS]] + _ensure_nonempty([vocab[t] for t in basic_split(tgt)], vocab) + [vocab[EOS]]
        pairs.append((x[:max_seq_len], y[:max_seq_len]))
    dataset = _Seq2SeqDataset(pairs, vocab, max_seq_len)
    dataset.collate_fn = lambda b, pad_idx=vocab[PAD]: _collate_s2s(b, pad_idx, max_seq_len)
    return dataset, vocab, vocab, 'CNNDM_SUM', len(dataset)


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size:int, embed_dim:int, hidden:int, layers:int,
                 directions:int, num_classes:int, dropout:float=0.1,
                 cell_type:str='lstm', pad_idx:int=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        kwargs = dict(input_size=embed_dim, hidden_size=hidden, num_layers=layers,
                      batch_first=True, dropout=(dropout if layers>1 else 0.0),
                      bidirectional=(directions==2))
        self.cell_type = cell_type.lower()
        if self.cell_type=='lstm':
            self.rnn = nn.LSTM(**kwargs)
        elif self.cell_type=='gru':
            self.rnn = nn.GRU(**kwargs)
        else:
            self.rnn = nn.RNN(**kwargs, nonlinearity='tanh')
        out_dim = hidden * directions
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x, lengths:Optional[torch.Tensor], pad_idx:int=0):
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        if isinstance(h, tuple): h = h[0]
        num_dirs = 2 if getattr(self.rnn, "bidirectional", False) else 1
        last_layer_start = (getattr(self.rnn, "num_layers", 1) - 1) * num_dirs
        h_last = h[last_layer_start:last_layer_start + num_dirs]
        last = h_last.transpose(0, 1).reshape(x.size(0), -1)
        return self.fc(last)

class LuongAttention(nn.Module):
    def __init__(self, q_dim:int, k_dim:int, d_attn:Optional[int]=None):
        super().__init__()
        d = d_attn if d_attn is not None else max(q_dim, k_dim)
        self.Wq = nn.Linear(q_dim, d, bias=False)
        self.Wk = nn.Linear(k_dim, d, bias=False)
        self.scale = 1.0 / math.sqrt(max(1, d))
        self._mask_fill_fp16_safe = -1e4
    def forward(self, q_top: torch.Tensor, enc_out: torch.Tensor,
                mask: Optional[torch.Tensor]=None, k_proj: Optional[torch.Tensor]=None):
        q_proj = self.Wq(q_top).float()
        if k_proj is None: k_proj = self.Wk(enc_out).float()
        scores = torch.bmm(q_proj, k_proj.transpose(1,2)) * float(self.scale)
        if mask is not None:
            mask_bool = mask > 0
            scores = scores.masked_fill(~mask_bool.unsqueeze(1), self._mask_fill_fp16_safe)
        w = torch.softmax(scores, dim=-1).to(enc_out.dtype)
        ctx = torch.bmm(w, enc_out)
        return ctx.squeeze(1)

def _residual(x, y): return x + y if x.shape == y.shape else y

class GNMTEncoder(nn.Module):
    def __init__(self, vocab_size:int, embed_dim:int, hidden:int, layers:int, dropout:float=0.1, pad_idx:int=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm0 = nn.LSTM(embed_dim, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.stack = nn.ModuleList([nn.LSTM(hidden*2 if i==0 else hidden, hidden, num_layers=1, batch_first=True)
                                    for i in range(layers-1)])
    def forward(self, src: torch.Tensor, src_len: torch.Tensor):
        B, Ts = src.size()
        emb = self.embedding(src)
        pack = pack_padded_sequence(emb, src_len.cpu(), batch_first=True, enforce_sorted=False)
        out0, _ = self.lstm0(pack)
        out0, _ = pad_packed_sequence(out0, batch_first=True, total_length=Ts)
        h = out0
        for i, rnn in enumerate(self.stack):
            out_i, _ = rnn(h)
            if i >= 1: out_i = _residual(h[:, :, :out_i.size(2)], out_i)
            h = out_i
        mask = (torch.arange(Ts, device=src_len.device).unsqueeze(0) < src_len.unsqueeze(1)).to(h.dtype)
        return h, mask

class GNMTDecoder(nn.Module):
    def __init__(self, vocab_size:int, embed_dim:int, hidden:int, layers:int, enc_out_dim:int, dropout:float=0.1, pad_idx:int=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.layers = layers; self.hidden = hidden
        self.rnns = nn.ModuleList()
        in_dim = embed_dim + enc_out_dim
        for i in range(layers):
            self.rnns.append(nn.LSTM(in_dim if i==0 else hidden, hidden, num_layers=1, batch_first=True))
        self.attn = LuongAttention(q_dim=hidden, k_dim=enc_out_dim)
        self.proj = nn.Linear(hidden, vocab_size)
    def forward(self, tgt: torch.Tensor, tgt_len: torch.Tensor,
                enc_out: torch.Tensor, enc_mask: torch.Tensor):
        B, Tt = tgt.size(); device = tgt.device
        emb = self.embedding(tgt)
        ctx_prev = torch.zeros(B, enc_out.size(2), device=device, dtype=enc_out.dtype)
        logits_steps = []; hiddens = [None for _ in range(self.layers)]
        k_proj = self.attn.Wk(enc_out).float()
        max_Tt = int(tgt_len.max().item())
        for t in range(max_Tt):
            step_in = torch.cat([emb[:, t, :], ctx_prev], dim=-1).unsqueeze(1)
            x = step_in
            for i, rnn in enumerate(self.rnns):
                x, hiddens[i] = rnn(x, hiddens[i])
                if i >= 2 and x.shape == step_in.shape:
                    x = x + step_in
            ctx = self.attn(q_top=x, enc_out=enc_out, mask=enc_mask, k_proj=k_proj)
            ctx_prev = ctx
            logits_steps.append(self.proj(x.squeeze(1)))
        logits = torch.stack(logits_steps, dim=1)
        return logits, max_Tt

class GNMT(nn.Module):
    def __init__(self, vocab_size:int, embed_dim:int, hidden:int, layers_enc:int, layers_dec:int, pad_idx:int=0, tie_embeddings:bool=True):
        super().__init__()
        self.enc = GNMTEncoder(vocab_size, embed_dim, hidden, layers_enc, pad_idx=pad_idx)
        self.dec = GNMTDecoder(vocab_size, embed_dim, hidden, layers_dec, enc_out_dim=hidden, pad_idx=pad_idx)
        if tie_embeddings:
            self.dec.embedding.weight = self.enc.embedding.weight
    def forward(self, src, src_len, tgt, tgt_len, pad_idx:int=0):
        enc_out, enc_mask = self.enc(src, src_len)
        logits, max_Tt = self.dec(tgt, tgt_len, enc_out, enc_mask)
        return logits, max_Tt


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def _safe_path_component(s: str, maxlen: int = 140) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', str(s)); s = re.sub(r'\s+', '_', s.strip()); return s[:maxlen]

def build_loader(ds, batch_size:int, shuffle:bool, num_workers:int=2):
    if platform.system() == "Windows": num_workers = 0


    collate = getattr(ds, "collate_fn", None)
    if collate is None and hasattr(ds, "dataset"):
        collate = getattr(ds.dataset, "collate_fn", None)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(DEVICE.type=='cuda'),
        persistent_workers=False,
        collate_fn=collate
    )


def extract_probe_features_cls(model, criterion, xb, lengths, yb, pad_idx=None):
    device = next(model.parameters()).device
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)

    was_training = model.training
    model.train(True)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(xb, lengths)
        loss = criterion(logits, yb)

        grads = torch.autograd.grad(
            loss,
            [p for p in model.parameters() if p.requires_grad],
            retain_graph=False,
            create_graph=False,
            allow_unused=True
        )
        flat = [g.reshape(-1) for g in grads if g is not None]
        gcat = torch.cat(flat) if flat else torch.zeros(1, device=device)

        grad_norm = torch.norm(gcat, p=2).item()
        ntk_trace_proxy = torch.sum(gcat * gcat).item()
        init_loss = loss.item()

        return {
            'gradient_norm_log10': float(np.log10(grad_norm + 1e-8)),
            'ntk_trace_proxy_log10': float(np.log10(ntk_trace_proxy + 1e-8)),
            'initial_loss_log10': float(np.log10(init_loss + 1e-8)),
        }
    finally:
        model.train(was_training)

def extract_probe_features_s2s(model: nn.Module, criterion: nn.Module,
                               Xs, Ls, Xt, Lt, pad_idx:int) -> Dict[str, float]:
    device = next(model.parameters()).device
    Xs = Xs.to(device, non_blocking=True); Ls = Ls.to(device, non_blocking=True)
    Xt = Xt.to(device, non_blocking=True); Lt = Lt.to(device, non_blocking=True)

    was_training = model.training
    model.train(True)
    try:
        model.zero_grad(set_to_none=True)
        logits, max_Tt = model(Xs, Ls, Xt, Lt, pad_idx=pad_idx)
        loss = criterion(
            logits[:, :max_Tt, :].reshape(-1, logits.size(-1)),
            Xt[:, :max_Tt].reshape(-1)
        )

        grads = torch.autograd.grad(
            loss,
            [p for p in model.parameters() if p.requires_grad],
            create_graph=False, retain_graph=False, allow_unused=True
        )
        flat = [g.reshape(-1) for g in grads if g is not None]
        gcat = torch.cat(flat) if flat else torch.zeros(1, device=device)

        gnorm = torch.norm(gcat, p=2).item()
        ntk_proxy = torch.sum(gcat * gcat).item()
        init_loss = loss.item()

        return {
            'gradient_norm_log10': float(np.log10(gnorm + 1e-8)),
            'ntk_trace_proxy_log10': float(np.log10(ntk_proxy + 1e-8)),
            'initial_loss_log10': float(np.log10(init_loss + 1e-8)),
        }
    finally:
        model.train(was_training)


FIXED = {
    "adamw": {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0},
    "sgd":   {"momentum": 0.9, "nesterov": False, "weight_decay": 0.0},
}

def _build_optimizer(params, name: str, lr: float):
    name_l = name.lower()
    if name_l == 'adamw':
        return torch.optim.AdamW(
            params, lr=lr,
            betas=FIXED["adamw"]["betas"],
            eps=FIXED["adamw"]["eps"],
            weight_decay=FIXED["adamw"]["weight_decay"]
        )
    elif name_l == 'sgd':
        return torch.optim.SGD(
            params, lr=lr,
            momentum=FIXED["sgd"]["momentum"],
            nesterov=FIXED["sgd"]["nesterov"],
            weight_decay=FIXED["sgd"]["weight_decay"]
        )
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def train_until_plateau_cls(model: nn.Module, train_loader, val_loader,
                            lr: float, precision_bits: int, optimizer_name: str,
                            max_epochs: int, patience: int, min_delta: float,
                            record_first_k: int = 20, pad_idx:int = 0):

    model.to(DEVICE); criterion = nn.CrossEntropyLoss()
    opt = _build_optimizer(model.parameters(), optimizer_name, lr)
    best = float('inf'); no_imp = 0; val_curve=[]
    for epoch in range(1, max_epochs+1):
        model.train(True)
        for xb, lens, yb in train_loader:
            xb, lens, yb = xb.to(DEVICE), lens.to(DEVICE), yb.to(DEVICE).long()
            opt.zero_grad(set_to_none=True)
            logits = model(xb, lens, pad_idx=pad_idx)
            loss = criterion(logits, yb)
            loss.backward(); opt.step()

        model.eval(); v=0.0; n=0
        with torch.no_grad():
            for xb, lens, yb in val_loader:
                xb, lens, yb = xb.to(DEVICE), lens.to(DEVICE), yb.to(DEVICE).long()
                logits = model(xb, lens, pad_idx=pad_idx)
                v += criterion(logits, yb).item(); n += 1
        v /= max(1,n)
        if len(val_curve) < record_first_k: val_curve.append(float(v))
        if v < best - min_delta: best = v; no_imp = 0
        else: no_imp += 1
        if no_imp >= patience: return epoch, val_curve
    return max_epochs, val_curve

def train_until_plateau_s2s(model: nn.Module, train_loader, val_loader,
                            lr: float, precision_bits: int, optimizer_name: str,
                            max_epochs: int, patience: int, min_delta: float,
                            record_first_k: int = 20, pad_idx:int = 0):

    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    opt = _build_optimizer(model.parameters(), optimizer_name, lr)
    best = float('inf'); no_imp = 0; val_curve=[]
    for epoch in range(1, max_epochs+1):
        model.train(True)
        for Xs, Ls, Xt, Lt in train_loader:
            Xs, Ls, Xt, Lt = Xs.to(DEVICE), Ls.to(DEVICE), Xt.to(DEVICE), Lt.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits, max_Tt = model(Xs, Ls, Xt, Lt, pad_idx=pad_idx)
            loss = criterion(logits[:, :max_Tt, :].reshape(-1, logits.size(-1)),
                             Xt[:, :max_Tt].reshape(-1))
            loss.backward(); opt.step()

        model.eval(); v=0.0; n=0
        with torch.no_grad():
            for Xs, Ls, Xt, Lt in val_loader:
                Xs, Ls, Xt, Lt = Xs.to(DEVICE), Ls.to(DEVICE), Xt.to(DEVICE), Lt.to(DEVICE)
                logits, max_Tt = model(Xs, Ls, Xt, Lt, pad_idx=pad_idx)
                v += criterion(logits[:, :max_Tt, :].reshape(-1, logits.size(-1)),
                               Xt[:, :max_Tt].reshape(-1)).item()
                n += 1
        v /= max(1,n)
        if len(val_curve) < record_first_k: val_curve.append(float(v))
        if v < best - min_delta: best = v; no_imp = 0
        else: no_imp += 1
        if no_imp >= patience: return epoch, val_curve
    return max_epochs, val_curve


def main():
    p = argparse.ArgumentParser("CAPE++ RNN Meta-Gen (Validation Early-Stop, Zero-Shot Probe)")

    p.add_argument('--datasets', nargs='+', default=['IMDB','AG_NEWS','SST2','WMT14_EN_DE','CNNDM_SUM'])

    p.add_argument('--models', nargs='+', default=['gnmt'])
    p.add_argument('--data-root', type=str, default='./data')
    p.add_argument('--logdir', type=str, default='./meta_logs')

    p.add_argument('--batch-sizes', nargs='+', type=int, default=[16,32,64])
    p.add_argument('--lrs', nargs='+', type=float, default=[5e-4, 1e-3, 2e-3])
    p.add_argument('--optimizers', nargs='+', default=['AdamW','SGD'])

    p.add_argument('--embedding-dims', nargs='+', type=int, default=[256])
    p.add_argument('--hidden-sizes',  nargs='+', type=int, default=[512])
    p.add_argument('--num-layers', type=int, default=2)
    p.add_argument('--enc-layers', type=int, default=4)
    p.add_argument('--dec-layers', type=int, default=4)


    p.add_argument('--vocab-size', type=int, default=30000)
    p.add_argument('--max-seq-len', type=int, default=256)


    p.add_argument('--max-epochs', type=int, default=150, help='Universal epoch cap used for all model/dataset pairs')
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--min-delta', type=float, default=1e-4)
    p.add_argument('--val-fraction', type=float, default=0.2, help='For datasets without explicit val split')


    p.add_argument('--probe-batch-size', type=int, default=64)
    p.add_argument('--record-first-k', type=int, default=20)

    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = p.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    abs_logdir = os.path.abspath(args.logdir)
    set_seed(args.seed)


    BUILT: Dict[str, Any] = {}
    def _build(name):
        if name == 'IMDB':            return build_imdb(args.data_root, args.max_seq_len, args.vocab_size)
        if name == 'AG_NEWS':         return build_agnews(args.data_root, args.max_seq_len, args.vocab_size)
        if name == 'SST2':            return build_sst2(args.data_root, args.max_seq_len, args.vocab_size)
        if name == 'WMT14_EN_DE':     return build_wmt14(args.data_root, args.max_seq_len, args.vocab_size)
        if name == 'CNNDM_SUM':       return build_cnndm(args.data_root, args.max_seq_len, args.vocab_size)
        raise ValueError(name)

    for name in args.datasets:
        try:
            out = _build(name)
            if name in ('WMT14_EN_DE','CNNDM_SUM'):
                ds, vs, vt, label, N = out
                BUILT[label] = (ds, vs, vt, N)
            else:
                ds, nc, vocab, label, N = out
                BUILT[label] = (ds, nc, vocab, N)
            print(f"[OK] Built dataset: {name}")
        except Exception as e:
            print(f"[WARN] Building dataset {name} failed: {e}")


    ALLOWED = {
        'lstm':   ['IMDB','AG_NEWS'],
        'gru':    ['IMDB','AG_NEWS'],
        'bilstm': ['SST2','IMDB'],
        'gnmt':   ['WMT14_EN_DE','CNNDM_SUM'],
    }

    out_rows: List[Dict[str, Any]] = []
    out_name = "RNN_Convergence.csv"

    for model_name in args.models:
        allowed = [d for d in ALLOWED.get(model_name.lower(), []) if d in BUILT.keys()]
        if not allowed:
            print(f"[WARN] No allowed datasets for model '{model_name}'. Skipping.")
            continue

        for ds_label in allowed:
            is_s2s = (model_name.lower() == 'gnmt')
            if is_s2s:
                ds, vs, vt, total_N = BUILT[ds_label]; vocab = vs
                num_classes = None
            else:
                ds, num_classes, vocab, total_N = BUILT[ds_label]
            pad_idx = vocab[PAD] if isinstance(vocab, Vocab) else getattr(vocab, 'stoi', {}).get(PAD, 0)


            n_total = len(ds); n_val = int(round(args.val_fraction * n_total)); n_train = n_total - n_val
            train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))


            base_collate = getattr(ds, "collate_fn", None)
            setattr(train_ds, "collate_fn", base_collate)
            setattr(val_ds,   "collate_fn", base_collate)


            probe_loader = build_loader(train_ds, batch_size=args.probe_batch_size, shuffle=True, num_workers=args.num_workers)
            probe_batch = next(iter(probe_loader))


            grid: List[Dict[str, Any]] = []
            for opt_name in args.optimizers:
                for lr in args.lrs:
                    for B in args.batch_sizes:
                        for E in args.embedding_dims:
                            for H in args.hidden_sizes:
                                grid.append({'optimizer':opt_name,'lr':lr,'batch_size':B,'E':E,'H':H})

            for t in tqdm(grid, desc=f"{ds_label} — {model_name}", unit="trial"):
                opt_name = t['optimizer']; lr = t['lr']; B = t['batch_size']; E = t['E']; H = t['H']
                precision = 32


                train_loader = build_loader(train_ds, batch_size=B, shuffle=True,  num_workers=args.num_workers)
                val_loader   = build_loader(val_ds,   batch_size=B, shuffle=False, num_workers=args.num_workers)


                try:
                    if is_s2s:
                        model = GNMT(vocab_size=len(vocab), embed_dim=E, hidden=int(H),
                                     layers_enc=max(1,args.enc_layers), layers_dec=max(1,args.dec_layers),
                                     pad_idx=pad_idx, tie_embeddings=True)
                    else:
                        if model_name.lower() == 'bilstm':
                            directions, cell = 2, 'lstm'
                        elif model_name.lower() == 'lstm':
                            directions, cell = 1, 'lstm'
                        else:
                            directions, cell = 1, 'gru'
                        model = RNNClassifier(vocab_size=len(vocab), embed_dim=E, hidden=int(H),
                                              layers=args.num_layers, directions=directions,
                                              num_classes=int(num_classes), cell_type=cell, pad_idx=pad_idx)
                except Exception as e:
                    print(f"[ERROR] Model build failed ({model_name}) on {ds_label}: {e}")
                    continue

                P_params = param_count(model)


                if is_s2s:
                    Xs, Ls, Xt, Lt = probe_batch
                    criterion_probe = nn.CrossEntropyLoss(ignore_index=pad_idx)
                    probe_feats = extract_probe_features_s2s(model, criterion_probe, Xs, Ls, Xt, Lt, pad_idx=pad_idx)
                else:
                    xb, lens, yb = probe_batch
                    criterion_probe = nn.CrossEntropyLoss()
                    probe_feats = extract_probe_features_cls(model, criterion_probe, xb, lens, yb, pad_idx=pad_idx)

                features = {
                    'param_count_log10': float(np.log10(P_params + 1e-8)),
                    'learning_rate_log10': float(np.log10(lr + 1e-12)),
                    'batch_size_log10': float(np.log10(B + 1e-12)),
                    **probe_feats,
                }


                try:
                    if is_s2s:
                        t_conv, val_prefix = train_until_plateau_s2s(
                            model=model, train_loader=train_loader, val_loader=val_loader,
                            lr=lr, precision_bits=precision, optimizer_name=opt_name,
                            max_epochs=args.max_epochs, patience=args.patience, min_delta=args.min_delta,
                            record_first_k=args.record_first_k, pad_idx=pad_idx
                        )
                    else:
                        t_conv, val_prefix = train_until_plateau_cls(
                            model=model, train_loader=train_loader, val_loader=val_loader,
                            lr=lr, precision_bits=precision, optimizer_name=opt_name,
                            max_epochs=args.max_epochs, patience=args.patience, min_delta=args.min_delta,
                            record_first_k=args.record_first_k, pad_idx=pad_idx
                        )
                except RuntimeError as e:
                    print(f"[ERROR] Training failed [{model_name}] on {ds_label} "
                          f"(opt={opt_name}, B={B}, lr={lr}, E={E}, H={H}): {e}")
                    del model
                    if DEVICE.type=='cuda':
                        torch.cuda.empty_cache()
                    continue

                t80 = int(max(1, math.ceil(0.8 * t_conv)))
                t90 = int(max(1, math.ceil(0.9 * t_conv)))

                out_rows.append({
                    'dataset': ds_label,
                    'dataset_size': int(total_N),
                    'model': model_name,
                    'architecture': 'RNN',
                    'precision': 32,
                    'optimizer': opt_name,
                    'learning_rate': float(lr),
                    'batch_size': int(B),
                    'embedding_dim': int(E),
                    'hidden_size': int(H),
                    'num_layers': (int(args.num_layers) if not is_s2s else None),
                    'enc_layers': (int(args.enc_layers) if is_s2s else None),
                    'dec_layers': (int(args.dec_layers) if is_s2s else None),
                    'param_count': int(P_params),

                    **features,

                    'T_conv': int(t_conv),
                    'T_80close': int(t80),
                    'T_90close': int(t90),

                    'patience_P': int(args.patience),
                    'min_delta': float(args.min_delta),
                    'max_epochs_used': int(args.max_epochs),

                    'val_loss_prefix': json.dumps([float(x) for x in val_prefix]),

                    'seed': int(args.seed),
                })

                del model
                if DEVICE.type=='cuda':
                    torch.cuda.empty_cache()

    df = pd.DataFrame(out_rows)
    os.makedirs(abs_logdir, exist_ok=True)
    out_path = os.path.join(abs_logdir, "RNN_Convergence.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[OK] Saved RNN meta-dataset to:\n  {out_path}\nRows: {len(df)}")

if __name__ == '__main__':
    main()
