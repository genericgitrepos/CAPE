import torch
import torch.nn as nn
import random
import numpy as np


def build_cnn(input_shape: tuple, num_classes: int) -> nn.Module:
    c, h, w = input_shape
    layers = []
    in_ch = c
    pool_count = 0

    num_blocks = random.randint(1, 4)
    for _ in range(num_blocks):
        out_ch = random.choice([16, 32, 64, 128])
        k = random.choice([3,5])
        layers += [
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k//2),
            nn.ReLU(inplace=True)
        ]

        if random.random() < 0.5:
            layers.append(nn.MaxPool2d(2))
            pool_count += 1
        in_ch = out_ch

    layers.append(nn.Flatten())

    h_f = h // (2**pool_count) if pool_count>0 else h
    w_f = w // (2**pool_count) if pool_count>0 else w
    feat_dim = in_ch * h_f * w_f

    num_fcs = random.randint(1,2)
    dims = [feat_dim] + [random.choice([64,128,256]) for _ in range(num_fcs)] + [num_classes]
    
    for in_d, out_d in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(in_d, out_d))
        if out_d != num_classes:
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def extract_cnn_probe_features(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion, device):
    model.to(device).train()
    
    logP = np.log(sum(p.numel() for p in model.parameters()))
    logB = np.log(min(32, X.size(0)))
    
    Xp, yp = X[:32].to(device), y[:32].to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    g2_list, tau_list = [], []
    
    for xi, yi in zip(Xp, yp):
        xi, yi = xi.unsqueeze(0), yi.unsqueeze(0)
        model.zero_grad()
        logits = model(xi)
        loss = criterion(logits, yi)
        grads = torch.autograd.grad(loss, params, retain_graph=True)
        gv = torch.cat([g.contiguous().view(-1) for g in grads])
        g2_list.append((gv**2).sum().item())
        
        model.zero_grad()
        true_logit = logits.view(-1)[yi.item()]
        grads_f = torch.autograd.grad(true_logit, params, retain_graph=True)
        fv = torch.cat([g.contiguous().view(-1) for g in grads_f])
        tau_list.append((fv**2).sum().item())

    logG2  = np.log(np.mean(g2_list))
    logTau = np.log(np.sum(tau_list))
    
    return logP, logB, logG2, logTau