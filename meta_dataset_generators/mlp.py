import torch
import torch.nn as nn
import random
import numpy as np


def build_mlp(input_shape: tuple, num_classes: int) -> nn.Module:
    input_dim = int(np.prod(input_shape))
    layers = [nn.Flatten()]
    depth = random.randint(1, 3)
    sizes = [random.choice([64, 128, 256]) for _ in range(depth)]
    dims = [input_dim] + sizes + [num_classes]
    for in_d, out_d in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(in_d, out_d))
        if out_d != num_classes:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def extract_mlp_probe_features(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion, device):
    model.to(device).train()
    
    X = X.view(X.size(0), -1)

    P = sum(p.numel() for p in model.parameters())
    logP = np.log(P)

    Bp = min(32, X.size(0))
    logB = np.log(Bp)

    Xp, yp = X[:Bp].to(device), y[:Bp].to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    g2_list, tau_list = [], []
    for i in range(Bp):
        xi, yi = Xp[i:i+1], yp[i:i+1]

        model.zero_grad()
        logits_i = model(xi)
        loss_i = criterion(logits_i, yi)
        grads = torch.autograd.grad(loss_i, params, retain_graph=True)
        grad_vec = torch.cat([g.contiguous().view(-1) for g in grads])
        g2_list.append((grad_vec**2).sum().item())

        model.zero_grad()
        true_logit = logits_i.view(-1)[yi.item()]
        grads_f = torch.autograd.grad(true_logit, params, retain_graph=True)
        grad_f_vec = torch.cat([g.contiguous().view(-1) for g in grads_f])
        tau_list.append((grad_f_vec**2).sum().item())

    logG2 = np.log(np.mean(g2_list))
    logTau = np.log(np.sum(tau_list))

    return logP, logB, logG2, logTau