import torch
import torch.nn as nn
import random
import numpy as np
import timm


def build_transformer(name, num_classes):
    model = timm.create_model(name, pretrained=True, num_classes=num_classes)
    for pname, p in model.named_parameters():
        if "head" not in pname and "fc" not in pname and "classifier" not in pname:
            p.requires_grad = False
    return model


def extract_transformer_probe_features(model, X, y, criterion, device):
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

    logG2 = np.log(np.mean(g2_list))
    logTau = np.log(np.sum(tau_list))

    return logP, logB, logG2, logTau