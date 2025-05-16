import torch
import torch.nn as nn
import random
import numpy as np


def build_rnn(input_shape: tuple, num_classes: int) -> nn.Module:
    C, H, W = input_shape
    input_size = C * W
    hidden_size = random.choice([64, 128, 256])
    num_layers = random.randint(1, 3)
    bidirectional = random.choice([False, True])
    cell_type = random.choice(['LSTM', 'GRU'])

    class RNNClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            Cell = nn.LSTM if cell_type == 'LSTM' else nn.GRU
            self.rnn = Cell(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=bidirectional)
            self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1),
                                num_classes)

        def forward(self, x):
            B = x.size(0)
            seq = x.view(B, C, H, W).permute(0, 2, 1, 3)   \
                   .contiguous().view(B, H, C * W)
            out, _ = self.rnn(seq)
            last = out[:, -1, :]
            return self.fc(last)

    return RNNClassifier()


def extract_rnn_probe_features(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion, device):
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