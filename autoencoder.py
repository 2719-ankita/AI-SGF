
from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

@dataclass
class AEConfig:
    input_dim: int
    hidden_dims: List[int]
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 128
    device: str = "cpu"

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list):
        super().__init__()
        enc = []
        last = input_dim
        for h in hidden_dims:
            enc += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.encoder = nn.Sequential(*enc)
        dec = []
        for h in reversed(hidden_dims[:-1] + [input_dim]):
            dec += [nn.Linear(last, h), nn.ReLU()]
            last = h
        dec[-1] = nn.Linear(dec[-2].out_features, input_dim)
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def train_autoencoder(X: np.ndarray, cfg: AEConfig):
    device = torch.device(cfg.device)
    model = Autoencoder(cfg.input_dim, cfg.hidden_dims).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    crit = nn.MSELoss()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(X_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    model.train()
    for _ in range(cfg.epochs):
        for (batch,) in loader:
            opt.zero_grad()
            recon = model(batch)
            loss = crit(recon, batch)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        recon = model(X_t)
        re = ((recon - X_t) ** 2).mean(dim=1).cpu().numpy()
    return model, re
