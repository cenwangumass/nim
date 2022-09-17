import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl


_C = math.log(2 * math.pi)


def sample(mu, logvar):
    eps = torch.randn_like(mu)
    return mu + (logvar / 2).exp() * eps


class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim):
        super().__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim

        self.lstm = nn.LSTM(x_dim, h_dim, batch_first=True)
        self.fc = nn.Linear(h_dim, h_dim)
        self.mu = nn.Linear(h_dim, x_dim)
        self.logvar = nn.Linear(h_dim, x_dim)

    def forward(self, x, hidden=None):
        h, _ = self.lstm(x, hidden)
        h = F.relu(self.fc(h))
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim):
        super().__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim

        self.lstm = nn.LSTM(x_dim + x_dim, h_dim, batch_first=True)
        self.fc = nn.Linear(h_dim, h_dim)
        self.mu = nn.Linear(h_dim, x_dim)
        self.logvar = nn.Linear(h_dim, x_dim)

    def forward(self, x_previous, z, hidden=None):
        z = torch.cat([x_previous, z], 2)
        h, hidden = self.lstm(z, hidden)
        h = F.relu(self.fc(h))
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar, hidden


class Net(pl.LightningModule):
    def __init__(self, x_dim, h_dim, lr, step_size=None, gamma=None):
        super().__init__()

        self.save_hyperparameters()
        self.encoder = Encoder(x_dim, h_dim)
        self.decoder = Decoder(x_dim, h_dim)
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma

    def forward(self, x, hidden_e=None, hidden_d=None):
        mu_e, logvar_e = self.encoder(x, hidden_e)
        z = sample(mu_e, logvar_e)
        x_previous = torch.zeros_like(x)
        x_previous[:, 1:] = x[:, :-1]
        mu_d, logvar_d, _ = self.decoder(x_previous, z, hidden_d)
        return mu_e, logvar_e, mu_d, logvar_d

    def training_step(self, batch, batch_idx):
        if isinstance(batch, list):
            x, mask = batch
            mu_e, logvar_e, mu_d, logvar_d = self(x)
            reconstruction_loss = 0.5 * (
                _C + logvar_d + 1 / logvar_d.exp() * (x - mu_d) ** 2
            )
            reconstruction_loss = reconstruction_loss.sum(dim=2)
            kl_loss = -0.5 * (1 + logvar_e - mu_e ** 2 - logvar_e.exp())
            kl_loss = kl_loss.sum(dim=2)

            loss = torch.mean(reconstruction_loss * mask) + torch.mean(kl_loss * mask)
            self.log("train_loss", loss)
            return loss
        else:
            x = batch
            mu_e, logvar_e, mu_d, logvar_d = self(x)
            reconstruction_loss = 0.5 * (
                _C + logvar_d + 1 / logvar_d.exp() * (x - mu_d) ** 2
            )
            kl_loss = -0.5 * (1 + logvar_e - mu_e ** 2 - logvar_e.exp())
            loss = torch.mean(reconstruction_loss) + torch.mean(kl_loss)
            self.log("train_loss", loss)
            return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        if self.step_size is None:
            return optimizer
        else:
            steplr = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
            return [optimizer], [steplr]


def generate(decoder, size):
    n, t = size

    y = torch.zeros((n, t + 1, decoder.x_dim), dtype=torch.float32)
    z = torch.randn(n, t, decoder.x_dim)

    with torch.no_grad():
        hidden = (torch.zeros(1, n, decoder.h_dim), torch.zeros(1, n, decoder.h_dim))
        for i in range(t):
            x_previous = y[:, i : i + 1, :]
            z_t = z[:, i : i + 1, :]
            mu_d, logvar_d, hidden = decoder(x_previous, z_t, hidden=hidden)
            y_t = sample(mu_d, logvar_d)
            y[:, i + 1, :] = y_t[:, 0, :]

    return y[:, 1:, :]
