import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_C = math.log(2 * math.pi)


class NVM(nn.Module):
    def __init__(self, x_dim, h_dim):
        super().__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim

        self.h_e = nn.Linear(x_dim, h_dim)
        self.mu_e = nn.Linear(h_dim, x_dim)
        self.logvar_e = nn.Linear(h_dim, x_dim)

        self.h_d = nn.Linear(x_dim, h_dim)
        self.mu_d = nn.Linear(h_dim, x_dim)
        self.logvar_d = nn.Linear(h_dim, x_dim)

    def _sample(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + (logvar / 2).exp() * eps

    def encode(self, x):
        h_e = F.relu(self.h_e(x))
        mu_e = self.mu_e(h_e)
        logvar_e = self.logvar_e(h_e)
        return mu_e, logvar_e

    def decode(self, z):
        h_d = F.relu(self.h_d(z))
        mu_d = self.mu_d(h_d)
        logvar_d = self.logvar_d(h_d)
        return mu_d, logvar_d

    def forward(self, x):
        mu_e, logvar_e = self.encode(x)
        z = self._sample(mu_e, logvar_e)
        mu_d, logvar_d = self.decode(z)
        return mu_e, logvar_e, mu_d, logvar_d

    def loss_function(self, x, mu_e, logvar_e, mu_d, logvar_d):
        reconstruction_loss = 0.5 * (
            _C + logvar_d + 1 / logvar_d.exp() * (x - mu_d) ** 2
        )
        kl_loss = -0.5 * (1 + logvar_e - mu_e ** 2 - logvar_e.exp())
        loss = torch.sum(reconstruction_loss) + torch.sum(kl_loss)
        return loss

    def save(self, name):
        with open(f"{name}.json", "w") as f:
            json.dump({"type": "nvm", "x_dim": self.x_dim, "h_dim": self.h_dim}, f)
        torch.save(self.state_dict(), f"{name}.pth")

    def generate(self, size):
        if isinstance(size, int):
            n = size
            t = 1
        elif isinstance(size, tuple):
            n, t = size
        else:
            raise ValueError("size must be an integer or a tuple")
        total = n * t

        with torch.no_grad():
            z = torch.randn(total, self.x_dim)
            mu_d, logvar_d = self.decode(z)
            return self._sample(mu_d, logvar_d).view(size)


class NVL(nn.Module):
    def __init__(self, x_dim, h_dim):
        super().__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim

        self.h_e = nn.LSTM(x_dim, h_dim, batch_first=True)
        self.mu_e = nn.Linear(h_dim, x_dim)
        self.logvar_e = nn.Linear(h_dim, x_dim)

        self.h_d = nn.LSTM(x_dim + x_dim, h_dim, batch_first=True)
        self.mu_d = nn.Linear(h_dim, x_dim)
        self.logvar_d = nn.Linear(h_dim, x_dim)

    def _sample(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + (logvar / 2).exp() * eps

    def encode(self, x, hidden=None):
        h_e, _ = self.h_e(x, hidden)
        mu_e = self.mu_e(h_e)
        logvar_e = self.logvar_e(h_e)
        return mu_e, logvar_e

    def decode(self, x_previous, z, hidden=None):
        z = torch.cat([x_previous, z], 2)
        h_d, hidden = self.h_d(z, hidden)
        mu_d = self.mu_d(h_d)
        logvar_d = self.logvar_d(h_d)
        return mu_d, logvar_d, hidden

    def forward(self, x, hidden_e=None, hidden_d=None):
        mu_e, logvar_e = self.encode(x, hidden_e)
        z = self._sample(mu_e, logvar_e)
        x_previous = torch.zeros_like(x)
        x_previous[:, 1:] = x[:, :-1]
        mu_d, logvar_d, _ = self.decode(x_previous, z, hidden_d)
        return mu_e, logvar_e, mu_d, logvar_d

    def loss_function(self, x, mu_e, logvar_e, mu_d, logvar_d):
        reconstruction_loss = 0.5 * (
            _C + logvar_d + 1 / logvar_d.exp() * (x - mu_d) ** 2
        )
        kl_loss = -0.5 * (1 + logvar_e - mu_e ** 2 - logvar_e.exp())
        loss = torch.sum(reconstruction_loss) + torch.sum(kl_loss)
        return loss

    def save(self, name):
        with open(f"{name}.json", "w") as f:
            json.dump({"type": "nvl", "x_dim": self.x_dim, "h_dim": self.h_dim}, f)
        torch.save(self.state_dict(), f"{name}.pth")

    def generate(self, size):
        n, t = size

        y = torch.zeros((n, t + 1, self.x_dim), dtype=torch.float32)
        z = torch.randn(n, t, self.x_dim)

        with torch.no_grad():
            hidden = (torch.zeros(1, n, self.h_dim), torch.zeros(1, n, self.h_dim))
            for i in range(t):
                x_previous = y[:, i : i + 1, :]
                z_t = z[:, i : i + 1, :]
                mu_d, logvar_d, hidden = self.decode(x_previous, z_t, hidden=hidden)
                y_t = self._sample(mu_d, logvar_d)
                y[:, i + 1, :] = y_t[:, 0, :]

        return y[:, 1:, :]
