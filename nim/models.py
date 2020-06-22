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

    def generate(self, n):
        with torch.no_grad():
            z = torch.randn(n, self.x_dim)
            mu_d, logvar_d = self.decode(z)
            return self._sample(mu_d, logvar_d)