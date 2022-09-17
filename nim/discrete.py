import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
import pytorch_lightning as pl


_C = math.log(2 * math.pi)


def sample(mu, logvar):
    eps = torch.randn_like(mu)
    return mu + (logvar / 2).exp() * eps


def sample_softmax(logits):
    distribution = Categorical(logits=logits)
    x = distribution.sample()
    return F.one_hot(x, logits.size(2))


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
        self.softmax = nn.Linear(h_dim, x_dim)

    def forward(self, x_previous, z, hidden=None):
        z = torch.cat([x_previous, z], 2)
        h, hidden = self.lstm(z, hidden)
        h = F.relu(self.fc(h))
        softmax = self.softmax(h)
        return softmax, hidden


class Net(pl.LightningModule):
    def __init__(self, x_dim, h_dim, lr):
        super().__init__()

        self.save_hyperparameters()
        self.encoder = Encoder(x_dim, h_dim)
        self.decoder = Decoder(x_dim, h_dim)
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x, hidden_e=None, hidden_d=None):
        mu_e, logvar_e = self.encoder(x, hidden_e)
        z = sample(mu_e, logvar_e)
        x_previous = torch.zeros_like(x)
        x_previous[:, 1:] = x[:, :-1]
        softmax_d, _ = self.decoder(x_previous, z, hidden_d)
        return mu_e, logvar_e, softmax_d

    def training_step(self, batch, batch_idx):
        x, y = batch
        mu_e, logvar_e, softmax_d = self(x)
        n_classes = softmax_d.size(2)
        softmax_d = softmax_d.view(-1, n_classes)
        y = y.view(-1)
        reconstruction_loss = self.loss(softmax_d, y)
        kl_loss = -0.5 * (1 + logvar_e - mu_e ** 2 - logvar_e.exp())
        loss = torch.mean(reconstruction_loss) + torch.mean(kl_loss)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


def generate(decoder, size):
    n, t = size

    y = torch.zeros((n, t + 1, decoder.x_dim), dtype=torch.float32)
    z = torch.randn(n, t, decoder.x_dim)

    with torch.no_grad():
        hidden = (torch.zeros(1, n, decoder.h_dim), torch.zeros(1, n, decoder.h_dim))
        for i in range(t):
            x_previous = y[:, i : i + 1, :]
            z_t = z[:, i : i + 1, :]
            softmax, hidden = decoder(x_previous, z_t, hidden=hidden)
            y_t = sample_softmax(softmax)
            y[:, i + 1, :] = y_t[:, 0, :]

    y = y[:, 1:, :]
    y = y.argmax(dim=2)
    return y
