import numpy as np
from scipy.optimize import curve_fit
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from nim.models import Net


class Data(Dataset):
    def __init__(self, x, log=False):
        self.x = x.astype(np.float32)
        self.log = log

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        if self.log:
            x = np.log(x)
            x = np.nan_to_num(x, nan=0, posinf=0, neginf=0)
        return x


def f(t, a, b, c, d, e):
    return a * np.cos(b * t + c) * np.exp(-d * t) + e


def load(filename):
    x = np.load(filename)
    n, T = x.shape
    x = np.log(x)

    mean = x.mean(axis=0)
    t = np.arange(T)
    params, _ = curve_fit(f, t, mean, p0=[0.1, 0.1, 0.1, 0.1, 0.1])
    x = x - f(t, *params)

    x = x.reshape(n, T, 1)
    np.save("experiments/4-5/params", params)

    return Data(x, log=False)


if __name__ == "__main__":
    train_dataset = load("experiments/4-5/train.npy")
    train_dataloader = DataLoader(train_dataset, batch_size=1000)

    net = Net(x_dim=1, h_dim=32, lr=0.002)
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=2000,
        precision=16,
        check_val_every_n_epoch=10,
    )
    trainer.fit(net, train_dataloader)
    trainer.save_checkpoint("experiments/4-5/model.ckpt")