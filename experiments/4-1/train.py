import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from nim.models import Net


class Data(Dataset):
    def __init__(self, x, lengths, log=False):
        self.x = x.astype(np.float32)

        n = x.shape[0]
        t = x.shape[1]
        self.mask = np.zeros([n, t])
        for i in range(n):
            self.mask[i, : lengths[i]] = 1

        self.log = log

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        mask = self.mask[index]
        if self.log:
            x = np.log(x)
            x = np.nan_to_num(x, nan=0, posinf=0, neginf=0)
        return x, mask


def load(filename):
    data = np.load(filename)
    x = data["x"]
    lengths = data["lengths"]
    n, t = x.shape
    x = x.reshape(n, t, 1)
    return Data(x, lengths, log=True)


if __name__ == "__main__":
    train_dataset = load("experiments/4-1/train.npz")
    train_dataloader = DataLoader(train_dataset, batch_size=1000)

    net = Net(x_dim=1, h_dim=32, lr=0.005)
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=4000,
        precision=16,
        check_val_every_n_epoch=10,
    )
    trainer.fit(net, train_dataloader)
    trainer.save_checkpoint("experiments/4-1/model.ckpt")
