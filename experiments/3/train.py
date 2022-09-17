import numpy as np
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
        return x


def load(filename):
    x = np.load(filename)
    x = np.diff(x, axis=1)
    n, t = x.shape
    x = x.reshape(n, t, 1)
    return Data(x, log=False)


if __name__ == "__main__":
    train_dataset = load("experiments/3/train.npy")
    train_dataloader = DataLoader(train_dataset, batch_size=1000)

    net = Net(x_dim=1, h_dim=32, lr=0.002)
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=1000,
        precision=16,
        check_val_every_n_epoch=10,
    )
    trainer.fit(net, train_dataloader)
    trainer.save_checkpoint("experiments/3/model.ckpt")
