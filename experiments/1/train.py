import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from nim.discrete import Net


class Data(Dataset):
    def __init__(self, x, n_classes):
        x = torch.from_numpy(x).long()
        self.x = F.one_hot(x, num_classes=n_classes).float()
        self.y = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y


def load(filename):
    x = np.load(filename)
    return Data(x, 3)


if __name__ == "__main__":
    train_dataset = load("experiments/1/train.npy")
    train_dataloader = DataLoader(train_dataset, batch_size=1000)

    net = Net(x_dim=3, h_dim=32, lr=0.002)
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=4000,
        precision=16,
        check_val_every_n_epoch=10,
    )
    trainer.fit(net, train_dataloader)
    trainer.save_checkpoint("experiments/1/model.ckpt")
