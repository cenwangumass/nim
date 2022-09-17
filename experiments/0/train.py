import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

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
    return Data(x, log=True)


if __name__ == "__main__":
    train_dataset = load("experiments/0/train.npy")
    train_dataloader = DataLoader(train_dataset, batch_size=1000)

    net = Net(x_dim=1, h_dim=32, lr=0.001)
    trainer = pl.Trainer(
        # gpus=1,
        max_epochs=4000,
        # precision=16,
        check_val_every_n_epoch=10,
    )
    trainer.fit(net, train_dataloader)
    trainer.save_checkpoint("experiments/0/model.ckpt")
