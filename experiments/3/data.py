import numpy as np


def create_dataset(n, t):
    x = np.zeros([n, t])
    x[:, 1:] = np.random.randn(n, t - 1) * 0.1
    x = np.cumsum(x, axis=1)
    return x


if __name__ == "__main__":
    data = create_dataset(1000, 100)
    np.save("experiments/3/train.npy", data)
