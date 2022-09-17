import numpy as np


def create_dataset(n, t):
    return np.random.gamma(2, 2, size=(n, t, 1))


if __name__ == "__main__":
    x = create_dataset(1000, 10)
    np.save("experiments/0/train.npy", x)
