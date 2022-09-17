import numpy as np


def _create_one(t, a, b):
    e = np.random.randn(t)
    x = np.zeros(t)
    x[:13] = e[:13]
    for j in range(13, len(x)):
        x[j] = x[j - 1] + a * x[j - 12] - a * x[j - 13] + e[j] + b * e[j - 1]
    return x


def create_dataset(n, t):
    a, b = 0.3, -0.4
    data = np.zeros([n, t])
    for i in range(n):
        data[i] = _create_one(t, a, b)
    return data


if __name__ == "__main__":
    x = data = create_dataset(2000, 100)
    np.save("experiments/5/train", x)
