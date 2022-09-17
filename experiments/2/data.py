import numpy as np


def sample_path(a, t):
    y = np.zeros([t, 2])
    e = np.random.randn(2)
    y[0] = e
    for i in range(1, t):
        y[i] = a @ y[i - 1] + np.random.randn(2)
    return y


def create_dataset(n, t):
    a = np.array([[0.3, -0.4], [-0.6, -0.2]])
    data = np.zeros([n, t, 2])
    for i in range(n):
        data[i] = sample_path(a, t)
    return data


if __name__ == "__main__":
    data = create_dataset(1000, 100)
    np.save("experiments/2/train.npy", data)
