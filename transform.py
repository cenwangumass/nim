import numpy as np


def log_nvm(x):
    return np.log(x).reshape(-1, 1)


def log_nvl(x):
    n, t = x.shape
    return np.log(x).reshape(n, t, 1)


def exp(x):
    return np.exp(x)
