import numpy as np


def compute_statistics(x, n_classes):
    count = np.zeros((n_classes, n_classes, n_classes))
    for i in range(len(x)):
        x_i = x[i]
        for j in range(2, len(x_i)):
            count[x_i[j - 2], x_i[j - 1], x_i[j]] += 1

    p = count / count.sum(axis=2, keepdims=True)
    return p
