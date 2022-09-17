import random

import numpy as np


def create_dataset(n, t):
    # fmt: off
    p = np.array([[[0.1, 0.5, 0.4],
                   [1 / 3, 1 / 3, 1 / 3],
                   [1 / 2, 1 / 3, 1 / 6]],
                  [[0.2, 0.7, 0.1],
                   [0.8, 0.1, 0.1],
                   [0.1, 0.8, 0.1]],
                  [[0.3, 0.3, 0.4],
                   [0.4, 0.4, 0.2],
                   [0.1, 1 / 3, 1 - 0.1 - 1 / 3]]])
    # fmt: on

    data = []
    for _ in range(n):
        d = [random.randrange(3), random.randrange(3)]
        for _ in range(t - 2):
            d.append(random.choices([0, 1, 2], weights=p[d[-2], d[-1]])[0])
        data.append(d)

    return np.array(data)


if __name__ == "__main__":
    data = create_dataset(1000, 50)
    # np.save("experiments/1/train.npy", data)
