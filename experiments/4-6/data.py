from math import sin, pi, log
import random

import numpy as np


# def create_dataset(n, time):
#     rate_function = lambda t: 0.5 * sin(pi / 5 * t) + 2
#     max_rate = 2.5

#     data = []
#     for _ in range(n):
#         d = [0]
#         t = 0
#         while t < time:
#             u1 = random.random()
#             t = t - 1 / max_rate * log(u1)
#             u2 = random.random()
#             if u2 < rate_function(t) / max_rate:
#                 d.append(t)
#         d = np.array(d)
#         d = np.diff(d)
#         data.append(d)

#     lengths = [len(d) for d in data]
#     max_length = max(lengths)

#     x = np.zeros((n, max_length))
#     for i in range(n):
#         x[i, : lengths[i]] = data[i]

#     return x, lengths


def create_dataset(n, n_limit):
    rate_function = lambda t: 0.5 * np.sin(np.pi / 5 * t) + 0.02 * t + 1
    max_rate = 10

    data = []
    for _ in range(n):
        d = [0]
        t = 0
        while len(d) < n_limit + 1:
            u1 = random.random()
            t = t - 1 / max_rate * log(u1)
            u2 = random.random()
            if u2 < rate_function(t) / max_rate:
                d.append(t)
        d = np.array(d)
        d = np.diff(d)
        data.append(d)

    return np.array(data)


if __name__ == "__main__":
    data = create_dataset(4000, 200)
    np.save("experiments/4-6/train", data)
