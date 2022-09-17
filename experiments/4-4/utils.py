import numpy as np


def count_arrival(data, interval, lengths=None, max_=None):
    data = np.cumsum(data, axis=1)

    if max_ is None:
        max_ = data.max() + interval
    else:
        max_ = max_ + interval
    mesh = np.arange(0, max_, interval)

    n, t = data.shape

    ys = []
    for i in range(n):
        d = data[i]
        if lengths is not None:
            d = d[: lengths[i]]
        ys.append(np.histogram(d, bins=mesh)[0])

    return mesh[:-1], np.array(ys) / interval
