#!/usr/bin/env python
# coding: utf-8

import math
import itertools
from time import time

import numpy as np
from numpy import asarray, array, ndarray
from scipy.signal import convolve
from sklearn.utils import check_random_state
import pandas as pd
from scipy.stats import reciprocal as loguniform


def _time_conv(x, h, mode="valid", method="auto"):
    times = []
    loop_start = time()
    while True:
        start = time()
        _ = convolve(x, h, mode=mode, method=method)
        times.append(time() - start)
        if time() - loop_start > 0.1:
            break
    return min(times)


def _get_data(a, b, ndim, rng):
    if ndim == 2:
        s1 = loguniform(min(3, a // 2), a * 2).rvs(random_state=rng).astype(int)
        s2 = loguniform(min(3, b // 2), b * 2).rvs(random_state=rng).astype(int)
        x = rng.randn(a, s1)
        h = rng.randn(b, s2)
        if mode == "valid":
            s = np.sort([a, b, s1, s2])
            h = rng.randn(*s[:2])
            x = rng.randn(*s[2:])
            assert all(h.shape[i] <= x.shape[j] for i in [0, 1] for j in [0, 1])
    elif ndim == 1:
        x = rng.randn(a)
        h = rng.randn(b)
    else:
        raise ValueError("ndim")
    return x, h


def time_conv(n, a, mode="full", ndim=2, random_state=None):
    rng = check_random_state(random_state)
    x, h = _get_data(n, a, ndim, rng)

    t_fft = _time_conv(x, h, mode=mode, method="fft")
    t_direct = _time_conv(x, h, mode=mode, method="direct")
    return {
        "fft_time": t_fft,
        "direct_time": t_direct,
        "mode": mode,
        "x.shape[0]": x.shape[0],
        "x.shape[1]": x.shape[1],
        "h.shape[0]": h.shape[0],
        "h.shape[1]": h.shape[1],
        "random_state": random_state,
    }


if __name__ == "__main__":
    sizes = np.logspace(np.log10(5), np.log10(500), num=70).astype("int")
    sizes = np.unique(sizes)
    print(sizes)
    today = "2019-11-06"
    fname = f"out/{today}-2d-train.parquet"
    data = []
    loop_start = time()
    for n, a, mode in itertools.product(sizes, sizes, ["full", "valid", "same"]):
        datum = time_conv(n, a, random_state=len(data), ndim=2, mode=mode)
        data.append(datum)
        if len(data) % 100 == 0:
            pd.DataFrame(data).to_parquet(fname, index=False)
        if len(data) % 100 == 0:
            print(f"{len(data)} {n} for 2d train. {time() - loop_start:0.2f}")
    pd.DataFrame(data).to_parquet(fname, index=False)
