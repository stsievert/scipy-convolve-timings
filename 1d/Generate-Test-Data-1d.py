#!/usr/bin/env python
# coding: utf-8

from time import time
import itertools

import numpy as np
from sklearn.utils import check_random_state
from scipy.signal import convolve
import pandas as pd
from scipy.stats import reciprocal as loguniform
# see https://github.com/scipy/scipy/pull/10815


def get_random_size(random_state=None, larger=False):
    rng = check_random_state(random_state)
    rv = loguniform(3, 100_000)
    x1, h = rv.rvs(size=2, random_state=rng).astype(int)
    return (x1,), (h,)


def time_convs(S1, S2, mode, random_state=None):
    rng = check_random_state(random_state)
    x = rng.randn(*S1)
    h = rng.randn(*S2)
    start = time()
    _ = convolve(x, h, mode=mode, method="fft")
    fft_time = time() - start

    start = time()
    _ = convolve(x, h, mode=mode, method="direct")
    direct_time = time() - start
    return {"fft_time": fft_time, "direct_time": direct_time}


if __name__ == "__main__":
    data = []
    loop_start = time()
    for seed, mode in itertools.product(range(20_000), ["valid", "full", "same"]):
        S1, S2 = get_random_size(random_state=seed, larger=mode == "valid")
        times = time_convs(S1, S2, mode, random_state=seed)
        datum = {
            "shape1[0]": S1[0],
            "shape2[0]": S2[0],
            "mode": mode,
            "seed": seed,
            **times,
        }
        data.append(datum)
        if seed % 10 == 0:
            today = "2019-11-06"
            df = pd.DataFrame(data)
            df.to_parquet(f"out/{today}-1d-test.parquet", index=False)
        if len(data) % 100 == 0:
            print(f"{len(data)}, {seed} for 1d test. {time() - loop_start:0.2f}")
    df = pd.DataFrame(data)
    df.to_parquet(f"out/{today}-1d-test.parquet", index=False)
