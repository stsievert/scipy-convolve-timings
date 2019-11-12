from time import time
import numpy as np
import pandas as pd
import scipy
from scipy.stats import loguniform, uniform
from scipy.signal import convolve, choose_conv_method
import itertools
from pprint import pprint
assert scipy.__file__ == '/home/scipy/scipy/__init__.py'

def _time_1d(seed, mode):
    rng = np.random.RandomState(2**31 - seed)
    n, k = loguniform(3, 5e4).rvs(size=2, random_state=rng).astype(int)
    x = rng.randn(n)
    h = rng.randn(k)
    assert x.ndim == 1 and h.ndim == 1
    datum = {"x_shape": n, "h_shape": k, "seed": seed, "mode": mode, "ndim": 1}
    datum["choose_conv_method"] = choose_conv_method(x, h, mode)
    for method in ["fft", "direct", "auto"]:
        start = time()
        y = convolve(x, h, mode=mode, method=method)
        datum[method + "_time"] = time() - start
    return datum 

def _time_2d(seed, mode):
    rng = np.random.RandomState(2**31 - seed)
    n = loguniform(1e1, 500).rvs(size=1, random_state=rng).astype(int).item()
    k = loguniform(3, 75).rvs(size=1, random_state=rng).astype(int).item()
    r1, r2 = uniform(1, 2).rvs(size=2, random_state=rng)
    n2, k2 = int(r1 * n), int(r2 * k)
    if mode == "valid":
        k, k2, n, n2 = np.sort([n, n2, k, k2])
    x = rng.randn(n, n2)
    h = rng.randn(k, k2)
    assert x.ndim == 2 and h.ndim == 2
    datum = {"x_shape0": n, "h_shape0": k, "x_shape1": n2, "h_shape1": k2,
             "seed": seed, "mode": mode, "ndim": 2}
    datum["choose_conv_method"] = choose_conv_method(x, h, mode)
    for method in ["fft", "direct", "auto"]:
        start = time()
        y = convolve(x, h, mode=mode, method=method)
        datum[method + "_time"] = time() - start
    return datum 

if __name__ == "__main__":
    seeds = range(50000)
    modes = ["full", "valid", "same"]
    data = []
    for seed in seeds:
        for mode in modes:
            datum = _time_1d(seed, mode)
            data.append(datum)
            datum = _time_2d(seed, mode)
            data.append(datum)
            if len(data) % 10 == 0:
                df = pd.DataFrame(data)
                df.to_parquet("2019-11-11-timings-v2.parquet")
                frac_fft_faster = {}
                for ndim in [1, 2]:
                    for mode in modes:
                        d = df[(df["mode"] == mode) & (df["ndim"] == ndim)]
                        frac_fft_faster[(ndim, mode)] = (d["fft_time"] < d["direct_time"]).sum() / len(d)
                msg = "{} {} {:0.2e} {:0.2e}"
                msg = msg.format(seed, len(data), df.fft_time.max(), df.direct_time.max())
                print(msg)
                pprint({k: f"{v:0.2f}" for k, v in frac_fft_faster.items()})

    df = pd.DataFrame(data)
    df.to_parquet("2019-11-11-timings.parquet")
