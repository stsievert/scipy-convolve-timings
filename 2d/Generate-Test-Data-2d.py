#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.utils import check_random_state
from scipy.signal import convolve
from time import time
from scipy.stats import reciprocal as loguniform  # see https://github.com/scipy/scipy/pull/10815


# In[6]:


def get_random_size(random_state=None, larger=False):
    rng = check_random_state(random_state)
    rv = loguniform(3, 400)
    x1, x2 = rv.rvs(size=2, random_state=rng).astype(int)
    h = loguniform(3, 100).rvs(random_state=rng).astype(int)
    if larger:
        h, x1, x2 = np.sort([h, x1, x2])
    return (x1, x2), (h, h)

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


# In[7]:


import pandas as pd
import itertools
data = []
for seed, mode in itertools.product(
    range(1000), ["valid", "full", "same"], 
):
    S1, S2 = get_random_size(random_state=seed, larger=mode == "valid")
    times = time_convs(
        S1, S2, mode, random_state=seed,
    )
    datum = {
        "shape1[0]": S1[0],
        "shape2[0]": S2[0],
        "shape1[1]": S1[1],
        "shape2[1]": S2[1],
        "mode": mode,
        "seed": seed,
        **times,
    }
    data.append(datum)
    if seed % 10 == 0:
        today = "2019-11-06"
        df = pd.DataFrame(data)
        df.to_parquet(f"out/{today}-2d-test.parquet", index=False)
        print(seed, mode)


# In[ ]:




