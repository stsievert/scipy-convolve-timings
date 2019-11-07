#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.utils import check_random_state
from scipy.signal import convolve
from time import time
from scipy.stats import reciprocal as loguniform  # see https://github.com/scipy/scipy/pull/10815


# In[2]:


def get_random_size(random_state=None, larger=False):
    rng = check_random_state(random_state)
    rv = loguniform(3, 100_000)
    x1, h = rv.rvs(size=2, random_state=rng).astype(int)
    return (x1, ), (h, )

def time_convs(S1, S2, mode, random_state=None):
    rng = check_random_state(random_state)
    x = rng.randn(*S1)
    h = rng.randn(*S2)
    fft_times = []
    direct_times = []
    loop_start = time()
    while True:
        start = time()
        _ = convolve(x, h, mode=mode, method="fft")
        fft_times.append(time() - start)
        
        start = time()
        _ = convolve(x, h, mode=mode, method="direct")
        direct_times.append(time() - start)
        if time() - loop_start > 0.01:
            break
            
    return {"fft_time": min(fft_times), "direct_time": min(direct_times)}


# In[ ]:


import pandas as pd
import itertools
data = []
for seed, mode in itertools.product(
    range(10_000), ["valid", "full", "same"], 
):
    S1, S2 = get_random_size(random_state=seed, larger=mode == "valid")
    times = time_convs(
        S1, S2, mode, random_state=seed,
    )
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
        print(seed, mode)


# In[ ]:





# In[ ]:




