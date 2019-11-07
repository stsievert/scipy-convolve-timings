#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import asarray, array, ndarray
from scipy.signal import convolve
import math


# In[2]:


randn = np.random.randn
import timeit
def time(x, h, mode='valid', method='auto', repeat=20, number=2):
    times = timeit.repeat(
        "convolve(x, h, mode='{}', method='{}')".format(mode, method), 
        "import numpy as np\n" +
        "from scipy.signal import convolve",
        repeat=repeat,
        number=number,
        globals={"x": x, "h": h}
    )
    return min(times) / number


# In[3]:


from sklearn.utils import check_random_state
import pandas as pd
from scipy.stats import reciprocal as loguniform

def _get_data(a, b, ndim, rng):
    if ndim == 2:
        s1 = loguniform(min(3, a//2), a*2).rvs(random_state=rng).astype(int)
        s2 = loguniform(min(3, b//2), b*2).rvs(random_state=rng).astype(int)
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
        
    t_fft = time(x, h, mode=mode, method="fft")
    t_direct = time(x, h, mode=mode, method="direct")
    return {
        "fft_time": t_fft,
        "direct_time": t_direct,
        "mode": mode,
        "x.shape[0]": x.shape[0],
        "x.shape[1]": x.shape[1],
        "h.shape[0]": h.shape[0],
        "h.shape[1]": h.shape[1],
        "random_state": random_state
    }


# In[16]:


import itertools
# sizes = np.logspace(np.log10(3), np.log10(500), num=100).astype("int")
# sizes = np.unique(sizes)
sizes = np.logspace(np.log10(5), np.log10(500), num=40).astype("int")
sizes = np.unique(sizes)
print(sizes)
today = "2019-11-06"

# data = []


# In[19]:


len(data)


# In[20]:


fname = f"out/{today}-2d-train.parquet" 
for n, a, mode in itertools.product(
    sizes, sizes, ["full", "valid", "same"],
):
    if n < 37:
        continue
    datum = time_conv(n, a, random_state=len(data), ndim=2, mode=mode)
    data.append(datum)
    if len(data) % 100 == 0:
        print(n, a, mode, len(data))
        pd.DataFrame(data).to_parquet(fname, index=False)
pd.DataFrame(data).to_parquet(fname, index=False)


# In[ ]:




