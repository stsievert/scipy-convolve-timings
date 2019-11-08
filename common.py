from functools import lru_cache
import numpy as np
import pandas as pd
from pathlib import Path

sum_builtin = sum

def _prod(iterable):
    """
    Product of a list of numbers.
    Faster than np.prod for short lists like array shapes.
    """
    product = 1
    for x in iterable:
        product *= x
    return product

def _direct_muls(S_1, S_2, mode="full"):
    """Prediction of number of multiplications for these shapes and mode"""
    import numpy as np
    if mode == "full":
        if len(S_1) == 1:
            return S_1[0] * S_2[0]
        else:
            return min(np.prod(S_1), np.prod(S_2)) * np.prod([n + k - 1 for n, k in zip(S_1, S_2)])
    elif mode == "valid":
        if len(S_1) == 1:
            S_1, S_2 = S_1[0], S_2[0]
            if S_2 < S_1:
                S_1, S_2 = S_2, S_1
            return (S_2 - S_1 + 1) * S_1
        else:
            return min(np.prod(S_1), np.prod(S_2)) * np.prod([max(n, k) - min(n, k) + 1 for n, k in zip(S_1, S_2)])
    elif mode == "same":
        if len(S_1) == 1:
            S_1, S_2 = S_1[0], S_2[0]
            if S_1 < S_2:
                return S_1 * S_2
            else:
                return S_1 * S_2 - (S_2 // 2) * ((S_2 + 1) // 2)
        else:
            return np.prod(S_1) * np.prod(S_2)

def _fftconv_faster(x_shape, x_size, h_shape, h_size, mode):
    """
    See if using fftconvolve or convolve is faster. The value returned (a
    boolean) depends on the sizes and shapes of the input values.
    The big O ratios were found to hold to different machines, which makes
    sense as it's the ratio that matters (the effective speed of the computer
    is found in both big O constants). Regardless, this had been tuned on an
    early 2015 MacBook Pro with 8GB RAM and an Intel i5 processor.
    """
    if mode == 'full':
        out_shape = [n + k - 1 for n, k in zip(x_shape, h_shape)]
    elif mode == 'same':
        out_shape = x_shape
    elif mode == 'valid':
        out_shape = [n - k + 1 for n, k in zip(x_shape, h_shape)]
    else:
        raise ValueError('mode is invalid')
    out_shape = [o if o > 0 else -o + 2 for o in out_shape]

    # see whether the Fourier transform convolution method or the direct
    # convolution method is faster (discussed in scikit-image PR #1792)
    direct_time = _direct_muls(x_shape, h_shape, mode)
    fft_time = sum_builtin(n * np.log(n) for n in (x_shape + h_shape +
                                               tuple(out_shape)))
    return fft_time, direct_time
    


# def _get_constant(mode, x_ndim, x_size, h_size):
#     if mode == 'full':
#         big_O_constant = 10963.92823819 if x_ndim == 1 else 8899.1104874
#     elif mode == 'same':
#         oneD_big_O = {True: 7183.41306773, False: 856.78174111}
#         big_O_constant = oneD_big_O[h_size <= x_size] if x_ndim == 1 \
#                                                       else 34519.21021589
#     elif mode == 'valid':
#         big_O_constant = 41954.28006344 if x_ndim == 1 else 66453.24316434
#     else:
#         raise ValueError('mode is invalid')
#     return big_O_constant

@lru_cache()
def _get_constant(mode, ndim, x_size, h_size):
    p = Path(__file__).parent / f"{ndim}d" / "constants.csv"
    df = pd.read_csv(str(p))
    if mode != "same":
        idx = (df["ndim"] == ndim) & (df["mode"] == mode)
    else: 
        cond = "smaller_kernel" if h_size <= x_size else "bigger_kernel"
        idx = (df["ndim"] == ndim) & (df["mode"] == "same")
        if "cond" in df:
            idx &= (df["cond"] == cond)
    assert idx.sum() == 1
    return df[idx]["constant"].values.item()


def _fftconv_faster_test(x_shape, h_shape, mode):
    """
    See if using fftconvolve or convolve is faster. The value returned (a
    boolean) depends on the sizes and shapes of the input values.
    The big O ratios were found to hold to different machines, which makes
    sense as it's the ratio that matters (the effective speed of the computer
    is found in both big O constants). Regardless, this had been tuned on an
    early 2015 MacBook Pro with 8GB RAM and an Intel i5 processor.
    """
    x_size = _prod(x_shape)
    x_ndim = len(x_shape)
    h_size = _prod(h_shape)
    h_ndim = len(h_shape)
    if mode == 'full':
        out_shape = [n + k - 1 for n, k in zip(x_shape, h_shape)]
    elif mode == 'same':
        out_shape = x_shape
    elif mode == 'valid':
        out_shape = [n - k + 1 for n, k in zip(x_shape, h_shape)]
    else:
        raise ValueError('mode is invalid')
    out_shape = [o if o > 0 else -o + 2 for o in out_shape]

    # see whether the Fourier transform convolution method or the direct
    # convolution method is faster (discussed in scikit-image PR #1792)
    direct_time = min(x_size, h_size) * _prod(out_shape)
    fft_time = sum_builtin(n * np.log(n) for n in (x_shape + h_shape +
                                               tuple(out_shape)))
    big_O_constant = _get_constant(mode, x_ndim, x_size, h_size)
    return "fft" if big_O_constant * fft_time < direct_time else "direct"