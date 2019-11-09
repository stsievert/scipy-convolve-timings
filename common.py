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

def _fftconv_faster(x_shape, h_shape, mode):
    """
    See if using fftconvolve or convolve is faster. The value returned (a
    boolean) depends on the sizes and shapes of the input values.
    The big O ratios were found to hold to different machines, which makes
    sense as it's the ratio that matters (the effective speed of the computer
    is found in both big O constants). Regardless, this had been tuned on an
    early 2015 MacBook Pro with 8GB RAM and an Intel i5 processor.
    """
    x_size = _prod(x_shape)
    h_size = _prod(h_shape)
    out_shapes = {
        "full": [n + k - 1 for n, k in zip(x_shape, h_shape)],
        "same": x_shape,
        "valid": [max(n, k) - min(n, k) + 1 for n, k in zip(x_shape, h_shape)],
    }
    out_shape = out_shapes[mode]
    assert all(o >= 0 for o in out_shape)

    S1, S2 = x_shape, h_shape
    if len(x_shape) == 1:
        S1, S2 = S1[0], S2[0]
        direct_muls = {
            "full": S1 *S2,
            "valid": (S2 - S1 + 1) * S1 if S2 >= S1 else (S1 - S2 + 1) * S2,
            "same": S1 * S2 if S1 <= S2 else S1 * S2 - (S2 // 2) * ((S2 + 1) // 2),
        }
    else:
        direct_muls = {
            "full": min(np.prod(S1), np.prod(S2)) * np.prod(out_shape),
            "valid": min(np.prod(S1), np.prod(S2)) * np.prod(out_shape),
            "same": np.prod(S1) * np.prod(S2),
        }
    direct_time = direct_muls[mode]

    # see whether the Fourier transform convolution method or the direct
    # convolution method is faster (discussed in scikit-image PR #1792)
    # direct_time = _direct_muls(x_shape, h_shape, mode)
    fft_time = sum_builtin(n * np.log(n) for n in (x_shape + h_shape +
                                               tuple(out_shape)))
    return fft_time, direct_time


@lru_cache()
def _get_constant(mode, ndim, x_size, h_size, test=False):
    # p = Path(__file__).parent / f"{ndim}d" / "constants.csv"
    # df = pd.read_csv(str(p))
    # if mode != "same":
    #     idx = (df["ndim"] == ndim) & (df["mode"] == mode)
    # else: 
    #     cond = "np_conv" if h_size <= x_size else "sp_conv"
    #     idx = (df["ndim"] == ndim) & (df["mode"] == "same")
    #     if "cond" in df:
    #         idx &= (df["cond"] == cond)
    # assert idx.sum() == 1
    # return df[idx]["constant"].values.item()
    if ndim == 1:
        constants = {
            "valid": 14.336458,
            "full": 11.548068,
            "same": 15.747428 if h_size <= x_size else 0.73367078,
        }
    else:
        constants = {
            "same": 16.487500,
            "valid": 11.680560,
            "full": 10.423440,
        }
    return constants[mode]
        


def _fftconv_faster_test(x_shape, h_shape, mode, test=True):
    """
    See if using fftconvolve or convolve is faster. The value returned (a
    boolean) depends on the sizes and shapes of the input values.
    The big O ratios were found to hold to different machines, which makes
    sense as it's the ratio that matters (the effective speed of the computer
    is found in both big O constants). Regardless, this had been tuned on an
    early 2015 MacBook Pro with 8GB RAM and an Intel i5 processor.
    """
    fft_time, direct_time = _fftconv_faster(x_shape, h_shape, mode)
    big_O_constant = _get_constant(mode, len(x_shape), _prod(x_shape), _prod(h_shape))
    return "fft" if big_O_constant * fft_time < direct_time else "direct"