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
            return min(np.prod(S_1), np.prod(S_2)) * np.prod(
                [n + k - 1 for n, k in zip(S_1, S_2)]
            )
    elif mode == "valid":
        if len(S_1) == 1:
            S_1, S_2 = S_1[0], S_2[0]
            if S_2 < S_1:
                S_1, S_2 = S_2, S_1
            return (S_2 - S_1 + 1) * S_1
        else:
            return min(np.prod(S_1), np.prod(S_2)) * np.prod(
                [max(n, k) - min(n, k) + 1 for n, k in zip(S_1, S_2)]
            )
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
        s1, s2 = S1[0], S2[0]
        direct_muls = {
            "full": s1 * s2,
            "valid": (s2 - s1 + 1) * s1 if s2 >= s1 else (s1 - s2 + 1) * s2,
            "same": s1 * s2 if s1 <= s2 else s1 * s2 - (s2 // 2) * ((s2 + 1) // 2),
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
    # fft_time = sum_builtin(n * np.log(n) for n in (x_shape + h_shape +
    #                                            tuple(out_shape)))
    # fft_time = sum(n * np.log(n) for n in (x_shape + h_shape + tuple(out_shape)))
    # N = _prod(n+k+1 for n, k in zip(S1, S2))
    # fft_time = N * np.log(N)
    full_out_shape = [n + k - 1 for n, k in zip(S1, S2)]
    N = _prod(full_out_shape)
    fft_time = 3 * N * np.log(N)  # 3 separate FFTs of size full_out_shape
    return fft_time, direct_time


@lru_cache()
def __read_constant(mode, ndim, np_conv):
    p = Path(__file__).parent / f"{ndim}d" / "constants.csv"
    df = pd.read_csv(str(p))
    if ndim == 2:
        if mode != "same":
            idx = (df["ndim"] == ndim) & (df["mode"] == mode)
        else:
            cond = "np_conv" if np_conv else "sp_conv"
            idx = (df["ndim"] == ndim) & (df["mode"] == "same")
        assert idx.sum() == 1
        return df.loc[idx, ["O_fft", "O_direct", "O_offset"]].values.flatten().tolist()
    elif ndim == 1:
        idx = (df["ndim"] == ndim) & (df["mode"] == "same")
        cond = "np_conv" if np_conv else "sp_conv"
        idx &= df["cond"] == cond
    return df.loc[idx, ["O_fft", "O_direct", "O_offset"]].values.flatten().tolist()


def _get_constant(mode, ndim, x_size, h_size, test=False):
    O_fft, O_direct, O_offset = __read_constant(mode, ndim, h_size <= x_size)
    if ndim == 1:
        O_offset = -1e-3
        if mode == "same" and x_size < h_size:
            O_offset = -1e-5
    else:
        O_offset = -1e-4

    #     if ndim == 1 and mode == "same" and x_size < h_size:
    #         O_offset = -2e-1
    #         O_fft = 1.5193e-5
    #         O_direct = 1.760897e-6

    #     if mode == "same" and ndim == 1 and x_size < h_size:
    # #         O_offset = 100e-6
    #         O_offset = -2e-4
    #         O_direct = 2e-5
    return O_fft, O_direct, O_offset


#     if ndim == 1:
#         offset = 0
#         constants = {
#             "valid": (7.2880e-6, 3.344823e-7, offset),
#             "full": (7.2673e-6, 2.01e-7, offset),
#             "same": (2.3223e-5, 1.51e-6, offset) if h_size <= x_size else\
#                     (2.3427e-5, 17e-6, offset),
#         }
#     else:
#         offset = -2e-4
#         constants = {
#             "valid": (4.24046e-9, 3.344823e-8, offset),
#             "full": (3.4457e-9, 2.06903e-8, offset),
#             "same": (4.14859e-9, 1.65125e-8, offset),
#         }
#     return constants[mode]


def _fftconv_faster_test(x_shape, h_shape, mode, test=True):
    """
    See if using fftconvolve or convolve is faster. The value returned (a
    boolean) depends on the sizes and shapes of the input values.
    The big O ratios were found to hold to different machines, which makes
    sense as it's the ratio that matters (the effective speed of the computer
    is found in both big O constants). Regardless, this had been tuned on an
    early 2015 MacBook Pro with 8GB RAM and an Intel i5 processor.
    """
    ndim = len(x_shape)
    fft_ops, direct_ops = _fftconv_faster(x_shape, h_shape, mode)
    [O_fft, O_direct, O_offset] = _get_constant(
        mode, len(x_shape), _prod(x_shape), _prod(h_shape)
    )
    fft_time = O_fft * fft_ops
    direct_time = O_direct * direct_ops + O_offset
    if mode == "same" and ndim == 1 and _prod(x_shape) < _prod(h_shape):
        if min(fft_time, direct_time) < 1e-4:
            return "direct"
    return "fft" if fft_time < direct_time else "direct"


#     else:
#         fft_time, direct_time = _fftconv_faster(x_shape, h_shape, mode)
#         big_O_constant = _get_constant(mode, len(x_shape), _prod(x_shape), _prod(h_shape))
#         return "fft" if big_O_constant * fft_time < direct_time else "direct"
