Times SciPy's [`convolve`][conv] function to determine the constants for
[`choose_conv_method`][ccm]

[conv]:https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html#scipy.signal.convolve
[ccm]:https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.choose_conv_method.html


**Goals**:

- [ ] update 2D convolve numbers for SciPy 1.4
- [ ] update 1D convolve numbers for SciPy 1.4

**TODO:**

- [ ] 2D timings: have constant written to disk in "train"; read in "test"
- [ ] Implement 1D timings
- [ ] Write 3 versions to disk:
    * native macOS, SciPy 1.3
    * docker, SciPy 1.3
    * docker, SciPy 1.4
- [ ] run tests on Docker and SciPy 1.4