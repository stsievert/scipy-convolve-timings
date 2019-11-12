These timings were run on r5a.2xlarge.

This runs convolutions under...

* SciPy PR 11031 (https://github.com/scipy/scipy/pull/11031) on commit
  1f2d323979404bc537bf507de7ca17d1b6093c9d.
* SciPy 1.2.0 (commit 722bfc3)

For the SciPy 1.2.0 tests, the 1D convolutions tended to error out with
`mode="valid"` (some bug in `choose_conv_method`. I threw everything in a
try-except block).
