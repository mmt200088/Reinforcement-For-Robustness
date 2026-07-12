# Normalized Replan-Delta Reuse Evidence

## Change

Source commit `8f5792a` adds a strict zero-copy fast path to
`_normalize_delta_overrides()`. An exact built-in `dict` whose keys are exact
built-in strings and whose values are exact built-in integers or the string
`"x2"` is already in the interface's normalized representation, so it is
returned unchanged.

All other inputs, including bools, numpy integer scalars, string subclasses,
non-string keys, custom mappings, and invalid values, continue through the
original conversion and validation path.

## TDD And Gates

- RED commit: `51c5830`; the focused test proved the old path called the delta
  parser and allocated a replacement dictionary.
- GREEN commit: `8f5792a`; the focused test passed.
- Related Rescale/fusion gate: 131 tests passed, with two existing optional
  torch-bridge skips.
- Python compilation passed for production and test files.

## Hot-Loop A/B

The real `block1_mrpc` template enumerated ranks `[0, 100000)` eleven times
per source.

| Source | Median / 100k | Mean / 100k |
| --- | ---: | ---: |
| Parent `51c5830` | 1.302531 s | 1.304152 s |
| Production `8f5792a` | 1.257134 s | 1.258217 s |

Production is `1.0361x` faster. Valid count, reduced rows, and result SHA256
`8d76a3ab286630d6ee68f7a341961c0673c0c34c264bccb293e8db790bd41b62`
are identical. In the 200,000-combination profile, normalization cumulative
time drops from about `0.315s` to `0.132s`.

## Builder A/B

The real 20-worker builder enumerated all 3,913,140 `block1_mrpc`
combinations three times per source.

| Source | Wall samples | Median | Median user CPU |
| --- | --- | ---: | ---: |
| Parent | 11.40s, 11.50s, 11.48s | 11.48s | 195.04s |
| Production | 11.45s, 11.41s, 11.37s | 11.41s | 193.30s |

The short builder's end-to-end wall improvement is `1.0061x`; median user CPU
drops by `1.74s`. The optimization primarily scales with long combination
loops rather than fixed process startup. All generated maps are equal after
excluding wall metadata; canonical map SHA256 is
`b96031eeff4f44ac76a1df3930493ef766847cf4cab23b59e5eed9911f10fc26`.

Raw RED/GREEN logs, microbenchmarks, profile, builder timings, generated maps,
hardware inventory, and hashes are retained here.
