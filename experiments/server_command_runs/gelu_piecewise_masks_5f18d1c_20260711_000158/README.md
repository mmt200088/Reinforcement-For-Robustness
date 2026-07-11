# Reduced GELU Piecewise-Mask Evidence

## Change

`_select_piecewise_gelu_output()` is shared by plaintext `PolynomialGELU` and
the noisy Block-5 GELU path. The legacy implementation built two compound
interval masks:

```python
(x >= -2.7) & (x < 0)
(x >= 0) & (x <= 2.7)
```

Production source `5f18d1c` selects the negative/positive polynomial with one
`x < 0` comparison, uses `x >= -2.7` to preserve the low and NaN zero branch,
then retains the existing `x > 2.7` identity branch. It removes four
full-shape comparison/boolean operations per GELU selection without changing
the polynomial, coefficients, interval boundaries, or output values.

## TDD And Tests

- RED source: `50e2ebdc40c5d616e003babe4e1bb05cd1236b15`
- RED command: the new source-allocation gate plus the legacy boundary/special
  value equivalence test in `tests.test_stage1_eval_accel`
- RED result: exit `1`; the implementation gate failed on the two old compound
  masks while the behavior test passed.
- GREEN source: `5f18d1cddf3924d33ca6604ca46de065205c2257`
- Compile result: exit `0` for `function_handler.py` and the test module.
- Relevant GREEN gate: 71 tests passed, 4 skipped, exit `0`, across
  `tests.test_stage1_eval_accel`, `tests.test_stage1_approx_reuse`,
  `tests.test_blb_fused_rescale_install`, and
  `tests.test_blb_chain_integrity`.

An intentionally broader first gate ran 110 tests: 108 passed and two
`BLBActionFinalEvalRegressionTests` failed. Those failures concern removed
Paean `apply_optimizer_output_to_cfg` plumbing and the old expectation that an
optimizer-invalid candidate must still run a model forward. Replaying exactly
those two tests on parent source `df9a853` produced the same one error and one
failure, proving they predate and are unrelated to this GELU change. Both logs
are retained rather than treating the broad gate as green.

## GPU Benchmark

All measurements used the server RTX 4090. Each arm was warmed up, arm order
alternated, and medians were taken across repeated samples. Legacy and
optimized tensors were compared before timing.

### Selector Kernel

| FFN shape | Legacy | Optimized | Speedup | Value parity |
| --- | ---: | ---: | ---: | --- |
| `16 x 128 x 3072` | 0.275784 ms | 0.219014 ms | 1.259x | bit-identical |
| `128 x 128 x 3072` | 3.565670 ms | 2.700070 ms | 1.321x | bit-identical |

The parity fixture also covers `-2.7`, negative and positive zero, `2.7`, both
out-of-band sides, positive/negative infinity, and NaN.

### Real MRPC Validation Full

The end-to-end arm loaded the cached fine-tuned
`textattack/bert-base-uncased-MRPC` checkpoint and the real 408-row GLUE MRPC
validation split. It used batch size 128 and dynamic batch sequence lengths
`[84, 86, 79, 72]`, with Softmax degree 6 in every layer.

| GELU configuration | Legacy | Optimized | Speedup | Saved/eval |
| --- | ---: | ---: | ---: | ---: |
| all degree 1 | 350.757 ms | 329.349 ms | 1.065x | 21.408 ms |
| all degree 2 | 403.087 ms | 382.084 ms | 1.055x | 21.002 ms |
| all degree 4 | 528.299 ms | 506.905 ms | 1.042x | 21.394 ms |
| mixed `[4,2,1] x 4` | 428.770 ms | 407.482 ms | 1.052x | 21.288 ms |

Every configuration produced bit-identical logits, predictions, and accuracy
between arms. At 50,000 cache-miss full-validation forwards, the mixed result
would avoid about 17.7 minutes of GPU wall time; cache hits already skip the
whole forward, so this is a hot-path projection rather than an end-to-end run
claim.

GPU sampling recorded 938 samples, 97.03% mean utilization, 100% peak
utilization, 5,267 MiB maximum memory, and 415.36 W maximum power. The measured
gain therefore occurred under a saturated GPU workload.

Raw samples are in `green/benchmark.jsonl`; hardware and utilization summaries,
RED/GREEN logs, the pre-existing parent replay, return codes, and timing files
are in the adjacent directories.
