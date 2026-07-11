# Gated Paired GELU Evaluation Evidence

## Change

For degree-2 and degree-4 `PolynomialGELU`, the negative and positive pieces
use the same input powers. The legacy path evaluated each piece independently,
launching two sets of `addcmul` kernels and recomputing powers.

Production source `faa1679` caches the tiny negative/positive coefficient pair
and evaluates both pieces in one leading dimension. It is deliberately gated
to CUDA float32 tensors with at least 12,000,000 elements and degrees 2 or 4.
CPU, non-float32, degree 0/1, and smaller CUDA tensors retain the legacy path.
The threshold is based on measured RTX 4090 crossovers; paired evaluation is
slower for small tensors.

The paired output layout has two contiguous blocks. Peak live activation
storage is no larger than the legacy second-piece phase, which already retains
the first piece, the second accumulator, and the current power.

## TDD And Tests

- RED source: `408da9e684d7dfa36124ef6b2db28de806ad2e7f`
- RED result: exit `1`; the source gate found no threshold/paired path and the
  behavior test found no `_poly_pair()` method.
- Production source: `faa1679e280f450c2f6319b0685418da50d73fa7`
- Final CUDA test commit: `0fcca30`
- Final compile result: exit `0`.
- Final related gate: 74 tests passed, 4 skipped, exit `0`, across
  `tests.test_stage1_eval_accel`, `tests.test_stage1_approx_reuse`,
  `tests.test_blb_fused_rescale_install`, and
  `tests.test_blb_chain_integrity`.

The final CUDA unit test proves that a small tensor does not populate the pair
cache, a threshold-sized tensor does, and production output exactly equals two
independent `_poly()` calls for degrees 2 and 4.

## Size Gate

Every sweep row was compared bit-for-bit before timing. Batch sizes 16, 24,
and 32 remain below the conservative threshold and stay within measurement
noise of `1.00x`. Degree 1 always stays on the legacy path. Above the gate:

| Batch, shape `B x 86 x 3072` | Degree 2 | Degree 4 |
| --- | ---: | ---: |
| 48 | 1.026x | 1.148x |
| 64 | 1.068x | 1.173x |
| 128 | 1.072x | 1.174x |

## Real MRPC Validation Full

The end-to-end benchmark loaded the cached fine-tuned
`textattack/bert-base-uncased-MRPC` checkpoint and the real 408-row GLUE MRPC
validation split. It used batch size 128, dynamic shapes
`[128x84, 128x86, 128x79, 24x72]`, and Softmax degree 6. The last 24-row batch
falls below the threshold and automatically uses the legacy path.

| GELU configuration | Legacy | Optimized | Speedup | Saved/eval |
| --- | ---: | ---: | ---: | ---: |
| all degree 1 | 329.459 ms | 329.982 ms | 0.998x | noise-range -0.523 ms |
| all degree 2 | 381.980 ms | 372.107 ms | 1.027x | 9.874 ms |
| all degree 4 | 506.963 ms | 466.881 ms | 1.086x | 40.082 ms |
| mixed `[4,2,1] x 4` | 407.610 ms | 390.940 ms | 1.043x | 16.670 ms |

Every configuration produced bit-identical logits, predictions, and accuracy.
The pair cache stayed empty for all-degree1, held 12 entries for all-degree2/4,
and held 8 entries for the mixed configuration. At 50,000 cache-miss
full-validation forwards, the mixed result projects to about 13.9 minutes of
GPU wall avoided; existing exact-config cache hits already skip whole forwards,
so this is a hot-path projection rather than an end-to-end training claim.

GPU sampling recorded 939 samples, 96.81% mean utilization, 100% peak
utilization, 4,169 MiB maximum memory, and 417.22 W maximum power.

Raw sweep and MRPC samples are in `green/benchmark.jsonl`. RED/GREEN return
codes, compact RED summary, final test log, hardware data, and raw GPU samples
are retained alongside it.
