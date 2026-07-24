# Verification

## Semantic Parity

- Final versus aggregate: 180/180 episode equality, strict diagnostic equality,
  and 2/2 PPO update equality.
- Inference-noise fast path ON versus OFF: the same three checks pass.
- CUDA unit coverage verifies exact output tensors and exact next RNG state.
- Public Rescale cache callers retain mutation isolation.

## Tests

Focused single-GPU optimization tests:

```text
33 passed
289 passed, 2 skipped, 35 subtests passed
```

Focused five-GPU run:

```text
322 passed, 1 failed, 35 subtests passed
```

The one failure is
`tests/test_blb_chain_integrity.py::ProbeRunnerTwoGPUTest::test_runner_returns_results_in_trial_order`.
It reproduces on aggregate baseline `accc27d6`: its stub requires labels while
the shared inference helper intentionally filters labels.

Broad RL regression:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m pytest -q \
  tests/test_blb*.py tests/test_rescale_optimizer*.py tests/test_stage1*.py
```

```text
1088 passed, 9 failed, 2 skipped, 1 warning, 76 subtests passed in 96.50s
```

The same nine historical failures are present on the aggregate baseline. They
cover RealReplan mock/config expectations, a fusion AST assertion, a Paean
legacy attribute, an all-max legacy assertion, and a skipped-reason string.
They were not changed because they are unrelated to the retained runtime work.

## Hardware

The Stage-2 A/B used five healthy GPUs. Utilization evidence records
`cuda:0..4` as visible, active, and used for terminal probes, with no idle-device
or warning entries.
