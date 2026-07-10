# GLUE Inference Candidate Profile

Profiled source: `c448ee201d710b31f1cc3d6e23fbd56ac0395234`

No production change was accepted from this profile.

## Kernel and Transfer Candidate

Real BERT-base MRPC test inference used 1,725 examples at batch 128.

- Current median: `1.3915s`.
- `inference_mode` median: `1.3846s`.
- Pinned-loader median: `1.3892s`.
- Combined candidate median: `1.3851s` (`1.0047x`).
- All logits were bit-identical.

The gain is too small to justify changing the established submission path.

## Batch-Size Candidate

- Batch 16 median: `1.3253s`.
- Batch 32 median: `1.2359s` (`1.0724x`).
- Batch 64 median: `1.2671s` (`1.0459x`).
- Batch 128 median: `1.4137s` (`0.9374x`).
- Batch 256 median: `1.5066s` (`0.8796x`).

Changing batch size changed logits by roughly `1e-5`, so it failed the strict
result-preservation gate. Larger batches also became slower on this workload.
The default remains unchanged.

## Retained Diagnostics

- `benchmark.rc=1`: the first fixture used an offline dataset cache key that
  was unavailable to `load_dataset`; the valid retry loaded the existing
  server Arrow artifact directly.
- `benchmark_retry.rc=0`: accepted kernel/transfer profile.
- `batch_size.rc=2`: measurements completed, then the intentional bitwise
  parity guard rejected the batch-size candidate.
