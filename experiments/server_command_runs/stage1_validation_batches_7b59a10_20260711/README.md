# Stage-1 Reusable Validation Batch Evidence

## Change

Source commit `7b59a10` materializes only `validation_full` dataloaders as
immutable tuples after their first collate. This includes the optional MNLI
mismatched validation loader. Training dataloaders remain lazy.

The cached tensors remain pinned CPU tensors, so one tuple can be reused by
single- or multi-GPU evaluator workers without reserving a copy on every GPU.
No dataset split, batch order, padding, model forward, loss averaging, metric,
reward, or exact-configuration eval-cache behavior changed.

## TDD And Gates

- RED test commit: `c84973f`.
- RED result: one focused test failed because the old
  `validation_full` registry retained the DataLoader and collated on every
  iteration; the training-loader contract test passed.
- GREEN source commit: `7b59a10`.
- GREEN focused result: 2 tests passed.
- Related gate: 90 tests passed across Stage-1 eval/approximation,
  shared inference, entropy-stop, parallel semantics, selection semantics,
  and Stage-1/Stage-2 alignment suites.
- `py_compile` passed for the changed source and test.

## Real MRPC A/B

The production benchmark used the cached
`textattack/bert-base-uncased-MRPC` checkpoint, the real 408-row GLUE MRPC
`validation_full` split, the exact `rl_tune.py` pre-tokenization and
`padding=max_length(128)` collator, batch size 128, and GELU
`[4,2,1] x 4` with Softmax degree 6.

The benchmark alternated legacy DataLoader and cached-batch evaluation for 18
rounds per mode. Loss, metrics, labels, and logits were bit-identical on every
round.

| Path | Median wall | Mean wall |
| --- | ---: | ---: |
| Legacy DataLoader | 0.698002 s | 0.697846 s |
| Cached pinned CPU batches | 0.690052 s | 0.689949 s |

This is a `1.0115x` speedup and saves `7.951ms` per cache-miss full-validation
evaluation. At 50,000 cache-miss evaluations it projects to `397.5s` of wall
time. Existing exact-configuration cache hits still skip the whole forward,
so the projection is not a claim that all 50,000 episodes will miss.

The four cached batches occupy 1,256,640 bytes. All cached tensors were
confirmed pinned. GPU sampling reached 100% utilization and 3,735 MiB maximum
memory during the complete load/map/benchmark process.

## Rejected Alternatives

- GPU-resident validation batches produced only another `57.9us` per eval
  beyond the CPU tuple in the pre-change screening. That does not justify
  per-worker VRAM copies or a per-device cache.
- Suppressing GELU/Softmax install-success logs projected only `0.052s` saved
  across 50,000 configurations. It would remove about 198,795 lines but is not
  a meaningful runtime optimization for the actual `nohup > logfile` path.

Raw RED/GREEN return codes, logs, benchmark JSON, hardware inventory, GPU
samples, and screening measurements are retained in this directory.
