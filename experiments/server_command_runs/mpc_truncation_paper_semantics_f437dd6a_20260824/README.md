# MPC Truncation Paper-Semantics Verification

- Source commit: `f437dd6afcd3b883bfccc533acb4759db56a56e3`
- Source tree: `d39f5bd32208f02bace53642634a3498aa5260f6`
- Baseline commit: `480e154053b1303e140077a05c46295cab95ef0a`
- Server: `f1ac06029e4a`

## Focused tests

The server ran 81 focused tests. Sixty-four executable and static-contract tests
passed; 17 torch-dependent truncation backend tests skipped because this shared
server image had no torch installation. Python compilation passed.

The passing tests covered the H/M/L preset table, K6-K13 decode, BERT-base and
BERT-large layer schedules, all five Block K slots, layer-0 Block1, fusion-map
splicing, boosted overrides, counterfactual materialization, reward-facing
resource costs, layerwise environment handoff, and shared training/final-eval
materialization contracts.

## Exact execution parity

The same snapshot script ran at the pre-change baseline and at the source
commit. It covered uniform H/M/L with Block4 fusion 0 and 1 plus one mixed
12-layer action. Each snapshot includes the complete legacy full vector,
decoded simulation K values, fusion option IDs, boosted overrides, and every
field of the variable-cost result.

`cmp` returned zero. Both files have SHA-256:

```text
68a50ef270d894f3995bd01437b6febcb0bd2b3c757b42edb03485ad2ceb63e7
```

Therefore the executable RL/materialization/cost state is byte-identical across
the source change. Only the shared human-readable action description gains the
paper-facing ciphertext K, reserve-bit, and ring-width metadata.

## Environment limitation

The server's NVML driver/library versions did not match and its Python
environment did not contain torch. No GPU process was started, stopped, or
modified. CUDA behavior was not rerun; the source diff intentionally excludes
all model truncation and CUDA implementation files.
