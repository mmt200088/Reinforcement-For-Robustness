# Block3 Runtime Verification

This bundle records the server-side verification of the Stage-2 Block3 runtime
wiring at source commit `ed2e89f2fa3f667eca1d06135c1f4748b9957076`.

## Environment

- Server host: `I29d99ecd500020146c`
- Server runroot: `/hy-tmp/block3_runtime_verify_ed2e89f2_20260721_173108`
- Source checkout: detached, clean, and equal to `SOURCE_SYNC_COMMIT`
- Hardware: five NVIDIA GeForce RTX 5090 GPUs; the replay used GPU 0
- Model: `textattack/bert-base-uncased-MRPC`
- Dataset: local GLUE MRPC parquet, `validation_full=408`
- Stage-1 functions: GELU degree 4 and Softmax degree 6 in every layer

## Test Gate

- `py_compile`: passed (`py_compile.rc=0`)
- Focused Torch-backed tests: 128 passed, 0 failed, 0 skipped
- The first pytest invocation returned `focused_tests.rc=1` only because this
  server image has no `pytest` module. The same six test modules were then run
  through `python3 -m unittest`; `focused_tests_unittest.rc=0` and the log ends
  with `Ran 128 tests ... OK`.

The focused tests cover exact RO baseline decoding, Block3 request generation,
K-preserving optimizer write-back, layerwise action splicing, bridge install and
restore, final-eval install verification, and a controlled tensor-level check
that Block3 K changes the post-polynomial output.

## Real Model Replay

The replay uses the production 12x6 layerwise environment. Both candidates have
Block2 fusion count fixed to 1, Block5 fusion count fixed to 1, Block4 fusion
count 0, and K=13 everywhere outside Block3. The only changed policy decision is
Block3 K in all 12 layers:

| Candidate | Block3 K | Loss | Accuracy | Weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| `block3_k13` | 13 | 0.3360243183 | 0.8848039227 | 0.8827562891 |
| `block3_k8` | 8 | 0.3320906630 | 0.8823529423 | 0.8804175574 |

Both candidates used seed `20260721` and one full-validation noise trial. The
comparison is a wiring proof, not a stability or quality ranking.

All 12 JSON gates passed:

- the runtime loaded the static-skeleton RO baseline once;
- Block3 baseline fusion count is 0 in all 12 layers;
- the base environment uses the calibrated per-layer RO SF table;
- all Block3 non-K action slots remain equal to the baseline vector;
- all 24 Block3 replans are valid and fully applied;
- all 24 Block3 configs are installed while the real model forward runs;
- K=13 and K=8 reach the installed config exactly;
- the installed non-K SF chain equals the RO baseline in every layer;
- both candidates execute a real model forward;
- changing only Block3 K changes all 816 logits.

Logit difference for K=13 versus K=8:

- shape: `408 x 2`
- changed values: `816 / 816`
- maximum absolute delta: `0.3084035516`
- mean absolute delta: `0.0391400568`
- not equal at absolute tolerance `1e-6`

## Installed Block3 Configuration

All 12 layers use the same RO baseline SF chain:

- graph: `block3_exp_n6`
- fusion count: `0`
- degree: `6`
- polynomial input fresh: `N=16384`, SF `31`
- inverse-2^n encode: `N=16384`, SF `15`
- six square rescale points: `N=16384`, SF `35` each
- `x_inv_2n_result_rescale`: absent (`None`)
- output truncation mode: binary
- output truncation K: the selected 13 or 8

The configuration above was captured from `block3_cfg_per_layer` inside the
actual model forward hook, after cfg decode and optimizer write-back. It is not
an action-vector-only projection.

## Decoder Caveat

`block3_baseline_probe.json` shows that the old standalone Paean generic
`load_max_sfs("mrpc")` decoder is not the current calibrated RL baseline. For
layer 0 it yields square-rescale SF 31 and x-fresh SF 28, while the static RO
baseline used by the production layerwise path yields 35 and 31. The real-model
gate therefore intentionally uses the production layerwise environment rather
than the old generic Paean decoder. A standalone Paean result must not be called
an exact replay of the current RL baseline unless it receives the calibrated
per-layer table or an equivalent explicit configuration.

## Files

- `block3_model_replay.py`: locally authored reproducibility driver
- `server_snapshot/artifacts/block3_model_replay.json`: full action, replan,
  installed-config, logits, metrics, map hashes, and gate evidence
- `server_snapshot/artifacts/block3_model_replay.log`: server execution log
- `server_snapshot/artifacts/focused_tests_unittest.log`: 128-test log
- `server_snapshot/artifacts/block3_baseline_probe.json`: generic versus
  calibrated baseline diagnostic
- `server_snapshot/SOURCE_SYNC_COMMIT`: server source provenance marker
