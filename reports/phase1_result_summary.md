# Phase-1 Result Summary + Claude Layer-0 Flow Refresh

## Status

Claude's layer-0 flow update, Codex regression coverage, local diagnostic F0 scan, registry export, and handoff package refresh are complete.

Server sync and F1 masked smoke were not executed in this refresh. Non-interactive SSH reaches authentication but returns `Permission denied (publickey,password)`.

## Layer-0 Flow Change

- Action layout stays unchanged: `73 * L + 1`; MRPC `L=12` still has `877` dims.
- Layer-0 block1 slots stay in the action vector but are `effective=False` and are not installed.
- `first_input_sf` stays as a tail compatibility slot but returns placeholder `0` and is not installed.
- Layer-0 block2 remains fully active.
- RO request count is now `5L - 1`; baseline handover accepts canonical `5L - 1` and tolerates old `5L`.

## Local F0 Result

- baseline optimizer_valid: `true`
- baseline total_bits_sum: `14779`
- baseline fusion_count: `0`
- baseline avg_k: `13.0`
- mask hash: `2dda3a20ce9339f9fc95f164b88226f92d543debc2a0f38762f7b1312f4e870e`
- action width / slot count: `877 / 877`
- required/effective slot count: `791`
- ineffective or compat-extra slot count: `86`

Masked random validity:

| mutation_count | attempted | valid | valid_rate |
|---:|---:|---:|---:|
| 1 | 50 | 50 | 1.0 |
| 2 | 50 | 50 | 1.0 |
| 4 | 50 | 50 | 1.0 |
| 8 | 50 | 50 | 1.0 |

Beam scan note: no cost-improving single-slot mutations were found, so beam attempted expansions at depths 1/2/4/8 are zero. The generated mask is therefore a feasible-neighborhood mask, not evidence of optimizer cost improvement.

## Identity

- git HEAD: `6341ceab2bb15cd6e4cb0b98805bc88d7343a984`
- local tracked diff hash at package-refresh time: `4c333753e92e5bfcdb2a97e9b288f37bef7b0413`
- registry hash: `6c3662ba26160952e27dca8a8e3ae164af8326ac01819677c7b1a453fe342412`
- max_sfs hash: `bee17f0ccab949b79b4ca011a97da4cebd1d749e6ad49bffa272a701895e09f6`
- stage1 config content hash: `6454e0556f54ddb4519d9d2998582bca40a41fe2910d2ece679e455f8854eed3`
- Rescale_optimizer canonical hash: `ed28392d4078e4eb7734740023d281d5b87f1abde68340d7776f4e2855e4278e`

## Verification

- focused unittest including baseline handover and flow regressions: `49 tests OK`
- py_compile: passed
- git diff check: passed; CRLF warnings only
- layer-0 flow smoke:
  - action_dim: `220`
  - block1_cfg_keys: `[1, 2]`
  - first_input_sf: `0`
  - request_count: `14`
  - has_block1_mrpc_L0: `False`
  - layer-0 block1 records: `9`, all `effective=False`
  - first_input_effective: `False`
- server bash/tests/smoke: not run in this refresh

## Result Boundary

The F0 result is current local optimizer-only diagnostic evidence. It is not formal feasible evidence and must not be mixed with old long-training results. No RL training has been run after Claude's layer-0 flow changes.
