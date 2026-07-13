# Stage-2 production/shared-path integration evidence

This bundle records the focused server verification performed after merging
the published Stage-2 production source (`24e919c`) into the runtime
optimization branch.

## Source sequence

- `cd6fb55`: clean merge of optimization source and the production branch.
- `013022f`: align Paean regression fixtures with the canonical replan helper.
- `442fcb3`: restore shared optimizer write-back, installed probe inference,
  and probe diagnostics after `16b68e3` had expanded them into local copies.
- `c786336`: keep deterministic probe forward protection intact while moving
  metric kernels and the batched device-to-host synchronization outside the
  shared-device lock; avoid repeated `model.eval()` when already in eval mode.
- `87a57a1`: correct the final batch-metric fixture so its per-batch accuracy
  values agree with its predictions and labels.

## Red/green progression

| Source | Focused result | Meaning |
| --- | --- | --- |
| `013022f` | 76 tests, 3 failures | Static guards found duplicated optimizer write-back, installed inference, and diagnostics payload construction. |
| `442fcb3` | 118 tests, 9 errors, 1 skip | The shared calls were restored, but bare-env compatibility and lock-scope tests exposed that metric work had moved under the device lock. One loaded module was pytest-only. |
| `c786336` | 117 tests, 1 failure, 1 skip | Production and lock-scope behavior passed; one inconsistent metric fixture remained. |
| `87a57a1` | 117 tests, OK, 1 skip | Final focused gate passed. |

For the final source, Bash syntax, Python compilation, and the no-GPU
effective-command preflight also returned `0`. The server worktree was clean
before and after verification. The server image does not currently include
`pytest`, recorded as `pytest_dependency_rc=3`; the pytest-only metric file was
compiled, while its equivalent contracts were exercised by the passing
`unittest` modules.

## Scope and remaining gate

This evidence is CPU/no-GPU integration verification. It does not claim a
1-GPU-versus-5-GPU speedup. At evidence packaging time, the isolated formal
Stage-2 run from source `24e919c` was still healthy at 12,360 of 60,000
episodes and owned all five RTX 5090 GPUs. The strict equality and throughput
A/B must wait for that process to exit and for the harness idle check to pass.

The server archive used for transfer was
`/hy-tmp/stage2_shared_path_integration_20260714.tar.gz`, SHA-256
`9f54cf2a3c560c795fd32755ef1ab3c8ff89b1f055755cf35aab8a3eb50ef4f6`.
`pycache` content was excluded before transfer.
