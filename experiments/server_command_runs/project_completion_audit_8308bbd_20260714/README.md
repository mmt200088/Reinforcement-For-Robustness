# Whole-project runtime completion audit

This bundle records the repository-wide CPU/no-GPU audit performed on the
five-RTX-5090 server while the separate formal Stage-2 run retained exclusive
use of all GPUs.

## Source progression

- `f7dc231`: initial post-production-integration audit source.
- `3de5244`: restore Stage-2 immutable probe assignment caching, preallocated
  result slots, static schedule/fusion-mask tensor caches, causal-prefix
  execution, and batched PPO tensor materialization.
- `66a4895`: restore the shared integer-list parser in the Stage-2 runner.
- `8308bbd`: align dynamic integration fixtures with the shared optimizer and
  diagnostics APIs, and use machine-precision comparison for weighted means.

No reward formula, statistical constraint, trial count, validation dataset, or
formal launch parameter changed in this sequence.

## Final gates

| Gate | Result | Evidence |
| --- | --- | --- |
| Six-stage source/artifact audit | 30/30 expected files; all 6 artifact classes present | `audit_final.md`, `audit_final.json` |
| Focused repaired pytest files | 28 passed | `pytest_targeted_8308bbd.log` |
| Full unittest discovery | 1,509 passed; 6 condition skips | `full_tests_66a4895.log.gz` |
| Full pytest | 1,599 passed; 6 condition skips; 5 warnings | `pytest_8308bbd.log` |
| Changed-source compilation | exit 0 | `gates_3de5244/py_compile.rc` |

The six skips require either visible CUDA devices or Python 3.9. CUDA was
intentionally hidden from this audit to avoid contaminating the formal run.
The server image had no global pytest and could not create a normal venv
because `ensurepip` was absent, so pytest 8.4.2 and pytest-xdist 3.8.0 were
installed into isolated `/hy-tmp/rfr_test_deps_f7dc231` and injected only via
`PYTHONPATH`. The installation transcript and failed venv attempt are retained
in this bundle.

## Red-to-green record

The initial broad `unittest` run exposed stale Stage-2 contracts and hot-path
regressions after the production merge. After the runtime restorations and
shared parser fix, full `unittest` passed. The first full pytest then exposed
eight pytest-function-only failures: one exact floating-point comparison and
seven dynamic fixture import failures. The final two-file gate passed 28 tests,
followed by the complete 1,599-test pytest pass. Compressed initial logs remain
for auditability.

## Active formal run

At the captured snapshot, PID `10089` from source `24e919c` was alive at
14,640/60,000 episodes (24.4%), with terminal priority P3 and
`last_invalid=false`. All five RTX 5090 GPUs were active at roughly 33-39%
sampled compute utilization and 3.1-3.2 GiB memory each. See
`formal_run_snapshot.json`, `formal_run_process.txt`, and
`nvidia_smi_after.csv`.

This bundle does not claim the strict 1GPU-versus-5GPU parity or speedup gate.
That remains pending until the formal process exits and the harness idle check
passes.

The transferred server archive was
`/hy-tmp/project_completion_audit_8308bbd_20260714.tar.gz`, SHA-256
`7d03f7f203a088ee9155e25d2c6706efa3617a792945d855532b8d1c37dc52b6`.
