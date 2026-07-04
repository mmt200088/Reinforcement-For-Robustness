# Stage-1 GPU A/B Gate (`4bca31a`)

Purpose: verify the Stage-1 1GPU vs 4GPU throughput gate before changing any
rollout/cache defaults.

Source commit: `4bca31a`

Server run: `/hy-tmp/rfr_stage1_ab_4bca31a_20260704_084330_min170_wait`

Result:

- Launcher audit import regression fixed and verified:
  `gpu_audit_import_red.log` fails on `877eefa`, while
  `gpu_audit_import_green.log` passes on the fixed source with `11` tests and
  direct CLI `CLI_RC=0`.
- Stage-1 MRPC 170-episode A/B completed for both `g1` and `g4`.
- Both runs had `launcher_rc=0`, `wait_rc=0`, and `COMPLETED`.
- `g4` used `cuda:0..3` with worker counts `43/43/42/42`.
- Do not promote Stage-1 defaults to 4GPU from this evidence:
  `g4` was slower overall.

Key numbers from `stage1_ab_out/comparison.json`:

- `g1` wall time: `106s`; parser throughput: `7453.416 ep/h`.
- `g4` wall time: `182s`; parser throughput: `3756.032 ep/h`.
- Wall-clock speedup `g4/g1`: `0.582`.
- Parser-throughput speedup `g4/g1`: `0.504`.
- `g1` model-forward timing: `68.735s`.
- `g4` model-forward timing: `559.123s`.

Conclusion: the four-worker path is active, but duplicated validation/model
forward cost dominates this short formal gate. The next Stage-1 optimization
should reduce that duplicated work before retrying default promotion.
