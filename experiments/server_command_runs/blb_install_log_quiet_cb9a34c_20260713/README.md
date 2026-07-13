# Default-Quiet BLB Install Logging Evidence

This bundle verifies the low-conflict shared-path change at source commit
`cb9a34c2fcf5425a3e7a413a464a6b8c85dab9b7` on the five-RTX-5090 server.
BLB install messages are now quiet when `BLB_NOISE_INSTALL_LOGS` is unset;
setting `BLB_NOISE_INSTALL_LOGS=1` still enables the original diagnostic
output.

## Profile

The profile streamed the still-running Stage-2 log produced by source
`24e919cea64f22b1b869b9bf22d57bbb776bac5e` at 8,520 completed episodes:

- 1,543,515 BLB install lines occupied 344,739,386 bytes.
- Install messages were 98.504% of all lines and 99.9865% of all bytes.
- Linear projection to 60,000 episodes is about 10.87 million lines and
  2.43 GB of avoidable log output.
- The scan processed the 344,785,931-byte file in 3.11 seconds without loading
  it into memory.

The active process started from the old source and was not modified or
restarted. This change applies to future runs and does not claim an isolated
wall-clock speedup before the pending production 1GPU-versus-5GPU gate.

## Verification

- RED source `d948d450426b4f97517eab50442355ca1d4f8b14` failed the new default-quiet
  regression because the old helper printed `hidden by default`.
- GREEN passed both BLB install-log regression tests.
- `python -m py_compile function_handler.py` passed.
- The direct behavior gate recorded `default_quiet=true` and
  `explicit_one_loud=true`.
- Server Git status was clean before and after both focused gates.

## Files

- `active_log_profile/`: streamed size/count profile and the sampled GPU state.
- `red/`: focused failing test and clean Git status at the RED commit.
- `green/`: focused GREEN tests, compile check, behavior gate, and clean Git
  status at the production source commit.
