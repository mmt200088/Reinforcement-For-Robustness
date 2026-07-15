# Probe lifecycle comparator contract

Source `b41355812c34453264ad07714b89325de1914190` classifies
`terminal_probe_install_skipped` and `terminal_probe_clear_skipped` as
execution diagnostics. They are expected to differ between the single-GPU
clear-each-episode path and the multi-GPU persistent-install path, so they must
not make research-output parity fail.

Server evidence:

- RED source `acd08a6`: the focused test failed on
  `terminal_probe_clear_skipped: False != True`.
- GREEN source `b413558`: all 13 comparator tests passed.
- Replaying the archived 170-episode A/B produced exact quality/effect,
  strict-diagnostic, and PPO equality PASS results. The measured `1.758x`
  speedup remains unchanged and below the final `3.4x` gate.

No GPU work was required for this contract correction.
