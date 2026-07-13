# Stage-2 A/B GPU Coverage Hardening

This bundle verifies source `1b8fa08408f67ab07f9f2de0dfc3d53d7bfc788c`
on the five-RTX-5090 server. It closes a benchmark-integrity gap before the
pending production 1GPU-versus-5GPU run; it is not itself a speed result.

## Change

- RED source `04f2b2e2` proves that the A/B comparator counted only the first
  device in each episode's `terminal_probe_devices` list, the utilization CLI
  had no strict sampled-activity mode, and the A/B runner did not enforce its
  collected `nvidia-smi` evidence.
- Source `1b8fa084` counts every participating probe device and its trial
  count.
- Each completed A/B case now writes JSON and Markdown GPU-utilization reports
  and fails unless every requested physical GPU exceeds the sampled utilization
  threshold. Reused 1GPU baselines must carry and revalidate their original
  `one_nvidia_smi.csv` evidence.
- Training, reward, constraints, seeds, validation data, and PPO behavior are
  unchanged.

## Verification

| Gate | Result | Evidence |
| --- | --- | --- |
| Focused RED | Expected failure, exit 1 | `red_tests.log`, `red_tests.rc` |
| Bash syntax | Exit 0 | `bash_n.rc` |
| Related tests | 38 passed | `related_tests.log`, `related_tests.rc` |
| Full pytest, CPU thread-capped and CUDA-hidden | 1,602 passed, 6 skipped, 5 warnings in 77.05s | `full_pytest.log`, `full_pytest.rc`, `full_pytest.time.txt` |
| Exact command preflight | Exit 0 without GPU query | `preflight.log`, `preflight/effective_commands.txt` |
| Artifact checksum | 29 server files verified | `SHA256SUMS.server` |

The first uncapped full-pytest attempt was intentionally interrupted after
330.28s and 282 passing tests because CPU Torch helper pools expanded enough to
compete with the formal run. Its log and exit code are retained as
`full_pytest_uncapped_interrupted.*`. The authoritative rerun used the same
thread caps as the production launcher and completed the whole suite in
77.05s with CUDA hidden.

## Concurrent Formal Run

No competing RL or A/B process was started. After verification, formal PID
`10089` remained alive and all five GPUs remained assigned to it. The captured
status was 16,832/60,000 episodes. The latest 100 rows contained 42 P1, 9 P2,
and 49 P3 outcomes, with zero invalid episodes and zero loss-cap sentinels;
this is an algorithm-health warning for the Stage-2 owner, not an efficiency
change made here.

The optimization goal still requires the isolated 600-episode production
1GPU-versus-5GPU equality and wall-clock gate after the formal process and its
post-run work release all GPUs.
