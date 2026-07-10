# Compact Replan Fusion Enumeration Evidence

## Change

Source commit `511b3f2` adds `ReplanSession.replan_compact()` for the
fusion-map hot loop. It returns only validity, fusion count, total bits, and
the deployable compact config. `fusion_enum_fast.eval_combo_fast()` consumes
that result directly instead of expanding the compatibility JSON document on
every Cartesian-product combination.

The existing `ReplanSession.replan()` default, full result dictionary, CLI,
JSON artifacts, fusion policy, graph mutation/restoration, compact-config
builder, and golden fallback are unchanged. The workload remains a pure-Python
integer replan path; GPU execution would require replacing the cost source of
truth and is therefore not appropriate.

## TDD And Gates

- RED test commit: `5f260d0`.
- RED result: exit 1. The compact result/API import failed and the fast enum
  test caught the old call to full `session.replan()`.
- GREEN source commit: `511b3f2`.
- GREEN focused result: 2 tests passed.
- Related Rescale/fusion gate: 128 tests ran, 126 passed and 2 pre-existing
  torch-bridge tests skipped because their optional import guard was
  unavailable. The directly changed live-session and fast-enum tests ran.
- `py_compile` passed for all changed production and test files.

## Hot-Loop A/B

The benchmark built and golden-verified the real `block1_mrpc` fast template,
then enumerated the same rank range `[0, 100000)` six times per source. Both
sources reported 769 valid configurations, one reduced row, and identical
result SHA256
`04b1fa904266bcc79ae3874cd29b7b8b6b253f90a59f7b1477edee8f3ab0e45c`.

| Source path | Median / 100k | Mean / 100k |
| --- | ---: | ---: |
| Full compatibility output (`5f260d0`) | 2.065481 s | 2.065252 s |
| Compact result (`511b3f2`) | 1.485932 s | 1.485859 s |

The production path is `1.3900x` faster and saves `0.579549s` per 100,000
combinations. A linear single-core projection over the current
2,207,205,000-combination block4 domain is 3.55 core-hours saved. With ideal
20-worker scaling that corresponds to 10.66 minutes; this is a projection,
not a completed block4 wall-clock claim.

## Builder A/B

The real `blb_build_fusion_count_map.py` entrypoint then built all 3,913,140
`block1_mrpc` combinations with 20 workers and 64 golden verification probes.

| Source path | End-to-end wall | Enum rate |
| --- | ---: | ---: |
| Full compatibility output | 14.17 s | 302,731 combos/s |
| Compact result | 12.34 s | 353,222 combos/s |

End-to-end wall improved `1.1483x`. After removing only measured wall fields
from `build_meta`, parent and production map JSON were equal.

Raw RED/GREEN logs, return codes, benchmark samples, generated maps, hardware
inventory, and comparison JSON are retained in this directory.
