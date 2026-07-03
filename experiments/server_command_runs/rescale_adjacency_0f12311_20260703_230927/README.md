# Rescale Stage-Edge Adjacency Reuse Evidence

Source commits: `ceb9216`, `0f12311`
Red-test commit: `ef1133d`

## Optimization

`Rescale_optimizer.rescale_optimizer.reachability.compute_reachability()` and
`Rescale_optimizer.rescale_optimizer.backward_level_dp.build_dp_table()` now
build per-source stage-edge adjacency once and reuse it in the forward,
backward, and DP loops.

This preserves the existing stage-edge order while avoiding repeated full
`graph.stage_edges` scans for each cut point and level during Rescale/fusion
map feasibility and DP work.

## Server Verification

Red run:

- Directory: `rfr_rescale_adjacency_red_ef1133d_20260703_230430/`
- Command: `PYTHONPATH="$PWD:$PWD/Rescale_optimizer" python -m unittest tests.test_rescale_optimizer_hotpaths -v`
- Result: `red_rc=1`
- Expected failure: reachability and DP still scanned `graph.stage_edges`
  directly inside hot loops.

Intermediate run:

- Directory: `rfr_rescale_adjacency_green_ceb9216_20260703_230707/`
- Result: `green_py_compile_rc=0`, `green_unittest_rc=1`
- Reason: the initial source guard was too strict and rejected the one allowed
  scan that builds the adjacency table.

Final green run:

- Directory: `rfr_rescale_adjacency_green_0f12311_20260703_230927/`
- Commands:
  - `PYTHONPATH="$PWD:$PWD/Rescale_optimizer" python -m py_compile Rescale_optimizer/rescale_optimizer/reachability.py Rescale_optimizer/rescale_optimizer/backward_level_dp.py tests/test_rescale_optimizer_hotpaths.py`
  - `PYTHONPATH="$PWD:$PWD/Rescale_optimizer" python -m unittest tests.test_rescale_optimizer_hotpaths tests.test_blb_fusion_enum_fast.ComboRangeTest -v`
- Results:
  - `green_py_compile_rc=0`
  - `green_unittest_rc=0`
  - `Ran 6 tests ... OK`
