# Feasibility Cut-Point Index Evidence

Source commit: `c48e63d` (`Cache feasibility cut point indices`)

This directory contains server-side red/green evidence for replacing repeated
linear cut-point index scans in
`Rescale_optimizer/rescale_optimizer/feasibility.py` with one precomputed
identity map inside `build_feasibility_dag()`.

## Red

Directory:
`rfr_feasibility_cutpoint_index_red_aa539af_20260703_233253/`

Command target:
`python -m unittest tests.test_rescale_optimizer_hotpaths -v`

Result:
`red_rc=1`

The new hot-path guard failed on the pre-change implementation because
`build_feasibility_dag()` still called `_cut_point_index(graph, node)` for
each cut-point node while collecting stage nodes.

## Green

Directory:
`rfr_feasibility_cutpoint_index_green_cfg_aa539af_20260703_233525/`

Command targets:

- `python -m py_compile Rescale_optimizer/rescale_optimizer/feasibility.py tests/test_rescale_optimizer_hotpaths.py`
- `python -m unittest tests.test_rescale_optimizer_hotpaths tests.test_blb_optimizer_cost_consistency tests.test_blb_fusion_enum_fast.ComboRangeTest -v`

Result:

- `green_py_compile_rc=0`
- `green_unittest_rc=0`
- `Ran 12 tests ... OK`

The green package included Python sources plus `Rescale_optimizer/configs/`
JSON files so the in-process optimizer bridge cost-consistency tests could run
against the same config lookup path used by server map tooling.
