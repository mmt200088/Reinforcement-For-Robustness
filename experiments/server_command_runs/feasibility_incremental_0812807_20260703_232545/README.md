# Feasibility DAG Incremental Accumulation Evidence

Source commit: `0812807` (`Optimize feasibility DAG accumulation`)

This directory contains server-side red/green evidence for optimizing
`Rescale_optimizer/rescale_optimizer/feasibility.py`.

## Red

Directory:
`rfr_feasibility_incremental_red2_cf738d1_20260703_231937/`

Command target:
`python -m unittest tests.test_rescale_optimizer_hotpaths -v`

Result:
`red_rc=1`

The strengthened hot-path guard failed on the pre-change implementation because
`build_feasibility_dag()` still rebuilt cumulative node lists and propagated
from the original start scale on each step.

## Green

Directory:
`rfr_feasibility_incremental_green_cfg_cf738d1_20260703_232545/`

Command targets:

- `python -m py_compile Rescale_optimizer/rescale_optimizer/feasibility.py tests/test_rescale_optimizer_hotpaths.py`
- `python -m unittest tests.test_rescale_optimizer_hotpaths tests.test_blb_optimizer_cost_consistency tests.test_blb_fusion_enum_fast.ComboRangeTest -v`

Result:

- `green_py_compile_rc=0`
- `green_unittest_rc=0`
- `Ran 11 tests ... OK`

The green package included Python sources plus `Rescale_optimizer/configs/`
JSON files so the neighboring optimizer cost-consistency tests could exercise
the in-process bridge path.
