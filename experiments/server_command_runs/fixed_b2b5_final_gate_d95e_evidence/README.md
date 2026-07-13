# Fixed Block2/Block5 Fusion Final Gate

Source commit: `d95e93483bbb9da91e3bc204ea079a8047b3296e`

## Accepted gates

- Python compilation: rc=0.
- Focused source/contract/Paean compatibility gate: 176 tests, rc=0.
- Canonical map audit: all 6 profiles pass; every required Block2/4/5 map has fusion counts exactly `[0,1]`; no count above 1.
- Canonical load/domain gate: MRPC's only policy-selectable fusion block is Block4. Block1 has one fusion-0 option; Block2 and Block5 each expose one policy-local option resolving to the real fusion-1 map option.
- Target-profile golden replay: all 25 newly completed other-profile Block2/4/5 maps reproduce validity/fusion/bits and are K-independent.
- Runtime install gate: 6 profiles x 5 boosted graph options = 30 checks, 0 problems.
- MRPC fixed-action comparison: all action/install gates pass; full 408-example validation, 5 noise trials, K=13.

## Scope limitation

The five newly completed directories were explicitly requested for Block2/4/5 map completion. They do not contain `block1_<profile>`, so they are not yet complete standalone Stage-2 RL map bundles. MRPC remains the complete current RL target. No unrequested Block1 enumeration was synthesized.

## Broad-suite residual

The 213-test broad gate retained two known pre-existing Paean regression failures in `tests.test_blb_stage2_rl_regressions` (`apply_optimizer_output_to_cfg` legacy test interface and the old invalid-optimizer forward expectation). The exact same two failures were already present before this feature branch. They are preserved in `tests.log`; all tests touched by this task and the merged Paean compatibility tests pass in `focused_tests.log`.
