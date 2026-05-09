# BLB Phase 0 Entrypoints

1. Main training entrypoint: `bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --fresh`.
2. Resume entrypoint: `bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl`.
3. Stage-2 variant switch: `--stage2-rl-variant blb_v3` in `presets/mrpc-blb-stage2-rl.conf`.
4. Runner implementation: `blb_stage2_rl/runner.py` (`BLBStage2RLRunner`).
5. Action registry/decode implementation: `blb_stage2_rl/action_space.py`.
6. Rescale optimizer path: `Rescale_optimizer`; BLB Stage-2 uses the in-process optimizer path.

| artifact | exists |
|---|---:|
| llama_7B_LayerImportance.sh | true |
| presets/mrpc-blb-stage2-rl.conf | true |
| blb_stage2_rl/runner.py | true |
| blb_stage2_rl/action_space.py | true |
| Rescale_optimizer | true |

Current defaults from the preserved operator surface are resolved by the launcher and preset; this report is an audit artifact, not a replacement command.
