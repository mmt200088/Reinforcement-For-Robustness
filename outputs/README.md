# Generated Search Outputs

Generated artifacts are grouped by algorithm and are not tracked by Git.

```text
outputs/<algorithm>/<model>/<dataset>/
```

RL keeps independent `stage1/<run>` and `stage2/<run>` directories. BO-RF,
Greedy, and COINN-GA keep each bound two-stage run under `two_stage/<run>`.
Each run owns its metadata, logs, resumable checkpoints, structured records,
and stage evidence. Normally completed Stage 1 runs write
`stage1_best_config.json`; complete strict two-stage runs write
`search_best_config.json`. Independent final evaluations are stored under
`outputs/evaluation/<algorithm>/<model>/<dataset>/<run>/evaluation/`.
