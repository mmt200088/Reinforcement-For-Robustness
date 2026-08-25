# Generated Search Outputs

Generated artifacts are grouped by algorithm and are not tracked by Git.

```text
outputs/<algorithm>/<model>/<dataset>/
```

RL keeps independent `stage1/<run>` and `stage2/<run>` directories. BO-RF,
Greedy, and COINN-GA keep each bound two-stage run under `two_stage/<run>`.
Each run owns its metadata, logs, checkpoints, structured records, stage
artifacts, and validation evaluation.
