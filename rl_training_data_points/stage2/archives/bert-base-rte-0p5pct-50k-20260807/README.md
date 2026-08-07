# BERT-base RTE Stage-2 0.5% / 50k training snapshot

This result-only archive preserves the complete server-side training state for
`bert-base-rte-stage2-0p5pct-50k-20260806`.

- Run source commit: `6c36532a47349ffc43d38616a030b37dd1b29153`
- Run source tree: `500c52b104391995602eca9b191e7c1fe7c8de33`
- Episodes: `50,000`
- PPO updates: `417`
- Stop phase: `max_episodes_reached`
- Precision constraint: `0.5%`
- Stability constraint: `200%`
- Stage-2 trials per online evaluation: `3`
- Probe size: `256`
- Stage-1 GELU: `[1, 1, 1, 4, 1, 1, 2, 1, 1, 1, 1, 1]`
- Stage-1 Softmax: `[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]`
- Source files: `49`
- Raw payload: `1.857 GiB`
- Compressed/copied payload: `210.2 MiB`

The source `RUN_IDENTITY.json` and migration receipt retain their launch-time
`active_training` lifecycle labels. The authoritative final status files in the
snapshot record `50,000/50,000`, `417` PPO updates, and
`max_episodes_reached`; raw files were intentionally not rewritten.

## Contents

The snapshot includes the persistent run directory, structured JSONL mirror,
all generated final-report assets, launch/migration receipts, model checkpoint,
candidate and Pareto stores, episode/PPO streams, action diagnostics, and exact
Stage-1 prerequisite vectors.

## Restore

From this archive directory:

```bash
python3 restore_snapshot.py --destination /path/to/empty/restore-dir
```

Each payload is checked against both its archived SHA-256 and its original raw
SHA-256. File sizes, text line counts/newline state, modes, and modification
times are validated/restored. `SHA256SUMS` verifies the archive itself.
