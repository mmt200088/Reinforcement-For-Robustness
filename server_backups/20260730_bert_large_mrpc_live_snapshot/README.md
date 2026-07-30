# BERT-large MRPC Stage-2 live snapshot

This directory is a Git-safe, validated snapshot of the active Stage-2 run.
It was captured without stopping training.

## Identity

- Source commit: `0764e71025ee887327d63364052935d770c489f9`
- Training PID at capture: `4022538`
- Snapshot time: `2026-07-30T09:14:17Z`
- Resume checkpoint: episode `12000`, PPO update `100`
- Complete analysis streams: episodes `0..12031`
- Planned maximum: `150000` episodes

`source_git_state.txt` records the exact server tree state. The only tracked
modification shown there is an unrelated generated weight-plot PNG.

## Contents

- `streams/`: gzip-compressed, complete-line JSONL prefixes from the live run.
- `small_files/progress/`: checkpoint, action vectors, Pareto state, run
  manifest, summaries, and diagnostics.
- `small_files/structured/`: structured-writer manifest.
- `snapshot_manifest.json`: source paths, row counts, first/last records, raw
  hashes, and compressed hashes.
- `resume_cut_manifest.json`: checkpoint-consistent byte boundaries and hashes.
- `verification_report.json`: JSON validity and mirror-semantic checks.
- `SHA256SUMS`: hashes for every archived file.

The diagnostics and structured PPO streams are semantically identical. The
structured episode stream adds only `best_reward_so_far` and `is_new_best`.

## Verify

From this directory:

```bash
shasum -a 256 -c SHA256SUMS
gzip -t streams/*.gz
```

## Restore

Create a checkpoint-consistent resume tree:

```bash
python3 restore_snapshot.py --mode resume --output-dir /path/to/restored
```

Create the full analysis snapshot through episode 12031:

```bash
python3 restore_snapshot.py --mode full --output-dir /path/to/restored
```

Resume mode truncates each append stream to the exact byte count stored in the
episode-12000 checkpoint and verifies the checkpoint fingerprint. Full mode
restores every complete record captured during the live snapshot.
