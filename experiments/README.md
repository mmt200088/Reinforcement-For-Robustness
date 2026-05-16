# Experiments index

This directory is the **single source of truth** for all RL / GA / search
runs we've done. Each training run produces a row in `registry.jsonl`
automatically (wired into `sequential_runner.py`), and `index.md` is the
human-readable view of that registry.

## Files

- `registry.jsonl` — append-only JSONL, one row per run. Schema in
  `tools/experiments_log.py` docstring.
- `index.md` — auto-regenerated index, sorted by registered_at desc,
  with a "best so far per dataset" mini-table at the top.
- `multi_seed/<run_name>/` — per multi-seed sweep output:
  - `seed_summary.{md,json}` — aggregated mean ± std across seeds
  - `seed_list.txt` — `<seed> <run_tag>` pairs that the sweep ran
  - `_failures.txt` — present iff some seed(s) failed

## Common operations

```bash
# View the latest index (regenerates first)
python3 tools/experiments_log.py rebuild
less experiments/index.md

# Filter from CLI
python3 tools/experiments_log.py query --dataset mrpc --min-reward 0.4
python3 tools/experiments_log.py query --status complete --last-n 20

# Annotate a run after the fact: edit registry.jsonl in your editor,
# find the row by run_id, change "notes": "" to your note. Then:
python3 tools/experiments_log.py rebuild

# Run a multi-seed sweep
bash tools/run_multi_seed.sh mrpc-blb-stage2-rl 1,2,3,4,5 myrun --fresh

# Aggregate seeds manually (useful if you ran some seeds by hand)
python3 tools/aggregate_seeds.py \
    --run-name myrun --seed-list experiments/multi_seed/myrun/seed_list.txt \
    --output-dir experiments/multi_seed/myrun
```

## What's in a row

A complete record looks like::

    {
      "run_id":           "20260516_143020_pid12345",
      "registered_at":    "2026-05-16T14:50:08+08:00",
      "git_commit":       "abc1234",
      "git_dirty":        false,
      "dataset":          "mrpc",
      "model_type":       "bert-base",
      "algorithm":        "rl",
      "preset":           "mrpc-blb-stage2-rl",
      "rl_variant":       "blb_v3_sequential",
      "seed":             42,
      "status":           "complete",            // complete / training_only / crashed
      "elapsed_sec":      6312.5,
      "completed_episodes": 6000,
      "total_episodes_planned": 6000,
      "best_reward":      0.4521,
      "final_eval":       {"loss": 0.3812, "metric1": 0.8623, ...},
      "persistent_dir":   "Parting Chapter/persistent/...",
      "artifact_paths":   {"best_action_full_md": "...", "report_md": "...", ...},
      "notes":            ""
    }

## What if I forgot to register manually?

`sequential_runner.py` auto-registers at training end. If for some reason
that didn't happen (crash before the hook, --fresh-stage2 wipe, etc.),
you can register an old run by hand:

```bash
python3 tools/experiments_log.py register \
    --run-id "20260515_pid99" \
    --dataset mrpc --algorithm rl \
    --preset mrpc-blb-stage2-rl --seed 42 \
    --status complete --elapsed-sec 5400 \
    --best-reward 0.42 \
    --persistent-dir "Parting Chapter/persistent/rl/bert-base/mrpc/<slug>" \
    --notes "manually registered after the fact"
```

## When to leave notes

- Any time you tweak a preset / change a hyperparameter mid-run
- When you spot a regression / bug discovered after training finished
- "This run is the figure for paper section 4.2"
- "Trained on stale stage1 config — ignore"

Add notes by editing `registry.jsonl` directly (it's append-only, but you
can edit existing lines safely as long as you don't break the JSON).
