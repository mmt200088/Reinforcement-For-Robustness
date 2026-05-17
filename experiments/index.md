# Experiments index

_Auto-generated from `/var/tmp/root-home/Reinforcement-For-Robustness/experiments/registry.jsonl` on 2026-05-17T09:02:38+08:00. Edit `notes` field in registry.jsonl to annotate a run; rerun `python3 tools/experiments_log.py rebuild` to refresh._

- Total registered run_ids: **1**
- By status: complete=1
- By dataset: mrpc=1

## Best so far (per dataset)

| Dataset | Best reward | Final loss | Final metric1 | Run ID |
|---|---:|---:|---:|---|
| mrpc | -152.5556 |  |  | `s1t0.005_s2t0.005_s` |

## All runs (most recent first)

| Run ID | Dataset | Algo | Preset | Seed | Status | Time | Best | Loss | Metric1 | Git | Persistent |
|---|---|---|---|---:|---|---:|---:|---:|---:|---|---|
| s1t0.005_s2t0.005_s | mrpc | rl | Stage-1 config (json) | 42 | complete | 7.43h | -152.5556 |  |  | `⚠dirty` | `progress` |

---

**How to use this file**:

- 想看某个具体 run 的细节：去 `persistent` 列对应的目录，看 `blb_stage2_best_action_full.md` / `diagnostics/diagnostics_summary.md`。
- 想做 cross-run 对比：用 `python3 tools/experiments_log.py query --dataset mrpc --min-reward 0.4`。
- 想给某个 run 加注释：直接编辑 `registry.jsonl` 那一行的 `notes` 字段，然后 `python3 tools/experiments_log.py rebuild`。