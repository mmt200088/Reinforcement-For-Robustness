# Experiments index

_Auto-generated from `/var/folders/4k/w3ccv4d14m964xjmws95x39h0000gn/T/blb_smoke_z4i934ct/registry.jsonl` on 2026-05-17T18:53:06+08:00. Edit `notes` field in registry.jsonl to annotate a run; rerun `python3 tools/experiments_log.py rebuild` to refresh._

- Total registered run_ids: **1**
- By status: complete=1
- By dataset: mrpc=1

## Best so far (per dataset)

| Dataset | Best reward | Final loss | Final metric1 | Run ID |
|---|---:|---:|---:|---|
| mrpc | +0.4200 | 0.4000 | 0.8000 | `20260516_smoke_pid1` |

## All runs (most recent first)

| Run ID | Dataset | Algo | Preset | Seed | Status | Time | Best | Loss | Metric1 | Git | Persistent |
|---|---|---|---|---:|---|---:|---:|---:|---:|---|---|
| 20260516_smoke_pid1 | mrpc | rl | smoke-test | 42 | complete | 0.03h | +0.4200 | 0.4000 | 0.8000 | `⚠dirty` | `fake` |

---

**How to use this file**:

- 想看某个具体 run 的细节：去 `persistent` 列对应的目录，看 `blb_stage2_best_action_full.md` / `diagnostics/diagnostics_summary.md`。
- 想做 cross-run 对比：用 `python3 tools/experiments_log.py query --dataset mrpc --min-reward 0.4`。
- 想给某个 run 加注释：直接编辑 `registry.jsonl` 那一行的 `notes` 字段，然后 `python3 tools/experiments_log.py rebuild`。