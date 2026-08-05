# Stage-2 six-model Stage-1 and cloud-backup audit

Overall status: PASS

All six installed GELU and Softmax vectors exactly match the authoritative Stage-1 HTML reports.
All six training histories are available on remote Git branches with episode/PPO streams, candidate data, checkpoints, manifests, and restore evidence.
BERT-large MRPC points to the repaired archive branch that restores the two previously omitted small logs.

| Model | Dataset | Episodes | PPO updates | Boundary | Stage-1 | Backup |
|---|---:|---:|---:|---|---|---|
| bert-base | mrpc | 60000 | 501 | max_episodes_reached | PASS | PASS |
| bert-base | rte | 50000 | 417 | max_episodes_reached | PASS | PASS |
| bert-base | sst2 | 50000 | 417 | max_episodes_reached | PASS | PASS |
| bert-large | mrpc | 24720 | 206 | checkpoint_boundary_graceful_stop | PASS | PASS |
| bert-large | rte | 33720 | 281 | checkpoint_boundary_graceful_stop | PASS | PASS |
| bert-large | sst2 | 35280 | 294 | checkpoint_boundary_graceful_stop | PASS | PASS |

See six_model_backup_audit.json for exact vectors, report hashes, branch commits, archive paths, byte counts, and checks.
