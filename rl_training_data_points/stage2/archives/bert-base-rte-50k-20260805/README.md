# BERT-base RTE Stage-2 full training archive

Status: PASS

- Episodes: 50,000 / 50,000
- PPO updates: 417
- Raw files: 39
- Raw bytes represented: 1780721758
- Archive bytes: 99168209
- Stage-1 configuration: exact match to authoritative base-model HTML
- Source commit: 1b34e94936ec19f92a1715ccaf7020049cf926c1
- Source tree: b43505118859b3de4a49a88530f835ae081e56fa
- Restore verification: RESTORE_OK

Restore with: python3 restore_snapshot.py /path/to/empty/output

The snapshot contains both the persistent run root and its structured rl_training_data_points mirror. Per-file source path, original SHA256, archived SHA256, line count, mode, and restore mapping are recorded in snapshot_manifest.json.
