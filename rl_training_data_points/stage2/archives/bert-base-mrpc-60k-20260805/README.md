# BERT-base MRPC Stage-2 full training archive

Status: PASS

- Episodes: 60,000 / 60,000
- PPO updates: 501
- Raw files: 64
- Raw bytes represented: 2245565240
- Archive bytes: 151296461
- Stage-1 configuration: exact match to authoritative base-model HTML
- Source commit: 5c222da6186b8a60244b46029bbc8dac79befb34
- Source tree: 20c3280a25fa4376471e846c2053ca03e8bb55c4
- Restore verification: RESTORE_OK

Restore with: python3 restore_snapshot.py /path/to/empty/output

The snapshot contains both the persistent run root and its structured rl_training_data_points mirror. Per-file source path, original SHA256, archived SHA256, line count, mode, and restore mapping are recorded in snapshot_manifest.json.
