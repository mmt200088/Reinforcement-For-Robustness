# Archived raw data

This run completed 114240 episodes and 952 PPO updates before a verified
graceful stop. Its original `manifest.json` is retained here for discovery.

The exact `episodes.jsonl` and `ppo_updates.jsonl` byte streams are split and
compressed under
`server_backups/20260720_stage2_mrpc_ep114240_full_recovery/archives/` so no
Git object exceeds GitHub's single-file size limit. Run that bundle's
`restore.sh` to reconstruct this directory with the two original JSONL files
and verify their SHA-256 hashes.
