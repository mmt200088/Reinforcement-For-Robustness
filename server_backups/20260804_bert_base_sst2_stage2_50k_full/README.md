# BERT-base SST-2 Stage-2 50k full training archive

This directory is a content-addressed, restorable snapshot of one completed BERT-base SST-2 Stage-2 PPO run.

## Captured training state

- Episodes: 50000 / 50000
- PPO updates: 417
- Terminal phase: max_episodes_reached
- Profile: sst2
- Policy: shared_gtrxl_small_v1
- Raw logical data: 1.821 GiB across 49 files
- Git payload: 134.0 MiB with large JSONL streams compressed independently

The six restored roots are run, structured, report, supplemental_report, stage1_reference, and tooling.

## Stage-1 provenance warning

Status: UNRESOLVED_PENDING_AUTHORITATIVE_JULY_ARTIFACT. This backup preserves historical raw evidence; it is not a scientific-validity endorsement. The archived run installed:

- GELU: [1, 1, 1, 1, 1, 4, 1, 2, 1, 1, 1, 1]
- Softmax: [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]

These vectors match the supplied June HTML (sst2_stage1_best_20260625). No authoritative July BERT-base SST-2 Stage-1 artifact was found in the audited local files, Git refs/history, five-card server, or single-card server. Do not promote this Stage-2 result until the authoritative July artifact is obtained and compared.

## Restore

Run restore_snapshot.py with a destination directory. Before restoration, run sha256sum -c SHA256SUMS and gzip -t streams/*.gz.

## Reconstructable analyses

The archive contains all episode rows, PPO-update rows, candidate-store rows, top-candidate and Pareto snapshots, live/final summaries, status and manifests, checkpoints, reports, launch evidence, logs, and the structured mirror. The checkpoint is loadable at episode 50,000/update 417.

## Integrity evidence

- snapshot_manifest.json: original and archived paths, SHA-256 hashes, sizes, row counts, permissions, and timestamps.
- verification_report.json: semantic and row-count gates (VERIFY_OK).
- restore_verification.json: full server-side restore rehearsal.
- SHA256SUMS: repository payload integrity.
- cloud_restore_verification.json: added after a fresh cloud clone and restore rehearsal.
