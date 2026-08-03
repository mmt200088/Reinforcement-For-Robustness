# BERT-large SST2 Stage-2 RL graceful-stop archive

This directory is the complete recoverable snapshot of the BERT-large SST2 Stage-2 PPO run stopped at a full checkpoint boundary.

## Resume cut

- Episodes: **35280 / 50000**
- PPO updates: **294**
- Status: **已停止：checkpoint-boundary graceful stop**
- Stopped at: **2026-08-02T23:45:00.662410**
- Checkpoint SHA256: 975ee52ebbd51069b8d5bd9c340aea45be5d9d15912da3f99e48360e418ee7dc
- Source model/profile: yoshitomo-matsubara/bert-large-uncased-sst2 / sst2_large
- Constraints: 0.1% precision tolerance and 200% stability multiplier for loss, Accuracy, and Weighted F1
- Policy: shared_gtrxl_small_v1; 24 layerwise decisions; Block4 fusion 0/1 plus high/medium/low truncation preset
- Online terminal trials: 3; promotion/final banks are retained in candidate and diagnostics streams

## Completeness

The archive contains **every regular file and empty directory** under the original run directory, the structured data-point mirror, and the final report snapshot directory. It preserves 44 files, 1.599 GiB of raw bytes, and 84.6 MiB of archived payload. Large JSONL streams are gzip-compressed individually; all other files are byte-for-byte copies.

snapshot_manifest.json records, for every source file, its root, relative path, raw size, row count where applicable, trailing-newline state, permissions, timestamp, raw SHA256, archive path, archive size, and archive SHA256. stream_map.tsv is the compact index.

## Restore

From this directory, run:

    python3 restore_snapshot.py /hy-tmp/restored_bert_large_sst2_stage2_35280

This recreates three exact trees:

- run/: the complete resumable training run, including checkpoint, status, baseline, best action, candidate store, diagnostics, launch command, logs, and empty runtime directories.
- structured/: the project structured writer output (episodes.jsonl, ppo_updates.jsonl, manifest, summary).
- report/: the standalone final HTML report; the authoritative JSONL streams can regenerate all training curves.

The restore command refuses a non-empty destination and verifies every restored file against the manifest.

## Reproducible figures and analyses

The preserved data supports reconstruction of reward, loss, Accuracy, Weighted F1, all three stability curves, P1/P2/P3 distributions, invalid/collapse rates, Block4 and truncation entropy, KL/clip/gradient/value diagnostics, throughput and per-GPU probe scaling, action histograms, per-layer fusion and K decisions, candidate/promotion/final-bank histories, Pareto/resource-frontier progress, best-so-far state, and baseline-versus-best comparisons. The HTML file is a convenience artifact; the JSONL streams remain authoritative.

## Integrity

Run sha256sum -c SHA256SUMS, gzip -t streams/*.gz, then execute the restore command. restore_verification.json records the server-side full restore rehearsal performed before commit.
