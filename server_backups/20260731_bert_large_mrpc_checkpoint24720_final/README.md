# BERT-large MRPC Stage-2 checkpoint-boundary archive

This directory is the complete, Git-safe archive of the Stage-2 run that was
stopped at an atomic PPO checkpoint boundary on 2026-07-31.

## Identity

- Source commit: `0764e71025ee887327d63364052935d770c489f9`
- Model/profile: BERT-large MRPC / `mrpc_large`
- Checkpoint: episode `24720`, PPO update `206`
- Original planned maximum: `150000` episodes
- Policy: `shared_gtrxl_small_v1`
- Algorithm: `network_weighted_hml_three_bank_convergence_v12`
- Stop state: process and reward-probe workers terminated after the checkpoint
  was atomically replaced

The raw status JSON reports episode 24600 because the external checkpoint
boundary stop froze the process before the following status refresh. The
checkpoint and append-only streams are authoritative and agree exactly at
episode 24720/update 206.

## Completeness

The archive contains:

- all 24,720 diagnostic and structured episode records;
- all 206 diagnostic and structured PPO update records;
- all 34,792 candidate-store records, including F1/F4 trials and promotion
  status;
- every Pareto and top-candidate event;
- the resumable Torch checkpoint and its optimizer/policy state;
- baseline, best-action, action-histogram, run-context, convergence, log, and
  structured-writer artifacts;
- the temporary status file present at the exact stop boundary;
- the original command line, stop evidence, source Git state, and GPU/process
  snapshot.

`verification_report.json` records the original byte counts, row counts, raw
SHA-256 hashes, mirror checks, and checkpoint fingerprints.
`snapshot_manifest.json` maps every compressed stream back to its original
relative path. `SHA256SUMS` covers every archived file except itself.

## Verify

Run from this directory:

```bash
shasum -a 256 -c SHA256SUMS
gzip -t streams/*.gz
```

## Restore

Restore a complete resume tree:

```bash
python3 restore_snapshot.py \
  --output-dir /path/to/restored-bert-large-mrpc
```

The output contains `run/` and `structured/` roots. The script decompresses all
streams, verifies their raw byte count, row count, newline boundary, and
SHA-256 hash, and refuses to overwrite a non-empty directory.

This stopped archive has no post-checkpoint tail, so `resume` and `full` modes
restore identical bytes.

## Report status

The accompanying HTML report uses the authoritative full-validation F4
evidence available at the stop point. The selected strict candidate was
promoted on Banks A+B (30 full-validation trials). The run had not reached its
90,000-episode convergence floor, and the final Bank-C revalidation was not yet
due; the report labels the result as an interim strict best rather than a
converged final model.
