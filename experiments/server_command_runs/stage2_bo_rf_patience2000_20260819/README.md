# Stage-2 BO-RF patience-2000 result

This artifact-only result preserves the complete BERT-base MRPC two-stage
BO-RF run rooted at:

`/hy-tmp/stage2_bo_rf_patience2000_v2_20260819_012847/bo_rf/bert-base/mrpc/s1t0.001_s2t0.001_s2st2.0`

The run used source commit `480e154053b1303e140077a05c46295cab95ef0a` and source tree `556ee67f7af1be94cdea349d18145fb032230bad`.

## Scientific status

- Stage-1: 1,072 evaluations; stopped after 1,000 without improvement; feasible.
- Stage-2: 2,076 inference-reaching evaluations; stopped after 2,000 without improvement.
- Strict validation: all eligible top-5 received 15-trial Bank-A evaluation.
- Final status: `complete_least_violating`.
- Strict feasible: `false`.
- Failed selected-candidate constraints: Loss mean, Accuracy mean and Weighted-F1 mean.
- Paean final evaluation: `skipped_ineligible`.

The selected configuration is the strict least-violating candidate and must not
be reported as a strict-feasible result.

## Files

- `stage2_bo_rf_patience2000_result.html`: human-readable report.
- `result_summary.json`: compact machine-readable result.
- `readable/`: browsable run artifacts except the large Stage-2 observation journal.
- `manifests/raw_files.json`: every raw file, size, mode, mtime and SHA-256.
- `archives/run.tar.zst.part-*`: lossless archive of all 29 raw files.
- `ARCHIVE_MANIFEST.json`: archive-part hashes and verification status.
- `archive_verification.json`: streaming restore-verification receipt.
- `source_parity.json`: local/remote/server source commit and tree parity.
- `SHA256SUMS`: hashes for every tracked artifact in this directory except itself.

## Restore

```bash
cat archives/run.tar.zst.part-* | zstd -dc | tar -xf -
sha256sum -c SHA256SUMS
```

The archive was stream-decoded and every member was rehashed against
`manifests/raw_files.json`; all 29 files and 536,151,200 raw bytes passed.
