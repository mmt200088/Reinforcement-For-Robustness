# Stage-2 COINN-GA 200-generation fresh result

This artifact-only result preserves the complete BERT-base MRPC two-stage
COINN-GA run rooted at:

`/hy-tmp/stage2_ga_fresh200_20260817_233430/coinn_ga/bert-base/mrpc/s1t0.001_s2t0.001_s2st2.0`

The run used source commit `4ca39159f03d6ccf5c1fc9cdf25027e3f97e784b` and source tree `0844404b4d16c4108c4507a9f48ee53cafddabbc`.

## Scientific status

- Stage-1: 200 update generations, 11,464 unique evaluations, feasible.
- Stage-2: 200 update generations, 11,464 inference-reaching candidates.
- Strict validation: all eligible top-5 received 15-trial Bank-A evaluation.
- Final status: `complete_least_violating`.
- Strict feasible: `false`.
- Failed selected-candidate constraints: Accuracy mean and Weighted-F1 mean.
- Paean final evaluation: `skipped_ineligible`.

The selected configuration is the strict least-violating candidate and must not
be reported as a strict-feasible result.

## Files

- `stage2_ga_200_fresh_result.html`: human-readable report.
- `result_summary.json`: compact machine-readable result.
- `readable/`: browsable run artifacts except the 2.73 GB Stage-2 observation journal.
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
`manifests/raw_files.json`; all 29 files and 2,785,069,075 raw bytes passed.
