# Stage-2 six-model final full report

This compact result artifact consolidates the final archived Stage-2 RL state
for BERT-base and BERT-large on MRPC, RTE, and SST-2.

Files:

- `stage2_six_model_final_full_report.html`: standalone human-readable report.
- `stage2_six_model_final_full_report_data.json`: machine-readable values and
  downsampled curves used by the report.
- `build_report.py`: reproducible builder that streams the immutable Git
  archive blobs without restoring the multi-gigabyte raw snapshots locally.
- `test_build_report.py`: focused regression test for legacy/new archive
  stream layouts.
- `SHA256SUMS`: artifact integrity manifest.

Metric scope:

- Training curves are reconstructed from the complete F1 online-probe episode
  and PPO streams.
- Final baseline/candidate metric comparisons use F4 `validation_full` banks.
- Strictly certified, Bank-B-only, and baseline-fallback outcomes are reported
  separately; the report does not promote provisional candidates to final
  certification.

The six archive references and Stage-1 configuration matches are anchored by
audit commit `2bb175defb4c042773650c26d7b853bdd81e0a59`.
