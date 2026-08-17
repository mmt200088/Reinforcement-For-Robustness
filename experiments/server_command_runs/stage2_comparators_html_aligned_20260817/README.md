# Stage-2 Comparator Runs Aligned To Authoritative BERT-base MRPC RL

This result-only artifact was produced from source commit
`b8584e4f7bbda2fb00010fb1570be3fec0fbcdf2` and source tree
`c3e4ec2c9b61bde00d5b2ddc204192d8939414e0` on one NVIDIA RTX 4090.
It does not change source, tests, launchers, or scientific configuration.

Authority

- Historical report: `stage2_six_model_final_full_report_updated_20260807.html`
- Report SHA-256: `0b0676ee40102562a4d1f313c16222196ba2b1dcbb03225b9ee431142a299734`
- Historical RL source: `5c222da6186b8a60244b46029bbc8dac79befb34`
- Historical RL evidence: `8c2a526dbf793c95c388b5f8544a793e83c733dc`
- Exact fusion-map tree: `390968c9c4b499d38c102df7805d3c869a64c84b`
- Exact max-SF tree: `7eb88e05680f38ed9ceed2edc76f947fa1ce344a`

Run Status

- BO-RF: complete, 176 online evaluations, strict top-5 evaluated, 75 fresh
  strict trials, `complete_least_violating`, 454.4852 seconds total.
- Greedy: user-interrupted by SIGINT after 20,217 observations. It did not
  complete exhaustive 1-opt/2-opt verification, did not enter strict
  validation, and must not be reported as a final scientific result.
- COINN-GA: complete, 18 generations and 1,090 online evaluations, strict
  top-5 evaluated, 75 fresh strict trials, `complete_least_violating`,
  2071.6983 seconds total.

All methods use their own latest Stage-1 result. All non-optimizer Stage-2
parameters match the authoritative RL contract, including model and dataset
identity, Stage-1/Stage-2 batches 16/64, tolerance 0.001, stability multiplier
2.0, baseline 5x3, online trials 3, strict top-5, strict bank trials, seeds,
precision presets, fusion-count maps, and binary truncation semantics.

The optional Paean final evaluation was intentionally skipped. The mandatory
authoritative internal strict-F4 search validation was not skipped.

Contents

- `final_run_receipt.json`: machine-readable alignment assertions and results.
- `html_authority_preflight.json`: authority, Stage-1 bindings, maps, K presets,
  seeds, and shared contract.
- `gates/focused.log`: focused canonical test gate.
- `readable/`: directly readable compact files, preserving runtime paths.
- `manifests/*_raw_files.json`: every original runtime file, byte size, SHA-256.
- `archives/*.tar.zst.part-*`: lossless full runtime directories.
- `ARCHIVE_MANIFEST.json`: archive-part hashes and streaming content validation.
- `SHA256SUMS`: hashes for every tracked artifact except `SHA256SUMS` itself.

Restore

From this directory, restore one complete method directory with:

```bash
cat archives/bo_rf.tar.zst.part-* | zstd -dc | tar -xf -
cat archives/greedy.tar.zst.part-* | zstd -dc | tar -xf -
cat archives/coinn_ga.tar.zst.part-* | zstd -dc | tar -xf -
```

Validate the tracked artifact set with:

```bash
sha256sum -c SHA256SUMS
```

The raw file manifests provide a second, per-original-file integrity layer and
were validated by streaming every compressed tar member and recomputing its
SHA-256 before this result branch was committed.
