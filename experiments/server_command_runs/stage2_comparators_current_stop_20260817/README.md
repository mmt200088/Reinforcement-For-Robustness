# Stage-2 Comparator Stop-Point Results Versus Authoritative RL

This result-only artifact preserves the current BERT-base MRPC Stage-2 runs
for BO-RF, Greedy, and COINN-GA. It was produced from source commit
`b8584e4f7bbda2fb00010fb1570be3fec0fbcdf2` and source tree
`c3e4ec2c9b61bde00d5b2ddc204192d8939414e0` on one NVIDIA RTX 4090. It does
not change project source, tests, launchers, or scientific configuration.

## Authority

- Historical report: `stage2_six_model_final_full_report_updated_20260807.html`
- Historical report SHA-256:
  `0b0676ee40102562a4d1f313c16222196ba2b1dcbb03225b9ee431142a299734`
- Historical RL source: `5c222da6186b8a60244b46029bbc8dac79befb34`
- Historical RL evidence: `8c2a526dbf793c95c388b5f8544a793e83c733dc`
- Exact fusion-map tree: `390968c9c4b499d38c102df7805d3c869a64c84b`
- Exact max-SF tree: `7eb88e05680f38ed9ceed2edc76f947fa1ce344a`

## Reported Results

- BO-RF completed 176 online evaluations and strict Bank-A evaluation of all
  top-5 candidates. Its reported configuration is the strict least-violating
  candidate. It is not strict-feasible and did not advance to full F4.
- Greedy was stopped by the user after 20,217 unique online observations. Its
  reported configuration is the best candidate observed at that stop point.
  Exhaustive 1-opt/2-opt completion and strict validation were not performed.
- COINN-GA was stopped after the request to stop at generation 50 arrived too
  late. The durable journal contains 59 complete update generations and 36 of
  57 offspring from partial generation 60, for 3,463 unique observations. Its
  reported configuration is the best candidate observed at that stop point.
  Strict validation for this continuation was not performed.
- RL is the only strict-F4-feasible result in this comparison. Greedy and GA
  online metrics use 256 examples and three trials, so they are not direct
  scientific substitutes for the RL 408-example, 45-trial result.

The readable report is
`stage2_three_comparators_vs_rl_current_stop_20260817.html`. Machine-readable
selection details are under `analysis/`.

## Superseded GA Files

The GA run directory still contained strict artifacts from the earlier
18-generation patience-stopped run. They are not the result of the later
59-generation continuation. Compact copies are therefore isolated under
`readable/coinn_ga/superseded_pre_ga200_extension/`. The lossless GA archive
preserves the runtime directory exactly, including those historical files.

## Full Data Archive

Every raw training and intermediate file from each runtime directory is
recorded in a per-file manifest and preserved in a zstd-compressed tar stream:

| Method | Raw files | Raw bytes | Archive bytes |
| --- | ---: | ---: | ---: |
| BO-RF | 32 | 78,108,546 | 745,735 |
| Greedy | 24 | 4,822,997,172 | 37,974,834 |
| COINN-GA | 37 | 874,172,308 | 8,940,873 |

Each archive was decompressed as a stream and every tar member was rehashed
against its raw-file manifest before the first result commit.

Restore one method from this directory with:

```bash
cat archives/bo_rf.tar.zst.part-* | zstd -dc | tar -xf -
cat archives/greedy.tar.zst.part-* | zstd -dc | tar -xf -
cat archives/coinn_ga.tar.zst.part-* | zstd -dc | tar -xf -
```

Validate the tracked artifact set with:

```bash
sha256sum -c SHA256SUMS
```

`SHA256SUMS` covers every tracked artifact in this directory except itself.
