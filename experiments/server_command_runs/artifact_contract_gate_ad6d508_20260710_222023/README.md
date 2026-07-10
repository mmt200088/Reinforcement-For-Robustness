# Structured Artifact Contract Gate

Source commit: `ad6d508ebcc50347af3152b82ace25e770feb2fe`

## Final Result

- Python compilation: rc=0.
- `tests.test_rl_data_points`: passed.
- `tests.test_stage2_persistent_output_verifier`: passed.
- `tests.test_optimization_evidence_bundle`: passed.
- Combined gate: 34 tests passed in 2.099s (`2.65s` process wall time).
- Server main worktree remained clean.
- The temporary `reports/` sparse inclusion was removed after the gate.

## Retained Diagnostics

- The first sparse-worktree run had 32 passing tests and two fixture errors
  because `reports/generate_blb_mapping_html_reports.py` was not checked out.
- A full-repository `git archive` retry was stopped before tests because it was
  copying the entire large experiment history. No result from that attempt is
  used.
- The accepted retry used Git sparse-checkout to add only `reports/`, ran the
  original command unchanged, then restored the exact original sparse paths.

Final evidence is under `sparse_retry/`; the earlier files remain for audit.
