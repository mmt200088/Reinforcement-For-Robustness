# Unrelated Experiment Archive

This branch preserves one-off experiment code, generated reports, HTML
artifacts, and non-production experiment outputs removed from the search-source
branch on 2026-08-25.

The archive is based on canonical commit
`f3411ecb913ff557b1315a8c12ec0f9f4acd15e7`. Production RL, BO-RF, Greedy,
COINN-GA, Paean, Rescale, and profile source files remain on the canonical
branch and are not duplicated here. Model checkpoints, model weights, raw
datasets, and PDF guidance files are intentionally excluded.

Inspect or recover an artifact in a separate worktree:

```bash
git fetch origin codex/experiment-unrelated-artifacts-20260825
git worktree add ../rfr-experiment-archive \
  origin/codex/experiment-unrelated-artifacts-20260825
```
