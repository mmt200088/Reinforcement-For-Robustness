# Training Artifact Archive

This branch preserves historical RL and comparator logs, intermediate search
state, and result artifacts that were removed from the production source tree
on 2026-08-25.

The archive is based on canonical commit
`f3411ecb913ff557b1315a8c12ec0f9f4acd15e7`. Model checkpoints, model weights,
raw datasets, and full server recovery bundles are intentionally excluded.

To inspect or recover an artifact without changing the production branch:

```bash
git fetch origin codex/archive-training-artifacts-20260825
git show origin/codex/archive-training-artifacts-20260825:<path>
```

Use a separate worktree when recovering a directory.
