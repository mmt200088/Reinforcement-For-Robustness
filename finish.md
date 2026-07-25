# Runtime Optimization Handoff

- Date: 2026-07-25
- Status: implementation, review, exactness verification, and server benchmark complete
- Branch: `codex/bert-large-mrpc-exact-probe-scheduling-20260725`
- Tested source commit: `201611d2`
- Tested source tree: `a390f5c2`
- Evidence commit: `21c09d0c`

The BERT-large MRPC Stage-2 terminal-probe scheduler now groups up to four
episodes without crossing PPO or run boundaries. It preserves episode actions,
trial seeds, trial indices, aggregation order, reward handling, PPO state, and
checkpoint state while balancing the 20 K=5 trial tasks evenly across the four
usable CUDA workers.

The matched 360-episode server A/B measured a 1.475091x training speedup
(765.592 to 1129.319 episodes/hour) and a 1.584673x terminal-probe speedup.
Episodes, PPO updates, semantic checkpoint state, and candidate records matched
exactly under the documented exclusions for timestamps and output metadata.

Evidence is archived under
`experiments/server_command_runs/stage2_exact_probe_batching_20260725_201609/`.
This file is a handoff marker, not an experiment-completion marker. Before any
downstream run, verify that this marker's commit is identical in the local
worktree, the remote Git branch, and `/hy-tmp/Reinforcement-For-Robustness`.
