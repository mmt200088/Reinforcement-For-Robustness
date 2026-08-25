# Repository Instructions

## Scope

This repository supports BERT-base and BERT-large on MRPC, RTE, and SST-2.
The production paths are Stage-1 PPO, Stage-2 layerwise robust PPO, BO-RF,
Greedy, COINN-GA, Paean final evaluation, fusion-map generation, and the
in-process Rescale optimizer.

Do not add compatibility paths, alternative policy networks, research
ablations, or support for another model or dataset without an explicit task.

## Git Protocol

Read `docs/GIT_MULTI_AGENT_PROTOCOL.md` before changing source.

- Ordinary task agents work in isolated `codex/task-*` branches and worktrees.
- Ordinary agents never update `jk_standard_rl` and never deploy source.
- Each completed task publishes one handoff under `agent_handoffs/tasks/`.
- Only the explicitly authorized aggregator may integrate completed handoffs,
  update `jk_standard_rl`, or synchronize the server checkout.
- Do not modify, reset, clean, or overwrite another worktree's dirty files.
- Never bypass `.githooks/pre-push`.

Source is edited locally. Server source is obtained only through Git at an
exact verified commit and tree. The server may run tests and experiments but
must not patch, format, or commit source.

## Scientific Invariants

- Search uses the fixed stratified 256-example GLUE train probe.
- Full GLUE validation is final-evaluation-only.
- Stage 2 uses the layerwise fusion plus H/M/L precision action, robust
  constrained reward, A/B/C search-gate banks, strict top-5 selection, and the
  small shared GTrXL policy.
- Paper ciphertext K metadata is distinct from executable simulation K.
- Rescale materialization uses the real in-process optimizer and fails closed.
- Elastic GPU scheduling may change resource assignment only. Seeds, actions,
  trial order, rewards, candidates, checkpoints, and scientific results must
  remain invariant.

## Verification

Use focused tests while editing and run the complete test suite before a
handoff. GPU-dependent tests and experiments run on the server. Report exactly
which tests ran, which were skipped, and why.
