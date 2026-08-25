# Agent Instructions

Follow `AGENTS.md` and `docs/GIT_MULTI_AGENT_PROTOCOL.md` for every task.
Run the required boundary checks through `scripts/repo_sync_guard.py`.

Work in an isolated task branch. Do not update the canonical branch or server
checkout unless the user has explicitly authorized this agent as the sole
aggregator. Preserve the train-probe protocol, action semantics, deterministic
seeds, reward and validation rules, checkpoint state, and result schemas.

Run source edits locally and runtime verification on the server from an exact
Git commit. Publish a completed handoff only after the requested tests pass.
