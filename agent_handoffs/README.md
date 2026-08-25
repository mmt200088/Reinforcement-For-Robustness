# Agent Handoffs

Task handoffs live in `agent_handoffs/tasks/`; aggregate manifests live in
`agent_handoffs/aggregates/`. Both use `agent_handoffs/schema.json`.

A completed task handoff records:

- a unique task ID and task branch;
- the exact source commit and source tree;
- base canonical commit and tree;
- changed paths and scientific invariants;
- local and server verification evidence;
- `aggregate_eligible=true` and `deployment_eligible=false`.

Only the authorized aggregator creates aggregate manifests or marks a canonical
deployment eligible. Result branches contain evidence only and are not source
merge inputs.
