# Stage-1 comparator batch16 formal run

This is the authoritative replacement for the prior batch64 Greedy, BO-RF,
and COINN-GA Stage-1 runs.

- Source commit: `9d833d90760b1bf85fca4c8650e8149f61119ad2`
- Source tree: `918fb6e4f5e6ea6fa659a30045331f99dc48800e`
- Model/dataset: `textattack/bert-base-uncased-MRPC` / GLUE MRPC validation (408 examples)
- Batch/micro-batch: 16 / 16
- Order: Greedy, BO-RF, COINN-GA
- Greedy: 958 evaluations, `verified_local_optimum`
- BO-RF: 1072 evaluations, `no_improvement_convergence`
- COINN-GA: 11464 evaluations, `completed_generations` (200 complete generations)

All persistent observations, histories, checkpoints, manifests, summaries,
model logs, launcher logs, GPU samples, queue state, and validation summaries
are included. `SHA256SUMS` verifies every archived file other than itself.
