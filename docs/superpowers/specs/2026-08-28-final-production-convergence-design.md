# Final Production Convergence Design

## Goal

Reduce the repository to the production preparation, search, and selected-config
evaluation paths while preserving the scientific behavior of PPO, BO-RF,
Greedy, and COINN-GA.

## Production Surface

The supported workflow consists of:

1. Build or validate the deterministic training probe, fusion-count maps, and
   Rescale optimizer baselines.
2. Run Stage 1 with PPO or one comparator.
3. Read the selected Stage-1 JSON and run Stage 2 with the same algorithm.
4. Read the selected two-stage JSON and evaluate that single configuration on
   the full GLUE validation split.

The supported model families are BERT-base and BERT-large. The supported GLUE
tasks are MRPC, RTE, and SST-2. Formal comparators remain limited to BERT-base
MRPC.

Git hooks, handoff records, and synchronization guards remain in the repository
as operational infrastructure, but are outside the runtime package.

## JSON Contracts

### Stage 1

Every completed Stage-1 search writes `<run>/stage1_best_config.json` using
schema `stage1_best_config_v1`. The editable scientific fields are:

- `algorithm`
- `model_type`
- `dataset`
- `num_layers`
- `stage1.gelu`
- `stage1.softmax`

Selection status, dataset-protocol identity, and search provenance are retained
as metadata. Stage 2 accepts exactly one input flag:
`--stage1-config <stage1_best_config.json>`. It rejects model, dataset, layer,
vector-length, and action-domain mismatches before model loading.

The launcher and evaluator no longer support all-max defaults, in-memory
handoff, record discovery, legacy reference files, or manual degree flags.
Comparator Stage 1 writes the same JSON and Stage 2 reads it back from disk;
the in-memory result is checked against the reloaded value but is not the
handoff mechanism.

### Complete Search

Every normally completed two-stage search writes
`<run>/search_best_config.json` using schema `search_best_config_v1`. Its
editable scientific fields are the Stage-1 vectors and the Stage-2
`action_matrix`, with one row per Transformer layer:

```json
{
  "stage2": {
    "action_matrix": [[0, 0], [1, 2]]
  }
}
```

The first coordinate selects Block 4 fusion count 0 or 1. The second selects
high, medium, or low precision. The full action vector, map option IDs, boost
overrides, and model configuration are deterministically rebuilt from this
matrix through the same materialization path used during search.

Completed least-violating searches still emit the JSON with
`final_eval_eligible=false`. Interrupted, failed, and test-only runs do not
emit a formal search-best JSON.

## Final Evaluation

The only selected-configuration entry point is:

```bash
bash run_search.sh eval --config <search_best_config.json>
```

The algorithm, model, dataset, and layer count come from the JSON. Final
evaluation validates eligibility and materializes the selected action through
the production fusion and Rescale chain. Users may edit the Stage-1 vectors or
Stage-2 action matrix; all derived state is recomputed.

Final evaluation repeats only that selected configuration. Checkpoint scanning,
run-directory recovery, reference JSONs, manual vectors, action templates,
ranges, fixed overrides, random configurations, and cost-matched controls are
removed.

## Termination Contracts

Graceful interruption remains available for operational safety but never
produces a formal best-config JSON. Each algorithm has one normal termination
condition:

- Stage-1 and Stage-2 PPO stop at their configured maximum episode count.
- COINN-GA stops after its configured update-generation count.
- BO-RF stops after its configured number of consecutive evaluations without
  improvement.
- Greedy stops after its configured number of complete 1-opt/2-opt neighborhood
  rounds without improvement. The default is one round, preserving the current
  verified-local-optimum result.

Entropy convergence, PPO plateau convergence, evaluation caps, GA patience,
and alternative completion branches are removed. Strict validation remains a
result-certification step, not a search termination condition.

## Data and Assets

Training uses the full GLUE training split. Preparation, Stage 1, and Stage 2
use the deterministic 256-example training probe. Selected configuration
evaluation uses the full GLUE validation split. The test split is not used.

The raw-row MRPC fixture is removed. MRPC ordering and identity are rebuilt from
the pinned Hugging Face dataset revision and deterministic seeds. The compact
probe positions and hashes remain because they are configuration metadata and
contain no raw examples.

No model weights or downloaded datasets are tracked. Empty
`local_assets/models` and `local_assets/datasets` directories remain as import
locations. One representative RL log may remain below 10 MB.

## Dead-Code Removal

Deletion requires all of the following evidence:

1. The module or symbol is unreachable from every supported launcher and
   preparation entry point in the static import graph.
2. It is absent from dynamic import and CUDA backend registration paths.
3. Server-side import tracing and production canaries do not execute it.
4. Removing it preserves focused execution snapshots and launcher contracts.

This removes retired final-eval generators, historical configuration loaders,
unused preparation CLIs, obsolete fallback branches, old result templates, and
the unused MRPC Block-1 fusion map. Resume, checkpoint, strict validation,
elastic GPU scheduling, CUDA kernels, and active Rescale code remain.

## Verification

The pre-cleanup commit is retained on a remote archive branch. Verification is
performed only on the GPU server and stored outside the final local checkout.

Required gates are:

- all supported launcher presets and termination settings;
- Stage-1 and search-best JSON generation, reload, edit, and rejection tests;
- deterministic old/new PPO and comparator trajectory parity;
- six-profile fusion-map audit and comparison with authoritative map bytes;
- H/M/L ciphertext, simulation, reserve, ring-width, and installed-K parity;
- Rescale baseline and full action materialization parity;
- real BERT/CUDA Stage-1, Stage-2, comparator, and final-eval canaries;
- no tracked personal data, raw dataset rows, model weights, PDFs, obsolete
  results, or unapproved logs;
- local, remote, and server commit/tree parity with clean tracked state.
