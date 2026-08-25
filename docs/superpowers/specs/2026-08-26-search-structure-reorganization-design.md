# Search Repository Structure Reorganization

## Goal

Reorganize the production repository around the search workflow without
changing scientific behavior. The final tree must make preparation, search,
runtime acceleration, validation evaluation, configuration, and generated
outputs easy to locate. The obsolete `Model_analysis/` and repository test
suite are removed from the final canonical branch.

## Non-Goals

- Do not change actions, random seeds, candidate order, rewards, constraints,
  baseline construction, trial counts, checkpoints, or evaluation rules.
- Do not optimize runtime behavior in this task.
- Do not restore retired algorithms, ablations, datasets, model families, or
  compatibility entrypoints.
- Do not restore the archived GLUE test-submission generator.
- Do not commit generated search results, model weights, or datasets.

## Current Findings

- `Model_analysis/` contains standalone approximation-distribution analysis
  scripts and is not imported by a production entrypoint.
- `tests/` is development-only and is not imported by the search runtime.
- The current launcher is `llama_7B_LayerImportance.sh`; its name no longer
  describes the BERT search pipeline.
- Production output is split across `Parting Chapter/`,
  `rl_training_data_points/`, and `Paean/outputs/`.
- The current production branch has no GLUE submission generator. A retired
  generator exists only on the experiment archive branch. It uses the old
  scaling-factor action, the GLUE test split, and datasets outside the current
  MRPC/RTE/SST-2 support matrix, so restoring it would not preserve current
  behavior.

## Target Repository Layout

```text
.
|-- run_search.sh
|-- README.md
|-- pyproject.toml
|-- src/rfr/
|   |-- common/
|   |-- preparation/
|   |   |-- data/
|   |   |-- fusion/
|   |   `-- rescale/
|   |-- search/
|   |   |-- common/
|   |   |-- runtime/
|   |   |-- rl/
|   |   |   |-- stage1/
|   |   |   `-- stage2/
|   |   `-- comparators/
|   |       |-- common/
|   |       |-- bo_rf/
|   |       |-- greedy/
|   |       `-- coinn_ga/
|   |-- evaluation/
|   `-- cli/
|-- configs/
|   |-- presets/
|   |-- preparation/
|   |   |-- fusion/
|   |   `-- rescale/
|   |-- evaluation/
|   `-- reference/
|-- fixtures/
|-- outputs/
|   |-- rl/
|   |-- bo_rf/
|   |-- greedy/
|   `-- coinn_ga/
|-- local_assets/
|-- examples/
|-- docs/
|-- agent_handoffs/
|-- scripts/
`-- .githooks/
```

Repository workflow scripts remain under `scripts/` because the Git protocol
depends on their stable paths. Search-facing command parsing moves to
`src/rfr/cli/`.

## Module Responsibilities

### Preparation

`rfr.preparation.data` owns the supported GLUE profile matrix, the fixed
stratified 256-example training probe, reproducibility fixtures, and fixture
generation.

`rfr.preparation.fusion` owns fusion-count map generation, auditing, loading,
enumeration, and fixed-action materialization. Generated production maps live
under `configs/preparation/fusion/`.

`rfr.preparation.rescale` owns the in-process Rescale optimizer, graph and
replan implementation, baseline bootstrap, optimizer bridges, and config
generation. Rescale JSON inputs live under `configs/preparation/rescale/`.

### Search

`rfr.search.common` owns shared scientific interfaces: candidate identity,
metrics, constraints, persistence, action serialization, reporting payloads,
and model-evaluation adapters.

`rfr.search.runtime` owns resource-only behavior: device discovery, elastic GPU
scheduling, CUDA kernels, persistent probe workers, model hook installation,
and graceful runtime control. This module may change resource assignment but
must not change search semantics.

`rfr.search.rl.stage1` contains only Stage-1 PPO policy, checkpoint, cache, seed,
and parallel rollout code. `rfr.search.rl.stage2` contains only the production
small GTrXL policy, layerwise environment, rollout, PPO training, validation
gate, and candidate promotion code.

`rfr.search.comparators.common` owns the exact Stage-1 and Stage-2 search spaces,
ranking rules, evaluator adapters, strict top-5 validation, persistence, and
resume journals shared by all comparators. Algorithm operators are separated:

- `bo_rf/`: categorical random-forest surrogate and acquisition logic.
- `greedy/`: exhaustive 1-opt and 2-opt traversal.
- `coinn_ga/`: population, selection, crossover, mutation, and generation
  replay.

Shared evaluator and constraint code is not copied into algorithm packages.

### Evaluation

`rfr.evaluation` owns full GLUE validation evaluation, selected-action loading,
Paean integration, result layout, and evaluation plots. Validation consumes the
same materialized action and model path used during search. It does not load
the GLUE test split or generate a submission archive.

### CLI and Configuration

`run_search.sh` is the only top-level launcher. It calls
`python -m rfr.cli.run` with `src/` on `PYTHONPATH`. The old launcher name is
removed without a compatibility wrapper.

Presets move to `configs/presets/`. Preset names, option values, defaults,
formal comparator contracts, fresh/resume behavior, graceful stop behavior,
and dry-run command semantics remain unchanged.

## Output Layout

Generated output is ignored by Git and rooted by algorithm:

```text
outputs/<algorithm>/<model>/<dataset>/<run-id>/
|-- metadata.json
|-- stage1/
|-- stage2/
|-- evaluation/
|-- checkpoints/
`-- logs/
```

`<algorithm>` is exactly one of `rl`, `bo_rf`, `greedy`, or `coinn_ga`.
Stage-only runs create only the stage they execute. Comparator runs keep their
own Stage-1 result, explicit Stage-2 binding, Stage-2 journal, strict top-5
selection, and validation evidence under the same algorithm run root.

Default run IDs remain deterministic functions of stage, constraints, seed,
and optional run tag so rerunning the same command without `--fresh` resumes
the same directory. No timestamp-only default is introduced.

Tracked final reference configurations remain source inputs under
`configs/reference/`; they are not mixed with generated output.

## Deletions

- Delete `Model_analysis/` and all of its contents.
- Delete `tests/` from the final production branch.
- Remove pytest-only project configuration and the test job from CI after the
  migration suite has been used for validation. Keep lint and dependency audit
  jobs.
- Remove old empty package paths and the old launcher after all imports and
  commands point to the new layout.

Git history and the existing remote archive branches remain recovery sources.

## Migration Rules

1. Move files with Git-aware renames before changing content.
2. Update imports and path constants mechanically, then make only the minimal
   manual changes required by the new package and output roots.
3. Split comparator algorithm implementations without changing function bodies
   or ordering. Shared code moves to `comparators/common`; dispatch selects the
   same implementation as before.
4. Keep JSON configuration bytes unchanged unless a path field must move.
5. Keep all runtime defaults and formal contract values unchanged.
6. Do not retain alias modules, legacy wrappers, or duplicate config copies.

## Verification

The migration is accepted only when all of the following hold:

1. Record the old launcher dry-run output for every production RL preset and
   each formal comparator command.
2. Record deterministic pure-function snapshots for action decoding, candidate
   hashes, precision presets, fusion mapping, Rescale materialization, search
   ranking, and output metadata.
3. Relocate test imports temporarily and run the complete existing server test
   suite against the reorganized source before deleting `tests/`.
4. Delete `tests/` in a separate final commit and prove that the only changes
   since the tested source commit are test/CI configuration removals.
5. From the exact final source commit, run Python compilation, shell syntax,
   preset validation, all production dry-runs, production-surface audit, and
   focused real Torch/GPU smoke for preparation, RL, all three comparators, and
   validation loading.
6. Compare old and new scientific snapshots exactly. Output directory strings
   and import/module names are the only permitted differences.
7. Verify generated output from every algorithm remains inside its own
   `outputs/<algorithm>/` root.
8. Complete task handoff, all-head aggregate review, server verification, and
   local/remote/server commit and tree parity before advancing canonical.

## Acceptance Criteria

- The final production tree matches the workflow-oriented layout.
- `Model_analysis/`, `tests/`, and the old launcher are absent.
- All four algorithms have isolated generated-output roots.
- Stage-1, Stage-2, resume, graceful stop, final validation, fusion mapping,
  Rescale materialization, and elastic GPU behavior are unchanged.
- No GLUE test-split submission behavior is introduced.
- The README contains concise English commands for preparation, Stage 1,
  Stage 2, comparators, resume, and validation.
- Canonical source is synchronized through Git with exact local, remote, and
  server commit/tree parity.
