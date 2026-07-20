# Stage-2 MRPC episode 114240 full recovery bundle

This directory preserves the complete recoverable state and raw evidence for the
BERT-base MRPC Stage-2 run stopped on 2026-07-19.

## Run identity

- Training source commit: `fa4ee9cbd27d6265238f8d1091b712e14ee86066`
- Submission runtime commit: `3889a9d0c215c2d4603ab4459fdf347599140cad`
- Structured run id: `Parting_Chapter_persistent_rl_bert-base_mrpc_s1t0.001_s2t0.001_s2st2.0__dual_resource_natconv_fresh_20260718_190924__20260718T111042090914Z__pid1943348`
- Graceful-stop checkpoint: episode `114240`, PPO update `952`
- Checkpoint SHA-256: `c039f3de3619261880aa3eb771d80318b1b17984cae926c4f6e819a3a03b1ab4`

## What is retained

The bundle keeps the raw byte streams, not only plots or summaries:

- all `114240` per-episode records, both diagnostics and structured-writer forms;
- all `952` PPO update records, both diagnostics and structured-writer forms;
- all `117168` candidate-store evidence rows: F1 trials, F4 promotions, validity,
  identity, rank keys, actions and trial groups;
- the loadable PPO checkpoint with policy, optimizer, episode/update counters,
  strict frontier, candidate-store fingerprints and Python/NumPy/Torch/CUDA RNG
  states;
- run and algorithm manifests, baseline and constraint evidence, launch command,
  health/status logs, action histogram, Pareto frontier, top candidates and best
  action descriptions;
- the live snapshot and checkpoint-safe stop snapshot, including the compact
  PPO stream and resume command;
- the two-seed GLUE submission outputs, TSV files, submission zips, logs, source
  hashes, fixed boosted/fused action and runtime install audits;
- `evidence/final_actual_installed_configuration.json`, a self-contained,
  machine-readable export of the selected checkpoint action, decoded per-layer
  choices, exact handler-installed configuration, every installed slot SF,
  truncation K, fusion choice, provenance and verification hashes;
- the five/six-profile SF audit result directories and the launch-wrapper result.
- the server worktree status and original `LATEST_*`/lock pointer values needed
  to audit exactly which result paths were server-only before evacuation. The
  stopped process's stale active lock is evidence only and is not reactivated by
  `restore.sh`.

The duplicated source checkout, Python bytecode, Hugging Face caches and model
weights are not copied: they are inputs/caches rather than run results. Exact code
versions and dataset/model input hashes are retained in the manifests.

## Archive layout

- `persistent_run_without_large_jsonl.tar.gz`: canonical run tree including the
  64 MB resumable checkpoint and all small/derived diagnostics.
- `candidate_store.jsonl.gz.part*`: exact 2.97 GB candidate store, compressed and
  split below the GitHub single-blob limit.
- `diagnostics_episodes.jsonl.gz.part*`: exact diagnostics episode stream.
- `structured_episodes.jsonl.gz.part*`: exact project-root structured mirror.
- `ppo_updates.jsonl.gz`: exact diagnostics PPO stream.
- `structured_ppo_updates.jsonl.gz`: exact structured-writer PPO stream.
- `structured_writer_metadata.tar.gz`: structured writer manifest.
- `report_and_graceful_stop_snapshots.tar.gz`: live and final stop snapshots.
- `final_glue_submission_results.tar.gz`: all final inference outputs and audits,
  excluding the duplicate source checkout.
- `server_experiment_results.tar.gz`: the related server audit result folders.
- `SHA256SUMS`: hash of every archive/blob committed to Git.
- `BUNDLE_CONTENTS.tsv`: SHA-256, byte size and relative path for every other
  file in this recovery bundle.
- `evidence/hy_tmp_top_level_inventory.tsv`: final size/path inventory of the
  server's top-level `/hy-tmp` objects, retained to audit the retirement scope.

The original structured-writer `manifest.json` is also exposed at its canonical
`rl_training_data_points/stage2/bert-base/mrpc/<run-id>/` path. Its large raw
streams remain in this split recovery bundle.

Run `./restore.sh /empty/output/directory` to reconstruct the original repo-relative
and `/hy-tmp`-relative trees and verify every restored raw stream.

## Completeness result

- Diagnostics episodes: IDs `0..114239`, no gaps, no malformed JSON.
- Structured episodes: IDs `0..114239`, no gaps, no malformed JSON.
- The two episode streams are row-equivalent after removing the structured
  writer's `is_new_best` and `best_reward_so_far` annotations.
- PPO updates: IDs `1..952`, no gaps, no malformed JSON; final completed episode
  is `114240`.
- Candidate store: `117168` valid JSON rows with terminal newline.
- The checkpoint hash matches the stop manifest and the checkpoint loads with all
  required resume state.

Machine-readable details and original-stream hashes are in
`integrity_manifest.json`.

## Runtime configuration finding

The two GLUE seeds installed byte-identical Block1/2/4/5 configs, and each run
proved `decoded == supplied_to_bridge == installed_in_handler` by SHA-256. The
raw source of truth is
`evidence/final_actual_installed_configuration.json`; it is rendered in
`reports/html_reports/20260720_stage2_mrpc_ep114240_actual_installed_sf_and_k.html`.

Block3 is not present in either handler install audit. The runtime bridge explicitly
ignores `block3_cfgs`; consequently Block3 SF and truncation K were not applied to
the model, although the RL summary counted those legacy-vector K values. The RL
summary reported 59 K positions and 101 removed bits; the actual installed config
contains 47 K positions and 73 removed bits. This bundle preserves both views so
the mapping issue remains auditable.
