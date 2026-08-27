+# Reinforcement for Robustness
+
+This repository searches approximation and MPC configurations for BERT
+sequence-classification models. The production surface contains two-stage PPO,
+BO-RF, Greedy, COINN-GA, fusion-map preparation, the in-process Rescale
+optimizer, and selected-configuration evaluation.
+
+## Supported Profiles
+
+- Models: `bert-base`, `bert-large`
+- GLUE tasks: `mrpc`, `rte`, `sst2`
+- Comparators: BERT-base MRPC
+
+Preparation and both search stages use the fixed stratified 256-example probe
+from the GLUE training split. Final evaluation uses the complete GLUE
+validation split. The test split is not used.
+
+## Setup
+
+Use Python 3.9-3.12, CUDA-enabled PyTorch, and Linux `flock`:
+
+```bash
+python3 -m venv .venv
+source .venv/bin/activate
+pip install -e .
+export RFR_PYTHON=python
+```
+
+`RFR_PYTHON` makes search, supervision, and evaluation use the same Python
+environment. Model weights and datasets may be cached by Hugging Face or placed
+under `local_assets/`; generated assets and run outputs are not tracked.
+
+## Preparation
+
+The repository includes the pinned no-text probe identities, fusion maps, and
+Rescale skeletons. Rebuild them only when their source definitions change:
+
+```bash
+$RFR_PYTHON -m rfr.preparation.data.build_probe_fixture
+
+$RFR_PYTHON -m rfr.preparation.fusion.build_map \
+  --profile mrpc \
+  --out-dir configs/preparation/fusion/maps/mrpc \
+  --workers 16
+```
+
+The fusion-map builder emits the five production graphs for each profile:
+Block 2, Block 4, and the three Block 5 GELU variants. Stage 2 loads these maps
+directly. Its baseline is materialized automatically from the selected Stage 1
+configuration through the in-process Rescale optimizer.
+
+## PPO Search
+
+List presets:
+
+```bash
+bash run_search.sh --list-presets
+```
+
+Run Stage 1:
+
+```bash
+bash run_search.sh run rl \
+  --preset bert-base-mrpc-stage1-rl \
+  --stage1-search-episodes 51000 \
+  --fresh
+```
+
+Normal Stage 1 termination occurs only after the configured maximum episode.
+On completion it writes:
+
+```text
+<stage1-run>/stage1_best_config.json
+```
+
+Run Stage 2 with that exact JSON:
+
+```bash
+bash run_search.sh run rl \
+  --preset bert-base-mrpc-stage2-rl \
+  --stage1-config <stage1-run>/stage1_best_config.json \
+  --stage2-search-episodes 150000 \
+  --fresh
+```
+
+Normal Stage 2 termination occurs only after the configured maximum episode.
+The action at each layer is `[fusion_count, precision]`, where fusion count is
+`0` or `1` and precision is H, M, or L. Completed strict selection writes:
+
+```text
+<stage2-run>/search_best_config.json
+```
+
+## Comparator Search
+
+Each comparator runs its own Stage 1, writes and reloads its own Stage 1 JSON,
+runs Stage 2 through the same fusion/Rescale/evaluation path, and strictly
+validates the eligible top five:
+
+```bash
+bash run_search.sh run bo_rf --fresh
+bash run_search.sh run greedy --fresh
+bash run_search.sh run coinn_ga --fresh
+```
+
+The only normal termination settings are:
+
+| Method | Stage 1 | Stage 2 |
+| --- | --- | --- |
+| BO-RF | `--bo-stage1-no-improvement 1000` | `--bo-stage2-no-improvement 2000` |
+| Greedy | `--greedy-stage1-no-improvement-rounds 1` | `--greedy-stage2-no-improvement-rounds 1` |
+| COINN-GA | `--ga-stage1-generations 200` | `--ga-stage2-generations 200` |
+
+BO-RF counts consecutive evaluated candidates without incumbent improvement.
+Greedy counts complete 1-opt and 2-opt neighborhood rounds without improvement;
+an accepted 2-opt move returns to 1-opt. COINN-GA always completes every
+configured generation and has no early-stop condition.
+
+Use `--comparator-stage1-only` to stop the workflow after a normally completed
+comparator Stage 1. This is a workflow boundary, not an alternate algorithm
+termination condition.
+
+## Outputs and Resume
+
+The launcher prints the exact run directory. Default roots are:
+
+```text
+outputs/rl/<model>/<dataset>/stage1/<run>/
+outputs/rl/<model>/<dataset>/stage2/<run>/
+outputs/<bo_rf|greedy|coinn_ga>/bert-base/mrpc/two_stage/<run>/
+```
+
+Every Stage 1 run writes `stage1_best_config.json` only after normal
+completion. Every complete two-stage run writes `search_best_config.json`
+after strict selection. Comparator observations, histories, checkpoints, and
+strict evidence remain under `stage1_comparator/<algorithm>/` and
+`stage2/progress/search_<algorithm>/`. PPO checkpoints, candidate journals,
+and diagnostics remain under the stage-specific run directory.
+
+Send `SIGINT` once for a graceful stop. The current candidate or episode is
+finished and the resumable checkpoint is flushed. Run the same command again
+without `--fresh` to resume. Interrupted, failed, or incomplete runs do not
+write a formal best-config JSON.
+
+## Final Evaluation
+
+Final evaluation is separate from search. It accepts only a completed,
+final-eval-eligible `search_best_config.json` and repeatedly evaluates that one
+configuration on the complete validation split:
+
+```bash
+bash run_search.sh eval \
+  --config <search-run>/search_best_config.json \
+  --output-root outputs/evaluation \
+  --run-name final-mrpc \
+  --repeat 50 \
+  --foreground
+```
+
+The editable scientific fields in the JSON are the Stage 1 GELU/Softmax vectors
+and the Stage 2 action matrix. The loader validates their domains and rebuilds
+the full MPC configuration through the production fusion and Rescale path.
+
+Evaluation output is written to:
+
+```text
+outputs/evaluation/<algorithm>/<model>/<dataset>/<run-name>/evaluation/
+  selected_config_final_eval.json
+  selected_config_final_eval.md
+```
+
+Omit `--foreground` to launch in the background. Use `--dry-run` with any
+search or evaluation command to inspect the resolved command without inference.
