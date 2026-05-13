# From Codex: Phase-1B Optimizer / Registry / Mask Trust Closure

## Task Executed

Codex executed GPT 5.5 Pro's taskbook `codex_phase1b_optimizer_consistency_registry_mask_taskbook_v1.md`.

Goal: close the Phase-1 evidence-integrity gap before any long BLB Stage-2 RL training. This included optimizer-cost convention repair, effective action identity, registry artifact consistency, F0 mask repair, runner/policy diagnostics, and a GPT 5.5 Pro handoff package.

No long training was started. Server sync and F1 GPU smoke are blocked because SSH to `100.84.74.99:8722` is reset during key exchange.

## Current Git HEAD

- branch: `jk_standard_rl`
- local HEAD: `6341ceab2bb15cd6e4cb0b98805bc88d7343a984`
- origin/jk_standard_rl: `6341ceab2bb15cd6e4cb0b98805bc88d7343a984`
- tracked diff hash: `5ceffb26b9a14856169b876fc9ffc3334b50de7b788da78e6bf5495b6243b9bd`
- uncommitted changes: Yes

## Result Credibility Page

Path: `RESULT_CREDIBILITY_PAGE.md`

Four-question summary:

- Current packaged code: Yes. Phase-1B local F0, registry, consistency reports, and tests come from this packaged code.
- Real in-process Rescale_optimizer: Yes. Hash `ed28392d4078e4eb7734740023d281d5b87f1abde68340d7776f4e2855e4278e`.
- Formal feasible vs diagnostic feasible: Limited. F0 optimizer-only diagnostic feasible; not F1/F2/F3 model feasibility.
- Comparable with previous handoff: Limited. Compare only under the Phase-1B identity tuple; do not mix old Trust0/Phase1 candidate stores.

## Files Changed

Main code and interface changes:

- `blb_stage2_rl/optimizer_cost.py`: added canonical `evaluate_action_for_cost(...)`.
- `blb_stage2_rl/env.py`: RL env step and baseline calibration use the canonical cfg-derived evaluator.
- `rescale_optimizer_bridge.py`: `evaluate_baseline_blocks(requests)` delegates to `evaluate_blocks(requests)`; optimizer-native empty baseline remains diagnostic/handover only.
- `blb_stage2_rl/candidate_store.py`: records raw/effective action identity and uses `effective_action_hash + identity_context` for candidate keys.
- `scripts/blb_eval_action.py`: F0 records now include raw/effective identity and use the canonical cost evaluator.
- `scripts/blb_f0_scan_feasible_domain.py`: baseline/candidates use the same canonical evaluator; inactive slots are baseline-only; K mask is conservative 13/12/11; random validity reports cost distribution; multi-random scan added.
- `scripts/blb_compare_optimizer_modes.py`: new Phase-1B optimizer consistency report.
- `scripts/blb_export_action_registry.py`: exports consistent embedded/full/effective registries.
- `blb_stage2_rl/action_mask.py`: rejects mask files that open ineffective slots to non-baseline values.
- `blb_stage2_rl/policy.py`: `per_dim_entropy(...)` accepts action mask/bias.
- `blb_stage2_rl/persistence.py` and `blb_stage2_rl/runner.py`: trace/log fields now distinguish raw/masked entropy and effective/ineffective mutation counts.

Tests and reports:

- Added/updated BLB tests including optimizer-cost consistency, registry artifact consistency, action mask, candidate identity, F0 scan, and stage2 RL regressions.
- Generated `reports/blb_opt/phase1b_consistency/*`, `reports/blb_opt/phase1b_registry/*`, `reports/blb_opt/phase1b_f0_scan/*`, and `reports/phase1b_result_summary.*`.

## Key Code Or Diff

Full tracked diff: `_sync_metadata/git_diff.patch`.
Diff summary: `_sync_metadata/git_diff_stat.txt`.

Canonical convention:

```text
all actions, including all-max, use:
action_vector_to_cfgs -> build_optimizer_requests -> evaluate_blocks
```

Optimizer consistency cases:

- all_max_raw: mode=evaluate_baseline_blocks, request_count=59, sends_block1_L0=False, sends_first_input=False, valid=True, total_bits_sum=14889, fusion_count=0, raw_hash=e18db2a9a1b3..., effective_hash=e18db2a9a1b3...
- all_max_via_candidate_path: mode=evaluate_blocks, request_count=59, sends_block1_L0=False, sends_first_input=False, valid=True, total_bits_sum=14889, fusion_count=0, raw_hash=e18db2a9a1b3..., effective_hash=e18db2a9a1b3...
- inactive_l0b1_mutation: mode=evaluate_blocks, request_count=59, sends_block1_L0=False, sends_first_input=False, valid=True, total_bits_sum=14889, fusion_count=0, raw_hash=e6bd245c733c..., effective_hash=e18db2a9a1b3...
- inactive_first_input_mutation: mode=evaluate_blocks, request_count=59, sends_block1_L0=False, sends_first_input=False, valid=True, total_bits_sum=14889, fusion_count=0, raw_hash=f74ae731417f..., effective_hash=e18db2a9a1b3...
- effective_single_mutation: mode=evaluate_blocks, request_count=59, sends_block1_L0=False, sends_first_input=False, valid=True, total_bits_sum=14873, fusion_count=0, raw_hash=37fbb939cbb9..., effective_hash=37fbb939cbb9...

## Commands Run

Local verification:

```bash
python -m py_compile blb_stage2_rl/candidate_store.py blb_stage2_rl/action_mask.py blb_stage2_rl/policy.py blb_stage2_rl/persistence.py blb_stage2_rl/optimizer_cost.py blb_stage2_rl/env.py blb_stage2_rl/runner.py scripts/blb_export_action_registry.py scripts/blb_eval_action.py scripts/blb_f0_scan_feasible_domain.py scripts/blb_compare_optimizer_modes.py rescale_optimizer_bridge.py
python -m unittest tests.test_blb_baseline_bootstrap tests.test_blb_stage2_rl_regressions tests.test_blb_action_mask tests.test_blb_candidate_store_identity tests.test_blb_cost_semantics tests.test_blb_f0_scan tests.test_blb_optimizer_cost_consistency tests.test_blb_registry_artifact_consistency -v
python -m py_compile blb_stage2_rl/env.py
python -m unittest tests.test_blb_stage2_rl_regressions.BLBOptimizerBaselineRegressionTests tests.test_blb_optimizer_cost_consistency -v
git diff --check
```

Report generation:

```bash
python scripts/blb_compare_optimizer_modes.py ... --output-dir reports/blb_opt/phase1b_consistency
python scripts/blb_export_action_registry.py ... --output-dir reports/blb_opt/phase1b_registry
python scripts/blb_f0_scan_feasible_domain.py ... --output-dir reports/blb_opt/phase1b_f0_scan --beam-mutation-limit 64 --random-samples 200 --multi-random-samples 500 --multi-mutation-counts 2,4,8,16,32
```

SSH/server probe:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 -o StrictHostKeyChecking=no -p 8722 root@100.84.74.99 "cd /var/tmp/root-home/Reinforcement-For-Robustness && pwd && git rev-parse HEAD"
```

## Test Results

- py_compile changed Python files: passed
- focused unittest: 55 tests OK
- post-cleanup targeted unittest: 6 tests OK
- git diff --check: passed with CRLF warnings only
- server focused tests: not run, SSH reset
- F1 GPU smoke: not run, SSH reset

## Errors Or Exceptions

- SSH reset: `kex_exchange_identification: read: Connection reset`; therefore no server sync, server tests, or F1 GPU smoke is claimed.
- One local verification command initially used `&&`, which this PowerShell does not support. It did not run tests; the equivalent PowerShell-native command was rerun and passed.
- `git diff --check` printed CRLF warnings but exited 0.

## Environment Variables

Local F0 did not require special environment variables.

Server smoke variables requested by taskbook but not used because SSH is blocked:

```bash
HF_ENDPOINT=https://hf-mirror.com
HF_HOME=/var/tmp/root-home/.cache/huggingface
GLUE_LOCAL_DATASET_DIR=/var/tmp/root-home/.cache/huggingface/datasets/glue/mrpc/0.0.0/1edab70c7fdff5d2
```

## Data Source

- profile/dataset: `mrpc`
- model: `bert-base`
- Stage-1 config source: `glue_final_configs_best_ppo.json`
- Stage-1 GELU degrees: `[1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1]`
- Stage-1 Softmax degrees: `[2, 2, 5, 5, 5, 2, 5, 2, 5, 5, 6, 2]`
- Stage-1 content hash: `6454e0556f54ddb4519d9d2998582bca40a41fe2910d2ece679e455f8854eed3`

## Rescale Optimizer Mode / Root / Hash

- mode: `in_process_real`
- root: `Rescale_optimizer`
- canonical hash: `ed28392d4078e4eb7734740023d281d5b87f1abde68340d7776f4e2855e4278e`

## Stage1 Config Source

- source file: `glue_final_configs_best_ppo.json`
- content hash: `6454e0556f54ddb4519d9d2998582bca40a41fe2910d2ece679e455f8854eed3`

## Action Registry Hash

- action_space_version: `current-code-v1`
- decode_version: `action_space_v1`
- action width: `877`
- full slot count: `877`
- effective slot count: `791`
- registry hash: `6c3662ba26160952e27dca8a8e3ae164af8326ac01819677c7b1a453fe342412`

## Max SFS Hash

- path: `blb_stage2_rl/max_sfs/mrpc.json`
- hash: `bee17f0ccab949b79b4ca011a97da4cebd1d749e6ad49bffa272a701895e09f6`

## Which Results Come From Current Code

Current-code evidence:

- `reports/blb_opt/phase1b_consistency/*`
- `reports/blb_opt/phase1b_registry/*`
- `reports/blb_opt/phase1b_f0_scan/*`
- `reports/phase1b_result_summary.*`
- local py_compile / unittest / git diff check

Old-code / reference only:

- `reports/blb_opt/trust0_*`
- `reports/trust0_*`
- `reports/blb_opt/phase1_*`
- any long-training result generated before Claude's layer-0 flow change and this Phase-1B patch set

## Training Commands And Results

No current-code long training. No F1/F2/F3 result.

F0 baseline:

- optimizer_valid: `True`
- total_bits_sum: `14889`
- fusion_count: `0`
- avg_k: `13.0`
- mask hash: `332b30d017d92e7bd5b27255b005413eb2a49cd147174750ac71e213b99f6d08`

Masked random:

- mutation_count=1: valid=50/50 (1.000), total_bits=14873/14887.28/14893, fusion=0/0.02/1
- mutation_count=2: valid=50/50 (1.000), total_bits=14873/14885.60/14895, fusion=0/0.10/1
- mutation_count=4: valid=50/50 (1.000), total_bits=14851/14881.88/14897, fusion=0/0.16/1
- mutation_count=8: valid=50/50 (1.000), total_bits=14843/14874.92/14891, fusion=0/0.42/3

Multi-random top candidates:

- mutation_count=32, total_bits_sum=14795, fusion_count=1, effective_hash=41e3ad0fee3e93b2e6ca50ee770ed308297c3dae4eb3cce1fb95cfda6782fe9d
- mutation_count=32, total_bits_sum=14803, fusion_count=0, effective_hash=cc9dd6ef0c6e57f413b675bb10732b145be9b7c978401a6b46c7aaf240c88e1a
- mutation_count=32, total_bits_sum=14803, fusion_count=0, effective_hash=25a87e6d9e982717426610335f833d2771cca7cfa862c0c82dae9eff36071a60
- mutation_count=16, total_bits_sum=14809, fusion_count=1, effective_hash=56a7fd7549ebe600215aa2951dfa930e504d4eb3ea131b8249fb7ca4d9351975
- mutation_count=32, total_bits_sum=14809, fusion_count=1, effective_hash=4995c4994bd113df62115012231b8c8f0d854334329eaf1dbdee4e3bb3147557

## Codex Summary And Suggestions

Phase-1B fixes the key evidence-integrity problem. The all-max action, inactive-slot mutations, and normal candidates can now be compared under one cfg-derived optimizer convention. Inactive layer-0 block1 and first_input mutations change raw hashes but not effective hashes or optimizer cost. The F0 mask no longer opens ineffective slots and no longer treats K=8/9/10 as optimizer-proven safe.

Next step should be server sync plus current-code F1 GPU smoke only. Do not start 50k+ RL training until that smoke passes.

## Supervisor Observations / Questions

- The remaining blocker is remote access, not a local code failure: SSH resets during key exchange.
- Phase-1B baseline is `14889` bits under the canonical cfg-derived all-max convention; do not compare it as the same identity as the previous `14779` Phase-1 baseline.
- Old Trust0/Phase1 candidate stores must not be mixed with this package unless the identity tuple matches exactly.
