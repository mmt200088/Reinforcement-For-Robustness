# Production Cleanup Phase 3 Repository Hygiene Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Remove non-runtime repository clutter and personal information, publish concise English documentation, and prove the final cleaned source is reproducible and scientifically equivalent to the Phase-1 production baseline.

**Architecture:** Phase 3 changes repository presentation and development metadata after the runtime slice has passed. It preserves all excluded data/result trees byte-for-byte, removes unused submodules and process history, then completes full CPU/GPU verification and the multi-agent aggregate workflow.

**Tech Stack:** Git submodules, Markdown, TOML, Python/Bash scans, unittest, pytest, CUDA/PyTorch, repo_sync_guard.

---

### Task 1: Remove unused Git submodules

**Files:**
- Delete: EzPC
- Delete: LLM-Adapters
- Delete: importance-aware-sparse-tuning-IST-paper
- Delete: .gitmodules
- Modify: .dockerignore
- Modify: pyproject.toml
- Test: tests/test_production_surface.py

- [ ] **Step 1: Add submodule absence tests**

The production guard rejects all three gitlink paths, .gitmodules entries, and
active-source references to those names.

- [ ] **Step 2: Run RED**

~~~bash
python -m unittest -v tests.test_production_surface
~~~

Expected: three submodule failures.

- [ ] **Step 3: Remove gitlinks and stale references**

~~~bash
git rm -f EzPC LLM-Adapters importance-aware-sparse-tuning-IST-paper
~~~

Remove all entries from .gitmodules and delete the file when empty. Remove
unused sys.path changes and Ruff/Docker excludes.

- [ ] **Step 4: Verify no submodules remain**

~~~bash
git ls-files -s | awk '$1 == 160000 {print}'
git grep -n -I -E 'EzPC|LLM-Adapters|importance-aware-sparse-tuning-IST-paper' \
  -- ':!server_backups/**' ':!rl_training_data_points/**' \
  ':!Parting Chapter/**' ':!Previous Chapter/**' \
  ':!experiment/outputs/**' ':!experiments/server_command_runs/**'
~~~

Expected: both commands have no output.

- [ ] **Step 5: Run tests and commit**

~~~bash
python -m unittest -v tests.test_production_surface
git add -A
git commit -m "chore: remove unused research submodules"
git push origin codex/task-production-code-cleanup-20260825
~~~

### Task 2: Remove process history and obsolete documentation

**Files:**
- Delete: docs/superpowers/
- Delete: docs/adr/
- Delete: docs/source_integration/
- Delete: docs/evidence/
- Delete: historical agent_handoffs/tasks/*.json
- Delete: historical agent_handoffs/aggregates/*.json
- Delete: README_FOR_GPT55PRO.md
- Delete: RESULT_CREDIBILITY_PAGE.md
- Delete: SERVER_COMMAND.md
- Delete: finish.md
- Delete: project_understanding_blb_stage2_rl.md
- Delete: CHANGELOG.md
- Delete: docs/GLOBALS.md
- Delete: docs/REORG_PLAN.md
- Delete: docs/stage2_policy_network_ablation_v10.md
- Delete: obsolete BLB flow/readme documents superseded by final docs

- [ ] **Step 1: Add process-file absence tests**

The production guard rejects:

~~~python
FORBIDDEN_PROCESS_PATHS = (
    "docs/superpowers/",
    "docs/adr/",
    "docs/source_integration/",
    "docs/evidence/",
    "README_FOR_GPT55PRO.md",
    "RESULT_CREDIBILITY_PAGE.md",
    "SERVER_COMMAND.md",
    "finish.md",
    "project_understanding_blb_stage2_rl.md",
)
~~~

Exclude the cleanup task handoff and final aggregate manifest created after
source verification.

- [ ] **Step 2: Run RED**

~~~bash
python -m unittest -v tests.test_production_surface
~~~

Expected: process-history paths fail.

- [ ] **Step 3: Remove historical records**

Retain only:

~~~text
agent_handoffs/README.md
agent_handoffs/schema.json
agent_handoffs/tasks/.gitkeep
agent_handoffs/aggregates/.gitkeep
docs/GIT_MULTI_AGENT_PROTOCOL.md
docs/ARCHITECTURE.md
docs/SETUP.md
~~~

The cleanup task handoff is added after the final source commit. The final
aggregate manifest is added by the aggregator.

- [ ] **Step 4: Verify preserved result paths were not matched**

~~~bash
git status --short | rg \
  '^(D| M|M |A ).*(server_backups|rl_training_data_points|Parting Chapter|Prelude Chapter|Previous Chapter|Paean/outputs|experiment/outputs|experiments/|gelu_analysis|glue_submission|reports|weight_hist_out|diagnose_certacc_output|glue_final_configs_best_)'
~~~

Expected: no output.

- [ ] **Step 5: Commit**

~~~bash
git add -A
git commit -m "docs: remove historical process records"
git push origin codex/task-production-code-cleanup-20260825
~~~

### Task 3: Sanitize development metadata

**Files:**
- Rewrite: AGENTS.md
- Rewrite: CLAUDE.md
- Rewrite: docs/GIT_MULTI_AGENT_PROTOCOL.md
- Rewrite: agent_handoffs/README.md
- Modify: agent_handoffs/schema.json
- Modify: pyproject.toml
- Delete: .claude/settings.local.json
- Delete: .claude/settings.json
- Delete: .vscode/settings.json
- Delete: .vscode/extensions.json

- [ ] **Step 1: Add personal-information scans**

Scan active source and retained documentation for:

~~~text
personal names
pengjunkai
mmt200088
/Users/
Windows user paths
gpushare.com
root@
literal IPv4 server addresses
wxid_
password
~~~

The word password is allowed only in generic security guidance and tests, never
beside a credential-like value.

- [ ] **Step 2: Run RED**

~~~bash
python scripts/production_surface_guard.py --json
~~~

Expected: active protocol/docs and pyproject report identifiers.

- [ ] **Step 3: Rewrite generic agent guidance**

AGENTS.md and CLAUDE.md contain only project purpose, production feature
matrix, mandatory task/aggregate Git workflow, local-edit/server-run boundary,
scientific-equivalence rule, current train-probe and MPC K invariants, and
links to retained docs.

- [ ] **Step 4: Sanitize package metadata**

Remove the personal authors entry and repository-owner URL from pyproject.toml.
Keep project name, dependencies, Python versions, and generic description.

- [ ] **Step 5: Remove local editor and permission files**

Delete tracked machine-local settings. Do not replace them with
machine-specific configuration.

- [ ] **Step 6: Run scans and commit**

~~~bash
python scripts/production_surface_guard.py --json
git diff --check
git add -A
git commit -m "chore: sanitize project metadata"
git push origin codex/task-production-code-cleanup-20260825
~~~

Expected: no active-source personal-information hit.

### Task 4: Write concise English user documentation

**Files:**
- Rewrite: README.md
- Rewrite: docs/ARCHITECTURE.md
- Rewrite: docs/SETUP.md
- Create: docs/METHOD.md
- Create: six Paean/presets/bert-*-*-final-eval.conf files
- Test: tests/test_readme_commands.py

- [ ] **Step 1: Add README command tests**

Extract fenced Bash commands from README.md. Run project commands with a fake
Python executable or --dry-run. Assert every referenced preset and file exists.

- [ ] **Step 2: Write the README**

Keep README under 220 lines. Include supported profiles, installation,
train-probe protocol, Stage-1 first-run/resume, Stage-2 first-run/resume,
three comparator commands, final evaluation, fixed parameters, outputs, and
graceful stop.

- [ ] **Step 3: Write architecture and method docs**

ARCHITECTURE.md documents retained modules and dependency direction. SETUP.md
uses no specific host or account. METHOD.md records train-probe, action,
reward, candidate-bank, MPC K, Rescale, and final-evaluation semantics.

- [ ] **Step 4: Add six final-eval presets**

Provide BERT-base/BERT-large times MRPC/RTE/SST-2 presets using the current
result schema and complete validation split.

- [ ] **Step 5: Run tests and commit**

~~~bash
python -m unittest -v \
  tests.test_readme_commands \
  tests.test_supported_profile_matrix \
  tests.test_stage2_persistent_launcher
git diff --check
git add README.md docs Paean/presets tests
git commit -m "docs: document the production workflow"
git push origin codex/task-production-code-cleanup-20260825
~~~

### Task 5: Remove non-result AI guidance and assets

**Files:**
- Delete: docs/assets/
- Modify: docs/METHOD.md
- Test: tests/test_production_surface.py

- [ ] **Step 1: Inventory active documents**

~~~bash
git ls-files | rg -i '\.(pdf|doc|docx|ppt|pptx|pages)$'
~~~

Classify each match under a preserved result root. Any document outside those
roots must be required by final README/METHOD or deleted.

- [ ] **Step 2: Remove copied papers and design assets**

Delete non-runtime paper copies, extracted text, design spreadsheets, old
learning-curve illustrations, and method-guidance assets from docs/assets.

- [ ] **Step 3: Verify result PDFs remain unchanged**

Compare the preserved-tree receipt with the pre-cleanup receipt.

Expected: identical tree IDs.

- [ ] **Step 4: Run guard and commit**

~~~bash
python -m unittest -v tests.test_production_surface
git add -A
git commit -m "docs: remove non-runtime guidance assets"
git push origin codex/task-production-code-cleanup-20260825
~~~

### Task 6: Complete static and CPU verification

**Files:**
- Test: tests/
- Validate: all retained source and docs

- [ ] **Step 1: Run final source scans**

~~~bash
python scripts/production_surface_guard.py --json
git grep -n -I -E \
  'pengjunkai|mmt200088|/Users/|gpushare\.com|root@|wxid_' \
  -- ':!server_backups/**' ':!rl_training_data_points/**' \
  ':!Parting Chapter/**' ':!Prelude Chapter/**' \
  ':!Previous Chapter/**' ':!Previous Chapter Server Reserve/**' \
  ':!Paean/outputs/**' ':!experiment/outputs/**' ':!experiments/**' \
  ':!gelu_analysis/**' ':!glue_submission/**' ':!reports/**' \
  ':!Model_analysis/model_statistics/weight_hist_out/**' \
  ':!Rescale_optimizer/diagnose_certacc_output/**' \
  ':!glue_final_configs_best_genetic.json' \
  ':!glue_final_configs_best_ppo.json'
git ls-files | rg -i '(^|/).*(\.bak|legacy_results|ablation)'
~~~

Expected: no output from either grep after a passing guard.

- [ ] **Step 2: Compile retained code**

~~~bash
python -m compileall -q \
  blb_stage2_rl stage1_rl Paean Model_analysis \
  Rescale_optimizer/rescale_optimizer config scripts \
  glue_data_protocol.py rl_tune.py rl_tune_genetic.py \
  layer_importance_evaluator.py final_evaluation_module.py
bash -n llama_7B_LayerImportance.sh Paean/run_final_eval.sh
~~~

Expected: exits zero.

- [ ] **Step 3: Run full server suites**

~~~bash
python -m unittest discover -v
python -m pytest -q
~~~

Expected: all tests pass or have explicit environment-only skips.

- [ ] **Step 4: Verify preserved trees**

Regenerate the six-path receipt and compare with the pre-cleanup receipt.

Expected: byte-identical receipts.

### Task 7: Run real GPU and equivalence acceptance

**Files:**
- Runtime evidence only under experiments/server_command_runs/
- No server source edits

- [ ] **Step 1: Require healthy CUDA**

~~~bash
nvidia-smi
python - <<'PY'
import torch
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 1
print(torch.cuda.get_device_name(0))
PY
~~~

Expected: CUDA is available. A driver/library mismatch blocks canonical
advancement.

- [ ] **Step 2: Run six Profile smokes**

Process the exact 256-example train probe for all supported profiles. Record
identity hash, example count, and real CUDA forward.

- [ ] **Step 3: Run six Stage-1 smokes**

Use fixed seeds. Verify action application, baseline/candidate metrics,
checkpoint save, graceful stop, and resume.

- [ ] **Step 4: Run six Stage-2 smokes**

Run one online candidate and strict search-gate path per profile with real
Rescale and CUDA. Verify fusion/K mapping, A/B/C seed separation, checkpoint
state, and elastic reward devices.

- [ ] **Step 5: Run comparator smokes**

Run minimal Stage-1 and Stage-2 BO-RF, Greedy, and COINN-GA searches on
BERT-base MRPC through shared data/model/materialization.

- [ ] **Step 6: Run final-evaluation canaries**

Evaluate one fixed configuration per profile on complete validation. Confirm
search state remains unchanged.

- [ ] **Step 7: Compare with Phase 1**

Require exact equality for actions, trial order, reward, cost, candidate rank,
checkpoint scientific state, selected config, and final-eval structured
metrics. Keep MPC execution SHA-256 equal to:

~~~text
68a50ef270d894f3995bd01437b6febcb0bd2b3c757b42edb03485ad2ceb63e7
~~~

- [ ] **Step 8: Publish final evidence**

Create a result branch from final source with compact evidence under
experiments/server_command_runs/production_cleanup_final_<short_sha>_20260825/
and pass result-check with remote parity.

### Task 8: Finish the task handoff

**Files:**
- Create: agent_handoffs/tasks/production-code-cleanup-20260825.json

- [ ] **Step 1: Commit and push final source**

Record full source commit/tree and require tracked-clean status.

- [ ] **Step 2: Create a completed handoff-only commit**

Record cloud archive, Phase-1/2/final evidence, changed scopes, preserved tree
hashes, CPU/GPU tests, strict equivalence, aggregate_eligible=true, and
deployment_eligible=false.

- [ ] **Step 3: Validate**

~~~bash
python3 scripts/repo_sync_guard.py agent-finish \
  --handoff agent_handoffs/tasks/production-code-cleanup-20260825.json \
  --remote origin
~~~

Expected: local/remote task parity passes.

### Task 9: Aggregate and synchronize canonical

**Files:**
- Create: agent_handoffs/aggregates/20260825-production-code-cleanup.json

- [ ] **Step 1: Refresh every remote head**

Run aggregate-preflight from a new clean aggregate branch based on remote
canonical.

- [ ] **Step 2: Review every disposition**

Mark the old train-probe task superseded by this cleanup task, the cloud backup
archive_only, and result branches result_only. Reuse prior dispositions only
for exact branch/commit matches. Leave no needs_review.

- [ ] **Step 3: Verify exact aggregate on the server**

Repeat production guard, full CPU tests, GPU evidence validation, README
commands, preserved trees, and scientific parity.

- [ ] **Step 4: Finalize and fast-forward canonical**

~~~bash
RFR_AGGREGATOR_AUTHORIZED=1 \
RFR_AGGREGATE_MANIFEST=agent_handoffs/aggregates/20260825-production-code-cleanup.json \
python3 scripts/repo_sync_guard.py aggregate-finalize \
  --manifest agent_handoffs/aggregates/20260825-production-code-cleanup.json \
  --remote origin --fetch
~~~

After success, push the aggregate branch to jk_standard_rl without force.

- [ ] **Step 5: Synchronize local and server canonical**

Use local-sync --apply and server-check --sync, followed by verify-only checks.

Acceptance requires identical full commit SHA, tree SHA, and tracked-clean
status for local canonical, remote canonical, and server canonical.
