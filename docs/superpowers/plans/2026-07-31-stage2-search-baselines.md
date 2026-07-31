# Stage-2 Search Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add constrained BO-RF, greedy, and COINN-style genetic baselines that search the current per-layer `(Block4 fusion, H/M/L truncation)` space through the same Stage-2 action materialization and model-evaluation path as PPO.

**Architecture:** A torch-free search core owns the discrete action space, six-constraint ranking, and the three optimizers. A runtime adapter evaluates an action matrix through `BLBStage2LayerwiseEnv`, extracts loss/metric means and standard deviations, and persists complete traces. The existing PPO backend remains the default; an explicit Stage-2 search-backend flag selects a baseline.

**Tech Stack:** Python, NumPy, scikit-learn random forests on the server, existing BLB Stage-2 environment and statistical-constraint modules, unittest/pytest.

---

### Task 1: Torch-Free Search Contract

**Files:**
- Create: `blb_stage2_rl/search_baselines.py`
- Create: `tests/test_blb_search_baselines.py`

- [ ] **Step 1: Write failing tests for the action space and six constraints**

Add tests asserting that a two-layer space has dimensions `(2, 3, 2, 3)`, produces only legal one-coordinate neighbors, rejects malformed actions, and treats loss/m1/m2 mean plus all three standard deviations as mandatory constraints.

- [ ] **Step 2: Run the focused tests and confirm the module is missing**

Run: `python3 -m pytest tests/test_blb_search_baselines.py -q`

Expected: collection fails because `blb_stage2_rl.search_baselines` does not exist.

- [ ] **Step 3: Implement immutable actions, observations, normalized margins, and deterministic ranking**

Use the canonical `compute_variable_cost_from_action_matrix()` helper for resource scores. Rank feasible candidates by weighted resource score, balance, safety margins, then a fixed lexicographic tie-break. Rank infeasible candidates by total normalized violation before resource score.

- [ ] **Step 4: Run the focused contract tests**

Run: `python3 -m pytest tests/test_blb_search_baselines.py -q`

Expected: action-space and constraint tests pass.

### Task 2: Three Optimizers

**Files:**
- Modify: `blb_stage2_rl/search_baselines.py`
- Modify: `tests/test_blb_search_baselines.py`

- [ ] **Step 1: Add failing deterministic optimizer tests**

Use a small synthetic evaluator whose unique feasible optimum is known. Assert that greedy, BO-RF, and COINN-style GA never evaluate duplicate actions, obey the evaluation budget, and return the best observed feasible action.

- [ ] **Step 2: Implement greedy search**

Start at all-high precision with Block4 fusion disabled. Enumerate resource-improving one-coordinate neighbors and accept the best feasible improvement; if the starting point is infeasible, first reduce normalized constraint violation.

- [ ] **Step 3: Implement constrained SMAC-style BO-RF**

Fit a multi-output `RandomForestRegressor` to six normalized constraint margins. Generate candidates from random configurations and incumbent neighborhoods, estimate feasibility from individual tree predictions, and maximize an EI-like acquisition combining feasibility probability, resource improvement, and uncertainty. Import scikit-learn lazily so torch-free tests can inject a small surrogate.

- [ ] **Step 4: Implement COINN-style GA**

Use a population of full layerwise configurations, feasibility-aware fitness-proportional parent selection, elite retention, and adjacent mesh mutation without crossover. Default to the COINN stopping rule of five generations without incumbent improvement.

- [ ] **Step 5: Run optimizer tests**

Run: `python3 -m pytest tests/test_blb_search_baselines.py -q`

Expected: all optimizer tests pass.

### Task 3: Production Runtime Adapter

**Files:**
- Create: `blb_stage2_rl/search_baseline_runner.py`
- Create: `tests/test_blb_search_baseline_runner.py`

- [ ] **Step 1: Add failing adapter tests with a recording layerwise environment**

Assert that every matrix row is passed to the existing environment, terminal model-forward and materialization gates are enforced, loss/m1/m2 means and standard deviations are returned, and JSONL persistence contains every evaluated action.

- [ ] **Step 2: Implement the evaluator adapter and persistence**

Reset the canonical layerwise environment with a deterministic seed, set the probe seed, call `env.step()` once per layer, consume the base environment terminal payload, and persist manifest, observations, incumbent history, final summary, full action vector, boosted overrides, and exact per-layer K/fusion descriptions.

- [ ] **Step 3: Add optional strict final certification**

For full comparison runs, route the selected candidates through the existing fixed validation-bank promotion/certification helpers. Allow explicit smoke mode to skip this expensive certification while marking the result non-scientific.

- [ ] **Step 4: Run adapter tests**

Run: `python3 -m pytest tests/test_blb_search_baseline_runner.py -q`

Expected: all adapter tests pass.

### Task 4: Explicit Backend Wiring

**Files:**
- Modify: `blb_stage2_rl/runner.py`
- Modify: `blb_stage2_rl/sequential_runner.py`
- Modify: `layer_importance_evaluator.py`
- Modify: `rl_tune.py`
- Modify: `llama_7B_LayerImportance.sh`
- Create: `tests/test_blb_search_backend_wiring.py`

- [ ] **Step 1: Add failing flag/config propagation tests**

Assert that the accepted values are `ppo`, `bo_rf`, `greedy`, and `coinn_ga`, PPO remains the default, unknown values fail, and the evaluator value reaches `BLBStage2TrainConfig`.

- [ ] **Step 2: Wire explicit search-backend and baseline-search settings**

Add backend, evaluation budget, initial-design/population size, patience, and strict-final flags. Keep `rl_algo=ppo` unchanged so retired GRPO cannot be re-enabled.

- [ ] **Step 3: Dispatch the layerwise branch**

After the common Stage-2 model, robust baseline, fusion maps, and calibrated action context are built, dispatch non-PPO backends to `search_baseline_runner`. Do not initialize or train a policy/value network for those backends.

- [ ] **Step 4: Run wiring and regression tests**

Run: `python3 -m pytest tests/test_blb_search_backend_wiring.py tests/test_blb_search_baselines.py tests/test_blb_search_baseline_runner.py -q`

Expected: all focused tests pass and PPO defaults remain unchanged.

### Task 5: Verification, Git, and Server Smoke

**Files:**
- Create during the run: `experiments/server_command_runs/stage2_search_baselines_smoke_<timestamp>/`

- [ ] **Step 1: Run local static and focused verification**

Run: `python3 -m py_compile blb_stage2_rl/search_baselines.py blb_stage2_rl/search_baseline_runner.py rl_tune.py layer_importance_evaluator.py`

Run: `python3 -m pytest tests/test_blb_search_baselines.py tests/test_blb_search_baseline_runner.py tests/test_blb_search_backend_wiring.py -q`

- [ ] **Step 2: Commit and push the isolated branch**

Commit only the search-baseline implementation, tests, and plan. Push `codex/stage2-search-baselines-20260731`, then fast-forward `jk_standard_rl` only if its remote head has not moved.

- [ ] **Step 3: Synchronize an isolated server source tree**

Fetch the verified commit through Git. If the server cannot fetch, upload a `git archive` of that exact commit, record commit/tree hashes, and leave the active RTE PPO source untouched.

- [ ] **Step 4: Run real-model smoke tests**

Run each backend with a tiny evaluation budget through the current BERT-base MRPC Stage-2 environment on an otherwise idle GPU. Require at least one real model forward, a valid replan/materialization fingerprint, complete loss/m1/m2 mean and standard-deviation fields, exact action/K/fusion output, and a clean process exit. Smoke mode may skip expensive A/B/C final certification, but it must label that omission and must not export a standard scientific-best action.

- [ ] **Step 5: Save smoke evidence and report**

Persist commands, source hashes, stdout/stderr, manifests, observation JSONL, summaries, and a concise comparison table. Commit compact evidence if its size is suitable; otherwise push a manifest and checksums while retaining the server path.
