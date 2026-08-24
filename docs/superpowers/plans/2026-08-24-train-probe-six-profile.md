# Six-Profile Train-Probe Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restrict the project to BERT-base/BERT-large on MRPC, RTE, and SST-2, and make Profile, Stage-1, and Stage-2 share one deterministic 256-example training probe while reserving the complete validation split for post-search final evaluation.

**Architecture:** A torch-free `glue_data_protocol.py` owns the six-profile registry, the pinned GLUE revision, the fixed 256-example identity fixture, protocol hashing, and dataset view construction. `rl_tune.py` creates `train_probe` and `validation_full` once; Stage-1, Stage-2, comparators, and Profile consume `train_probe`, while Paean final evaluation alone consumes `validation_full`. Protocol hashes are persisted in manifests and checkpoints so old validation-probe runs fail closed.

**Tech Stack:** Python 3.10, Hugging Face `datasets` and `transformers`, NumPy, scikit-learn, PyTorch, Bash launcher, unittest/pytest, Git worktrees, `scripts/repo_sync_guard.py`.

---

## File Structure

### New Files

- `glue_data_protocol.py`: supported-profile registry, deterministic probe selection, fixture loading, identity hashing, dataset view validation, protocol payload generation.
- `fixtures/reproducibility/glue_train_probe_v1.json`: MRPC/RTE/SST-2 ordered train-probe IDs and label histograms at the pinned GLUE revision.
- `scripts/build_glue_train_probe_fixture.py`: server-only fixture generator using the same selector as runtime.
- `tests/test_glue_data_protocol.py`: selector, fixture, profile-matrix, identity, and error tests.
- `tests/test_search_split_isolation.py`: Stage-1/Stage-2 search cannot access validation; final evaluation must use validation.

### Modified Runtime Files

- `rl_tune.py`: restrict GLUE loading/tokenization to three tasks, build protocol views, pass protocol context to the evaluator.
- `layer_importance_evaluator.py`: register `train_probe`, route Stage-1 search to it, persist protocol provenance, preserve validation-only final evaluation.
- `blb_stage2_rl/runner.py`: build Stage-2 probe and search-validation batches from `train_probe` without resampling.
- `blb_stage2_rl/sequential_runner.py`: bind calibration, promotion banks, strict top-5, evidence tiers, and checkpoint identity to `train_probe`; bump schemas.
- `blb_stage2_rl/eval_metrics.py`: retain accuracy and weighted-F1 only; remove STS-B and CoLA metric paths.
- `stage1_rl/search_runner.py`: persist and validate `train_probe` provenance.
- `llama_7B_LayerImportance.sh`: expose only MRPC/RTE/SST-2 and BERT-base/BERT-large.
- `Paean/config.py`, `Paean/run_final_eval.py`: restrict public task/model choices while keeping validation-full final evaluation.
- `general_policy_module.py`, `rl_tune_general.py`: restrict task validation and tokenization to the three tasks.
- `generate_glue_submission.py`: keep only the three BERT task registrations.
- `Model_analysis/analyze_all_distribution_new.py`: keep six BERT profile entries and consume the shared probe fixture.
- `Model_analysis/analyze_all_distribution_approx.py`: consume the same shared profile probe.
- `Model_analysis/analyze_gelu_distribution.py`: remove unsupported task/model registrations and consume the shared probe.
- `glue_configs.json`: remove unsupported datasets and GPT-2 model sections.
- Active tests that assert old validation-probe or unsupported-task behavior.

### Deleted Active Configuration

- `Rescale_optimizer/configs/wnli/`
- Unsupported task entries in `Model_analysis/configs/approx_per_dataset.json`
- Presets or fixtures whose only supported target is outside the six-profile matrix.
- MRPC search-probe fields in `fixtures/reproducibility/mrpc_validation_v1.json`; validation-order data remains only if final-evaluation compatibility still requires it.

Historical result directories are not deleted in this implementation phase.

---

## Task 1: Shared Supported-Profile Registry

**Files:**
- Create: `glue_data_protocol.py`
- Create: `tests/test_glue_data_protocol.py`

- [ ] **Step 1: Write failing registry tests**

Add tests that require exactly six profiles and reject every other public task/model pair:

```python
from glue_data_protocol import (
    SUPPORTED_DATASETS,
    SUPPORTED_MODEL_FAMILIES,
    supported_profiles,
    validate_supported_profile,
)


def test_supported_matrix_contains_only_six_bert_profiles():
    assert SUPPORTED_DATASETS == ("mrpc", "rte", "sst2")
    assert SUPPORTED_MODEL_FAMILIES == ("bert-base", "bert-large")
    assert supported_profiles() == (
        ("bert-base", "mrpc"),
        ("bert-base", "rte"),
        ("bert-base", "sst2"),
        ("bert-large", "mrpc"),
        ("bert-large", "rte"),
        ("bert-large", "sst2"),
    )


def test_unsupported_profile_fails_closed():
    for model, dataset in (
        ("gpt-2", "mrpc"),
        ("bert-base", "stsb"),
        ("bert-large", "qnli"),
        ("bert-base", "mnli"),
    ):
        with pytest.raises(ValueError, match="unsupported profile"):
            validate_supported_profile(model, dataset)
```

- [ ] **Step 2: Commit and push the RED test**

```bash
git add tests/test_glue_data_protocol.py
git commit -m "test: define six supported BERT profiles"
git push
```

- [ ] **Step 3: Run RED on the server**

```bash
python -m pytest tests/test_glue_data_protocol.py -q
```

Expected: import failure because `glue_data_protocol.py` does not exist.

- [ ] **Step 4: Implement the registry**

Create the module with this public surface:

```python
from __future__ import annotations

from dataclasses import dataclass


PROTOCOL_SCHEMA = "glue_train_probe_protocol_v1"
GLUE_DATASET_REPO = "nyu-mll/glue"
GLUE_DATASET_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
TRAIN_PROBE_SPLIT = "train_probe"
TRAIN_PROBE_SOURCE_SPLIT = "train"
FINAL_EVAL_SPLIT = "validation_full"
TRAIN_PROBE_SIZE = 256
TRAIN_PROBE_SEED = 42

SUPPORTED_DATASETS = ("mrpc", "rte", "sst2")
SUPPORTED_MODEL_FAMILIES = ("bert-base", "bert-large")


@dataclass(frozen=True)
class TaskSpec:
    input_columns: tuple[str, ...]
    metric_names: tuple[str, str] = ("accuracy", "weighted_f1")


TASK_SPECS = {
    "mrpc": TaskSpec(("sentence1", "sentence2")),
    "rte": TaskSpec(("sentence1", "sentence2")),
    "sst2": TaskSpec(("sentence",)),
}


def supported_profiles() -> tuple[tuple[str, str], ...]:
    return tuple(
        (model, dataset)
        for model in SUPPORTED_MODEL_FAMILIES
        for dataset in SUPPORTED_DATASETS
    )


def validate_supported_profile(model_family: str, dataset: str) -> None:
    profile = (str(model_family).strip().lower(), str(dataset).strip().lower())
    if profile not in supported_profiles():
        raise ValueError(f"unsupported profile: {profile[0]}/{profile[1]}")
```

- [ ] **Step 5: Run GREEN on the server**

```bash
python -m pytest tests/test_glue_data_protocol.py -q
```

Expected: registry tests pass.

- [ ] **Step 6: Commit and push**

```bash
git add glue_data_protocol.py tests/test_glue_data_protocol.py
git commit -m "feat: centralize six-profile registry"
git push
```

---

## Task 2: Deterministic 256-Example Probe Selector

**Files:**
- Modify: `glue_data_protocol.py`
- Modify: `tests/test_glue_data_protocol.py`

- [ ] **Step 1: Write failing selector tests**

Use a small fake Hugging Face-style dataset with `.select()` and stable `idx`
fields. Require determinism, uniqueness, sorted physical order, stratification,
and identical base/large identities:

```python
def test_train_probe_is_deterministic_stratified_and_ordered():
    dataset = fake_binary_dataset(size=600, zero_count=240)
    first, first_identity = build_train_probe(dataset, dataset="mrpc")
    second, second_identity = build_train_probe(dataset, dataset="mrpc")
    assert len(first) == 256
    assert first_identity == second_identity
    assert list(first_identity.positions) == sorted(first_identity.positions)
    assert len(set(first_identity.positions)) == 256
    assert first_identity.label_histogram == {0: 102, 1: 154}


def test_model_family_does_not_change_probe_identity():
    dataset = fake_binary_dataset(size=600, zero_count=240)
    _, base = build_train_probe(dataset, dataset="rte")
    _, large = build_train_probe(dataset, dataset="rte")
    assert base.ordered_identity_hash == large.ordered_identity_hash
```

Also test fewer than 256 rows, missing labels, duplicate `idx`, and a single
label all fail rather than falling back to random sampling.

- [ ] **Step 2: Commit/push RED and run it on the server**

```bash
git add tests/test_glue_data_protocol.py
git commit -m "test: require deterministic train probes"
git push
python -m pytest tests/test_glue_data_protocol.py -q
```

Expected: failure because selector APIs are absent.

- [ ] **Step 3: Implement the selector and protocol payload**

Add immutable identity and view types and use the existing stratified method:

```python
@dataclass(frozen=True)
class TrainProbeIdentity:
    dataset: str
    positions: tuple[int, ...]
    raw_ids: tuple[int, ...]
    label_histogram: tuple[tuple[int, int], ...]
    ordered_identity_hash: str


@dataclass(frozen=True)
class GlueProtocolViews:
    train_full: object
    train_probe: object
    validation_full: object
    identity: TrainProbeIdentity


def build_train_probe(raw_train, *, dataset: str):
    validate_dataset(dataset)
    if len(raw_train) < TRAIN_PROBE_SIZE:
        raise GlueDataProtocolError("training split has fewer than 256 rows")
    labels = [int(value) for value in raw_train["label"]]
    if set(labels) != {0, 1}:
        raise GlueDataProtocolError("formal training labels must contain 0 and 1")
    shuffled = raw_train.shuffle(seed=TRAIN_PROBE_SEED)
    positions = np.arange(len(shuffled))
    selected, _ = train_test_split(
        positions,
        train_size=TRAIN_PROBE_SIZE,
        random_state=TRAIN_PROBE_SEED,
        shuffle=True,
        stratify=np.asarray(shuffled["label"], dtype=int),
    )
    selected = tuple(sorted(int(value) for value in selected))
    probe = shuffled.select(list(selected))
    raw_ids = tuple(int(value) for value in probe["idx"])
    identity_payload = {
        "schema": PROTOCOL_SCHEMA,
        "dataset": dataset,
        "positions": selected,
        "raw_ids": raw_ids,
        "labels": tuple(int(value) for value in probe["label"]),
    }
    identity = TrainProbeIdentity(
        dataset=dataset,
        positions=selected,
        raw_ids=raw_ids,
        label_histogram=tuple(sorted(Counter(identity_payload["labels"]).items())),
        ordered_identity_hash=stable_json_hash(identity_payload),
    )
    return probe, identity
```

Use `json_utils.stable_json_hash`; do not add a local JSON hashing helper.

- [ ] **Step 4: Run GREEN and commit**

```bash
python -m pytest tests/test_glue_data_protocol.py -q
git add glue_data_protocol.py tests/test_glue_data_protocol.py
git commit -m "feat: build deterministic stratified train probes"
git push
```

---

## Task 3: Generate and Validate the Fixed Probe Fixture

**Files:**
- Create: `scripts/build_glue_train_probe_fixture.py`
- Create: `fixtures/reproducibility/glue_train_probe_v1.json`
- Modify: `glue_data_protocol.py`
- Modify: `tests/test_glue_data_protocol.py`

- [ ] **Step 1: Add failing fixture round-trip tests**

Require the fixture to contain exactly MRPC/RTE/SST-2, the pinned dataset
revision, 256 IDs per task, label histograms, and selector hashes. Tampered IDs,
wrong revisions, duplicates, or an extra task must fail.

- [ ] **Step 2: Commit/push RED and verify failure on the server**

```bash
git add tests/test_glue_data_protocol.py
git commit -m "test: specify fixed GLUE probe fixture"
git push
python -m pytest tests/test_glue_data_protocol.py -q
```

- [ ] **Step 3: Implement fixture serialization and loading**

The fixture schema is:

```json
{
  "schema_version": "glue_train_probe_protocol_v1",
  "dataset_repo": "nyu-mll/glue",
  "dataset_revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
  "probe_size": 256,
  "probe_seed": 42,
  "tasks": {
    "mrpc": {"raw_ids": [], "label_histogram": {}, "ordered_identity_hash": ""},
    "rte": {"raw_ids": [], "label_histogram": {}, "ordered_identity_hash": ""},
    "sst2": {"raw_ids": [], "label_histogram": {}, "ordered_identity_hash": ""}
  }
}
```

The builder imports `build_train_probe`, loads every task at the pinned
revision, writes through `json_utils.write_json_file`, then reads it back with
the runtime loader before returning success.

- [ ] **Step 4: Run the fixture builder on the server**

```bash
python scripts/build_glue_train_probe_fixture.py \
  --output fixtures/reproducibility/glue_train_probe_v1.json
```

Expected: three tasks, 256 rows each, no duplicate IDs, schema validation pass.

- [ ] **Step 5: Commit the generated fixture and implementation**

```bash
git add glue_data_protocol.py scripts/build_glue_train_probe_fixture.py \
  fixtures/reproducibility/glue_train_probe_v1.json \
  tests/test_glue_data_protocol.py
git commit -m "feat: freeze three GLUE training probes"
git push
```

---

## Task 4: Load Only Three GLUE Tasks and Build Protocol Views

**Files:**
- Modify: `rl_tune.py`
- Modify: `tests/test_glue_dataset_loading.py`
- Modify: `tests/test_glue_data_protocol.py`

- [ ] **Step 1: Write failing loader tests**

Tests require:

- `GLUE_PARQUET_SPLITS` and `GLUE_REQUIRED_COLUMNS` contain only three tasks;
- primary and equivalent routes use `GLUE_DATASET_REVISION`;
- `train_probe` is selected before tokenization and validation remains full;
- model family does not affect IDs;
- unsupported tasks fail before `load_dataset` is called.

- [ ] **Step 2: Commit/push RED and run on the server**

```bash
git add tests/test_glue_dataset_loading.py tests/test_glue_data_protocol.py
git commit -m "test: require train-probe dataset loading"
git push
python -m pytest tests/test_glue_dataset_loading.py \
  tests/test_glue_data_protocol.py -q
```

- [ ] **Step 3: Replace the dataset dispatch**

In `rl_tune.py`:

```python
GLUE_PARQUET_SPLITS = {
    "mrpc": ("train", "validation", "test"),
    "rte": ("train", "validation", "test"),
    "sst2": ("train", "validation", "test"),
}

GLUE_REQUIRED_COLUMNS = {
    "mrpc": ("sentence1", "sentence2", "label", "idx"),
    "rte": ("sentence1", "sentence2", "label", "idx"),
    "sst2": ("sentence", "label", "idx"),
}
```

Call `validate_supported_profile` before model/dataset loading. Load the pinned
revision, build raw protocol views, tokenize `train`, `train_probe`, and
`validation_full` independently, and pass `GlueDataProtocolContext` into
`LayerImportanceEvaluator`.

Delete MNLI matched/mismatched, STS-B regression, QNLI, CoLA, WNLI, QQP, GPT-2,
Llama tokenizer, causal-LM, and unsupported tokenization branches from this
entrypoint.

- [ ] **Step 4: Run GREEN and commit**

```bash
python -m pytest tests/test_glue_dataset_loading.py \
  tests/test_glue_data_protocol.py -q
git add rl_tune.py tests/test_glue_dataset_loading.py \
  tests/test_glue_data_protocol.py
git commit -m "feat: load six supported train-probe profiles"
git push
```

---

## Task 5: Register the Probe and Persist Protocol Identity

**Files:**
- Modify: `layer_importance_evaluator.py`
- Modify: `rl_data_points.py`
- Create: `tests/test_search_split_isolation.py`
- Modify: `tests/test_stage1_eval_accel.py`

- [ ] **Step 1: Write failing evaluator tests**

Construct an evaluator seam with separate sentinel datasets. Assert:

```python
assert evaluator.dataset_splits["train_probe"] is protocol.train_probe
assert evaluator.dataset_splits["validation_full"] is protocol.validation_full
assert evaluator.get_reward_reference_split_name() == "train_probe"
assert evaluator.get_online_reward_split_name() == "train_probe"
```

Require `dataset_protocol.json` to exist before any candidate evaluation and to
round-trip through `json_utils.read_json_file`.

- [ ] **Step 2: Commit/push RED and run it on the server**

```bash
git add tests/test_search_split_isolation.py tests/test_stage1_eval_accel.py
git commit -m "test: isolate search data from validation"
git push
python -m pytest tests/test_search_split_isolation.py \
  tests/test_stage1_eval_accel.py -q
```

- [ ] **Step 3: Implement split registration**

Replace `USE_VALIDATION_FOR_REWARD` and validation-proxy compatibility with a
single explicit contract:

```python
def _prepare_rl_datasets(self, train_data, train_probe, validation_data):
    self._register_dataset_split("train", train_data)
    self._register_dataset_split("train_probe", train_probe)
    self._register_dataset_split("validation_full", validation_data)


def get_reward_reference_split_name(self):
    return "train_probe"


def get_online_reward_split_name(self):
    return "train_probe"
```

Write the protocol payload atomically at evaluator initialization and expose
`dataset_protocol_hash` for Stage-1/Stage-2 identity contexts. Remove
validation proxy and train-anchor branches that are no longer reachable.

- [ ] **Step 4: Run GREEN and commit**

```bash
python -m pytest tests/test_search_split_isolation.py \
  tests/test_stage1_eval_accel.py -q
git add layer_importance_evaluator.py rl_data_points.py \
  tests/test_search_split_isolation.py tests/test_stage1_eval_accel.py
git commit -m "feat: register and persist train-probe identity"
git push
```

---

## Task 6: Move All Stage-1 Search Evaluation to `train_probe`

**Files:**
- Modify: `layer_importance_evaluator.py`
- Modify: `stage1_rl/search_runner.py`
- Modify: `stage1_rl/search_baselines.py`
- Modify: `tests/test_search_split_isolation.py`
- Modify: `tests/test_stage1_search_baselines.py`

- [ ] **Step 1: Write failing Stage-1 isolation tests**

Use a validation sentinel that raises on access. Exercise PPO and each
comparator for one candidate. Require baseline, constraints, reward, candidate
metrics, and manifests to state `train_probe`. Then invoke final evaluation and
require the complete validation sentinel to be consumed.

- [ ] **Step 2: Commit/push RED and run it on the server**

```bash
git add tests/test_search_split_isolation.py tests/test_stage1_search_baselines.py
git commit -m "test: bind Stage-1 search to train probe"
git push
python -m pytest tests/test_search_split_isolation.py \
  tests/test_stage1_search_baselines.py -q
```

- [ ] **Step 3: Update Stage-1 baseline and search contracts**

Replace hard-coded `validation_full` search metadata with
`TRAIN_PROBE_SPLIT`. The Stage-1 baseline and tolerance thresholds use the
train probe. Include `dataset_protocol_hash` in invocation, checkpoint,
candidate identity, result, and two-stage binding.

Do not change GELU actions, Softmax degree, costs, candidate ranking, seeds, or
optimizer behavior.

- [ ] **Step 4: Run GREEN and commit**

```bash
python -m pytest tests/test_search_split_isolation.py \
  tests/test_stage1_search_baselines.py -q
git add layer_importance_evaluator.py stage1_rl/search_runner.py \
  stage1_rl/search_baselines.py tests/test_search_split_isolation.py \
  tests/test_stage1_search_baselines.py
git commit -m "feat: evaluate Stage-1 search on train probe"
git push
```

---

## Task 7: Move Stage-2 Calibration and Search Gates to `train_probe`

**Files:**
- Modify: `blb_stage2_rl/runner.py`
- Modify: `blb_stage2_rl/sequential_runner.py`
- Modify: `blb_stage2_rl/layerwise_runner.py`
- Modify: `blb_stage2_rl/search_baseline_runner.py`
- Modify: `tests/test_search_split_isolation.py`
- Modify: `tests/test_blb_layerwise_runner.py`
- Modify: `tests/test_blb_search_baseline_runner.py`

- [ ] **Step 1: Write failing Stage-2 isolation tests**

Require `_build_probe_batches` and the search-validation environment to use the
same `train_probe` object. Validate all baseline groups and Bank A/B/C receive
identical sample IDs in identical order. Make any search-time access to
`validation_full` raise.

- [ ] **Step 2: Commit/push RED and run it on the server**

```bash
git add tests/test_search_split_isolation.py tests/test_blb_layerwise_runner.py \
  tests/test_blb_search_baseline_runner.py
git commit -m "test: bind Stage-2 gates to train probe"
git push
python -m pytest tests/test_search_split_isolation.py \
  tests/test_blb_layerwise_runner.py \
  tests/test_blb_search_baseline_runner.py -q
```

- [ ] **Step 3: Implement one shared batch set**

`BLBStage2Runner._build_probe_batches` reads `dataset_splits["train_probe"]`
directly and never calls `_get_stability_probe`. Replace
`_build_authoritative_validation_env` with a search-gate environment built from
the same frozen batches. Register one batch set with the shared probe runner and
use it for online, calibration, promotion, and strict evaluation.

Version evidence names and schemas:

```python
SEARCH_EVIDENCE_SPLIT = "train_probe"
DATASET_PROTOCOL_SCHEMA = "glue_train_probe_protocol_v1"
LAYERWISE_RUN_SCHEMA = "stage2_layerwise_train_probe_run_v1"
```

Remove `F4_validation_full` and `validation_full_stratified_probe` from search
artifacts. Preserve validation-only final evaluation APIs.

- [ ] **Step 4: Run GREEN and commit**

```bash
python -m pytest tests/test_search_split_isolation.py \
  tests/test_blb_layerwise_runner.py \
  tests/test_blb_search_baseline_runner.py -q
git add blb_stage2_rl/runner.py blb_stage2_rl/sequential_runner.py \
  blb_stage2_rl/layerwise_runner.py \
  blb_stage2_rl/search_baseline_runner.py \
  tests/test_search_split_isolation.py tests/test_blb_layerwise_runner.py \
  tests/test_blb_search_baseline_runner.py
git commit -m "feat: evaluate Stage-2 search on train probe"
git push
```

---

## Task 8: Preserve Validation-Full Final Evaluation

**Files:**
- Modify: `final_evaluation_module.py`
- Modify: `Paean/embedded.py`
- Modify: `Paean/blb_action_eval.py`
- Modify: `tests/test_search_split_isolation.py`
- Modify: `tests/test_final_evaluation_config_cache.py`
- Modify: `tests/test_final_eval_normalize_arrays.py`
- Modify: `tests/test_blb_paean_handoff_ordinary.py`
- Modify: `tests/test_blb_two_stage_binding_ordinary.py`

- [ ] **Step 1: Write failing final-evaluation tests**

Require final evaluation to consume every validation example exactly once per
repeat, reject `train_probe` as a final source, and preserve existing batch and
loss aggregation behavior.

- [ ] **Step 2: Commit/push RED and verify it fails**

```bash
git add tests/test_search_split_isolation.py \
  tests/test_final_evaluation_config_cache.py \
  tests/test_final_eval_normalize_arrays.py \
  tests/test_blb_paean_handoff_ordinary.py \
  tests/test_blb_two_stage_binding_ordinary.py
git commit -m "test: reserve validation for final evaluation"
git push
python -m pytest tests/test_search_split_isolation.py \
  tests/test_final_evaluation_config_cache.py \
  tests/test_final_eval_normalize_arrays.py \
  tests/test_blb_paean_handoff_ordinary.py \
  tests/test_blb_two_stage_binding_ordinary.py -q
```

- [ ] **Step 3: Make final split explicit**

Use `FINAL_EVAL_SPLIT` in final-evaluation entrypoints. Require the persisted
search result and protocol hash, but do not feed validation metrics back into
candidate ranking or checkpoint state.

- [ ] **Step 4: Run GREEN and commit**

```bash
python -m pytest tests/test_search_split_isolation.py \
  tests/test_final_evaluation_config_cache.py \
  tests/test_final_eval_normalize_arrays.py \
  tests/test_blb_paean_handoff_ordinary.py \
  tests/test_blb_two_stage_binding_ordinary.py -q
git add final_evaluation_module.py Paean/embedded.py Paean/blb_action_eval.py \
  tests/test_search_split_isolation.py \
  tests/test_final_evaluation_config_cache.py \
  tests/test_final_eval_normalize_arrays.py \
  tests/test_blb_paean_handoff_ordinary.py \
  tests/test_blb_two_stage_binding_ordinary.py
git commit -m "feat: reserve validation for final plaintext evaluation"
git push
```

---

## Task 9: Align Cleartext Profile With the Shared Probe

**Files:**
- Modify: `Model_analysis/analyze_all_distribution_new.py`
- Modify: `Model_analysis/analyze_all_distribution_approx.py`
- Modify: `Model_analysis/analyze_gelu_distribution.py`
- Create: `tests/test_profile_train_probe_protocol.py`

- [ ] **Step 1: Write failing Profile identity tests**

Patch model loading and capture DataLoader examples. Require all three profile
programs to load the fixture IDs from `train`, process exactly 256 rows, and
produce the same ordered identity hash as Stage-1/Stage-2.

- [ ] **Step 2: Commit/push RED and run it on the server**

```bash
git add tests/test_profile_train_probe_protocol.py
git commit -m "test: align Profile with the train probe"
git push
python -m pytest tests/test_profile_train_probe_protocol.py -q
```

- [ ] **Step 3: Replace Profile sampling**

Reduce `TASK_REGISTRY` to six BERT entries. `_prepare_bert_data` loads the
pinned GLUE revision, selects rows by the shared fixture before tokenization,
and writes the identity payload beside profile output. Remove GPT-2 model,
WikiText, causal-LM, and unsupported BERT task branches.

- [ ] **Step 4: Run GREEN and commit**

```bash
python -m pytest tests/test_profile_train_probe_protocol.py -q
git add Model_analysis/analyze_all_distribution_new.py \
  Model_analysis/analyze_all_distribution_approx.py \
  Model_analysis/analyze_gelu_distribution.py \
  tests/test_profile_train_probe_protocol.py
git commit -m "feat: profile the shared train probe"
git push
```

---

## Task 10: Restrict Launcher, Paean, General RL, and Submission Entrypoints

**Files:**
- Modify: `llama_7B_LayerImportance.sh`
- Modify: `Paean/config.py`
- Modify: `Paean/run_final_eval.py`
- Modify: `general_policy_module.py`
- Modify: `rl_tune_general.py`
- Modify: `generate_glue_submission.py`
- Create: `tests/test_supported_profile_matrix.py`
- Modify: `tests/test_stage2_persistent_launcher.py`
- Modify: `tests/test_run_layout.py`

- [ ] **Step 1: Write failing public-entrypoint tests**

Run each parser with all six supported profiles and representative unsupported
values. Supported commands must preserve their existing effective parameters;
unsupported values must exit before Python/model launch.

- [ ] **Step 2: Commit/push RED and run on the server**

```bash
git add tests/test_supported_profile_matrix.py \
  tests/test_stage2_persistent_launcher.py tests/test_run_layout.py
git commit -m "test: restrict public profile matrix"
git push
python -m pytest tests/test_supported_profile_matrix.py \
  tests/test_stage2_persistent_launcher.py tests/test_run_layout.py -q
```

- [ ] **Step 3: Remove unsupported parser and registry branches**

All entrypoints import or mirror the shared six-profile registry. Delete MNLI
mismatch options, unsupported tokenizers, GPT-2 model switches, regression
flags, and task names from help text and error messages. General RL may combine
only MRPC/RTE/SST-2 and still uses the same model family across one run.

- [ ] **Step 4: Run GREEN and commit**

```bash
python -m pytest tests/test_supported_profile_matrix.py \
  tests/test_stage2_persistent_launcher.py tests/test_run_layout.py -q
bash -n llama_7B_LayerImportance.sh Paean/run_final_eval.sh
git add llama_7B_LayerImportance.sh Paean/config.py Paean/run_final_eval.py \
  general_policy_module.py rl_tune_general.py generate_glue_submission.py \
  tests/test_supported_profile_matrix.py \
  tests/test_stage2_persistent_launcher.py tests/test_run_layout.py
git commit -m "refactor: expose only six supported profiles"
git push
```

---

## Task 11: Remove Unsupported Metrics and Dedicated Configuration

**Files:**
- Modify: `blb_stage2_rl/eval_metrics.py`
- Modify: `glue_configs.json`
- Modify: `Model_analysis/configs/approx_per_dataset.json`
- Delete: `Rescale_optimizer/configs/wnli/`
- Modify: `tests/test_blb_inference_eval_shared.py`
- Modify: `tests/test_glue_dataset_loading.py`

- [ ] **Step 1: Write failing inventory and metric tests**

Require accuracy and weighted-F1 for all three supported tasks. Assert active
runtime/config paths contain no unsupported dataset keys or GPT-2 model entries.

- [ ] **Step 2: Commit/push RED and verify failure**

```bash
git add tests/test_supported_profile_matrix.py \
  tests/test_blb_inference_eval_shared.py tests/test_glue_dataset_loading.py
git commit -m "test: remove unsupported metrics and configs"
git push
python -m pytest tests/test_supported_profile_matrix.py \
  tests/test_blb_inference_eval_shared.py tests/test_glue_dataset_loading.py -q
```

- [ ] **Step 3: Delete unsupported metric and config paths**

`metric_pair_for_dataset` validates the task then returns accuracy and
weighted-F1. Remove Pearson, Spearman, Matthews, regression-logit handling, and
all unsupported-only configuration files. Do not delete shared optimizer or
action code used by the six profiles.

- [ ] **Step 4: Run GREEN and commit**

```bash
python -m pytest tests/test_supported_profile_matrix.py \
  tests/test_blb_inference_eval_shared.py tests/test_glue_dataset_loading.py -q
git add -A blb_stage2_rl/eval_metrics.py glue_configs.json \
  Model_analysis/configs Rescale_optimizer/configs presets Paean/presets tests
git commit -m "refactor: remove unsupported task configuration"
git push
```

---

## Task 12: Reject Old Resume State and Bind New Provenance

**Files:**
- Modify: `blb_stage2_rl/persistence.py`
- Modify: `blb_stage2_rl/sequential_runner.py`
- Modify: `stage1_rl/search_runner.py`
- Modify: `layer_importance_evaluator.py`
- Modify: `tests/test_blb_layerwise_runner.py`
- Modify: `tests/test_blb_search_baseline_runner.py`
- Modify: `tests/test_blb_search_baseline_runner_ordinary.py`
- Modify: `tests/test_blb_two_stage_binding_ordinary.py`
- Modify: `tests/test_stage2_ga_extension_preflight.py`

- [ ] **Step 1: Write failing resume tests**

Require missing, old, or mismatched protocol schema/hash to fail before replay
or model inference. Matching protocols must preserve ordered replay and RNG.

- [ ] **Step 2: Commit/push RED and verify failure**

```bash
git add tests/test_blb_layerwise_runner.py \
  tests/test_blb_search_baseline_runner.py \
  tests/test_blb_search_baseline_runner_ordinary.py \
  tests/test_blb_two_stage_binding_ordinary.py \
  tests/test_stage2_ga_extension_preflight.py
git commit -m "test: reject validation-probe checkpoints"
git push
python -m pytest tests/test_blb_layerwise_runner.py \
  tests/test_blb_search_baseline_runner.py \
  tests/test_blb_search_baseline_runner_ordinary.py \
  tests/test_blb_two_stage_binding_ordinary.py \
  tests/test_stage2_ga_extension_preflight.py -q
```

- [ ] **Step 3: Version and validate persisted state**

Add `dataset_protocol_schema` and `dataset_protocol_hash` to every Stage-1 and
Stage-2 checkpoint/invocation contract. Validation compares exact strings and
raises a message directing users to start a fresh run under the train-probe
protocol.

- [ ] **Step 4: Run GREEN and commit**

```bash
python -m pytest tests/test_blb_layerwise_runner.py \
  tests/test_blb_search_baseline_runner.py \
  tests/test_blb_search_baseline_runner_ordinary.py \
  tests/test_blb_two_stage_binding_ordinary.py \
  tests/test_stage2_ga_extension_preflight.py -q
git add blb_stage2_rl/persistence.py blb_stage2_rl/sequential_runner.py \
  stage1_rl/search_runner.py layer_importance_evaluator.py tests
git commit -m "feat: bind checkpoints to train-probe identity"
git push
```

---

## Task 13: Comments, Schemas, and Focused Full-Flow Verification

**Files:**
- Modify: comments/docstrings in files touched by Tasks 1-12.
- Modify: `AGENTS.md` with the final six-profile train-probe contract.
- Modify: affected tests.

- [ ] **Step 1: Remove stale prompt-like and old split language**

Within touched runtime files, replace references to user requests, PDFs,
validation-guided search, validation proxies, or unsupported tasks with concise
technical comments. Do not perform repository-wide publication cleanup yet.

- [ ] **Step 2: Run static source scans**

```bash
rg -n "validation_full_stratified_probe|F4_validation_full|Validation Guided|PDF优化|用户要求|gpt-?2|stsb|cola|qnli|wnli|mnli" \
  glue_data_protocol.py rl_tune.py layer_importance_evaluator.py \
  blb_stage2_rl stage1_rl Paean Model_analysis general_policy_module.py \
  rl_tune_general.py generate_glue_submission.py llama_7B_LayerImportance.sh
```

Expected: no stale search-split labels, prompt-like comments, or unsupported
active task/model branches. Technical historical migration error messages may
name the old protocol only where required to reject it.

- [ ] **Step 3: Run the complete affected server suite**

```bash
python -m unittest -v \
  tests.test_glue_data_protocol \
  tests.test_glue_dataset_loading \
  tests.test_search_split_isolation \
  tests.test_profile_train_probe_protocol \
  tests.test_supported_profile_matrix \
  tests.test_stage1_search_baselines \
  tests.test_blb_search_baselines \
  tests.test_blb_search_baseline_runner \
  tests.test_blb_layerwise_runner \
  tests.test_stage2_persistent_launcher
python -m py_compile glue_data_protocol.py rl_tune.py \
  layer_importance_evaluator.py blb_stage2_rl/*.py stage1_rl/*.py \
  Paean/*.py Model_analysis/*.py
bash -n llama_7B_LayerImportance.sh Paean/run_final_eval.sh
git diff --check
```

- [ ] **Step 4: Run the complete project test suite on the server**

```bash
python -m unittest discover -v
python -m pytest -q
```

Expected: no unsupported-task tests remain, and every retained test passes or
has an explicit environment-only skip.

- [ ] **Step 5: Commit and push**

```bash
git add AGENTS.md glue_data_protocol.py rl_tune.py \
  layer_importance_evaluator.py blb_stage2_rl stage1_rl Paean Model_analysis \
  general_policy_module.py rl_tune_general.py generate_glue_submission.py \
  llama_7B_LayerImportance.sh tests
git commit -m "docs: align comments with train-probe protocol"
git push
```

---

## Task 14: Six Real-GPU Profile and Search Smokes

**Files:**
- Runtime evidence only under `/hy-tmp`; no server source edits.
- Handoff verification entries reference exact evidence paths.

- [ ] **Step 1: Verify aggregate candidate source parity before experiments**

```bash
python scripts/repo_sync_guard.py server-check \
  --expected-commit "$SOURCE_COMMIT" \
  --expected-tree "$SOURCE_TREE" \
  --remote origin --canonical jk_standard_rl
```

For task-branch verification, use an isolated Git checkout at the exact task
source commit; do not update the canonical server checkout yet.

- [ ] **Step 2: Generate and compare dataset identities**

For MRPC, RTE, and SST-2, run the fixture verifier twice and assert identical
256 IDs/hashes. Confirm base/large profile invocations report the same hash.

- [ ] **Step 3: Run six minimal Profile smokes**

For each `(bert-base|bert-large) x (mrpc|rte|sst2)`, run one formal Profile
forward over the fixed 256 examples. Require real CUDA forward, exact example
count, and the expected protocol hash.

- [ ] **Step 4: Run six minimal Stage-1 smokes**

Use one-candidate comparator/PPO smoke per profile. Require baseline and
candidate evidence to state `train_probe`, with no validation access.

- [ ] **Step 5: Run six minimal Stage-2 smokes**

Use the canonical layerwise action path, one online candidate, real
Rescale_optimizer, and real CUDA forward. Require baseline, online, and strict
smoke evidence to use the same probe hash and preserve fusion/K mapping.

- [ ] **Step 6: Run six validation-full final-evaluation canaries**

After fixing each smoke-selected configuration, run one final evaluation over
the complete validation split. Record exact validation count and prove its
metrics do not feed back into search state.

- [ ] **Step 7: Record the server evidence summary**

The summary must include all six profiles, source commit/tree, train-probe
hashes, validation counts, CUDA forward evidence, and pass/fail status. Any
profile failure blocks handoff completion.

---

## Task 15: Complete Handoff, Aggregate, and Advance Existing Canonical

**Files:**
- Create: `agent_handoffs/tasks/glue-train-probe-six-profile-20260824.json`
- Create: `agent_handoffs/aggregates/20260824-glue-train-probe-six-profile.json`

- [ ] **Step 1: Commit and push the final source commit**

Record its full commit and tree IDs. Ensure the task worktree has no tracked
changes.

- [ ] **Step 2: Create the completed handoff-only commit**

The handoff records all changed scopes, RED/GREEN evidence, complete server
tests, six-profile smoke paths, source commit/tree, and
`deployment_eligible=false`.

- [ ] **Step 3: Validate the handoff**

```bash
python scripts/repo_sync_guard.py agent-finish \
  --handoff agent_handoffs/tasks/glue-train-probe-six-profile-20260824.json \
  --remote origin
```

- [ ] **Step 4: As the authorized aggregator, refresh every remote head**

Create a fresh aggregate from the current `origin/jk_standard_rl`, integrate
all completed non-superseded handoffs, classify every remote head, and leave no
`needs_review` entry.

- [ ] **Step 5: Verify the exact aggregate source on the server**

Repeat the affected/full suites and source provenance checks from the exact
aggregate commit/tree.

- [ ] **Step 6: Finalize and fast-forward `jk_standard_rl`**

```bash
RFR_AGGREGATOR_AUTHORIZED=1 \
RFR_AGGREGATE_MANIFEST=agent_handoffs/aggregates/20260824-glue-train-probe-six-profile.json \
python scripts/repo_sync_guard.py aggregate-finalize \
  --manifest agent_handoffs/aggregates/20260824-glue-train-probe-six-profile.json \
  --remote origin --fetch

RFR_AGGREGATOR_AUTHORIZED=1 \
RFR_AGGREGATE_MANIFEST=agent_handoffs/aggregates/20260824-glue-train-probe-six-profile.json \
git push origin codex/aggregate-glue-train-probe-six-profile-20260824:jk_standard_rl
```

- [ ] **Step 7: Synchronize local and server canonical through Git**

Use `local-sync --apply`, then `server-check --sync`, followed by verify-only
checks. Acceptance requires identical local/remote/server full commit and tree
IDs and tracked-clean status.

- [ ] **Step 8: Do not start formal training automatically**

Report that all six profiles are ready under the new data protocol. Formal
long-running experiments require a separate user instruction because all old
validation-probe results are scientifically historical.

---

## Plan Self-Review

- Every approved requirement is assigned to a task.
- The 512-example encrypted-inference set is absent from implementation.
- Data/result cleanup and README work are deferred.
- Only MRPC/RTE/SST-2 and BERT-base/BERT-large remain active.
- Profile, Stage-1, and Stage-2 share one fixed probe identity.
- Validation is final-evaluation-only.
- Old resume state fails closed.
- Every behavior change follows RED/GREEN TDD on the server.
- Final deployment advances the existing `jk_standard_rl`; no alternate
  canonical is introduced.
