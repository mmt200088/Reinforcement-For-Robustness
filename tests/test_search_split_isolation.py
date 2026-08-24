from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from glue_data_protocol import (
    GLUE_DATASET_REVISION,
    PROTOCOL_SCHEMA,
    TRAIN_PROBE_SIZE,
    GlueDataProtocolContext,
    TrainProbeIdentity,
)
from json_utils import read_json_file
from layer_importance_evaluator import LayerImportanceEvaluator
from final_evaluation_module import require_final_evaluation_protocol
from rl_data_points import write_dataset_protocol


def _identity(dataset="mrpc"):
    labels = tuple(index % 2 for index in range(TRAIN_PROBE_SIZE))
    return TrainProbeIdentity(
        dataset=dataset,
        source_size=512,
        positions=tuple(range(TRAIN_PROBE_SIZE)),
        raw_ids=tuple(range(1000, 1000 + TRAIN_PROBE_SIZE)),
        labels=labels,
        label_histogram=((0, 128), (1, 128)),
        ordered_identity_hash="a" * 64,
    )


def _context():
    return GlueDataProtocolContext(
        model_family="bert-base",
        dataset="mrpc",
        train_probe=[object() for _ in range(TRAIN_PROBE_SIZE)],
        validation_full=[object() for _ in range(408)],
        identity=_identity(),
    )


def test_protocol_payload_records_search_and_final_split_identity():
    context = _context()

    payload = context.as_payload()

    assert payload["schema_version"] == PROTOCOL_SCHEMA
    assert payload["dataset_revision"] == GLUE_DATASET_REVISION
    assert payload["source_split"] == "train"
    assert payload["search_split"] == "train_probe"
    assert payload["final_eval_split"] == "validation_full"
    assert payload["probe_size"] == TRAIN_PROBE_SIZE
    assert payload["ordered_identity_hash"] == "a" * 64
    assert payload["dataset_protocol_hash"] == context.dataset_protocol_hash


def test_dataset_protocol_file_round_trips_atomically(tmp_path):
    context = _context()

    path = write_dataset_protocol(tmp_path, context.as_payload())

    assert path == Path(tmp_path) / "dataset_protocol.json"
    assert read_json_file(path) == context.as_payload()
    assert not path.with_name("dataset_protocol.json.tmp").exists()


def test_prepare_rl_datasets_registers_explicit_probe_and_validation_views():
    registered = {}
    train = object()
    probe = object()
    validation = object()

    def register(split_name, dataset):
        registered[split_name] = dataset
        evaluator.dataloaders[split_name] = f"{split_name}-loader"

    evaluator = SimpleNamespace(
        _register_dataset_split=register,
    )

    LayerImportanceEvaluator._prepare_rl_datasets(
        evaluator,
        train_data=train,
        train_probe=probe,
        validation_data=validation,
    )

    assert registered == {
        "train": train,
        "train_probe": probe,
        "validation_full": validation,
    }
    assert evaluator.dataloader_train == "train-loader"
    assert evaluator.dataloader_test == "validation_full-loader"


def test_search_reward_split_names_are_always_train_probe():
    evaluator = SimpleNamespace()

    assert (
        LayerImportanceEvaluator.get_reward_reference_split_name(evaluator)
        == "train_probe"
    )
    assert (
        LayerImportanceEvaluator.get_online_reward_split_name(evaluator)
        == "train_probe"
    )


def test_stage1_search_source_has_no_validation_guided_branch():
    source = Path("layer_importance_evaluator.py").read_text(encoding="utf-8")
    stage1_flow = source[source.index("    def on_evaluate("):]

    assert "if USE_VALIDATION_FOR_REWARD" not in stage1_flow
    assert "if not USE_VALIDATION_FOR_REWARD" not in stage1_flow
    assert "reward_reference_split != TRAIN_PROBE_SPLIT" in stage1_flow
    assert '"split": TRAIN_PROBE_SPLIT' in stage1_flow
    assert stage1_flow.count(
        '"dataset_protocol_hash": self.dataset_protocol_hash'
    ) >= 3
    assert "eval_split_name=online_reward_split" in stage1_flow
    assert stage1_flow.count(
        "dataset_protocol_hash=self.dataset_protocol_hash"
    ) == 4


def test_stage2_search_artifacts_name_only_train_probe_evidence():
    sequential = Path("blb_stage2_rl/sequential_runner.py").read_text(
        encoding="utf-8"
    )
    layerwise = Path("blb_stage2_rl/layerwise_runner.py").read_text(
        encoding="utf-8"
    )
    baseline = Path("blb_stage2_rl/search_baseline_runner.py").read_text(
        encoding="utf-8"
    )

    assert "SEARCH_EVIDENCE_SPLIT = TRAIN_PROBE_SPLIT" in sequential
    assert "DATASET_PROTOCOL_SCHEMA = PROTOCOL_SCHEMA" in sequential
    assert (
        'LAYERWISE_RUN_SCHEMA = "stage2_layerwise_train_probe_run_v1"'
        in sequential
    )
    for source in (sequential, layerwise, baseline):
        assert "validation_full_stratified_probe" not in source
        assert "F4_validation_full" not in source
    assert '"split": SEARCH_EVIDENCE_SPLIT' in baseline


def test_final_evaluation_requires_validation_full_and_matching_search_hash(tmp_path):
    context = _context()
    protocol_path = write_dataset_protocol(tmp_path, context.as_payload())
    validation_loader = ["validation-batch-1", "validation-batch-2"]
    evaluator = SimpleNamespace(
        dataset_protocol_hash=context.dataset_protocol_hash,
        dataset_protocol_path=protocol_path,
        dataset_splits={
            "train_probe": context.train_probe,
            "validation_full": context.validation_full,
        },
        dataloaders={"validation_full": validation_loader},
    )
    search_results = (
        {"dataset_protocol_hash": context.dataset_protocol_hash},
        {"dataset_protocol_hash": context.dataset_protocol_hash},
    )

    resolved = require_final_evaluation_protocol(
        evaluator,
        search_results=search_results,
        requested_split="validation_full",
    )

    assert resolved["split_name"] == "validation_full"
    assert resolved["dataset"] is context.validation_full
    assert resolved["dataloader"] is validation_loader
    assert resolved["example_count"] == 408

    with pytest.raises(RuntimeError, match="validation_full"):
        require_final_evaluation_protocol(
            evaluator,
            search_results=search_results,
            requested_split="train_probe",
        )
    with pytest.raises(RuntimeError, match="protocol hash"):
        require_final_evaluation_protocol(
            evaluator,
            search_results=({"dataset_protocol_hash": "wrong"},),
            requested_split="validation_full",
        )


def test_blb_final_eval_consumes_full_validation_once_per_repeat():
    from Paean.blb_action_eval import BLBActionFinalEvaluationModule

    validation_ids = tuple(range(408))
    passes = []

    class ValidationLoader:
        def __iter__(self):
            passes.append([])
            for start in range(0, len(validation_ids), 64):
                batch = validation_ids[start:start + 64]
                passes[-1].extend(batch)
                yield batch

    evaluator = SimpleNamespace(
        apply_configuration=lambda *_args, **_kwargs: None,
        dataloaders={"validation_full": ValidationLoader()},
        _resolve_eval_split=lambda *, use_train, split: split,
        _run_evaluation=lambda loader, **_kwargs: (
            tuple(loader) and (0.5, 0.8, 0.7, 1.0)
        ),
    )
    module = BLBActionFinalEvaluationModule.__new__(
        BLBActionFinalEvaluationModule
    )
    module.evaluator = evaluator
    module.final_eval_split = "validation_full"
    module._clear_all_noise = lambda: None

    trials = module._run_clean_baseline_trials(
        baseline_stage1_gelu=[4],
        baseline_stage1_softmax=[6],
        repeats=2,
    )

    assert len(trials) == 2
    assert passes == [list(validation_ids), list(validation_ids)]
