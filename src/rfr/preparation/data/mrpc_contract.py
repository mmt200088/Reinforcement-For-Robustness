"""Pinned runtime contract for the BERT-base MRPC comparators."""

from __future__ import annotations

from typing import Any, Mapping


MRPC_MODEL_ID = "textattack/bert-base-uncased-mrpc"
MRPC_MODEL_REVISION = "d421614df8fbeb22d6826a24d6397809fdc1e3ff"
MRPC_TOKENIZER_REVISION = MRPC_MODEL_REVISION
MRPC_COMPARATOR_BATCH_SIZE = 16
MRPC_STAGE2_RL_ALIGNMENT_BATCH_SIZE = 64
MRPC_MAX_LENGTH = 128
MRPC_NUM_LAYERS = 12
MRPC_VALIDATION_EXAMPLE_COUNT = 408
MRPC_TRAIN_PROBE_EXAMPLE_COUNT = 256


def comparator_pretrained_revision_kwargs(
        *,
        enabled: bool,
        data_path: str,
        model_id: str,
        ) -> tuple[dict[str, str], dict[str, str]]:
    if not enabled:
        return {}, {}
    if str(data_path or "").strip().lower() != "mrpc":
        raise ValueError("comparators require the MRPC dataset")
    if str(model_id or "").strip().lower() != MRPC_MODEL_ID:
        raise ValueError(f"comparators require model {MRPC_MODEL_ID}")
    return (
        {"revision": MRPC_MODEL_REVISION},
        {"revision": MRPC_TOKENIZER_REVISION},
    )


def validate_mrpc_comparator_runtime(
        *,
        model: Any,
        tokenizer: Any,
        collator: Any,
        validation_full: Any,
        train_probe: Any,
        batch_size: int,
        ) -> None:
    config = getattr(model, "config", None)
    model_id = str(getattr(config, "_name_or_path", "") or "").lower()
    if model_id != MRPC_MODEL_ID:
        raise ValueError(f"comparators require model {MRPC_MODEL_ID}")
    if str(getattr(config, "_commit_hash", "") or "") != MRPC_MODEL_REVISION:
        raise ValueError("MRPC model revision does not match the pinned revision")
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if not isinstance(init_kwargs, Mapping):
        raise ValueError("MRPC tokenizer revision metadata is unavailable")
    tokenizer_id = str(
        getattr(tokenizer, "name_or_path", "")
        or init_kwargs.get("name_or_path")
        or ""
    ).lower()
    if tokenizer_id != MRPC_MODEL_ID:
        raise ValueError(f"comparators require tokenizer {MRPC_MODEL_ID}")
    tokenizer_revision = str(init_kwargs.get("_commit_hash") or "")
    if tokenizer_revision and tokenizer_revision != MRPC_TOKENIZER_REVISION:
        raise ValueError("MRPC tokenizer revision does not match")
    if int(batch_size) != MRPC_COMPARATOR_BATCH_SIZE:
        raise ValueError(
            f"MRPC comparator batch size must be {MRPC_COMPARATOR_BATCH_SIZE}"
        )
    if int(getattr(config, "num_hidden_layers", 0)) != MRPC_NUM_LAYERS:
        raise ValueError(f"MRPC comparators require {MRPC_NUM_LAYERS} layers")
    if len(validation_full) != MRPC_VALIDATION_EXAMPLE_COUNT:
        raise ValueError("MRPC full validation size does not match")
    if len(train_probe) != MRPC_TRAIN_PROBE_EXAMPLE_COUNT:
        raise ValueError("MRPC training probe size does not match")

    collator_type = type(collator)
    expected = {
        "padding": "max_length",
        "max_length": MRPC_MAX_LENGTH,
        "return_tensors": "pt",
        "pad_to_multiple_of": 8,
    }
    actual = {name: getattr(collator, name, None) for name in expected}
    if (
            collator_type.__name__ != "DataCollatorWithPadding"
            or not collator_type.__module__.startswith("transformers")
            or actual != expected
    ):
        raise ValueError("MRPC comparator collator configuration does not match")


__all__ = [
    "MRPC_COMPARATOR_BATCH_SIZE",
    "MRPC_STAGE2_RL_ALIGNMENT_BATCH_SIZE",
    "comparator_pretrained_revision_kwargs",
    "validate_mrpc_comparator_runtime",
]
