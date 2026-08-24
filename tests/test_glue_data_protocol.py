from __future__ import annotations

import pytest

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


@pytest.mark.parametrize(
    ("model_family", "dataset"),
    (
        ("gpt-2", "mrpc"),
        ("bert-base", "stsb"),
        ("bert-large", "qnli"),
        ("bert-base", "mnli"),
    ),
)
def test_unsupported_profile_fails_closed(model_family, dataset):
    with pytest.raises(ValueError, match="unsupported profile"):
        validate_supported_profile(model_family, dataset)
