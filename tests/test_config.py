from pathlib import Path

import pytest
from pydantic import ValidationError

from sentimetro.config import apply_overrides, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_project_config_is_valid() -> None:
    cfg = load_config(ROOT / "configs" / "params.yaml")
    assert cfg.transformer.model_name.startswith("dccuchile/")
    assert cfg.data.star_to_label[2] == "neutro"
    assert cfg.evaluation.models["baseline"].path == Path("models/baseline")


def test_overrides_are_parsed_with_yaml_types() -> None:
    cfg = load_config(
        ROOT / "configs" / "params.yaml",
        ["transformer.epochs=3", "transformer.train_samples=null", "tracking.enabled=false"],
    )
    assert cfg.transformer.epochs == 3
    assert cfg.transformer.train_samples is None
    assert cfg.tracking.enabled is False


def test_apply_overrides_creates_sections_and_does_not_mutate() -> None:
    original = {"a": {"b": 1}}
    result = apply_overrides(original, ["a.c=x", "d.e=[1, 2]"])
    assert result == {"a": {"b": 1, "c": "x"}, "d": {"e": [1, 2]}}
    assert original == {"a": {"b": 1}}


@pytest.mark.parametrize(
    "override",
    [
        "transformer.learning_rate=-1",
        "transformer.max_length=4096",
        "baseline.c_grid=[]",
        "data.unknown_key=1",
        "data.star_to_label={0: malo, 1: negativo, 2: neutro, 3: positivo, 4: positivo}",
    ],
)
def test_invalid_values_are_rejected(override: str) -> None:
    with pytest.raises(ValidationError):
        load_config(None, [override])


def test_malformed_override() -> None:
    with pytest.raises(ValueError, match="expected key=value"):
        load_config(None, ["transformer.epochs"])
