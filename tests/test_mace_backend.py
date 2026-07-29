from __future__ import annotations

import sys
from types import ModuleType

import pytest

from dopingflow import ml_backends


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("m3gnet", ("m3gnet", "default", "")),
        ("uma", ("uma", "uma-s-1p2", "omat")),
        ("mace", ("mace", "small", "")),
        ("grace", ("grace", "GRACE-1L-OMAT", "")),
    ],
)
def test_blank_backend_model_and_task_use_compatible_defaults(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    expected: tuple[str, str, str],
) -> None:
    monkeypatch.setattr(ml_backends, "get_mace_model_choices", lambda: ("small", "mh-1"))

    assert ml_backends.normalize_backend_config(
        backend=backend,
        model="",
        task="",
        section_name="scan",
    ) == expected


def test_mace_discovery_filters_internal_none_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mace_package = ModuleType("mace")
    calculators = ModuleType("mace.calculators")
    foundations = ModuleType("mace.calculators.foundations_models")
    foundations.mace_mp_names = [None, "small", "mh-1"]  # type: ignore[attr-defined]
    mace_package.calculators = calculators  # type: ignore[attr-defined]
    calculators.foundations_models = foundations  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mace", mace_package)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)
    monkeypatch.setitem(sys.modules, "mace.calculators.foundations_models", foundations)

    ml_backends.get_mace_model_choices.cache_clear()
    try:
        assert ml_backends.get_mace_model_choices() == ("small", "mh-1")
    finally:
        ml_backends.get_mace_model_choices.cache_clear()


def test_mace_mh_alias_and_head_are_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ml_backends,
        "get_mace_model_choices",
        lambda: ("small", "mh-0", "mh-1"),
    )

    assert ml_backends.normalize_backend_config(
        backend="mace",
        model="mh-1",
        task="omat_pbe",
        section_name="scan",
    ) == ("mace", "mh-1", "omat_pbe")


@pytest.mark.parametrize("task", ["", "   "])
def test_mace_mh1_uses_compatible_head_when_task_is_blank(
    monkeypatch: pytest.MonkeyPatch,
    task: str,
) -> None:
    monkeypatch.setattr(
        ml_backends,
        "get_mace_model_choices",
        lambda: ("small", "mh-1"),
    )

    assert ml_backends.normalize_backend_config(
        backend="mace",
        model="mh-1",
        task=task,
        section_name="scan",
    ) == ("mace", "mh-1", "omat_pbe")


def test_mace_custom_checkpoint_path_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ml_backends, "get_mace_model_choices", lambda: ("small",))

    assert ml_backends.normalize_backend_config(
        backend="mace",
        model="models/custom.model",
        task="custom_head",
        section_name="relax",
    ) == ("mace", "models/custom.model", "custom_head")


def test_mace_unknown_non_path_alias_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ml_backends, "get_mace_model_choices", lambda: ("small", "mh-1"))

    with pytest.raises(ValueError, match="alias supported by the installed MACE"):
        ml_backends.normalize_backend_config(
            backend="mace",
            model="not-a-mace-model",
            task="",
            section_name="references",
        )


def test_mace_calculator_receives_optional_head(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    mace_package = ModuleType("mace")
    calculators = ModuleType("mace.calculators")

    def fake_mace_mp(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    calculators.mace_mp = fake_mace_mp  # type: ignore[attr-defined]
    mace_package.calculators = calculators  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mace", mace_package)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)

    ml_backends.build_ase_calculator(
        backend="mace",
        model="mh-1",
        task="omat_pbe",
        device="cpu",
    )

    assert captured == {
        "model": "mh-1",
        "device": "cpu",
        "default_dtype": "float64",
        "head": "omat_pbe",
    }


def test_mace_calculator_omits_blank_head(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    mace_package = ModuleType("mace")
    calculators = ModuleType("mace.calculators")
    calculators.mace_mp = lambda **kwargs: captured.update(kwargs)  # type: ignore[attr-defined]
    mace_package.calculators = calculators  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mace", mace_package)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)

    ml_backends.build_ase_calculator(
        backend="mace",
        model="small",
        task="",
        device="cuda",
    )

    assert "head" not in captured


def test_mace_calculator_defaults_mh1_head_when_called_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    mace_package = ModuleType("mace")
    calculators = ModuleType("mace.calculators")
    calculators.mace_mp = lambda **kwargs: captured.update(kwargs)  # type: ignore[attr-defined]
    mace_package.calculators = calculators  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mace", mace_package)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)

    ml_backends.build_ase_calculator(
        backend="mace",
        model="mh-1",
        task="",
        device="cpu",
    )

    assert captured["head"] == "omat_pbe"
