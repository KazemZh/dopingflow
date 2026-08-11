from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from dopingflow.workflow_convergence_guards import (
    _get_candidate_poscars_converged_only,
    _read_ranking_relax_converged_only,
    _relax_one_candidate_convergence_aware,
)


def _write_meta(path: Path, *, converged: bool, energy: float = -1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "converged": converged,
                "energy_relaxed_eV": energy,
                "final_fmax_eV_per_A": 0.2 if not converged else 0.01,
                "fmax_target_eV_per_A": 0.05,
                "optimizer_steps": 300,
                "max_steps": 300,
            }
        ),
        encoding="utf-8",
    )


def test_filter_excludes_old_status_ok_but_unconverged_rows(tmp_path: Path) -> None:
    folder = tmp_path / "Sb5"
    folder.mkdir()
    ranking = folder / "ranking_relax.csv"
    with ranking.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate",
                "rank_relax",
                "energy_relaxed_eV",
                "rank_sp",
                "energy_sp_eV",
                "signature",
                "status",
                "walltime_s",
                "converged",
                "final_fmax_eV_per_A",
                "optimizer_steps",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "candidate": "candidate_001",
                "energy_relaxed_eV": -10.0,
                "status": "ok",
            }
        )
        writer.writerow(
            {
                "candidate": "candidate_002",
                "energy_relaxed_eV": -11.0,
                "status": "ok",
            }
        )

    _write_meta(folder / "candidate_001/02_relax/meta.json", converged=True, energy=-10.0)
    _write_meta(folder / "candidate_002/02_relax/meta.json", converged=False, energy=-11.0)

    rows = _read_ranking_relax_converged_only(ranking)
    assert [row["candidate"] for row in rows] == ["candidate_001"]


def test_formation_guard_skips_unconverged_selected_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "Sb5"
    good = folder / "candidate_001/02_relax/POSCAR"
    bad = folder / "candidate_002/02_relax/POSCAR"
    good.parent.mkdir(parents=True)
    bad.parent.mkdir(parents=True)
    good.write_text("placeholder", encoding="utf-8")
    bad.write_text("placeholder", encoding="utf-8")
    _write_meta(good.parent / "meta.json", converged=True)
    _write_meta(bad.parent / "meta.json", converged=False)

    import dopingflow.workflow_convergence_guards as guards

    monkeypatch.setattr(guards, "_BASE_GET_CANDIDATE_POSCARS", lambda _folder: [good, bad])
    assert _get_candidate_poscars_converged_only(folder) == [good]


def test_relax_result_is_relabelled_when_not_converged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dopingflow.workflow_convergence_guards as guards

    monkeypatch.setattr(
        guards,
        "_BASE_RELAX_ONE_CANDIDATE",
        lambda _job: {
            "candidate": "candidate_015",
            "status": "ok",
            "converged": False,
            "final_fmax_eV_per_A": 0.12,
            "optimizer_steps": 300,
        },
    )
    result = _relax_one_candidate_convergence_aware(object())
    assert result["status"] == "not_converged"
    assert "optimizer_completed_without_positive_convergence" in result["error"]
