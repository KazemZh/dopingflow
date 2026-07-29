# gui/io_project.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

RANKING_FILES = [
    "ranking_scan.csv",
    "ranking_relax.csv",
    "ranking_relax_filtered.csv",
]

VACANCY_DATABASE_FILES = ["vacancies_database.csv", "vacancies_database.json"]
VACANCY_ANALYSIS_FILES = [
    "vacancy_minima_by_composition.csv",
    "vacancy_stability_intervals.csv",
    "vacancy_best_counts.csv",
]

@dataclass
class ProjectIndex:
    root: Path
    outdir: Path  # e.g., root / "random_structures"

    def compositions(self) -> list[str]:
        if not self.outdir.exists():
            return []
        return sorted([p.name for p in self.outdir.iterdir() if p.is_dir()])

    def composition_path(self, comp: str) -> Path:
        return self.outdir / comp

    def list_rankings(self, comp: str) -> dict[str, Path]:
        base = self.composition_path(comp)
        found = {}
        for fn in RANKING_FILES:
            p = base / fn
            if p.exists():
                found[fn] = p
        return found

    def read_csv_safe(self, path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    def selected_candidates(self, comp: str) -> list[str]:
        p = self.composition_path(comp) / "selected_candidates.txt"
        if not p.exists():
            return []
        return [line.strip() for line in p.read_text().splitlines() if line.strip()]

    def find_candidate_dir(self, comp: str, candidate_name: str) -> Path | None:
        base = self.composition_path(comp)
        cand = base / candidate_name
        return cand if cand.exists() else None

    def find_structure_files(self, cand_dir: Path) -> dict[str, Path]:
        """
        Find 'before' and 'after' structure files based on dopingflow layout.

        Expected workflow layout (yours):
          - before: <cand_dir>/01_scan/POSCAR
          - after:  <cand_dir>/02_relax/POSCAR
        """
        cand_dir = Path(cand_dir)

        found: dict[str, Path] = {}

        # ---- Preferred (matches your workflow) ----
        before = cand_dir / "01_scan" / "POSCAR"
        after = cand_dir / "02_relax" / "POSCAR"

        if before.exists():
            found["before"] = before

        if after.exists():
            found["after"] = after

        # ---- Secondary candidates (if you later add CONTCAR etc.) ----
        if "before" not in found:
            candidates_before = [
                cand_dir / "POSCAR",
                cand_dir / "before.POSCAR",
                cand_dir / "before.vasp",
                cand_dir / "structure_before.vasp",
                cand_dir / "initial.vasp",
            ]
            for p in candidates_before:
                if p.exists():
                    found["before"] = p
                    break

        if "after" not in found:
            candidates_after = [
                cand_dir / "CONTCAR",
                cand_dir / "after.CONTCAR",
                cand_dir / "after.vasp",
                cand_dir / "structure_after.vasp",
                cand_dir / "final.vasp",
                cand_dir / "relaxed.vasp",
            ]
            for p in candidates_after:
                if p.exists():
                    found["after"] = p
                    break

        # ---- Fallback: pick *any* vasp-like file if nothing found ----
        if "before" not in found:
            for p in cand_dir.glob("*.vasp"):
                found["before"] = p
                break

        if "after" not in found:
            for p in cand_dir.glob("*.vasp"):
                found["after"] = p
                break

        return found

    def vacancy_database(self) -> Path | None:
        path = self.outdir / "vacancies_database.csv"
        return path if path.exists() else None

    def vacancy_analysis_files(self) -> dict[str, Path]:
        return {
            filename: self.outdir / filename
            for filename in VACANCY_ANALYSIS_FILES
            if (self.outdir / filename).exists()
        }

    def vacancy_parents(self, comp: str) -> list[str]:
        base = self.composition_path(comp)
        return sorted(
            candidate.name
            for candidate in base.glob("candidate_*")
            if (candidate / "05_vacancies" / "vacancy_results.csv").exists()
        )

    def vacancy_rankings(self, comp: str, candidate: str) -> dict[str, Path]:
        root = self.composition_path(comp) / candidate / "05_vacancies"
        found: dict[str, Path] = {}
        for group in sorted(root.glob("V_*_*")):
            for filename in ("ranking_scan.csv", "ranking_relax.csv"):
                path = group / filename
                if path.exists():
                    found[f"{group.name}/{filename}"] = path
        return found

    def find_vacancy_structure_files(
        self, comp: str, candidate: str, vacancy_count: int, configuration: str
    ) -> dict[str, Path]:
        root = self.composition_path(comp) / candidate / "05_vacancies"
        groups = sorted(root.glob(f"V_*_{vacancy_count:02d}"))
        if not groups:
            return {}
        config_dir = groups[0] / configuration
        candidates = {
            "parent": root / "parent_reference" / "relaxed" / "POSCAR",
            "generated": config_dir / "00_generate" / "POSCAR",
            "relaxed": config_dir / "02_relax" / "POSCAR",
        }
        return {label: path for label, path in candidates.items() if path.exists()}
