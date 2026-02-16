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
