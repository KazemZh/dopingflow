from __future__ import annotations

import copy
import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from dopingflow.generate import run_generate, enumerate_compositions
from dopingflow.scan import run_scan
from dopingflow.relax import run_relax
from dopingflow.filtering import run_filtering
from dopingflow.bandgap import run_bandgap
from dopingflow.formation import run_formation
from dopingflow.collect_relative import run_collect
from dopingflow.relative_energy import populate_relative_energy_columns

log = logging.getLogger(__name__)


def _load_raw_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _composition_label(comp: Dict[str, float]) -> str:
    parts = []
    for el, pct in sorted(comp.items()):
        pct = float(pct)
        if abs(pct - round(pct)) < 1e-9:
            pct_str = str(int(round(pct)))
        else:
            pct_str = f"{pct:.4g}".replace(".", "p")
        parts.append(f"{el}{pct_str}")
    return "_".join(parts)


def _get_sequential_compositions(raw_cfg: dict[str, Any]) -> List[Dict[str, float]]:
    dop = raw_cfg.get("doping", {}) or {}
    mode = str(dop.get("mode", "explicit")).lower().strip()

    if mode == "explicit":
        compositions = dop.get("compositions", []) or []

    elif mode == "enumerate":
        compositions = enumerate_compositions(
            dopants=[str(x) for x in dop.get("dopants", [])],
            must_include=[str(x) for x in dop.get("must_include", [])],
            max_dopants_total=int(dop.get("max_dopants_total", 2)),
            allowed_totals=[float(x) for x in dop.get("allowed_totals", [])],
            levels=[float(x) for x in dop.get("levels", [])],
        )

    else:
        raise ValueError("[doping].mode must be 'explicit' or 'enumerate'")

    compositions = [{str(k): float(v) for k, v in comp.items()} for comp in compositions]

    if not compositions:
        raise ValueError("No compositions found from [doping] section.")

    compositions.sort(key=lambda c: (sum(c.values()), tuple(sorted(c.items()))))
    return compositions


def _best_relaxed_poscar(comp_dir: Path) -> Path:
    ranking_path = comp_dir / "ranking_relax.csv"

    if not ranking_path.exists():
        raise FileNotFoundError(f"Missing ranking_relax.csv in {comp_dir}")

    with ranking_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ok_rows = [r for r in rows if r.get("status") == "ok"]

    if not ok_rows:
        raise RuntimeError(f"No successful relaxed candidates found in {comp_dir}")

    best = sorted(ok_rows, key=lambda r: float(r["energy_relaxed_eV"]))[0]
    candidate = best["candidate"]

    poscar = comp_dir / candidate / "02_relax" / "POSCAR"

    if not poscar.exists():
        raise FileNotFoundError(f"Best relaxed POSCAR not found: {poscar}")

    return poscar


def _merge_step_databases(
    sequential_root: Path,
    root: Path,
    raw_cfg: dict[str, Any],
) -> Path:
    import pandas as pd

    csv_files = sorted(sequential_root.glob("step_*/results_database.csv"))
    if not csv_files:
        raise FileNotFoundError("No step results_database.csv files found to merge.")

    dfs = []
    for csv_path in csv_files:
        step_root = csv_path.parent
        df = pd.read_csv(csv_path)

        if "sequential_step" not in df.columns:
            df.insert(0, "sequential_step", step_root.name)

        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    out = root / "results_database.csv"
    merged.to_csv(out, index=False)

    # Per-step collection has only one composition and therefore cannot define
    # a meaningful global endpoint. Recalculate relative energies once all
    # sequential compositions have been merged.
    populate_relative_energy_columns(out, raw_cfg)
    return out


def run_sequential(
    raw_cfg: dict[str, Any],
    root: Path,
    *,
    config_path: Path | None = None,
) -> Path:
    seq = raw_cfg.get("sequential", {}) or {}

    sequential_root = (root / str(seq.get("outdir", "sequential_structures"))).resolve()
    sequential_root.mkdir(parents=True, exist_ok=True)

    seq_mode = str(seq.get("mode", "full")).strip().lower()

    if seq_mode not in {"full", "recompute_energies"}:
        raise ValueError(
            "[sequential].mode must be either 'full' or 'recompute_energies'"
        )

    compositions = _get_sequential_compositions(raw_cfg)

    previous_best: Path | None = None

    log.info("Starting sequential doping workflow")
    log.info("Sequential mode: %s", seq_mode)
    log.info("Sequential output directory: %s", sequential_root)
    log.info("Compositions are taken from [doping] section.")
    log.info("Number of sequential steps: %d", len(compositions))

    for step_index, comp in enumerate(compositions, start=1):
        comp_label = _composition_label(comp)
        step_name = f"step_{step_index:03d}_{comp_label}"
        step_root = sequential_root / step_name
        step_root.mkdir(parents=True, exist_ok=True)

        log.info("=" * 80)
        log.info("Sequential step %d/%d: %s", step_index, len(compositions), comp)
        log.info("Step directory: %s", step_root)

        cfg = copy.deepcopy(raw_cfg)

        cfg.setdefault("structure", {})
        cfg.setdefault("generate", {})
        cfg.setdefault("doping", {})
        cfg.setdefault("scan", {})
        cfg.setdefault("relax", {})

        step_outdir = step_root / "random_structures"
        cfg["structure"]["outdir"] = str(step_outdir)

        # Internally each sequential step is treated as one explicit target composition.
        # The user-facing composition list is still defined by [doping].
        cfg["doping"]["mode"] = "explicit"
        cfg["doping"]["compositions"] = [comp]

        cfg["generate"]["incremental"] = True
        cfg["generate"]["clean_outdir"] = seq_mode == "full"

        if seq_mode == "full":
            if previous_best is not None:
                cfg["generate"]["base_poscar"] = str(previous_best.relative_to(root))
                log.info("Using previous best relaxed POSCAR as base: %s", previous_best)
            else:
                cfg["generate"].pop("base_poscar", None)
                log.info("First step: using relaxed host supercell from refs-build")
        else:
            log.info("Recompute mode: using existing relaxed structures in %s", step_outdir)

        cfg["scan"]["skip_if_done"] = False
        cfg["relax"]["skip_if_done"] = False
        cfg["relax"]["skip_candidate_if_done"] = False

        step_config_path = step_root / "input_step.json"
        step_config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        if seq_mode == "full":
            log.info("Running generate for %s", comp_label)
            run_generate(cfg, root, config_path=step_config_path)

            log.info("Running scan for %s", comp_label)
            run_scan(cfg, root, config_path=step_config_path)

            log.info("Running relax for %s", comp_label)
            run_relax(cfg, root, config_path=step_config_path)

            log.info("Running filter for %s", comp_label)
            run_filtering(cfg, root)

            bandgap_cfg = cfg.get("bandgap", {}) or {}
            if bool(bandgap_cfg.get("enabled", True)):
                log.info("Running bandgap for %s", comp_label)
                run_bandgap(cfg, root)
            else:
                log.info("Skipping bandgap for %s because [bandgap].enabled=false", comp_label)

        else:
            if not step_outdir.exists():
                raise FileNotFoundError(
                    f"Recompute mode requires existing step output directory: {step_outdir}"
                )

            log.info("Recompute mode: skipping generate, scan, relax, filter, and bandgap for %s", comp_label)

        log.info("Running formation for %s", comp_label)
        run_formation(cfg, root)

        log.info("Running collect for %s", comp_label)
        run_collect(cfg, root)

        project_db = root / "results_database.csv"

        if not project_db.exists():
            raise FileNotFoundError(f"collect step did not create {project_db}")

        step_db = step_root / "results_database.csv"
        shutil.copy2(project_db, step_db)

        if seq_mode == "full":
            comp_dirs = sorted([p for p in step_outdir.iterdir() if p.is_dir()])
            if len(comp_dirs) != 1:
                raise RuntimeError(
                    f"Expected exactly one composition directory in {step_outdir}, "
                    f"but found {len(comp_dirs)}."
                )

            best_poscar = _best_relaxed_poscar(comp_dirs[0])

            best_copy_dir = step_root / "best_relaxed"
            best_copy_dir.mkdir(parents=True, exist_ok=True)

            copied_best = best_copy_dir / "POSCAR"
            shutil.copy2(best_poscar, copied_best)
            previous_best = copied_best

            summary = {
                "step_index": step_index,
                "composition": comp,
                "composition_label": comp_label,
                "step_root": str(step_root),
                "composition_directory": str(comp_dirs[0]),
                "step_results_database": str(step_db),
                "best_relaxed_poscar_original": str(best_poscar),
                "best_relaxed_poscar_copied": str(copied_best),
                "next_step_base_poscar": str(previous_best),
                "mode": seq_mode,
            }

            (step_root / "sequential_step_summary.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )

            log.info("Best relaxed POSCAR copied to: %s", copied_best)

        else:
            summary = {
                "step_index": step_index,
                "composition": comp,
                "composition_label": comp_label,
                "step_root": str(step_root),
                "step_results_database": str(step_db),
                "mode": seq_mode,
                "note": "Recomputed formation/mixing energies from existing relaxed structures.",
            }

            (step_root / "sequential_step_summary_recompute.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )

        log.info("Step database copied to: %s", step_db)

    log.info("Sequential doping workflow finished.")
    merged_csv = _merge_step_databases(sequential_root, root, raw_cfg)
    log.info("Merged sequential database written to: %s", merged_csv)

    return sequential_root


def run_sequential_from_toml(config_path: Path) -> Path:
    raw = _load_raw_toml(config_path)
    root = config_path.resolve().parent
    return run_sequential(raw, root, config_path=config_path)
