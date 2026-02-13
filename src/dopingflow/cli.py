# src/dopingflow/cli.py
from __future__ import annotations

from pathlib import Path
import typer

from typing import Optional

from dopingflow.logging import setup_logging
from dopingflow.refs import run_refs_build_from_toml
from dopingflow.generate import run_generate_from_toml
from dopingflow.scan import run_scan_from_toml
from dopingflow.relax import run_relax_from_toml
from dopingflow.filtering import run_filtering_from_toml
from dopingflow.bandgap import run_bandgap_from_toml
from dopingflow.formation import run_formation_from_toml
from dopingflow.collect import run_collect_from_toml



app = typer.Typer(help="dopingflow: ML doping workflow pipeline")


def _init(config: Path, verbose: bool) -> None:
    root = config.resolve().parent
    setup_logging(root, verbose=verbose)


@app.command("refs-build")
def refs_build_cmd(
    config: Path = typer.Option(Path("input.toml"), "-c", "--config", exists=True),
    verbose: bool = typer.Option(False, "--verbose", help="More detailed logs"),
):
    """Step 00: Build/cache reference energies."""
    _init(config, verbose)
    run_refs_build_from_toml(config)


@app.command("generate")
def generate_cmd(
    config: Path = typer.Option(Path("input.toml"), "-c", "--config", exists=True),
    verbose: bool = typer.Option(False, "--verbose", help="More detailed logs"),
):
    """Step 01: Generate random doped structures."""
    _init(config, verbose)
    run_generate_from_toml(config)

@app.command("scan")
def scan_cmd(
    config: Path = typer.Option(Path("input.toml"), "-c", "--config", exists=True),
    verbose: bool = typer.Option(False, "--verbose", help="More detailed logs"),
):
    """Step 02: Symmetry-unique scan + M3GNet single-point energies (top-k)."""
    _init(config, verbose)
    run_scan_from_toml(config)

@app.command("relax")
def relax_cmd(
    config: Path = typer.Option(Path("input.toml"), "-c", "--config", exists=True),
    verbose: bool = typer.Option(False, "--verbose", help="More detailed logs"),
):
    """Step 03: Relax scanned candidates with M3GNet Relaxer."""
    _init(config, verbose)
    run_relax_from_toml(config)

@app.command("filter")
def filter_cmd(
    config: Path = typer.Option(Path("input.toml"), "-c", "--config", exists=True),
    only: str = typer.Option(None, "--only", help="Run only one folder name (e.g. Sb5_Zr5)"),
    force: bool = typer.Option(False, "--force", help="Rerun even if outputs already exist"),
    window_meV: float = typer.Option(None, "--window-mev", help="Override window filter (forces mode=window)"),
    topn: int = typer.Option(None, "--topn", help="Override top-N filter (forces mode=topn)"),
    verbose: bool = typer.Option(False, "--verbose", help="More detailed logs"),
):
    """Step 04: Filter relaxed candidates (window or top-N)."""
    _init(config, verbose)
    run_filtering_from_toml(config, only=only, force=force, window_meV=window_meV, topn=topn)

@app.command("bandgap")
def bandgap_cmd(
    config: Path = typer.Option(Path("input.toml"), "-c", "--config", exists=True),
    verbose: bool = typer.Option(False, "--verbose", help="More detailed logs"),
):
    """Step 05: Predict bandgap for filtered relaxed candidates (ALIGNN)."""
    _init(config, verbose)
    run_bandgap_from_toml(config)

@app.command("formation")
def formation_cmd(
    config: Path = typer.Option(Path("input.toml"), "-c", "--config", exists=True),
    verbose: bool = typer.Option(False, "--verbose", help="More detailed logs"),
):
    """Step 06: Compute formation energies using cached references."""
    _init(config, verbose)
    run_formation_from_toml(config)

@app.command("collect")
def collect_cmd(
    config: Path = typer.Option(Path("input.toml"), "-c", "--config", exists=True),
    verbose: bool = typer.Option(False, "--verbose", help="More detailed logs"),
):
    """Step 07: Collect selected candidates into one CSV database."""
    _init(config, verbose)
    run_collect_from_toml(config)


@app.command("run-all")
def run_all_cmd(
    config: Path = typer.Option(Path("input.toml"), "-c", "--config", exists=True),
    start: str = typer.Option("refs", "--from", help="Start step key (refs, generate, scan, relax, filter, bandgap, formation, collect)"),
    stop: str = typer.Option("collect", "--until", help="Stop step key (inclusive)"),
    only: Optional[str] = typer.Option(None, "--only", help="Comma-separated list of step keys to run (subset). Example: refs,generate,scan"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print planned steps without running"),
    # filter-specific passthroughs (optional but very helpful)
    filter_only: Optional[str] = typer.Option(None, "--filter-only", help="Filter step: run only one folder name (e.g. Sb5_Zr5)"),
    force: bool = typer.Option(False, "--force", help="Filter step: rerun even if outputs exist"),
    window_meV: Optional[float] = typer.Option(None, "--window-mev", help="Filter step: override window filter (forces mode=window)"),
    topn: Optional[int] = typer.Option(None, "--topn", help="Filter step: override top-N filter (forces mode=topn)"),
    verbose: bool = typer.Option(False, "--verbose", help="More detailed logs"),
):
    """
    Run the full pipeline in order, with optional step selection.

    Step keys:
      refs -> generate -> scan -> relax -> filter -> bandgap -> formation -> collect
    """
    _init(config, verbose)

    # Step registry (single source of truth)
    steps = [
        ("refs", "00 refs-build", lambda: run_refs_build_from_toml(config)),
        ("generate", "01 generate", lambda: run_generate_from_toml(config)),
        ("scan", "02 scan", lambda: run_scan_from_toml(config)),
        ("relax", "03 relax", lambda: run_relax_from_toml(config)),
        ("filter", "04 filter", lambda: run_filtering_from_toml(
            config, only=filter_only, force=force, window_meV=window_meV, topn=topn
        )),
        ("bandgap", "05 bandgap", lambda: run_bandgap_from_toml(config)),
        ("formation", "06 formation", lambda: run_formation_from_toml(config)),
        ("collect", "07 collect", lambda: run_collect_from_toml(config)),
    ]

    key_to_idx = {k: i for i, (k, _, _) in enumerate(steps)}
    valid_keys = list(key_to_idx.keys())

    def _parse_key(name: str, opt: str) -> int:
        name = (name or "").strip().lower()
        if name not in key_to_idx:
            raise typer.BadParameter(f"{opt}: unknown step '{name}'. Valid: {valid_keys}")
        return key_to_idx[name]

    i_start = _parse_key(start, "--from")
    i_stop = _parse_key(stop, "--until")
    if i_start > i_stop:
        raise typer.BadParameter("Require --from to be <= --until in pipeline order.")

    selected_indices = list(range(i_start, i_stop + 1))

    # Subset selection via --only
    if only:
        only_keys = [x.strip().lower() for x in only.split(",") if x.strip()]
        unknown = [k for k in only_keys if k not in key_to_idx]
        if unknown:
            raise typer.BadParameter(f"--only: unknown step(s) {unknown}. Valid: {valid_keys}")
        selected_indices = [i for i in selected_indices if steps[i][0] in set(only_keys)]
        if not selected_indices:
            raise typer.BadParameter("--only removed all steps in the selected --from/--until range.")

    # Plan print
    typer.echo("\nPlanned steps:")
    for i in selected_indices:
        k, title, _ = steps[i]
        typer.echo(f"  - {k:9s} {title}")

    if dry_run:
        typer.echo("\n(dry-run) Nothing executed.")
        return

    # Execute
    for i in selected_indices:
        k, title, fn = steps[i]
        typer.echo(f"\n=== {title} ({k}) ===")
        fn()
