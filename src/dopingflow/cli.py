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
    start: int = typer.Option(0, "--start", help="Start step number (0..7)"),
    stop: int = typer.Option(7, "--stop", help="Stop step number (0..7)"),
    verbose: bool = typer.Option(False, "--verbose", help="More detailed logs"),
):
    """Run the full pipeline (steps 00..07) in order."""
    _init(config, verbose)

    steps = [
        ("refs-build", lambda: run_refs_build_from_toml(config)),
        ("generate", lambda: run_generate_from_toml(config)),
        ("scan", lambda: run_scan_from_toml(config)),
        ("relax", lambda: run_relax_from_toml(config)),
        ("filter", lambda: run_filtering_from_toml(config, only=None, force=False, window_meV=None, topn=None)),
        ("bandgap", lambda: run_bandgap_from_toml(config)),
        ("formation", lambda: run_formation_from_toml(config)),
        ("collect", lambda: run_collect_from_toml(config)),
    ]

    if not (0 <= start <= 7 and 0 <= stop <= 7 and start <= stop):
        raise typer.BadParameter("Require 0 <= start <= stop <= 7")

    for i in range(start, stop + 1):
        name, fn = steps[i]
        typer.echo(f"\n=== Step {i:02d}: {name} ===")
        fn()
