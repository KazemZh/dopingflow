"""Configure, preview and run the optional electronic transport stage."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import toml

from dopingflow.conductivity import discover_targets, parse_config, select_targets

st.set_page_config(page_title="Electronic conductivity", layout="wide")
st.title("Electronic conductivity")
st.caption(
    "Band conductivity/tau for relaxed doped and oxygen-vacancy structures. Compatible GPAW results are shared with oxidation analysis."
)
root = (
    Path(st.sidebar.text_input("Project root", str(Path.cwd()), key="conductivity_root"))
    .expanduser()
    .resolve()
)
path = root / "input.toml"
if not path.exists():
    st.error(f"No input.toml at {path}")
    st.stop()
raw = toml.load(path)
section = dict(raw.get("conductivity", {}) or {})
section["enabled"] = st.checkbox("Enable conductivity stage", value=section.get("enabled", False))
options = ["favorable", "manual", "all"]
section["selection"] = st.selectbox(
    "Structure selection", options, index=options.index(section.get("selection", "favorable"))
)
section["top_k_per_group"] = int(
    st.number_input(
        "Lowest-energy structures per composition and vacancy count",
        min_value=1,
        value=int(section.get("top_k_per_group", 1)),
    )
)
st.caption(
    "Favorable selection compares identical atom counts and recorded energy models. Manual selections are added even when they are not lowest in energy."
)
section["include_vacancy_free"] = st.checkbox(
    "Include vacancy-free structures", section.get("include_vacancy_free", True)
)
section["include_oxygen_vacancies"] = st.checkbox(
    "Include oxygen-vacancy structures", section.get("include_oxygen_vacancies", True)
)
selectors = st.text_area(
    "Additional target IDs or wildcard patterns (one per line)",
    "\n".join(section.get("target_include", [])),
)
paths = st.text_area(
    "Additional structure paths (one per line, relative to project root or absolute)",
    "\n".join(section.get("structure_paths", [])),
)
section["target_include"] = [s.strip() for s in selectors.splitlines() if s.strip()]
section["structure_paths"] = [s.strip() for s in paths.splitlines() if s.strip()]
with st.expander("Available structure IDs"):
    try:
        preview = {**section, "selection": "all"}
        cfg, _ = parse_config(
            {
                **raw,
                "conductivity": {**preview, "dft": {**section.get("dft", {}), "kpts": [4, 4, 4]}},
            },
            root,
        )
        targets, _ = discover_targets(cfg)
        st.dataframe(
            pd.DataFrame(
                [
                    {"target_id": t.target_id, "kind": t.kind, "path": str(t.structure_path)}
                    for t in targets
                ]
            ),
            hide_index=True,
        )
    except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
        st.info(str(exc))
st.subheader("Transport settings")
temps = st.text_input(
    "Temperatures (K, comma separated)", ", ".join(map(str, section.get("temperatures_K", [300])))
)
excess = st.text_input(
    "Excess electrons (cm⁻³, comma separated)",
    ", ".join(map(str, section.get("excess_electrons_cm3", [0]))),
)
st.caption(
    "Zero keeps the explicit structure's electron count. Positive adds electrons; negative adds holes. This does not predict defect ionization or mobile carrier concentration."
)
use_tau = st.checkbox(
    "Also estimate conductivity using an assumed relaxation time",
    section.get("relaxation_time_fs") is not None,
)
if use_tau:
    section["relaxation_time_fs"] = st.number_input(
        "Assumed relaxation time (fs)",
        min_value=0.001,
        value=float(section.get("relaxation_time_fs", 10.0)),
    )
else:
    section.pop("relaxation_time_fs", None)
section["interpolation_factor"] = int(
    st.number_input(
        "Interpolation factor", min_value=2, value=int(section.get("interpolation_factor", 5))
    )
)
section["dos_points"] = int(
    st.number_input(
        "DOS integration points", min_value=100, value=int(section.get("dos_points", 4000))
    )
)
st.subheader("GPAW and shared results")
dft = dict(section.get("dft", {}) or {})
dft["execute"] = st.checkbox(
    "Run GPAW when compatible results are missing", dft.get("execute", False)
)
kpts = st.text_input("Uniform k-point mesh", ", ".join(map(str, dft.get("kpts", [4, 4, 4]))))
dft["save_wavefunctions"] = st.checkbox(
    "Save wavefunctions for later oxidation analysis", dft.get("save_wavefunctions", True)
)
st.caption(
    "Other DFT settings inherit from oxidation.dft_electronic; overrides can be supplied in conductivity.dft in input.toml. Exact geometry and electronic settings must match for reuse. A denser mesh requires a different calculation."
)
st.warning(
    "These results assume band-like transport. Check electron localization before interpreting them. Small-polaron hopping, scattering lifetimes and grain boundaries are not modeled."
)
try:
    section["temperatures_K"] = [float(x.strip()) for x in temps.split(",")]
    section["excess_electrons_cm3"] = [float(x.strip()) for x in excess.split(",")]
    dft["kpts"] = [int(x.strip()) for x in kpts.split(",")]
    section["dft"] = dft
    updated = {**raw, "conductivity": section}
    cfg, validated = parse_config(updated, root)
except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
    st.error(str(exc))
    st.stop()
if st.button("Preview selected structures"):
    try:
        chosen, warnings = select_targets(cfg, validated)
        st.dataframe(
            pd.DataFrame([{"target_id": t.target_id, **t.metadata} for t in chosen]),
            hide_index=True,
        )
        for warning in warnings:
            st.warning(warning)
    except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
        st.error(str(exc))
if st.button("Save settings"):
    path.write_text(toml.dumps(updated))
    st.success("Settings saved")
if st.button("Save and run conductivity", disabled=not section["enabled"]):
    path.write_text(toml.dumps(updated))
    with st.spinner("Running conductivity; GPAW may take a long time"):
        result = subprocess.run(
            [sys.executable, "-m", "dopingflow", "conductivity", "-c", str(path)],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
    st.code(result.stdout + result.stderr)
    if result.returncode:
        st.error("Some targets did not complete. Inspect the reported errors.")
    else:
        st.success("Conductivity analysis completed")
output = cfg.output_dir / "conductivity_results.json"
if output.exists():
    payload = json.loads(output.read_text())
    st.subheader("Results")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "target_id": r["target_id"],
                    "status": r["status"],
                    "DFT reused": r.get("dft_reused"),
                    "error": r.get("error"),
                }
                for r in payload.get("results", [])
            ]
        ),
        hide_index=True,
    )
    for row in payload.get("results", []):
        if row.get("rows"):
            with st.expander(row["target_id"]):
                st.dataframe(pd.DataFrame(row["rows"]), hide_index=True)
    st.download_button("Download results JSON", output.read_bytes(), file_name=output.name)
    csv = cfg.output_dir / "conductivity.csv"
    if csv.exists():
        st.download_button("Download transport CSV", csv.read_bytes(), file_name=csv.name)
