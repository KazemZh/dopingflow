from __future__ import annotations

import contextlib
import hashlib
import importlib
import sys
from pathlib import Path
from typing import Any, Iterator

from pymatgen.core import Composition, Structure

from dopingflow.oxidation import (
    OptionalMethodUnavailable,
    OxidationConfig,
    StructureTarget,
    base_method_result,
    cached_model,
    site_records,
)

TOSS_VERIFIED_COMMIT = "c45582a3cd3088480b5d83b1440d360bd4577b80"
TOSS_DEFAULT_LP = "models/pyg_Hetero_GCN_s_0608.pth"
TOSS_DEFAULT_NC = "models/pyg_GCN_s_0609.pth"
CHGNET_MAX_ATOMIC_NUMBER = 94
CHGNET_UPSTREAM_MN_RANGES = (
    (0.5, 1.5, 2),
    (1.5, 2.5, 3),
    (2.5, 3.5, 4),
    (3.5, 4.2, 3),
    (4.2, 5.0, 2),
)


@contextlib.contextmanager
def _prepend_sys_path(path: Path) -> Iterator[None]:
    text = str(path)
    sys.path.insert(0, text)
    try:
        yield
    finally:
        try:
            sys.path.remove(text)
        except ValueError:
            pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repo(settings: dict[str, Any], *, name: str, url: str) -> Path:
    value = str(settings.get("repo_path") or "").strip()
    if not value:
        raise OptionalMethodUnavailable(
            f"{name} requires repo_path pointing to a checkout of {url}."
        )
    repo = Path(value).expanduser().resolve()
    if not repo.is_dir():
        raise OptionalMethodUnavailable(f"{name} repo_path does not exist: {repo}")
    return repo


def _resolve_checkpoint(repo: Path, value: Any, default: str, *, label: str) -> Path:
    raw = str(value or default).strip()
    path = Path(raw).expanduser()
    path = path if path.is_absolute() else repo / path
    path = path.resolve()
    if not path.is_file():
        raise OptionalMethodUnavailable(f"TOSS-GNN {label} checkpoint not found: {path}")
    return path


def _toss_supported_elements(repo: Path) -> set[str]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise OptionalMethodUnavailable(
            "TOSS-GNN requires pandas/openpyxl to read upstream pre_set.xlsx."
        ) from exc
    preset = repo / "pre_set.xlsx"
    if not preset.is_file():
        raise OptionalMethodUnavailable(f"TOSS pre_set.xlsx not found: {preset}")
    try:
        frame = pd.read_excel(preset, sheet_name="Radii_X")
        return {str(item).strip() for item in frame["symbol"].dropna().tolist()}
    except Exception as exc:
        raise OptionalMethodUnavailable(f"Could not read TOSS element presets: {exc}") from exc


def _load_toss_gnn_models(
    repo: Path,
    lp_checkpoint: Path,
    nc_checkpoint: Path,
    device: str,
) -> tuple[Any, Any, Any]:
    # The verified upstream Predict.py converts tensors directly with
    # .detach().numpy() and builds CPU graph tensors. Running that interface on
    # CUDA would require changing upstream inference semantics. Keep this
    # adapter explicit rather than pretending a CUDA selection was honored.
    if device != "cpu":
        raise OptionalMethodUnavailable(
            "The verified upstream TOSS-GNN Predict.py interface is CPU-only in this adapter "
            "because it converts prediction tensors directly with .detach().numpy(). "
            "Set [oxidation.toss_gnn].device='cpu'."
        )
    try:
        import torch
    except ImportError as exc:
        raise OptionalMethodUnavailable("TOSS-GNN requires PyTorch.") from exc

    gnn_dir = repo / "toss_GNN"
    required = [gnn_dir / "Predict.py", gnn_dir / "model_utils_pyg.py", repo / "pre_set.xlsx"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise OptionalMethodUnavailable(
            "TOSS-GNN checkout is incomplete; missing: " + ", ".join(missing)
        )
    try:
        with _prepend_sys_path(gnn_dir):
            models = importlib.import_module("model_utils_pyg")
            predict = importlib.import_module("Predict")
        lp_model = models.pyg_Hetero_GCNPredictor(
            atom_feats=13,
            bond_feats=13,
            hidden_feats=[256, 256, 256, 256],
            predictor_hidden_feats=64,
            n_tasks=2,
            predictor_dropout=0.3,
        )
        nc_model = models.pyg_GCNPredictor(
            in_feats=15,
            hidden_feats=[256, 256, 256, 256],
            predictor_hidden_feats=64,
            n_tasks=12,
            predictor_dropout=0.3,
        )
        lp_state = torch.load(lp_checkpoint, map_location="cpu")
        nc_state = torch.load(nc_checkpoint, map_location="cpu")
        lp_model.load_state_dict(lp_state)
        nc_model.load_state_dict(nc_state)
        lp_model.eval()
        nc_model.eval()
        return lp_model, nc_model, predict.Get_OS_by_models
    except Exception as exc:
        raise OptionalMethodUnavailable(
            "Could not load the verified TOSS-GNN PyG inference stack. Install the upstream "
            f"TOSS requirements in this environment. Original error: {type(exc).__name__}: {exc}"
        ) from exc


def _run_toss_gnn(target: StructureTarget, settings: dict[str, Any]) -> dict[str, Any]:
    repo = _resolve_repo(
        settings,
        name="TOSS-GNN",
        url="https://github.com/yueyin19960520/TOSS",
    )
    device = str(settings.get("device", "cpu")).strip().lower()
    if device not in {"cpu", "cuda"}:
        raise ValueError("[oxidation.toss_gnn].device must be 'cpu' or 'cuda'")
    lp_checkpoint = _resolve_checkpoint(
        repo,
        settings.get("lp_checkpoint"),
        TOSS_DEFAULT_LP,
        label="link-prediction",
    )
    nc_checkpoint = _resolve_checkpoint(
        repo,
        settings.get("nc_checkpoint"),
        TOSS_DEFAULT_NC,
        label="node-classification",
    )
    structure = Structure.from_file(target.structure_path)
    supported = _toss_supported_elements(repo)
    unsupported = sorted({site.specie.symbol for site in structure if site.specie.symbol not in supported})
    if unsupported:
        return base_method_result(
            method="toss-gnn",
            target=target,
            scope="site-resolved",
            status="unsupported",
            provenance={
                "upstream": "yueyin19960520/TOSS",
                "verified_interface_commit": TOSS_VERIFIED_COMMIT,
                "repo_path": str(repo),
                "lp_checkpoint": str(lp_checkpoint),
                "nc_checkpoint": str(nc_checkpoint),
            },
            limitations=[f"Unsupported elements in upstream TOSS presets: {', '.join(unsupported)}"],
        )

    cache_key = (
        "toss-gnn",
        str(repo),
        str(lp_checkpoint),
        _sha256(lp_checkpoint),
        str(nc_checkpoint),
        _sha256(nc_checkpoint),
        device,
    )
    lp_model, nc_model, predictor_cls = cached_model(
        cache_key,
        lambda: _load_toss_gnn_models(repo, lp_checkpoint, nc_checkpoint, device),
    )
    try:
        predictor = predictor_cls(
            target.structure_path.name,
            lp_model,
            nc_model,
            server=True,
            filepath=str(target.structure_path),
        )
        predicted = predictor.NC_predict()
        elements = [str(item) for item in predicted.loc["Elements"].tolist()]
        values = [int(item) for item in predicted.loc["Valence"].tolist()]
        coordination_values = [int(item) for item in predicted.loc["Coordination Number"].tolist()]
    except Exception as exc:
        raise RuntimeError(f"TOSS-GNN inference failed: {exc}") from exc

    input_elements = [site.specie.symbol for site in structure]
    if elements != input_elements:
        raise RuntimeError(
            "TOSS-GNN returned a site order that does not match the original relaxed structure"
        )
    if len(values) != len(structure) or len(coordination_values) != len(structure):
        raise RuntimeError("TOSS-GNN returned an unexpected number of site predictions")

    coordination = [
        {
            "site_index": index,
            "element": input_elements[index],
            "coordination_number": coordination_values[index],
        }
        for index in range(len(structure))
    ]
    return base_method_result(
        method="toss-gnn",
        target=target,
        scope="site-resolved",
        status="assigned",
        formal_oxidation_states=site_records(structure, values),
        coordination=coordination,
        provenance={
            "upstream": "yueyin19960520/TOSS",
            "interface": "toss_GNN.Predict.Get_OS_by_models(...).NC_predict()",
            "verified_interface_commit": TOSS_VERIFIED_COMMIT,
            "repo_path": str(repo),
            "device": device,
            "lp_checkpoint": str(lp_checkpoint),
            "lp_checkpoint_sha256": _sha256(lp_checkpoint),
            "nc_checkpoint": str(nc_checkpoint),
            "nc_checkpoint_sha256": _sha256(nc_checkpoint),
        },
        limitations=[
            "TOSS-GNN is distinct from conventional Bayesian/MAP TOSS.",
            "The verified upstream NC_predict interface returns oxidation-state classes and coordination numbers but does not expose calibrated probabilities; no confidence values are invented here.",
            "Published broad-database performance does not establish accuracy for defective co-doped oxides; validate the target chemistry independently.",
        ],
    )


def _parse_moment_ranges(settings: dict[str, Any]) -> dict[str, list[tuple[float, float, int]]]:
    raw = settings.get("moment_oxidation_ranges", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("[oxidation.chgnet].moment_oxidation_ranges must be a table/dictionary")
    parsed: dict[str, list[tuple[float, float, int]]] = {}
    for element, entries in raw.items():
        if not isinstance(entries, list):
            raise ValueError(f"CHGNet moment ranges for {element} must be an array")
        ranges: list[tuple[float, float, int]] = []
        for entry in entries:
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                raise ValueError(
                    f"CHGNet moment range for {element} must be [min, max, oxidation_state]"
                )
            lower, upper, oxidation = float(entry[0]), float(entry[1]), int(entry[2])
            if upper <= lower:
                raise ValueError(f"CHGNet moment range for {element} has max <= min")
            ranges.append((lower, upper, oxidation))
        parsed[str(element)] = ranges
    if bool(settings.get("use_upstream_mn_mapping", True)) and "Mn" not in parsed:
        parsed["Mn"] = list(CHGNET_UPSTREAM_MN_RANGES)
    return parsed


def _run_chgnet(target: StructureTarget, settings: dict[str, Any]) -> dict[str, Any]:
    try:
        import chgnet
        from chgnet.model.model import CHGNet
    except ImportError as exc:
        raise OptionalMethodUnavailable(
            "CHGNet analysis requires the optional 'chgnet' package."
        ) from exc

    device = str(settings.get("device", "cpu")).strip().lower()
    model_name = str(settings.get("model_name", "0.3.0")).strip()
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("[oxidation.chgnet].device must be cpu, cuda, or mps")

    structure = Structure.from_file(target.structure_path)
    unsupported = sorted(
        {
            site.specie.symbol
            for site in structure
            if int(site.specie.Z) > CHGNET_MAX_ATOMIC_NUMBER
        }
    )
    if unsupported:
        return base_method_result(
            method="chgnet",
            target=target,
            scope="site-resolved",
            status="unsupported",
            provenance={
                "implementation": "chgnet.model.model.CHGNet.predict_structure",
                "model_name": model_name,
                "requested_device": device,
                "verified_atom_embedding_max_atomic_number": CHGNET_MAX_ATOMIC_NUMBER,
            },
            limitations=[
                "The upstream CHGNet AtomEmbedding default supports atomic numbers up to 94; "
                "unsupported elements present: " + ", ".join(unsupported)
            ],
        )

    cache_key = ("chgnet", model_name, device)

    def _load() -> Any:
        try:
            return CHGNet.load(model_name=model_name, use_device=device)
        except TypeError:
            # Compatibility with older CHGNet releases that did not expose
            # use_device in load(). The model still performs its own device
            # selection; provenance records the requested value.
            return CHGNet.load(model_name=model_name)

    model = cached_model(cache_key, _load)
    try:
        prediction = model.predict_structure(structure)
    except Exception as exc:
        raise RuntimeError(f"CHGNet predict_structure failed: {exc}") from exc

    moments_raw = prediction.get("m") if isinstance(prediction, dict) else None
    if moments_raw is None:
        raise RuntimeError("CHGNet prediction did not contain site magnetic moments under key 'm'")
    moments = [float(value) for value in moments_raw]
    if len(moments) != len(structure):
        raise RuntimeError(
            f"CHGNet returned {len(moments)} moments for a {len(structure)}-site structure"
        )

    moment_records = [
        {
            "site_index": index,
            "element": site.specie.symbol,
            "magnetic_moment": moments[index],
            "descriptor": "chgnet_predicted_magnetic_moment",
        }
        for index, site in enumerate(structure)
    ]
    ranges = _parse_moment_ranges(settings)
    use_absolute_moment = bool(settings.get("use_absolute_moment", False))
    inferred: list[int | None] = []
    ambiguous_elements: set[str] = set()
    for site, moment in zip(structure, moments):
        element = site.specie.symbol
        mapped_moment = abs(moment) if use_absolute_moment else moment
        value: int | None = None
        for lower, upper, oxidation in ranges.get(element, []):
            if lower <= mapped_moment < upper:
                value = oxidation
                break
        if value is None:
            ambiguous_elements.add(element)
        inferred.append(value)

    n_assigned = sum(value is not None for value in inferred)
    status = "assigned" if n_assigned == len(structure) else (
        "partial" if n_assigned else "descriptors-only"
    )
    limitations = [
        "CHGNet directly predicts local magnetic moments; any formal oxidation state in this "
        "result is an inference from an explicit moment-to-state mapping, not a native universal "
        "CHGNet oxidation-state output.",
        "Near-zero magnetic moments cannot reliably distinguish closed-shell alternatives such as "
        "Sn2+/Sn4+ or Sb3+/Sb5+; those sites are deliberately not forced into an oxidation state.",
    ]
    if ambiguous_elements:
        limitations.append(
            "No unambiguous configured magnetic-moment mapping was available for: "
            + ", ".join(sorted(ambiguous_elements))
        )
    if bool(settings.get("use_upstream_mn_mapping", True)):
        limitations.append(
            "The default Mn ranges are the ranges used by CHGNet's upstream solve_charge_by_mag "
            "helper. By default they are applied to the raw predicted moment, matching upstream."
        )
    if use_absolute_moment:
        limitations.append(
            "use_absolute_moment=true was explicitly requested; this differs from the upstream "
            "solve_charge_by_mag comparison, which uses the raw magnetic moment."
        )

    return base_method_result(
        method="chgnet",
        target=target,
        scope="site-resolved",
        status=status,
        formal_oxidation_states=site_records(
            structure,
            inferred,
            status="inferred-from-moment",
        ),
        magnetic_moments=moment_records,
        provenance={
            "implementation": "chgnet.model.model.CHGNet.predict_structure",
            "chgnet_package_version": getattr(chgnet, "__version__", None),
            "model_name": model_name,
            "model_version": getattr(model, "version", None),
            "requested_device": device,
            "verified_atom_embedding_max_atomic_number": CHGNET_MAX_ATOMIC_NUMBER,
            "use_absolute_moment": use_absolute_moment,
            "moment_oxidation_ranges": {
                element: [list(item) for item in entries] for element, entries in ranges.items()
            },
        },
        limitations=limitations,
    )


def _resolve_bertos_paths(settings: dict[str, Any]) -> tuple[Path, Path, Path]:
    repo = _resolve_repo(
        settings,
        name="BERTOS",
        url="https://github.com/usccolumbia/BERTOS",
    )
    model_raw = str(settings.get("model_path") or "trained_models/ICSD_CN").strip()
    tokenizer_raw = str(settings.get("tokenizer_path") or "tokenizer").strip()
    model_path = Path(model_raw).expanduser()
    tokenizer_path = Path(tokenizer_raw).expanduser()
    model_path = (model_path if model_path.is_absolute() else repo / model_path).resolve()
    tokenizer_path = (
        tokenizer_path if tokenizer_path.is_absolute() else repo / tokenizer_path
    ).resolve()
    if not model_path.is_dir():
        archive = model_path.with_suffix(".zip")
        hint = f" Extract {archive} first." if archive.is_file() else ""
        raise OptionalMethodUnavailable(
            f"BERTOS pretrained model directory not found: {model_path}.{hint}"
        )
    if not tokenizer_path.is_dir():
        raise OptionalMethodUnavailable(f"BERTOS tokenizer directory not found: {tokenizer_path}")
    return repo, model_path, tokenizer_path


def _load_bertos(model_path: Path, tokenizer_path: Path, device: str) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoConfig, AutoModelForTokenClassification, BertTokenizerFast
    except ImportError as exc:
        raise OptionalMethodUnavailable(
            "BERTOS requires PyTorch and transformers."
        ) from exc
    try:
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path, do_lower_case=False)
        config = AutoConfig.from_pretrained(model_path, num_labels=14)
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            config=config,
            ignore_mismatched_sizes=True,
        )
        model.to(device)
        model.eval()
        return model, tokenizer, torch
    except Exception as exc:
        raise OptionalMethodUnavailable(f"Could not load BERTOS checkpoint/tokenizer: {exc}") from exc


def _run_bertos(target: StructureTarget, settings: dict[str, Any]) -> dict[str, Any]:
    repo, model_path, tokenizer_path = _resolve_bertos_paths(settings)
    device = str(settings.get("device", "cpu")).strip().lower()
    if device not in {"cpu", "cuda"}:
        raise ValueError("[oxidation.bertos].device must be 'cpu' or 'cuda'")
    model, tokenizer, torch = cached_model(
        ("bertos", str(model_path), str(tokenizer_path), device),
        lambda: _load_bertos(model_path, tokenizer_path, device),
    )

    structure = Structure.from_file(target.structure_path)
    formula = structure.composition.reduced_formula
    comp_dict = Composition(formula).to_reduced_dict
    tokens: list[str] = []
    for element, count in comp_dict.items():
        # This mirrors the upstream getOS.py inference interface: reduced
        # composition is expanded into repeated element tokens. These are not
        # crystallographic site identifiers.
        tokens.extend([str(element)] * int(count))
    if not tokens:
        raise RuntimeError(f"BERTOS could not construct composition tokens for {formula}")
    input_seq = " ".join(tokens) + " "
    encoded = tokenizer.encode(input_seq, add_special_tokens=True)
    input_ids = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predictions = outputs.logits.argmax(dim=-1)
    predicted = predictions[0][1:-1].detach().cpu().tolist()
    maxima = probabilities[0][1:-1].max(dim=-1).values.detach().cpu().tolist()
    if len(predicted) != len(tokens):
        raise RuntimeError(
            "BERTOS token output length does not match its reduced-composition input sequence"
        )

    formal = []
    scores = []
    for index, (element, label, score) in enumerate(zip(tokens, predicted, maxima)):
        oxidation = int(label) - 5
        formal.append(
            {
                "composition_token_index": index,
                "element": element,
                "formal_oxidation_state": oxidation,
                "assignment_status": "composition-token-prediction",
            }
        )
        scores.append(
            {
                "composition_token_index": index,
                "element": element,
                "score": float(score),
                "score_kind": "model_softmax_max_probability_uncalibrated",
            }
        )

    return base_method_result(
        method="bertos",
        target=target,
        scope="composition-level",
        status="assigned",
        formal_oxidation_states=formal,
        method_scores=scores,
        provenance={
            "upstream": "usccolumbia/BERTOS",
            "interface": "getOS.py-equivalent pretrained token-classification inference",
            "repo_path": str(repo),
            "model_path": str(model_path),
            "tokenizer_path": str(tokenizer_path),
            "device": device,
            "reduced_formula": formula,
            "composition_tokens": tokens,
        },
        limitations=[
            "BERTOS is composition-based. Its repeated element tokens are stoichiometric "
            "composition tokens, not crystallographic sites, so no site mapping is invented.",
            "Structures with the same composition but different oxygen-vacancy arrangements are "
            "indistinguishable to this composition-only model.",
            "BERTOS therefore cannot identify which specific atom is reduced near a vacancy.",
            "Reported scores are the model's own maximum softmax probabilities; they are not "
            "presented as calibrated physical confidence.",
        ],
    )


def run_ml_method(
    method: str,
    target: StructureTarget,
    cfg: OxidationConfig,
    settings: dict[str, Any],
) -> dict[str, Any]:
    del cfg
    if method == "toss-gnn":
        return _run_toss_gnn(target, settings)
    if method == "chgnet":
        return _run_chgnet(target, settings)
    if method == "bertos":
        return _run_bertos(target, settings)
    raise ValueError(f"Unknown ML oxidation method: {method}")
