"""Optional parent selection for staged vacancy Monte Carlo workflows.

``[vacancies].parent_include`` limits the staged GRACE-search/MACE-finalize
workflow to named compositions or exact parent IDs. ``parent_pick`` controls
whether all selected candidates are used or only the first (lowest-energy)
filtered candidate for each composition.

DopingFlow's filtering stage writes ``selected_candidates.txt`` in ascending
relaxed-energy order, so the first discovered parent for a composition is its
lowest-energy selected candidate.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import logging
import re
from typing import Any, Callable

from dopingflow import vacancies as _base
from dopingflow import vacancy_mc_staged as _staged

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"([A-Z][a-z]?)_?(\d+(?:[p.]\d+)?)")

_ORIGINAL_SEARCH = getattr(
    _staged, "_parent_selection_original_search", _staged.run_vacancy_mc_search
)
_ORIGINAL_FINALIZE = getattr(
    _staged, "_parent_selection_original_finalize", _staged.run_vacancy_finalize
)


def parse_parent_include(section: dict[str, Any]) -> tuple[str, ...] | None:
    """Return validated ``parent_include`` selectors, or ``None`` for all parents."""

    if "parent_include" not in section:
        return None
    raw = section.get("parent_include")
    if not isinstance(raw, list) or not raw:
        raise ValueError("[vacancies].parent_include must be a non-empty string array")
    selectors: list[str] = []
    for index, value in enumerate(raw):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"[vacancies].parent_include[{index}] must be a non-empty string"
            )
        selectors.append(value.strip())
    if len(set(selectors)) != len(selectors):
        raise ValueError("[vacancies].parent_include must not contain duplicates")
    return tuple(selectors)


def parse_parent_pick(section: dict[str, Any]) -> str:
    """Return parent-selection mode: all or one lowest-energy parent/composition."""

    raw = str(section.get("parent_pick", "all")).strip().lower().replace("-", "_")
    aliases = {
        "all": "all",
        "lowest": "lowest_energy",
        "lowest_energy": "lowest_energy",
        "first": "lowest_energy",
        "first_selected": "lowest_energy",
    }
    if raw not in aliases:
        raise ValueError(
            "[vacancies].parent_pick must be 'all' or 'lowest_energy'"
        )
    return aliases[raw]


def _canonical_composition(label: str) -> tuple[tuple[str, Decimal], ...] | None:
    """Canonicalize common composition labels independently of element order."""

    text = str(label).strip()
    if not text or "/" in text:
        return None
    matches = list(_TOKEN_RE.finditer(text))
    if not matches:
        return None
    remainder = _TOKEN_RE.sub("", text).replace("_", "").replace("-", "")
    if remainder:
        return None

    values: dict[str, Decimal] = {}
    for match in matches:
        element = match.group(1)
        number = match.group(2).replace("p", ".")
        try:
            amount = Decimal(number).normalize()
        except InvalidOperation:
            return None
        if element in values:
            return None
        values[element] = amount
    return tuple(sorted(values.items()))


def _selector_matches_parent(selector: str, parent: dict[str, Any]) -> bool:
    parent_id = str(parent.get("parent_id", "")).strip()
    composition = str(parent.get("composition", "")).strip()
    candidate = str(parent.get("candidate", "")).strip()
    exact_id = f"{composition}/{candidate}" if composition and candidate else ""

    if "/" in selector:
        return selector == parent_id or selector == exact_id
    if selector == composition:
        return True
    selector_comp = _canonical_composition(selector)
    parent_comp = _canonical_composition(composition)
    return selector_comp is not None and selector_comp == parent_comp


def filter_selected_parents(
    parents: list[dict[str, Any]], selectors: tuple[str, ...] | None
) -> list[dict[str, Any]]:
    """Filter discovered parents and fail clearly when a requested selector is absent."""

    if selectors is None:
        return parents

    matched_by_selector: dict[str, list[dict[str, Any]]] = {s: [] for s in selectors}
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for parent in parents:
        matching = [s for s in selectors if _selector_matches_parent(s, parent)]
        if not matching:
            continue
        for selector in matching:
            matched_by_selector[selector].append(parent)
        parent_id = str(parent.get("parent_id", ""))
        if parent_id not in seen_ids:
            selected.append(parent)
            seen_ids.add(parent_id)

    missing = [s for s, matches in matched_by_selector.items() if not matches]
    if missing:
        available = sorted({str(parent.get("composition", "")) for parent in parents})
        raise ValueError(
            "[vacancies].parent_include contains selector(s) that matched no discovered "
            f"parent: {missing}. Available composition directories: {available}"
        )
    if not selected:
        raise ValueError("[vacancies].parent_include selected no vacancy parents")

    log.info(
        "Vacancy parent filter selected %d/%d discovered parent(s): %s",
        len(selected),
        len(parents),
        ", ".join(selectors),
    )
    return selected


def pick_parents(
    parents: list[dict[str, Any]], mode: str
) -> list[dict[str, Any]]:
    """Optionally keep only the lowest-energy selected parent per composition.

    ``discover_selected_parents`` preserves the line order of each
    ``selected_candidates.txt`` file. The filtering stage writes those lines in
    ascending relaxed-energy order, hence the first parent encountered for a
    composition is the lowest-energy filtered candidate.
    """

    if mode == "all":
        return parents
    if mode != "lowest_energy":
        raise ValueError(f"Unsupported parent_pick mode: {mode}")

    chosen: list[dict[str, Any]] = []
    seen_compositions: set[str] = set()
    for parent in parents:
        composition = str(parent.get("composition", ""))
        if composition in seen_compositions:
            continue
        chosen.append(parent)
        seen_compositions.add(composition)

    log.info(
        "Vacancy parent_pick=lowest_energy reduced %d selected parent(s) to %d "
        "composition representative(s)",
        len(parents),
        len(chosen),
    )
    return chosen


def _run_with_parent_filter(
    original: Callable[..., Any],
    raw: dict[str, Any],
    root,
    *,
    config_path=None,
):
    section = raw.get("vacancies") or {}
    selectors = parse_parent_include(section)
    pick_mode = parse_parent_pick(section)
    if selectors is None and pick_mode == "all":
        return original(raw, root, config_path=config_path)

    discover_original = _base.discover_selected_parents

    def discover_filtered(parent_root):
        parents = discover_original(parent_root)
        parents = filter_selected_parents(parents, selectors)
        return pick_parents(parents, pick_mode)

    _base.discover_selected_parents = discover_filtered
    try:
        return original(raw, root, config_path=config_path)
    finally:
        _base.discover_selected_parents = discover_original


def run_vacancy_mc_search(raw, root, *, config_path=None):
    return _run_with_parent_filter(
        _ORIGINAL_SEARCH, raw, root, config_path=config_path
    )


def run_vacancy_finalize(raw, root, *, config_path=None):
    return _run_with_parent_filter(
        _ORIGINAL_FINALIZE, raw, root, config_path=config_path
    )


def install_extensions() -> None:
    if getattr(_staged, "_parent_selection_extension_installed", False):
        return
    _staged._parent_selection_original_search = _ORIGINAL_SEARCH
    _staged._parent_selection_original_finalize = _ORIGINAL_FINALIZE
    _staged.run_vacancy_mc_search = run_vacancy_mc_search
    _staged.run_vacancy_finalize = run_vacancy_finalize
    _staged._parent_selection_extension_installed = True


install_extensions()


__all__ = [
    "filter_selected_parents",
    "install_extensions",
    "parse_parent_include",
    "parse_parent_pick",
    "pick_parents",
    "run_vacancy_finalize",
    "run_vacancy_mc_search",
]
