"""Optional parent-selection filter for staged vacancy Monte Carlo workflows.

The staged GRACE-search/MACE-finalize commands normally process every parent
returned by ``discover_selected_parents``.  ``[vacancies].parent_include`` lets a
study name only the desired compositions or exact parent IDs without copying or
symlinking directories.

Composition selectors are order/notation insensitive for the common dopingflow
labels.  For example ``Ti_2.5Sb_5``, ``Ti2p5_Sb5`` and ``Sb5_Ti2p5`` resolve to
the same composition.  A selector containing ``/`` is treated as an exact parent
ID such as ``Sb5_Ti2p5/candidate_003``.
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


def _canonical_composition(label: str) -> tuple[tuple[str, Decimal], ...] | None:
    """Canonicalize common composition labels independently of element order."""

    text = str(label).strip()
    if not text or "/" in text:
        return None
    matches = list(_TOKEN_RE.finditer(text))
    if not matches:
        return None

    # Only separators may remain outside recognized element/percentage tokens.
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


def _run_with_parent_filter(
    original: Callable[..., Any],
    raw: dict[str, Any],
    root,
    *,
    config_path=None,
):
    selectors = parse_parent_include(raw.get("vacancies") or {})
    if selectors is None:
        return original(raw, root, config_path=config_path)

    discover_original = _base.discover_selected_parents

    def discover_filtered(parent_root):
        return filter_selected_parents(discover_original(parent_root), selectors)

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
    "run_vacancy_finalize",
    "run_vacancy_mc_search",
]
