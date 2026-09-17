"""Pure helpers used by the staged GRACE -> MACE vacancy GUI page."""

from __future__ import annotations

from pathlib import Path


_STAGE_COMMANDS = {
    "search": "vacancies-mc-search",
    "finalize": "vacancies-finalize",
}


def parse_positive_int_list(text: str, *, field_name: str = "values") -> list[int]:
    """Parse comma/space/newline separated positive integers without duplicates."""

    tokens = str(text).replace(",", " ").split()
    if not tokens:
        raise ValueError(f"{field_name} must contain at least one positive integer")
    values: list[int] = []
    for token in tokens:
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"{field_name} contains a non-integer value: {token!r}") from exc
        if value <= 0:
            raise ValueError(f"{field_name} values must be positive integers")
        if value not in values:
            values.append(value)
    return values


def parse_parent_selectors(text: str) -> list[str]:
    """Parse one parent selector per line, ignoring blank lines and duplicates."""

    selectors: list[str] = []
    for raw in str(text).splitlines():
        value = raw.strip()
        if value and value not in selectors:
            selectors.append(value)
    return selectors


def build_staged_command(
    stage: str,
    *,
    env_name: str,
    config_path: str | Path = "input.toml",
    conda_executable: str = "conda",
    verbose: bool = True,
) -> list[str]:
    """Build a ``conda run`` command for one staged vacancy entry point."""

    key = str(stage).strip().lower()
    if key not in _STAGE_COMMANDS:
        raise ValueError("stage must be 'search' or 'finalize'")
    env = str(env_name).strip()
    if not env:
        raise ValueError("env_name must be non-empty")
    executable = str(conda_executable).strip()
    if not executable:
        raise ValueError("conda_executable must be non-empty")

    command = [
        executable,
        "run",
        "-n",
        env,
        "dopingflow",
        _STAGE_COMMANDS[key],
        "-c",
        str(config_path),
    ]
    if verbose:
        command.append("--verbose")
    return command


def format_int_list(values) -> str:
    return ", ".join(str(int(value)) for value in (values or []))


__all__ = [
    "build_staged_command",
    "format_int_list",
    "parse_parent_selectors",
    "parse_positive_int_list",
]
