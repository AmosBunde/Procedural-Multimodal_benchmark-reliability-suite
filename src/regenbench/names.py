"""Validation of names that end up in file paths.

Suite ids, run ids and model keys are read from the command line and from
manifest and result files, and are then used to build output paths. Anything
that is not a plain token is rejected so a crafted name cannot escape the
output directory.
"""

from __future__ import annotations

import re

_NAME = re.compile(r"[A-Za-z0-9._-]+")


def validate_name(name: object, what: str = "name") -> str:
    """Return ``name`` if it is a safe path component, otherwise raise ValueError."""
    if not isinstance(name, str) or _NAME.fullmatch(name) is None:
        raise ValueError(f"invalid {what}: {name!r} (allowed: letters, digits, '.', '_', '-')")
    if name in {".", ".."}:
        raise ValueError(f"invalid {what}: {name!r}")
    return name
