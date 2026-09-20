"""Path display helpers."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def display_path(path, base: Path = REPO_ROOT) -> str:
    """Shorten a path for display, falling back to the absolute form.

    Never raises: a path outside ``base`` is rendered as-is rather than
    producing a wall of ``../..`` segments.
    """
    path = Path(path)
    try:
        return str(path.resolve().relative_to(Path(base).resolve()))
    except ValueError:
        return str(path)
