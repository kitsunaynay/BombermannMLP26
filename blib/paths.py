"""Path display helpers.

``Path.relative_to`` raises ``ValueError`` when the target is not under the base,
so using it to shorten a path for a log message turns a legitimate
``--output /tmp/assets`` into a crash *after* the work is already done. These
helpers only ever affect how a path is printed.
"""

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
