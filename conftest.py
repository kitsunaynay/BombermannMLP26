"""Pytest bootstrap.

Puts the repository root on ``sys.path`` so tests can import both the framework
modules (``settings``, ``items``, ``environment``) and our own packages
(``shared.kit``, ``blib``) regardless of where pytest is invoked from.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
