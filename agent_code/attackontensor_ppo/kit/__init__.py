# --------------------------------------------------------------------------
# GENERATED FILE -- DO NOT EDIT.
# Vendored from shared/kit/__init__.py by tools/sync_kit.py.
# Edit the original, then re-run:  python tools/sync_kit.py
# --------------------------------------------------------------------------
"""Inference-critical game utilities shared by both AttackOnTensor agents.

``shared/kit`` is the single source of truth. The graders copy exactly one
agent directory into their framework, so each agent has to be self-contained:
this package is vendored into both by ``tools/sync_kit.py`` instead of being
imported from the repository root. ``tools/sync_kit.py --check`` runs as a unit
test and fails if a vendored copy has drifted.

Edit the files here, never the copies under ``agent_code/*/kit/``.
"""

from . import actions, geometry, pathfind, safety, symmetry  # noqa: F401

__all__ = ["actions", "geometry", "pathfind", "safety", "symmetry"]
