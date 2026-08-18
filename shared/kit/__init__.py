"""Inference-critical game utilities shared by both AttackOnTensor agents.

``shared/kit`` is the single source of truth. The project brief requires the
submitted agent directory to be self-contained -- the graders copy exactly one
directory containing ``callbacks.py`` into their framework -- so this package is
*vendored* into each agent by ``tools/sync_kit.py`` rather than imported from
the repository root. ``tools/sync_kit.py --check`` (also run as a unit test)
fails if a vendored copy has drifted from this original.

Edit the files here, never the copies under ``agent_code/*/kit/``.
"""

from . import actions, geometry, pathfind, safety, symmetry  # noqa: F401

__all__ = ["actions", "geometry", "pathfind", "safety", "symmetry"]
