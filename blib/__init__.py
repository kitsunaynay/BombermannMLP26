"""AttackOnTensor training and evaluation infrastructure.

Everything in this package is *development-only*. The brief has the graders copy
a single agent directory into their own checkout of the framework, so nothing
here exists during a tournament game and no submitted agent module may import
from it. Inference-critical code lives in ``shared/kit`` and is vendored into
each agent instead (see ``tools/sync_kit.py``).

Contents:

``fast_env``    a ``BombeRLeWorld`` subclass driven by injected actions
``opponents``   the provided baseline agents, called in-process
``vec_env``     multiprocessing fan-out over many environments
``tracking``    CSV/JSON metric sinks, with optional Weights & Biases
``metrics``     per-episode statistics and aggregation
``curriculum``  Tasks 1-4 from the brief, expressed as data
``benchmark``   headless evaluation against the baselines over N seeds
``plots``       figures and tables for the report
``seeding``     reproducible seed derivation
"""

__all__ = [
    "benchmark",
    "curriculum",
    "fast_env",
    "metrics",
    "opponents",
    "plots",
    "seeding",
    "tracking",
    "vec_env",
]
