"""AttackOnTensor training and evaluation infrastructure.
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
