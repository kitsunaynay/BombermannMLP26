"""Experiment tracking sinks.

Local CSV and JSON are the source of truth and are always written. Weights &
Biases is optional, behind a guarded import and a flag, because a training run
must not die because a laptop is offline or a package is missing -- and because
every figure in the report has to be reproducible from files in the repository,
not from a cloud dashboard the graders cannot open.

A tracking failure never propagates. Losing a metric row is an annoyance;
losing a training run to it is not acceptable.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

#: Rows buffered before the CSV is rewritten.
FLUSH_EVERY = 50


class MetricLogger:
    """Interface for a metric sink."""

    def log_config(self, config: Dict[str, Any]) -> None: ...

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class CsvJsonLogger(MetricLogger):
    """Append-style metric log backed by a CSV, plus a JSON config dump.

    Rows are buffered and the file is rewritten on flush rather than appended
    to. That costs a little I/O but means a metric that only appears later in a
    run (a PPO update statistic, say) still gets a column, instead of being
    silently dropped because the header was fixed by the first row.
    """

    def __init__(self, run_dir: Path, name: str = "metrics"):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / f"{name}.csv"

        self._rows: List[Dict[str, Any]] = []
        self._fieldnames: List[str] = []
        self._since_flush = 0

    def log_config(self, config: Dict[str, Any]) -> None:
        payload = dict(config)
        payload.setdefault("_created", time.strftime("%Y-%m-%d %H:%M:%S"))
        (self.run_dir / "config.json").write_text(json.dumps(payload, indent=2, default=str))

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        row = dict(metrics)
        if step is not None:
            row.setdefault("step", step)

        for key in row:
            if key not in self._fieldnames:
                self._fieldnames.append(key)

        self._rows.append(row)
        self._since_flush += 1
        if self._since_flush >= FLUSH_EVERY:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        with self.path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self._fieldnames, restval="")
            writer.writeheader()
            writer.writerows(self._rows)
        self._since_flush = 0

    def close(self) -> None:
        self.flush()

    @property
    def rows(self) -> List[Dict[str, Any]]:
        return list(self._rows)


class JsonSummaryLogger(MetricLogger):
    """Writes a single summary JSON, for benchmark results rather than curves."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.payload: Dict[str, Any] = {}

    def log_config(self, config: Dict[str, Any]) -> None:
        self.payload["config"] = config

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        self.payload.setdefault("entries", []).append(metrics)

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.payload, indent=2, default=str))

    def close(self) -> None:
        self.flush()


class WandbLogger(MetricLogger):
    """Weights & Biases sink. Degrades to a no-op if wandb is unavailable."""

    def __init__(self, project: str, run_name: str, config: Optional[Dict[str, Any]] = None):
        self.run = None
        try:
            import wandb  # noqa: PLC0415 - optional dependency, imported on demand
        except ImportError:
            self._wandb = None
            return

        self._wandb = wandb
        try:
            self.run = wandb.init(project=project, name=run_name, config=config or {})
        except Exception:  # noqa: BLE001 - offline, bad key, no network: keep training
            self.run = None

    @property
    def active(self) -> bool:
        return self.run is not None

    def log_config(self, config: Dict[str, Any]) -> None:
        if self.active:
            try:
                self.run.config.update(config, allow_val_change=True)
            except Exception:  # noqa: BLE001
                pass

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.active:
            try:
                self._wandb.log(metrics, step=step)
            except Exception:  # noqa: BLE001
                pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        if self.active:
            try:
                self.run.finish()
            except Exception:  # noqa: BLE001
                pass


class MultiLogger(MetricLogger):
    """Fan-out to several sinks; one failing sink never stops the others."""

    def __init__(self, sinks: Sequence[MetricLogger]):
        self.sinks = list(sinks)

    def _each(self, method: str, *args, **kwargs) -> None:
        for sink in self.sinks:
            try:
                getattr(sink, method)(*args, **kwargs)
            except Exception:  # noqa: BLE001 - tracking must never kill a run
                pass

    def log_config(self, config: Dict[str, Any]) -> None:
        self._each("log_config", config)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        self._each("log", metrics, step)

    def flush(self) -> None:
        self._each("flush")

    def close(self) -> None:
        self._each("close")


def make_run_id(prefix: str) -> str:
    """Timestamped, sortable, filesystem-safe run identifier."""
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"


def make_logger(
    run_id: str,
    config: Optional[Dict[str, Any]] = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    use_wandb: bool = False,
    wandb_project: str = "attackontensor-bomberman",
) -> MultiLogger:
    """Standard sink set: CSV/JSON always, W&B on request."""
    run_dir = Path(results_dir) / run_id
    sinks: List[MetricLogger] = [CsvJsonLogger(run_dir)]

    if use_wandb:
        sinks.append(WandbLogger(wandb_project, run_id, config))

    logger = MultiLogger(sinks)
    if config:
        logger.log_config(config)
    return logger
