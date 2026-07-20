"""Self-evolution stub: the agent modifying its own behaviour when eval breaks.

This closes the loop the harness is built to demonstrate:

    detect shift  ->  eval is now invalid  ->  reduce exposure
        ->  re-evaluate  ->  adapt (restore when back in-distribution)

It is deliberately a *stub*: the "evolution" is a single, legible control law —
when a distribution shift fires, shrink position size by ``SHRINK_FACTOR``; when
severity falls back below ``RECOVERY_THRESHOLD``, restore it. The point is not
the sophistication of the response but the *shape* of the loop: an agent that
reacts to the invalidation of its own evaluation is safer than one that keeps
optimizing a number that has quietly stopped meaning anything.

Every self-modification is logged in a structured, auditable format.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .. import config

logger = logging.getLogger(__name__)


@dataclass
class EvolutionEvent:
    """A single self-modification event.

    Attributes:
        timestamp: when the agent changed its own behaviour.
        event: ``"size_reduction"`` or ``"size_restored"``.
        regime: the regime label in force at that time.
        severity: shift severity (normalized KS statistic) that triggered it.
        size_factor: the new position-size factor after the change.
    """

    timestamp: pd.Timestamp
    event: str
    regime: str
    severity: float
    size_factor: float

    def to_log(self) -> dict:
        """Render the event as the structured log record."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "event": self.event,
            "regime": self.regime,
            "severity": round(float(self.severity), 4),
            "size_factor": round(float(self.size_factor), 4),
        }


@dataclass
class EvolutionResult:
    """Output of the self-evolution loop.

    Attributes:
        size_factor: per-timestamp position-size factor the agent chose.
        events: the ordered list of self-modification events.
        shrink_factor: the shrink factor used.
        recovery_threshold: the recovery threshold used.
    """

    size_factor: pd.Series
    events: list[EvolutionEvent] = field(default_factory=list)
    shrink_factor: float = config.SHRINK_FACTOR
    recovery_threshold: float = config.RECOVERY_THRESHOLD

    def events_log(self) -> list[dict]:
        """Return all events as structured log records."""
        return [e.to_log() for e in self.events]

    def events_frame(self) -> pd.DataFrame:
        """Return the events as a DataFrame timeline (empty-safe)."""
        columns = ["timestamp", "event", "regime", "severity", "size_factor"]
        if not self.events:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame([e.to_log() for e in self.events])[columns]


def run_self_evolution(
    shift_frame: pd.DataFrame,
    regimes: pd.Series,
    *,
    shrink_factor: float = config.SHRINK_FACTOR,
    recovery_threshold: float = config.RECOVERY_THRESHOLD,
    base_size: float = 1.0,
    log_path: Optional[Path] = None,
) -> EvolutionResult:
    """Walk the timeline, adjusting position size in response to shifts.

    State machine, evaluated once per timestamp:

    - **not yet reduced** and a shift fires -> shrink to
      ``base_size * shrink_factor`` and log a ``size_reduction``.
    - **currently reduced** and severity falls below ``recovery_threshold``
      -> restore to ``base_size`` and log a ``size_restored``.

    Args:
        shift_frame: output of the shift detector, with columns ``ks_stat`` and
            ``shift_detected`` (indexed by timestamp).
        regimes: per-timestamp regime labels (aligned by index).
        shrink_factor: multiplier applied to size when a shift fires.
        recovery_threshold: severity below which size is restored.
        base_size: the nominal (un-shrunk) position size.
        log_path: if given, append each event as a JSON line to this file.

    Returns:
        A populated :class:`EvolutionResult`.
    """
    regimes = regimes.reindex(shift_frame.index)
    severity = shift_frame["ks_stat"]
    detected = shift_frame["shift_detected"].fillna(False)

    size = base_size
    reduced = False
    sizes: list[float] = []
    events: list[EvolutionEvent] = []

    for ts in shift_frame.index:
        sev = severity.loc[ts]
        regime = regimes.loc[ts]
        regime = str(regime) if pd.notna(regime) else "unknown"

        if not reduced and bool(detected.loc[ts]):
            size = base_size * shrink_factor
            reduced = True
            events.append(EvolutionEvent(ts, "size_reduction", regime, sev, size))
        elif reduced and pd.notna(sev) and sev < recovery_threshold:
            size = base_size
            reduced = False
            events.append(EvolutionEvent(ts, "size_restored", regime, sev, size))

        sizes.append(size)

    size_series = pd.Series(sizes, index=shift_frame.index, name="size_factor")

    for event in events:
        logger.info("self-evolution: %s", event.to_log())
    if log_path is not None:
        _write_log(events, log_path)

    return EvolutionResult(
        size_factor=size_series,
        events=events,
        shrink_factor=shrink_factor,
        recovery_threshold=recovery_threshold,
    )


def _write_log(events: list[EvolutionEvent], log_path: Path) -> None:
    """Append events to ``log_path`` as newline-delimited JSON (JSONL)."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.to_log()) + "\n")
    logger.info("Wrote %d evolution events to %s", len(events), log_path)
