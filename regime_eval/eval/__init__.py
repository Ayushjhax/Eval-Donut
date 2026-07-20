"""Per-regime, heavy-tail evaluation + distribution-shift detection."""

from __future__ import annotations

from .metrics import (
    aggregate_metrics,
    cvar,
    max_drawdown,
    per_regime_metrics,
    rolling_max_drawdown,
    sharpe_ratio,
    tail_ratio,
)
from .shift import (
    ShiftResult,
    choose_reference_regime,
    detect_shift,
    reference_returns,
    run_shift_detection,
)

__all__ = [
    "aggregate_metrics",
    "cvar",
    "max_drawdown",
    "per_regime_metrics",
    "rolling_max_drawdown",
    "sharpe_ratio",
    "tail_ratio",
    "ShiftResult",
    "choose_reference_regime",
    "detect_shift",
    "reference_returns",
    "run_shift_detection",
]
