"""
eval/metrics.py -- Evaluation metrics for the SafeEvolve experiment.

Research motivation
-------------------
In a heavy-tailed setting, average return is a misleading scorecard: a strategy can
post a great mean right up until the one draw that ruins it. So we grade on survival
and risk-adjusted terms -- max drawdown, Sharpe, and explicit per-tail-event survival
counts -- not on terminal value alone. All functions here are pure and stateless so
the runner stays easy to audit.
"""
from __future__ import annotations

import numpy as np

STEPS_PER_YEAR = 252  # treat each step as a trading day for annualization


def drawdown_series(pv: np.ndarray) -> np.ndarray:
    """Fractional drawdown from the running peak at every step (0 = at peak)."""
    pv = np.asarray(pv, dtype=float)
    running_peak = np.maximum.accumulate(pv)
    return 1.0 - pv / running_peak


def max_drawdown(pv: np.ndarray) -> float:
    """Worst peak-to-trough loss -- the number a risk desk actually cares about."""
    return float(drawdown_series(pv).max())


def total_return(pv: np.ndarray) -> float:
    return float(pv[-1] / pv[0] - 1.0)


def sharpe(pv: np.ndarray) -> float:
    """Annualized Sharpe of the step-over-step returns (risk-free assumed 0)."""
    pv = np.asarray(pv, dtype=float)
    rets = pv[1:] / pv[:-1] - 1.0
    if rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(STEPS_PER_YEAR))


def tail_survival(pv: np.ndarray, tail_events: list[dict],
                  window: int = 5, tol: float = 0.10) -> tuple[int, int]:
    """Count tail events the agent 'survived'.

    Survival = the portfolio did not lose more than `tol` over the `window` steps
    following the shock, measured from the value just before it hit. This rewards
    agents that were small or flat going into the tail, which is precisely what the
    safety guards produce. Returns (survived, total).
    """
    pv = np.asarray(pv, dtype=float)
    survived = 0
    total = 0
    for ev in tail_events:
        te = ev["t"]
        pre = te - 1
        post = min(te + window, len(pv) - 1)
        if pre < 0 or post <= te:
            continue
        total += 1
        # Deepest point over [shock, shock+window] vs the value just before the shock.
        trough = pv[te:post + 1].min()
        drop = 1.0 - trough / pv[pre]
        if drop <= tol:
            survived += 1
    return survived, total


def rolling_accuracy(true_labels: list[str], est_labels: list[str],
                     window: int = 50) -> np.ndarray:
    """Rolling fraction of steps where the estimated regime matched the truth."""
    hits = np.array([1.0 if t == e else 0.0
                     for t, e in zip(true_labels, est_labels)], dtype=float)
    if len(hits) == 0:
        return hits
    out = np.empty_like(hits)
    for i in range(len(hits)):
        lo = max(0, i - window + 1)
        out[i] = hits[lo:i + 1].mean()
    return out
