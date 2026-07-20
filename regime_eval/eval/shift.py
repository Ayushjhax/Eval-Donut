"""Distribution-shift detector — the "is my eval still valid?" sensor.

The strategy is optimized on a particular regime with a particular return
distribution. When the live distribution drifts away from that reference, the
backtest that justified the strategy no longer describes reality: the evaluation
has gone stale. We detect this with a two-sample Kolmogorov-Smirnov test between
a rolling window of recent returns and the reference regime's returns.

KS is used instead of KL divergence because it is non-parametric (no density
estimation, no binning), its statistic is already normalized to ``[0, 1]`` — a
ready-made severity score — and it ships with a calibrated p-value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .. import config
from .metrics import cvar


def choose_reference_regime(
    returns: pd.Series,
    regimes: pd.Series,
    cvar_level: float = config.CVAR_LEVEL,
) -> str:
    """Pick the in-distribution reference regime for the shift detector.

    We anchor eval-validity to the regime where evaluation is most *trustworthy*
    — the benign, thinnest-tailed regime with the least-negative CVaR. This is
    the subtle but important point: a backtest's numbers only mean what they
    claim under roughly-stationary, thin-tailed returns, which holds in the calm
    regime and breaks in the high-vol one. So the calm regime is the natural
    "in-distribution" baseline, and the detector fires as the live distribution
    drifts *away* from it into heavier-tailed territory.

    (Using the best-Sharpe regime instead would be perverse here: the high-vol
    regime posts the highest Sharpe precisely because it also carries the worst
    tail — anchoring to it would make the sensor fire during calm markets.)

    Args:
        returns: per-period returns whose per-regime tails are compared.
        regimes: per-period regime labels.
        cvar_level: confidence level for the CVaR used to rank regimes.

    Returns:
        The reference regime label (thinnest left tail).
    """
    frame = pd.DataFrame({"r": returns}).join(regimes.rename("regime"), how="inner")
    tail_risk = frame.groupby("regime")["r"].apply(lambda s: cvar(s, cvar_level))
    # Least-negative CVaR == thinnest tail == most benign / in-distribution.
    return str(tail_risk.idxmax())


def reference_returns(
    market_returns: pd.Series,
    regimes: pd.Series,
    reference_regime: str,
) -> np.ndarray:
    """Return the sample of market returns drawn from the reference regime.

    Args:
        market_returns: per-period market (asset) returns.
        regimes: per-period regime labels.
        reference_regime: the regime to sample.

    Returns:
        1-D array of returns observed during ``reference_regime``.
    """
    aligned = pd.DataFrame({"r": market_returns}).join(
        regimes.rename("regime"), how="inner"
    )
    return aligned.loc[aligned["regime"] == reference_regime, "r"].dropna().to_numpy()


def detect_shift(
    market_returns: pd.Series,
    reference: np.ndarray,
    window: int = config.SHIFT_DETECTION_WINDOW,
    alpha: float = config.SHIFT_ALPHA,
) -> pd.DataFrame:
    """Run the rolling KS test of recent returns vs the reference distribution.

    Args:
        market_returns: per-period market returns to monitor.
        reference: reference-regime return sample.
        window: trailing window length for the current distribution.
        alpha: p-value threshold below which a shift is *detected*.

    Returns:
        DataFrame indexed like ``market_returns`` with columns:

        - ``ks_stat``: the KS statistic in ``[0, 1]`` — the shift *severity*.
        - ``p_value``: KS two-sample p-value.
        - ``shift_detected``: ``p_value < alpha`` (bool).
    """
    r = market_returns.to_numpy()
    n = len(r)
    ks_stat = np.full(n, np.nan)
    p_value = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_sample = r[i - window + 1 : i + 1]
        window_sample = window_sample[~np.isnan(window_sample)]
        if len(window_sample) < 2 or len(reference) < 2:
            continue
        stat, p = ks_2samp(window_sample, reference)
        ks_stat[i] = stat
        p_value[i] = p

    detected = (p_value < alpha) & ~np.isnan(p_value)
    return pd.DataFrame(
        {
            "ks_stat": ks_stat,
            "p_value": p_value,
            "shift_detected": detected,
        },
        index=market_returns.index,
    )


@dataclass
class ShiftResult:
    """Bundle of everything the shift detector produced.

    Attributes:
        frame: per-timestep DataFrame (``ks_stat``, ``p_value``, ``shift_detected``).
        reference_regime: the regime used as the in-distribution reference.
        reference_sample: the reference return sample.
        window: rolling window used.
        alpha: p-value threshold used.
    """

    frame: pd.DataFrame
    reference_regime: str
    reference_sample: np.ndarray
    window: int
    alpha: float

    @property
    def severity(self) -> pd.Series:
        """Per-timestep shift severity (normalized KS statistic, 0-1)."""
        return self.frame["ks_stat"]

    @property
    def detected(self) -> pd.Series:
        """Per-timestep boolean shift flag."""
        return self.frame["shift_detected"]

    def shift_dates(self) -> pd.DatetimeIndex:
        """Timestamps where a shift fired."""
        return self.frame.index[self.frame["shift_detected"].fillna(False)]


def run_shift_detection(
    market_returns: pd.Series,
    regimes: pd.Series,
    window: int = config.SHIFT_DETECTION_WINDOW,
    alpha: float = config.SHIFT_ALPHA,
    reference_regime: str | None = None,
) -> ShiftResult:
    """End-to-end shift detection: pick the reference regime, then test against it.

    Args:
        market_returns: per-period market returns to monitor.
        regimes: per-period regime labels.
        window: rolling window length.
        alpha: p-value threshold for firing a shift.
        reference_regime: override the auto-selected reference regime.

    Returns:
        A populated :class:`ShiftResult`.
    """
    if reference_regime is None:
        reference_regime = choose_reference_regime(market_returns, regimes)
    reference = reference_returns(market_returns, regimes, reference_regime)
    frame = detect_shift(market_returns, reference, window=window, alpha=alpha)
    return ShiftResult(
        frame=frame,
        reference_regime=reference_regime,
        reference_sample=reference,
        window=window,
        alpha=alpha,
    )
