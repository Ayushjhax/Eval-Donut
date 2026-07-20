"""Heavy-tail performance metrics, evaluated per regime.

Standard backtesting collapses a return series into a single Sharpe ratio. In
non-stationary, heavy-tailed markets that number is not just noisy — it is
*invalid*, because it silently assumes one stationary, roughly-Gaussian
distribution. This module does two things differently:

1. It reports tail-aware metrics (CVaR / expected shortfall, a *distribution*
   of rolling drawdowns, and a tail ratio) instead of leaning on Sharpe.
2. It never aggregates across regimes. The unit of evaluation is the regime.

Sharpe is still computed, but only ever "shown for comparison only" — it is the
number that lies, kept in frame so the reader can watch it lie.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = config.TRADING_PERIODS_PER_YEAR,
    annualize: bool = True,
) -> float:
    """Annualized Sharpe ratio. **Shown for comparison only.**

    This assumes i.i.d. roughly-Gaussian returns — an assumption that fails
    hardest exactly when it matters (in the high-vol regime). Use the tail
    metrics for decisions; keep Sharpe only to demonstrate its blind spots.

    Args:
        returns: per-period returns.
        periods_per_year: periods per year, for annualization.
        annualize: whether to scale by ``sqrt(periods_per_year)``.

    Returns:
        The Sharpe ratio, or ``nan`` if it is undefined.
    """
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    ratio = r.mean() / r.std(ddof=1)
    if annualize:
        ratio *= np.sqrt(periods_per_year)
    return float(ratio)


def cvar(returns: pd.Series, level: float = config.CVAR_LEVEL) -> float:
    """Conditional Value-at-Risk (expected shortfall) at ``level``.

    CVaR is the *mean* return in the worst ``1 - level`` tail — the expected
    loss given that you are already in the tail. Unlike VaR (a single quantile),
    CVaR looks past the threshold into the tail's shape, which is where heavy
    tails hide. Returned as a signed return (negative = loss).

    Args:
        returns: per-period returns.
        level: confidence level, e.g. ``0.95`` inspects the worst 5%.

    Returns:
        Expected shortfall, or ``nan`` if undefined.
    """
    r = returns.dropna()
    if r.empty:
        return float("nan")
    var_threshold = np.quantile(r, 1.0 - level)
    tail = r[r <= var_threshold]
    return float(tail.mean()) if len(tail) else float(var_threshold)


def _equity_curve(returns: pd.Series) -> pd.Series:
    """Convert a log-return series into a positive equity curve (start = 1.0)."""
    return np.exp(returns.fillna(0.0).cumsum())


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough drawdown over the whole series (a single number).

    Args:
        returns: per-period log returns.

    Returns:
        The most negative drawdown (e.g. ``-0.62`` = a 62% drawdown), or ``nan``.
    """
    r = returns.dropna()
    if r.empty:
        return float("nan")
    equity = _equity_curve(r)
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def rolling_max_drawdown(
    returns: pd.Series,
    window: int = config.SHIFT_DETECTION_WINDOW,
) -> pd.Series:
    """Drawdown as a rolling *distribution*, not one summary number.

    For each timestamp this is the worst drawdown within the trailing
    ``window``. Reporting the whole series (and its distribution) keeps the
    heavy tail visible instead of averaging it away.

    Args:
        returns: per-period log returns.
        window: trailing window length.

    Returns:
        Series of rolling max drawdowns (<= 0), indexed like ``returns``.
    """
    r = returns.fillna(0.0)
    equity = _equity_curve(r)
    rolling_peak = equity.rolling(window, min_periods=1).max()
    drawdown = equity / rolling_peak - 1.0
    return drawdown.rolling(window, min_periods=1).min().rename("rolling_max_drawdown")


def tail_ratio(
    returns: pd.Series,
    upper_q: float = config.TAIL_UPPER_Q,
    lower_q: float = config.TAIL_LOWER_Q,
) -> float:
    """Ratio of upside tail to downside tail.

    ``quantile(upper_q) / |quantile(lower_q)|`` — e.g. the 95th-percentile gain
    divided by the absolute 5th-percentile loss. ``> 1`` means the right tail is
    fatter than the left (asymmetry in your favour); ``< 1`` means losses tail
    harder than gains.

    Args:
        returns: per-period returns.
        upper_q: upper quantile (default 0.95).
        lower_q: lower quantile (default 0.05).

    Returns:
        The tail ratio, or ``nan`` if the downside quantile is zero/undefined.
    """
    r = returns.dropna()
    if r.empty:
        return float("nan")
    gain = np.quantile(r, upper_q)
    loss = abs(np.quantile(r, lower_q))
    return float(gain / loss) if loss > 0 else float("nan")


def _metric_row(returns: pd.Series, cvar_level: float) -> dict[str, float]:
    """Compute the full metric set for one return series."""
    r = returns.dropna()
    return {
        "n_periods": int(len(r)),
        "Sharpe (ref only)": sharpe_ratio(r),
        f"CVaR_{int(cvar_level * 100)}": cvar(r, cvar_level),
        "max_drawdown": max_drawdown(r),
        "tail_ratio": tail_ratio(r),
    }


def per_regime_metrics(
    strategy_return: pd.Series,
    regimes: pd.Series,
    cvar_level: float = config.CVAR_LEVEL,
) -> pd.DataFrame:
    """Evaluate the strategy **separately** within each regime.

    This table is the whole point of the harness. The columns after
    ``n_periods`` are the tail-aware metrics; ``Sharpe`` is retained only as a
    reference (hence its column label) so the reader can see how much it
    disagrees with itself across regimes.

    Args:
        strategy_return: realized per-period strategy returns.
        regimes: per-period regime labels (aligned by index).
        cvar_level: confidence level for CVaR.

    Returns:
        DataFrame indexed by regime label with columns
        ``[n_periods, Sharpe (ref only), CVaR_95, max_drawdown, tail_ratio]``.
    """
    frame = pd.DataFrame({"strategy_return": strategy_return}).join(
        regimes.rename("regime"), how="inner"
    )

    rows: dict[str, dict[str, float]] = {}
    for regime, group in frame.groupby("regime"):
        rows[regime] = _metric_row(group["strategy_return"], cvar_level)

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "regime"

    # Order rows by the canonical regime order when present.
    canonical = ["trending", "mean-reverting", "high-vol"]
    ordered = [r for r in canonical if r in table.index]
    ordered += [r for r in table.index if r not in canonical]
    return table.loc[ordered]


def aggregate_metrics(
    strategy_return: pd.Series,
    cvar_level: float = config.CVAR_LEVEL,
) -> pd.Series:
    """Compute the *single-number* metrics over the whole series.

    This is the naive "one backtest number" view — the thing the per-regime
    table exists to refute. Provided so the notebook can juxtapose the two.

    Args:
        strategy_return: realized per-period strategy returns.
        cvar_level: confidence level for CVaR.

    Returns:
        Series of the aggregate metrics.
    """
    row = _metric_row(strategy_return, cvar_level)
    return pd.Series(row, name="ALL (aggregate)")
