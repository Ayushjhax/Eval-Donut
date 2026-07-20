"""A deliberately simple momentum strategy.

The point of this harness is *evaluation*, not alpha, so the strategy is kept
minimal and transparent: a fixed-lookback momentum rule that goes long when the
price is above its value ``MOMENTUM_WINDOW`` periods ago and short otherwise.

Positions are lagged by one period so the backtest is causal (no lookahead):
the signal formed at the close of day *t* earns the return from *t* to *t+1*.
Position size can be a scalar or a per-timestamp Series — the latter is how the
self-evolution loop feeds its dynamic sizing back into the strategy.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from .. import config

Sizing = Union[float, pd.Series]


def momentum_signal(
    prices: pd.DataFrame,
    window: int = config.MOMENTUM_WINDOW,
) -> pd.Series:
    """Compute the raw momentum signal in ``{-1, 0, +1}``.

    Args:
        prices: OHLCV DataFrame with a ``close`` column.
        window: lookback period.

    Returns:
        Series of ``+1`` (long), ``-1`` (short) or ``0`` (undefined / flat),
        indexed like ``prices``.
    """
    momentum = prices["close"].pct_change(window)
    signal = np.sign(momentum)
    return signal.fillna(0.0).rename("signal")


def strategy_returns(
    prices: pd.DataFrame,
    window: int = config.MOMENTUM_WINDOW,
    position_size: Sizing = 1.0,
) -> pd.DataFrame:
    """Run the momentum backtest and return per-period series.

    Args:
        prices: OHLCV DataFrame with a ``close`` column.
        window: momentum lookback period.
        position_size: fixed size (float) or a per-timestamp Series aligned to
            ``prices.index``. Values are the *notional* size applied on top of
            the ``{-1, +1}`` signal.

    Returns:
        DataFrame indexed like ``prices`` with columns:

        - ``log_return``: the asset's log return.
        - ``signal``: the raw momentum signal.
        - ``position``: the lagged, size-scaled position actually held.
        - ``strategy_return``: ``position * log_return`` (the realized P&L).
    """
    log_return = np.log(prices["close"]).diff()
    signal = momentum_signal(prices, window)

    if isinstance(position_size, pd.Series):
        size = position_size.reindex(prices.index).ffill().fillna(0.0)
    else:
        size = pd.Series(float(position_size), index=prices.index)

    # Lag by one period: the position decided at t earns the return over t -> t+1.
    position = (signal * size).shift(1).fillna(0.0)
    strat_ret = (position * log_return).rename("strategy_return")

    return pd.DataFrame(
        {
            "log_return": log_return,
            "signal": signal,
            "position": position,
            "strategy_return": strat_ret,
        }
    ).dropna(subset=["log_return"])
