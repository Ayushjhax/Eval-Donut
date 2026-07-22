"""Central configuration for the regime-aware eval harness.

Every tunable lives here so that a researcher can change the behaviour of the
whole pipeline from a single file. Paths are derived from ``__file__`` — there
are no hardcoded absolute paths anywhere in the codebase.
"""

from __future__ import annotations

from pathlib import Path


# Market / data
SYMBOL: str = "SOL/USDT"          # SOL/USDT has deeper Binance history than USDC
TIMEFRAME: str = "1d"             # daily OHLCV
HISTORY_START: str = "2020-01-01"  # earliest candle to request (pre-listing is Ok)


# Regime model (GaussianHMM) 
HMM_STATES: int = 3               # trending / mean-reverting / high-vol
REALIZED_VOL_WINDOW: int = 20     # window (days) for the realized-vol feature


# Strategy
MOMENTUM_WINDOW: int = 20         # lookback for the momentum signal

 
# Distribution-shift detector
SHIFT_DETECTION_WINDOW: int = 30  # rolling window (days) compared to the reference
SHIFT_ALPHA: float = 0.05         # KS 2-sample p-value threshold to *fire* a shift

 
# Self-evolution loop
SHRINK_FACTOR: float = 0.5        # multiply position size by this when shift fires
RECOVERY_THRESHOLD: float = 0.2   # restore size once severity (KS stat) falls below

 
# Heavy-tail metrics
CVAR_LEVEL: float = 0.95          # confidence level for CVaR / expected shortfall
TAIL_UPPER_Q: float = 0.95        # upper quantile for the tail ratio
TAIL_LOWER_Q: float = 0.05        # lower quantile for the tail ratio
TRADING_PERIODS_PER_YEAR: int = 365  # crypto trades every day; used to annualize Sharpe

 
# Reproducibility
RANDOM_SEED: int = 42

 
# Paths (derived — never hardcode an absolute path elsewhere)
PACKAGE_ROOT: Path = Path(__file__).resolve().parent
CACHE_DIR: Path = PACKAGE_ROOT / "data" / "cache"
LOG_DIR: Path = PACKAGE_ROOT / "evolution" / "logs"


def cache_path(symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> Path:
    """Return the CSV cache path for a given symbol/timeframe.

    The symbol's ``/`` is replaced with ``_`` so the filename is filesystem-safe
    (e.g. ``SOL/USDT`` -> ``SOL_USDT_1d.csv``).
    """
    safe_symbol = symbol.replace("/", "_")
    return CACHE_DIR / f"{safe_symbol}_{timeframe}.csv"