"""regime_eval — a regime-aware evaluation harness for a trading agent.

The thesis of this package: in non-stationary, heavy-tailed markets a single
backtest number is meaningless. Evaluation must be conditioned on *regime*, and
an agent that can detect when its own evaluation has gone stale (a distribution
shift) is safer than one that simply optimizes harder.

The sub-packages are the engine; ``notebooks/main.ipynb`` is the demo surface.
"""

from __future__ import annotations

from . import config

__all__ = ["config"]
__version__ = "0.1.0"
