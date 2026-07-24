"""
To show that the safety layer is doing real work (and not just leaving money on the
table), we need a baseline that is competent but UNGUARDED. This agent is a standard
full-size momentum trader: it reads the raw signal and takes maximum exposure in its
direction. It has:
  - no regime detection       (it never knows the process changed),
  - no drawdown breaker       (it keeps trading straight through a crash),
  - no uncertainty gate       (it sizes the same whether calm or chaotic),
  - no per-trade cap          (it can flip from full long to full short in one step),
  - no self-evolution         (nothing to gate; it is the fixed reference point).

It is not a strawman -- momentum genuinely makes money in trending regimes. It fails
precisely where the thesis predicts: heavy-tailed shocks and regime switches, which is
exactly what the SafeAgent's guards are built to survive.
"""
from __future__ import annotations

import numpy as np


class NaiveAgent:
    """Unconstrained full-size momentum trader."""

    def __init__(self, start_value: float = 1.0, txn_cost: float = 0.0005) -> None:
        self.portfolio = start_value
        self.peak = start_value
        self.exposure = 0.0
        self.txn_cost = txn_cost
        self.pv_history = [start_value]
        self.log: list[dict] = []
        self._last_delta = 0.0

    def decide(self, t: int, features: dict) -> dict:
        """Go fully long or short on the sign of recent momentum. No guards, no caps."""
        target = float(np.sign(features["mom"]))  # +1 / -1 / 0, always maximum size
        new_exposure = target
        self._last_delta = new_exposure - self.exposure
        action = ("buy" if self._last_delta > 1e-9 else
                  "sell" if self._last_delta < -1e-9 else "hold")
        self.exposure = new_exposure
        entry = {"t": t, "action": action, "trade_size": self._last_delta,
                 "exposure": self.exposure, "portfolio": self.portfolio,
                 "reason": "raw_momentum_no_safety"}
        self.log.append(entry)
        return entry

    def settle(self, realized_return: float) -> float:
        """Same P&L mechanics as SafeAgent so the comparison is apples-to-apples."""
        pnl = self.exposure * realized_return - self.txn_cost * abs(self._last_delta)
        self.portfolio *= (1.0 + pnl)
        self.peak = max(self.peak, self.portfolio)
        self.pv_history.append(self.portfolio)
        return pnl
