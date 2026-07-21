"""
agent/safe_agent.py -- Trading agent with safety as an ARCHITECTURAL layer.

Research motivation
-------------------
The central claim of this prototype is that safety in an agentic system is not a
post-hoc filter you bolt on after the policy decides -- it is a set of hard
constraints that sit *between* intent and execution and can veto or shrink any
action. The policy here is intentionally simple and interpretable (regime-conditioned
momentum / mean-reversion). What matters is that EVERY action passes through three
guards before it can touch the portfolio, and every decision is logged with a
machine-readable reason string so a researcher can audit exactly why the agent did
or did not act:

  1. Drawdown circuit breaker -- >15% below peak halts trading (safe mode: hold only).
  2. Uncertainty gate        -- low regime confidence OR a volatility spike halves size.
  3. Per-trade position cap   -- no single trade may move more than 10% of the book.

The agent also carries EVOLVABLE per-regime size scales (see agent/evolution.py); the
policy proposes, the safety layer disposes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from env.market import REGIMES


@dataclass
class SafetyConfig:
    max_trade: float = 0.10        # <=10% of the book may change per trade (hard cap)
    dd_halt: float = 0.15          # >15% drawdown from peak -> safe mode
    dd_resume: float = 0.05        # re-enable trading once back within 5% of peak
    conf_thresh: float = 0.60      # below this confidence -> "uncertain regime"
    vol_spike_mult: float = 2.0    # recent vol > 2x rolling avg -> spike
    uncertainty_haircut: float = 0.50  # each triggered gate halves position size
    txn_cost: float = 0.0005       # per unit of turnover; makes churning costly


class SafeAgent:
    """Regime-conditioned policy wrapped in hard safety constraints."""

    def __init__(self, config: SafetyConfig | None = None, start_value: float = 1.0) -> None:
        self.cfg = config or SafetyConfig()
        self.portfolio = start_value
        self.peak = start_value
        self.exposure = 0.0            # signed fraction of book in the asset, [-1, 1]
        self.safe_mode = False
        # Evolvable knobs: how aggressive to be per regime. Uncertain -> 0 (sit out).
        self.size_scale = {"trending": 0.8, "mean_reverting": 0.6,
                           "high_volatility": 0.2, "uncertain": 0.0}
        self.pv_history = [start_value]
        self.log: list[dict] = []
        self._last_delta = 0.0

    def _raw_signal(self, features: dict, regime: str) -> float:
        """Policy intent in [-1, 1] BEFORE any safety adjustment.

        Trend -> ride momentum; mean-revert -> fade the deviation from the local mean;
        high-vol -> flatten. These are the classic regime-appropriate stances; the
        research point is not the alpha but that intent is separable from safety.
        """
        if regime == "trending":
            return float(np.tanh(features["mom"] * 80.0))
        if regime == "mean_reverting":
            return float(-np.tanh(features["price_dev"] * 12.0))
        # high_volatility or anything unmodeled: flatten and wait.
        return 0.0

    def decide(self, t: int, features: dict, regime: str, confidence: float) -> dict:
        """Choose the next exposure, enforcing all safety constraints first."""
        cfg = self.cfg
        drawdown = 1.0 - self.portfolio / self.peak
        reasons: list[str] = []

        # --- Guard 1: drawdown circuit breaker (state machine) -------------------
        if not self.safe_mode and drawdown > cfg.dd_halt:
            self.safe_mode = True
        elif self.safe_mode and drawdown < cfg.dd_resume:
            # Only leave safe mode once genuinely recovered -- not on the first bounce.
            self.safe_mode = False

        # Treat a low-confidence estimate as the "uncertain" regime for sizing.
        effective_regime = regime if confidence >= cfg.conf_thresh else "uncertain"

        if self.safe_mode:
            target = 0.0                                   # hold-only: bleed to cash
            reasons.append("circuit_breaker_safe_mode")
        else:
            base = self._raw_signal(features, effective_regime)
            target = base * self.size_scale.get(effective_regime, 0.0)
            # --- Guard 2: uncertainty gate ---------------------------------------
            if confidence < cfg.conf_thresh:
                target *= cfg.uncertainty_haircut
                reasons.append(f"low_confidence({confidence:.2f})")
            if features["avg_vol"] > 0 and features["recent_vol"] > cfg.vol_spike_mult * features["avg_vol"]:
                target *= cfg.uncertainty_haircut
                reasons.append("vol_spike")

        # --- Guard 3: per-trade position cap -------------------------------------
        raw_delta = target - self.exposure
        delta = float(np.clip(raw_delta, -cfg.max_trade, cfg.max_trade))
        if abs(raw_delta) > cfg.max_trade:
            reasons.append("trade_capped_10pct")
        new_exposure = float(np.clip(self.exposure + delta, -1.0, 1.0))

        action = "buy" if delta > 1e-9 else "sell" if delta < -1e-9 else "hold"
        self._last_delta = new_exposure - self.exposure
        self.exposure = new_exposure

        entry = {
            "t": t, "action": action, "trade_size": self._last_delta,
            "exposure": self.exposure, "regime": regime, "confidence": confidence,
            "portfolio": self.portfolio, "drawdown": drawdown,
            "safe_mode": self.safe_mode,
            "reason": ", ".join(reasons) if reasons else "nominal",
        }
        self.log.append(entry)
        return entry

    def settle(self, realized_return: float) -> float:
        """Apply the next-step market return to the position and update peak.

        P&L is exposure * return minus turnover cost. Called AFTER decide(), so the
        decision never sees the return it will be graded on (no look-ahead).
        """
        pnl = self.exposure * realized_return - self.cfg.txn_cost * abs(self._last_delta)
        self.portfolio *= (1.0 + pnl)
        self.peak = max(self.peak, self.portfolio)
        self.pv_history.append(self.portfolio)
        return pnl
