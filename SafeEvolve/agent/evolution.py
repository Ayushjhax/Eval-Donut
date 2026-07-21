"""
agent/evolution.py -- Safety-GATED self-evolution.

Research motivation
-------------------
Self-evolution needs feedback, and financial markets supply an endless non-stationary
stream of it. But that feedback is noisy, delayed, and regime-dependent, which creates
a specific and dangerous failure mode:

    If the agent updates its policy using performance measured DURING a tail event or
    an unrecognized regime, it fits to a signal that is about to invert. It adapts
    *into* the crash -- e.g. it learns "leverage paid off recently" moments before the
    payoff reverses -- and the update makes the next tail event worse.

So the rule we demonstrate is: evolution is a privilege that must be EARNED by a
confident, non-distressed state. Concretely, an evolution step is BLOCKED whenever the
agent is in safe mode (a drawdown breaker tripped) or the current regime confidence is
below threshold. Learning resumes only once the agent is solvent and sure of its
footing. Every attempt -- allowed or blocked -- is logged with its reason and the
performance signal, so the researcher can audit the learning trajectory.

The update itself is deliberately simple and interpretable: an EMA of the realized
per-regime Sharpe ratio nudges that regime's position-size scale up (when recent
risk-adjusted performance is good) or down (when it is poor).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


@dataclass
class EvolutionConfig:
    interval: int = 50          # re-evaluate policy every N steps
    ema_alpha: float = 0.30     # smoothing for the per-regime Sharpe estimate
    learning_rate: float = 0.15  # how far the size scale can move per evolution
    min_samples: int = 8        # need this many recent obs in a regime to trust it
    conf_thresh: float = 0.60   # must match the safety layer's uncertainty threshold
    scale_floor: float = 0.05   # never fully disable a regime
    scale_ceil: float = 1.00    # never exceed full intended size


class Evolver:
    """Online, safety-gated updater for the SafeAgent's per-regime size scales."""

    def __init__(self, config: EvolutionConfig | None = None) -> None:
        self.cfg = config or EvolutionConfig()
        self.sharpe_ema: dict[str, float] = defaultdict(float)
        # Buffer of realized per-step returns since the last evolution, keyed by the
        # regime the agent BELIEVED it was in (it can only learn from its own view).
        self._buffer: dict[str, list[float]] = defaultdict(list)
        self.events: list[dict] = []
        self.n_allowed = 0
        self.n_blocked = 0

    def record(self, regime: str, step_return: float) -> None:
        """Accumulate one step of realized P&L, tagged by the agent's regime belief."""
        self._buffer[regime].append(float(step_return))

    def maybe_evolve(self, t: int, agent, confidence: float) -> dict | None:
        """Attempt an evolution step at the interval; gate it on safety state.

        Returns the logged event dict (allowed or blocked), or None on non-interval
        steps. The gate is the point of the whole module: NO policy update happens
        while the agent is distressed or uncertain.
        """
        cfg = self.cfg
        if t == 0 or t % cfg.interval != 0:
            return None

        # --- The safety gate -----------------------------------------------------
        if agent.safe_mode:
            return self._block(t, "safe_mode", confidence)
        if confidence < cfg.conf_thresh:
            return self._block(t, f"low_confidence({confidence:.2f})", confidence)

        # --- Allowed: update each regime that has enough recent evidence ----------
        changes = {}
        for regime, rets in self._buffer.items():
            if regime == "uncertain" or len(rets) < cfg.min_samples:
                continue
            arr = np.asarray(rets)
            sharpe = float(arr.mean() / (arr.std() + 1e-9))  # per-step risk-adjusted
            prev = self.sharpe_ema[regime]
            self.sharpe_ema[regime] = (1 - cfg.ema_alpha) * prev + cfg.ema_alpha * sharpe
            old = agent.size_scale.get(regime, 0.0)
            # tanh keeps the nudge bounded even if a single window looks spectacular.
            new = float(np.clip(old + cfg.learning_rate * np.tanh(self.sharpe_ema[regime]),
                                cfg.scale_floor, cfg.scale_ceil))
            agent.size_scale[regime] = new
            changes[regime] = {"old": round(old, 3), "new": round(new, 3),
                               "sharpe": round(sharpe, 3),
                               "sharpe_ema": round(self.sharpe_ema[regime], 3)}

        self._buffer.clear()
        self.n_allowed += 1
        event = {"t": t, "status": "allowed", "confidence": confidence,
                 "changes": changes, "reason": "confident_and_solvent"}
        self.events.append(event)
        return event

    def _block(self, t: int, reason: str, confidence: float) -> dict:
        """Record a blocked evolution attempt and DROP the buffered feedback.

        We discard the buffer on a block on purpose: the returns it holds were earned
        during a distressed/uncertain window and are exactly the samples we must not
        learn from later.
        """
        self._buffer.clear()
        self.n_blocked += 1
        event = {"t": t, "status": "blocked", "confidence": confidence,
                 "changes": {}, "reason": reason}
        self.events.append(event)
        return event
