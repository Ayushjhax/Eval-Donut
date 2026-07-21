"""
env/market.py -- Non-stationary, heavy-tailed market simulator.

Research motivation
-------------------
Real markets are *non-stationary*: the data-generating process changes over time
(bull trends, choppy mean-reversion, volatility blow-ups) with no announcement that
a switch has occurred. They are also *heavy-tailed*: extreme moves happen far more
often than a Gaussian predicts -- the 2010 flash crash, the Feb-2018 vol spike, the
March-2020 COVID gap. An agent that assumes stationarity and Gaussian noise will
under-price tail risk and keep adapting to a regime that has already ended.

This simulator is deliberately built to punish both assumptions so we can test
whether a *safety-gated* agent behaves better than a naive one. The hidden regime
labels are exposed ONLY for evaluation -- the agent never sees them.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# The three regimes we stitch together. The agent must infer these from returns.
REGIMES = ("trending", "mean_reverting", "high_volatility")


@dataclass
class MarketConfig:
    """Knobs for the market. Defaults are tuned so tail risk is visible but not absurd."""

    n_steps: int = 1000
    # Default seed chosen so the run exercises EVERY safety guard, including the
    # drawdown circuit breaker (it trips once, ~15% down, then recovers). Change it
    # freely: SAFEEVOLVE_SEED overrides it at run time.
    seed: int = 34
    # Student-t degrees of freedom for innovations. df=3 -> finite variance but
    # INFINITE kurtosis, i.e. genuine fat tails. Gaussian is the df->inf limit.
    t_dof: float = 3.0
    min_regime_len: int = 60          # regimes persist long enough to be learnable
    max_regime_len: int = 160
    tail_hazard: float = 0.015        # per-step probability of an exogenous shock
    tail_drop: float = 0.20           # a crash shock is a ~-20% single-step jump
    vol_spike_mult: float = 4.0       # a vol-spike shock inflates that step's sigma
    crash_share: float = 0.6          # of shocks, 60% are crashes, 40% vol spikes


def standardized_t(rng: np.random.Generator, dof: float, size: int) -> np.ndarray:
    """Draw Student-t noise rescaled to unit variance.

    Why Student-t and not Gaussian: with dof=3 the fourth moment diverges, so
    isolated 6+ sigma moves appear on their own. Under Gaussian noise a -20% day is
    a ~1-in-10^20 event, so a Gaussian-trained agent would never learn to respect
    tails. Rescaling by sqrt((dof-2)/dof) fixes the variance to 1 for any dof, which
    lets us dial tail-fatness (dof) without also changing volatility.
    """
    raw = rng.standard_t(dof, size=size)
    return raw * np.sqrt((dof - 2.0) / dof)


class MarketSimulator:
    """Generates a full non-stationary return path plus hidden regime labels.

    We generate returns with a per-regime AR(1) structure so each regime leaves a
    *statistical signature* a detector can pick up (see agent/regime_detector.py):
      - trending        : positive drift + positive autocorrelation (momentum)
      - mean_reverting  : zero drift + negative autocorrelation (fades moves)
      - high_volatility : zero drift + large sigma (risk-off / panic)
    """

    def __init__(self, config: MarketConfig | None = None) -> None:
        self.cfg = config or MarketConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.returns, self.regimes, self.tail_events = self._generate()
        # Price is exp of cumulative log-returns, anchored at 100.
        self.prices = 100.0 * np.exp(np.cumsum(self.returns))

    def _regime_params(self, name: str) -> tuple[float, float, float]:
        """Return (drift mu, AR(1) phi, sigma) for a regime.

        Small randomization keeps every occurrence of a regime slightly different,
        so the detector cannot memorize exact constants -- it must generalize.
        """
        r = self.rng
        if name == "trending":
            # Drift can be bullish or bearish; the *signature* is positive momentum.
            mu = r.choice([1.0, -1.0]) * r.uniform(0.0004, 0.0010)
            return mu, r.uniform(0.10, 0.25), r.uniform(0.008, 0.012)
        if name == "mean_reverting":
            # No drift; negative phi means moves get faded -> choppy, range-bound.
            return 0.0, r.uniform(-0.45, -0.25), r.uniform(0.010, 0.014)
        # high_volatility: no drift, no autocorr, but a much larger sigma.
        return 0.0, 0.0, r.uniform(0.028, 0.040)

    def _generate(self):
        cfg = self.cfg
        returns = np.zeros(cfg.n_steps)
        labels = np.empty(cfg.n_steps, dtype=object)
        tails: list[dict] = []
        prev_r = 0.0
        t = 0
        while t < cfg.n_steps:
            # Pick the next regime and how long it lasts. The agent gets NO signal
            # that this switch is about to happen -- that is the whole point.
            name = str(self.rng.choice(REGIMES))
            length = int(self.rng.integers(cfg.min_regime_len, cfg.max_regime_len))
            mu, phi, sigma = self._regime_params(name)
            for _ in range(length):
                if t >= cfg.n_steps:
                    break
                eps = standardized_t(self.rng, cfg.t_dof, 1)[0]
                r = mu + phi * prev_r + sigma * eps
                # Exogenous tail shock, independent of the current regime. Modeling it
                # separately (rather than just relying on fat-tailed eps) lets us count
                # discrete "tail events" and measure survival per event.
                if self.rng.random() < cfg.tail_hazard:
                    if self.rng.random() < cfg.crash_share:
                        r += np.log(1.0 - cfg.tail_drop)  # ~-20% crash in log space
                        tails.append({"t": t, "type": "crash"})
                    else:
                        spike = cfg.vol_spike_mult * sigma
                        r += spike * standardized_t(self.rng, cfg.t_dof, 1)[0]
                        tails.append({"t": t, "type": "vol_spike"})
                returns[t] = r
                labels[t] = name
                prev_r = r
                t += 1
        return returns, labels, tails

    def summary(self) -> str:
        """One-line description for logs/README reproducibility."""
        n_crash = sum(1 for e in self.tail_events if e["type"] == "crash")
        n_spike = len(self.tail_events) - n_crash
        return (
            f"MarketSimulator(seed={self.cfg.seed}, steps={self.cfg.n_steps}, "
            f"t_dof={self.cfg.t_dof}, tail_events={len(self.tail_events)} "
            f"[{n_crash} crashes, {n_spike} vol spikes])"
        )
