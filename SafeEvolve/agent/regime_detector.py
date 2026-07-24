"""
The agent must answer two coupled questions in real time:
  1. "Has the regime just changed?"  (changepoint detection)
  2. "Which regime am I in now, and how sure am I?"  (classification + confidence)

Why BOCPD and not a plain EWMA / fixed-window vol estimate:
  - An EWMA silently mixes data from before and after a switch, so right after a
    regime change its estimate is a stale blend -- exactly when the agent most needs
    to KNOW it is uncertain. BOCPD instead maintains a full posterior over the
    "run length" (time since the last changepoint). When a switch happens, mass
    collapses toward run length 0, which we surface as a *calibrated drop in
    confidence*. That confidence signal is what the safety layer and the evolution
    gate consume -- an EWMA gives us a point estimate but no honest uncertainty.

Model: Adams & MacKay (2007) with a Gaussian observation model whose unknown mean
and variance carry a Normal-Inverse-Gamma prior; the posterior predictive is a
Student-t, which is itself heavy-tailed -- appropriate for this market.
"""

from __future__ import annotations

from dataclasses import dataclass     
import numpy as np
from scipy import stats

from env.market import REGIMES


@dataclass
class DetectorConfig:
    hazard: float = 1.0 / 120.0   # prior mean run length ~ regime length (~110 steps)
    r_max: int = 300              # truncate run-length posterior for O(r_max) cost/step
    conf_window: int = 8          # "just changed" = mass on run lengths < this
    high_vol_thresh: float = 0.020  # segment sigma above this => high_volatility
    ar_thresh: float = 0.12         # |lag-1 autocorr| needed to call trend vs revert
    min_seg: int = 12               # never classify on fewer than this many points
    # Normal-Inverse-Gamma prior (weakly informative, centered on zero return).
    mu0: float = 0.0
    kappa0: float = 1.0
    alpha0: float = 1.0
    beta0: float = 1e-4


@dataclass
class RegimeEstimate:
    regime: str
    confidence: float          # P(run length >= conf_window): high => stable regime
    run_length: int            # MAP time-since-changepoint
    seg_vol: float
    seg_mean: float
    seg_autocorr: float
    changepoint_prob: float    # P(run length == 0) this step


class BOCPDRegimeDetector:
    """Online detector: feed it one return per step, get a regime + confidence out."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.cfg = config or DetectorConfig()
        c = self.cfg
        # Run-length posterior; starts certain that run length is 0.
        self.rl = np.array([1.0])
        # NIG sufficient statistics, one entry per surviving run length.
        self.mu = np.array([c.mu0])
        self.kappa = np.array([c.kappa0])
        self.alpha = np.array([c.alpha0])
        self.beta = np.array([c.beta0])
        self.history: list[float] = []

    def _predictive(self, x: float) -> np.ndarray:
        """Posterior-predictive prob of x under each run length (Student-t)."""
        c = self.cfg
        df = 2.0 * self.alpha
        scale = np.sqrt(self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa))
        return stats.t.pdf(x, df=df, loc=self.mu, scale=scale)

    def _grow_stats(self, x: float):
        """NIG conjugate update: extend every run length by one observation."""
        c = self.cfg
        new_mu = (self.kappa * self.mu + x) / (self.kappa + 1.0)
        new_kappa = self.kappa + 1.0
        new_alpha = self.alpha + 0.5
        new_beta = self.beta + (self.kappa * (x - self.mu) ** 2) / (2.0 * (self.kappa + 1.0))
        # Prepend the prior (this is the "a changepoint just happened" hypothesis).
        self.mu = np.concatenate([[c.mu0], new_mu])
        self.kappa = np.concatenate([[c.kappa0], new_kappa])
        self.alpha = np.concatenate([[c.alpha0], new_alpha])
        self.beta = np.concatenate([[c.beta0], new_beta])

    def _truncate(self):
        """Cap run-length support so per-step cost stays bounded."""
        r = self.cfg.r_max
        if len(self.rl) > r:
            self.rl = self.rl[:r]
            self.mu, self.kappa = self.mu[:r], self.kappa[:r]
            self.alpha, self.beta = self.alpha[:r], self.beta[:r]
            self.rl /= self.rl.sum()

    def update(self, x: float) -> RegimeEstimate:
        """Ingest one return, advance the posterior, return a regime estimate."""
        self.history.append(float(x))
        h = self.cfg.hazard
        pred = self._predictive(x)
        # Growth: stay in the current run (no changepoint) with prob (1 - hazard).
        growth = self.rl * pred * (1.0 - h)
        # Changepoint: collapse to run length 0 with prob hazard.
        cp = float(np.sum(self.rl * pred * h))
        new_rl = np.concatenate([[cp], growth])
        new_rl /= new_rl.sum()
        self.rl = new_rl
        self._grow_stats(x)
        self._truncate()
        return self._estimate()

    def _estimate(self) -> RegimeEstimate:
        run_length = int(np.argmax(self.rl))
        changepoint_prob = float(self.rl[0])
        # Confidence = probability we are NOT freshly after a changepoint. Right after
        # a switch, mass piles onto small run lengths, so this drops -> the safety
        # layer reads it as "uncertain regime" and de-risks. This is the key output.
        w = min(self.cfg.conf_window, len(self.rl))
        confidence = float(1.0 - self.rl[:w].sum())
        regime, vol, mean, ac = self._classify(run_length)
        return RegimeEstimate(regime, confidence, run_length, vol, mean, ac, changepoint_prob)

    def _classify(self, run_length: int) -> tuple[str, float, float, float]:
        """Map the current segment's statistics onto one of the three regimes.

        We use only data since the estimated changepoint so a stale prior regime does
        not leak in. High volatility dominates (it is the risk-off state we most need
        to catch); otherwise the sign of lag-1 autocorrelation separates momentum
        (trend) from fading (mean-reversion).
        """
        n = max(run_length, self.cfg.min_seg)
        seg = np.asarray(self.history[-n:], dtype=float)
        if seg.size < 3:
            return REGIMES[0], 0.0, 0.0, 0.0
        vol = float(seg.std(ddof=1)) if seg.size > 1 else 0.0
        mean = float(seg.mean())
        ac = self._autocorr1(seg)
        if vol > self.cfg.high_vol_thresh:
            return "high_volatility", vol, mean, ac
        if ac < -self.cfg.ar_thresh:
            return "mean_reverting", vol, mean, ac
        return "trending", vol, mean, ac

    @staticmethod
    def _autocorr1(seg: np.ndarray) -> float:
        """Lag-1 autocorrelation; the sign is the trend-vs-revert discriminator."""
        if seg.size < 3 or seg.std() == 0:
            return 0.0
        a, b = seg[:-1], seg[1:]
        return float(np.corrcoef(a, b)[0, 1])
