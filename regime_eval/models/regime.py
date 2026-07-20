"""Regime segmentation with a Gaussian Hidden Markov Model.

We fit a 3-state ``GaussianHMM`` on ``[log_return, realized_vol_20d]`` and then
*automatically* attach human-readable labels (``trending`` / ``mean-reverting``
/ ``high-vol``) by inspecting each state's mean return and volatility. There is
no hardcoded ``state_index -> name`` mapping: the HMM assigns arbitrary integer
state ids on every fit, and ``label_regimes`` recovers the semantics from the
data. This is what makes the labelling robust to re-seeding and re-fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from .. import config

#: Features fed to the HMM, in column order.
FEATURE_COLUMNS: list[str] = ["log_return", "realized_vol_20d"]

#: The three regime labels this harness reasons about.
REGIME_LABELS: tuple[str, ...] = ("trending", "mean-reverting", "high-vol")


def compute_features(
    prices: pd.DataFrame,
    vol_window: int = config.REALIZED_VOL_WINDOW,
) -> pd.DataFrame:
    """Compute the HMM feature matrix from OHLCV prices.

    Args:
        prices: DataFrame with at least a ``close`` column, datetime-indexed.
        vol_window: window (in periods) for the realized-volatility feature.

    Returns:
        A copy of ``prices`` with added ``log_return`` and ``realized_vol_20d``
        columns, with the leading rows containing NaNs dropped.
    """
    frame = prices.copy()
    frame["log_return"] = np.log(frame["close"]).diff()
    frame["realized_vol_20d"] = frame["log_return"].rolling(vol_window).std()
    return frame.dropna(subset=FEATURE_COLUMNS)


def state_statistics(features: pd.DataFrame, states: np.ndarray) -> pd.DataFrame:
    """Summarise each hidden state by its return/volatility profile.

    Args:
        features: DataFrame containing ``FEATURE_COLUMNS``.
        states: integer state id per row (same length as ``features``).

    Returns:
        DataFrame indexed by state id with columns ``mean_return``,
        ``return_std``, ``mean_vol`` and ``n_periods``.
    """
    frame = features.copy()
    frame["state"] = states
    grouped = frame.groupby("state")
    stats = pd.DataFrame(
        {
            "mean_return": grouped["log_return"].mean(),
            "return_std": grouped["log_return"].std(),
            "mean_vol": grouped["realized_vol_20d"].mean(),
            "n_periods": grouped.size(),
        }
    )
    return stats


def label_regimes(stats: pd.DataFrame) -> dict[int, str]:
    """Map hidden-state ids to regime names from their statistics.

    Heuristic (no hardcoded mapping):

    1. The state with the **highest mean volatility** is ``high-vol`` — the
       chaotic, heavy-tailed regime where evaluation is least trustworthy.
    2. Of the two calmer states, the one with the **larger absolute mean
       return** is ``trending`` (a persistent directional drift that momentum
       is built to exploit).
    3. The remaining flat, low-drift state is ``mean-reverting``.

    Args:
        stats: output of :func:`state_statistics`.

    Returns:
        Dict mapping each state id to a regime label.
    """
    labels: dict[int, str] = {}

    high_vol_state = stats["mean_vol"].idxmax()
    labels[high_vol_state] = "high-vol"

    remaining = stats.drop(index=high_vol_state)
    if len(remaining) > 0:
        trending_state = remaining["mean_return"].abs().idxmax()
        labels[trending_state] = "trending"
        for state in remaining.drop(index=trending_state).index:
            labels[state] = "mean-reverting"

    return labels


def structural_breaks(
    series: pd.Series,
    penalty: float = 3.0,
    cost_model: str = "rbf",
    min_size: int = 15,
) -> list[pd.Timestamp]:
    """Detect change points in a series with ruptures (PELT).

    This is an *independent* cross-check on the HMM regime boundaries: PELT with
    an RBF cost knows nothing about the HMM, so agreement between the two methods
    is evidence the regimes are real structure rather than a modelling artefact.

    Args:
        series: the series to segment (e.g. realized volatility), datetime-indexed.
        penalty: PELT penalty; higher -> fewer breakpoints.
        cost_model: ruptures cost model ("rbf", "l2", ...).
        min_size: minimum segment length between change points.

    Returns:
        List of change-point timestamps (the final boundary is dropped).
    """
    import ruptures as rpt

    clean = series.dropna()
    values = clean.to_numpy().reshape(-1, 1)
    algo = rpt.Pelt(model=cost_model, min_size=min_size).fit(values)
    breakpoints = algo.predict(pen=penalty)
    return [clean.index[b] for b in breakpoints[:-1]]  # last index == len(series)


@dataclass
class RegimeDetector:
    """A fitted regime detector wrapping a GaussianHMM and its auto-labels.

    Typical use::

        detector = RegimeDetector().fit(prices)
        regimes = detector.regime_series()      # pd.Series of labels
        detector.labeled_statistics()           # per-regime profile table

    Attributes:
        n_states: number of HMM states.
        vol_window: realized-vol feature window.
        seed: random seed for reproducible fits.
    """

    n_states: int = config.HMM_STATES
    vol_window: int = config.REALIZED_VOL_WINDOW
    seed: int = config.RANDOM_SEED
    n_iter: int = 200

    # populated by fit()
    features_: pd.DataFrame = field(default=None, repr=False)
    model_: GaussianHMM = field(default=None, repr=False)
    scaler_: StandardScaler = field(default=None, repr=False)
    states_: np.ndarray = field(default=None, repr=False)
    stats_: pd.DataFrame = field(default=None, repr=False)
    label_map_: dict[int, str] = field(default=None, repr=False)

    def fit(self, prices: pd.DataFrame) -> "RegimeDetector":
        """Fit the HMM on price data and auto-label the resulting states.

        Args:
            prices: OHLCV DataFrame with a ``close`` column.

        Returns:
            ``self`` (fitted), for chaining.
        """
        self.features_ = compute_features(prices, self.vol_window)
        raw = self.features_[FEATURE_COLUMNS].to_numpy()

        # Standardize so the HMM's Gaussian emissions see comparable scales.
        self.scaler_ = StandardScaler().fit(raw)
        design = self.scaler_.transform(raw)

        self.model_ = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.seed,
        )
        self.model_.fit(design)
        self.states_ = self.model_.predict(design)

        # Label from statistics computed on the *original* (unscaled) features.
        self.stats_ = state_statistics(self.features_, self.states_)
        self.label_map_ = label_regimes(self.stats_)
        return self

    def _check_fitted(self) -> None:
        if self.model_ is None:
            raise RuntimeError("RegimeDetector is not fitted; call .fit(prices) first.")

    def regime_series(self) -> pd.Series:
        """Return a per-timestamp Series of regime labels."""
        self._check_fitted()
        labels = [self.label_map_[s] for s in self.states_]
        return pd.Series(labels, index=self.features_.index, name="regime")

    def labeled_statistics(self) -> pd.DataFrame:
        """Return the per-state statistics table with a ``regime`` label column.

        Rows are ordered by the canonical regime order for readability.
        """
        self._check_fitted()
        table = self.stats_.copy()
        table["regime"] = [self.label_map_[s] for s in table.index]
        order = {name: i for i, name in enumerate(REGIME_LABELS)}
        table = table.sort_values("regime", key=lambda s: s.map(order))
        return table[["regime", "mean_return", "return_std", "mean_vol", "n_periods"]]
