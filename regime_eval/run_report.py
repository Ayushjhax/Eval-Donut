"""Headless entry point: run the whole harness and write the README figure.

``notebooks/main.ipynb`` is the exploratory surface. This module is the same
pipeline with no display server and no Jupyter kernel required: it prints every
number the README quotes and writes ``result.png`` next to this file.

Run from the repository root::

    python -m regime_eval.run_report
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: render to PNG without a display server

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .data import load_price_data
from .eval import (
    aggregate_metrics,
    cvar,
    max_drawdown,
    per_regime_metrics,
    sharpe_ratio,
    tail_ratio,
    run_shift_detection,
)
from .evolution import run_self_evolution
from .models import RegimeDetector, structural_breaks
from .strategy import strategy_returns

REGIME_ORDER = ["trending", "mean-reverting", "high-vol"]
REGIME_COLORS = {"trending": "#2a9d8f", "mean-reverting": "#4c6ef5", "high-vol": "#e63946"}
FIGURE_PATH = config.PACKAGE_ROOT / "result.png"


def _equity(returns: pd.Series) -> pd.Series:
    return np.exp(returns.fillna(0.0).cumsum())


def build_figure(
    per_regime: pd.DataFrame,
    aggregate: pd.Series,
    compare: pd.DataFrame,
    cvar_col: str,
    avg_size: float,
    out_path=FIGURE_PATH,
) -> None:
    """Render the one figure the README embeds.

    Left and middle panels are the finding: Sharpe ranks the regimes in the
    opposite order to the tail metric, and the aggregate (dashed) describes
    neither. The right panel is the honest control -- adaptive de-risking versus
    uniform de-risking at the same average exposure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    colors = [REGIME_COLORS[r] for r in per_regime.index]

    ax = axes[0]
    ax.bar(per_regime.index, per_regime["Sharpe (ref only)"], color=colors)
    ax.axhline(aggregate["Sharpe (ref only)"], ls="--", color="0.25", lw=1.2)
    ax.text(
        0.98,
        aggregate["Sharpe (ref only)"],
        f" aggregate {aggregate['Sharpe (ref only)']:+.2f}",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=9,
        color="0.25",
    )
    for i, v in enumerate(per_regime["Sharpe (ref only)"]):
        ax.text(i, v, f"{v:+.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Sharpe ranks high-vol BEST", fontsize=12, fontweight="bold")
    ax.set_ylabel("annualized Sharpe")
    ax.set_ylim(0, max(per_regime["Sharpe (ref only)"]) * 1.25)

    ax = axes[1]
    ax.bar(per_regime.index, per_regime[cvar_col] * 100, color=colors)
    ax.axhline(aggregate[cvar_col] * 100, ls="--", color="0.25", lw=1.2)
    ax.text(
        0.02,
        aggregate[cvar_col] * 100,
        f"aggregate {aggregate[cvar_col]:.1%}",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="top",
        fontsize=9,
        color="0.25",
    )
    for i, v in enumerate(per_regime[cvar_col] * 100):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="top", fontsize=10)
    ax.set_title("Tail risk ranks it WORST", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{cvar_col} (expected shortfall, %)")
    ax.set_ylim(min(per_regime[cvar_col] * 100) * 1.18, 0)

    ax = axes[2]
    labels = ["baseline\n(1.00x)", f"uniform\n({avg_size:.2f}x)", "adaptive\n(safety loop)"]
    bars = ax.bar(labels, compare["max_drawdown"] * 100, color=["#adb5bd", "#e9a33c", "#2a9d8f"])
    for rect, v in zip(bars, compare["max_drawdown"] * 100):
        ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.1f}%", ha="center", va="top", fontsize=10)
    ax.set_title("Targeted beats blunt de-risking", fontsize=12, fontweight="bold")
    ax.set_ylabel("max drawdown (%)")
    ax.set_ylim(min(compare["max_drawdown"] * 100) * 1.2, 0)

    for ax in axes:
        ax.axhline(0, color="0.6", lw=0.8)
        ax.tick_params(labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "One backtest number cannot rank regimes: same strategy, opposite verdicts",
        fontsize=13.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run() -> None:
    prices = load_price_data()
    detector = RegimeDetector().fit(prices)
    regimes = detector.regime_series()

    data = prices.loc[regimes.index].copy()
    data["regime"] = regimes
    data["log_return"] = detector.features_["log_return"]

    print("=" * 72)
    print("regime_eval report")
    print(
        f"{len(prices)} daily candles | {prices.index.min().date()} -> "
        f"{prices.index.max().date()} | source: {prices.attrs.get('source')}"
    )
    print("=" * 72)

    print("\n[1] Auto-derived regime labels (no hardcoded state -> name map)")
    print(detector.labeled_statistics().to_string(float_format=lambda v: f"{v:.4f}"))
    share = data["regime"].value_counts(normalize=True).reindex(REGIME_ORDER) * 100
    print("\nregime share of days:")
    for name, pct in share.items():
        print(f"  {name:<16}{pct:5.1f}%")

    breaks = structural_breaks(detector.features_["realized_vol_20d"])
    transitions = data.index[data["regime"].ne(data["regime"].shift()).fillna(False)][1:]
    agree = sum(min(abs((t - b).days) for b in breaks) <= 7 for t in transitions)
    print(
        f"\n[2] PELT cross-check: {len(transitions)} HMM transitions, {len(breaks)} PELT "
        f"change points, {agree}/{len(transitions)} ({agree / len(transitions):.0%}) of HMM "
        "transitions within 7 days of a PELT breakpoint"
    )

    bt = strategy_returns(prices)
    data["strat_return"] = bt["strategy_return"].reindex(data.index)
    per_regime = per_regime_metrics(data["strat_return"], data["regime"])
    aggregate = aggregate_metrics(data["strat_return"])
    cvar_col = f"CVaR_{int(config.CVAR_LEVEL * 100)}"

    print("\n[3] Per-regime evaluation vs the single aggregate number")
    combined = pd.concat([per_regime, aggregate.to_frame().T])
    print(combined.to_string(float_format=lambda v: f"{v:.4f}"))

    sharpes = per_regime["Sharpe (ref only)"]
    print(
        f"\naggregate Sharpe {aggregate['Sharpe (ref only)']:+.2f} | per-regime "
        f"{sharpes.min():+.2f} ({sharpes.idxmin()}) -> {sharpes.max():+.2f} "
        f"({sharpes.idxmax()}) = {sharpes.max() / sharpes.min():.1f}x spread"
    )
    print(
        f"high-vol: Sharpe {per_regime.loc['high-vol', 'Sharpe (ref only)']:+.2f} (best) but "
        f"{cvar_col} {per_regime.loc['high-vol', cvar_col]:.1%} vs "
        f"{per_regime.loc['mean-reverting', cvar_col]:.1%} in the calm regime "
        f"({per_regime.loc['high-vol', cvar_col] / per_regime.loc['mean-reverting', cvar_col]:.1f}x "
        f"the tail loss), max drawdown {per_regime.loc['high-vol', 'max_drawdown']:.1%}"
    )

    shift = run_shift_detection(data["log_return"], data["regime"])
    data["severity"] = shift.severity
    fired = shift.detected.fillna(False)
    onsets = data.index[fired & ~fired.shift(1, fill_value=False)]
    print(f"\n[4] Distribution shift: reference regime = {shift.reference_regime}")
    print(
        f"fired on {int(fired.sum())} days ({fired.mean():.0%}) across {len(onsets)} episodes; "
        "mean severity by regime = "
        + str({k: round(float(v), 3) for k, v in data.groupby("regime")["severity"].mean().reindex(REGIME_ORDER).items()})
    )

    evo = run_self_evolution(shift.frame, data["regime"], log_path=config.LOG_DIR / "evolution_events.jsonl")
    n_red = sum(e.event == "size_reduction" for e in evo.events)
    n_res = sum(e.event == "size_restored" for e in evo.events)
    print(f"\n[5] Self-evolution: {len(evo.events)} logged events ({n_red} reductions / {n_res} restorations)")

    avg_size = float(evo.size_factor.mean())
    baseline = data["strat_return"]
    uniform = strategy_returns(prices, position_size=avg_size)["strategy_return"].reindex(data.index)
    adaptive = strategy_returns(prices, position_size=evo.size_factor)["strategy_return"].reindex(data.index)

    def row(r: pd.Series) -> list[float]:
        return [sharpe_ratio(r), cvar(r), max_drawdown(r), tail_ratio(r), float(_equity(r).iloc[-1])]

    compare = pd.DataFrame(
        [row(baseline), row(uniform), row(adaptive)],
        index=["baseline (1.00x)", f"uniform ({avg_size:.2f}x)", "adaptive (safety loop)"],
        columns=["Sharpe (ref only)", cvar_col, "max_drawdown", "tail_ratio", "terminal_wealth"],
    )
    print(f"\n[6] Safety loop vs the honest control (uniform deleverage at {avg_size:.2f}x average exposure)")
    print(compare.to_string(float_format=lambda v: f"{v:.4f}"))
    print(
        f"\nadaptive - uniform: {cvar_col} {compare.loc['adaptive (safety loop)', cvar_col] - compare.loc[f'uniform ({avg_size:.2f}x)', cvar_col]:+.4f}, "
        f"max_drawdown {compare.loc['adaptive (safety loop)', 'max_drawdown'] - compare.loc[f'uniform ({avg_size:.2f}x)', 'max_drawdown']:+.4f} "
        "(both positive = smaller loss)"
    )

    build_figure(per_regime, aggregate, compare, cvar_col, avg_size)
    print(f"\nFigure written to {FIGURE_PATH}")
    print(f"Evolution audit log written to {config.LOG_DIR / 'evolution_events.jsonl'}\n")


if __name__ == "__main__":
    run()
