"""
A researcher should be able to see the whole story in one glance: where the naive
agent blows up, how the safe agent behaves through the same shocks, whether the regime
detector tracks reality, and -- the crux -- when self-evolution was allowed vs blocked.
Everything is drawn on a shared time axis so cause and effect line up vertically.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless: save a PNG without needing a display server
import matplotlib.pyplot as plt
import numpy as np

from eval.metrics import drawdown_series


def build_figure(out_path: str, market, safe, naive, evolver,
                 est_regimes, roll_acc, summary: dict) -> None:
    """Render the six-panel comparison figure to `out_path`."""
    T = market.cfg.n_steps
    x = np.arange(T)
    tail_x = [e["t"] for e in market.tail_events]
    fig, ax = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("SafeEvolve vs Naive Baseline -- non-stationary, heavy-tailed market",
                 fontsize=15, fontweight="bold")

    # (0,0) Portfolio value, log scale, with tail events marked.
    a = ax[0, 0]
    a.plot(safe.pv_history, label="SafeEvolve", color="#1b7837", lw=1.8)
    a.plot(naive.pv_history, label="Naive baseline", color="#b2182b", lw=1.3, alpha=0.9)
    for tx in tail_x:
        a.axvline(tx, color="gray", alpha=0.12, lw=1)
    a.set_yscale("log")
    a.set_title("Portfolio value (log scale) -- gray lines = tail events")
    a.set_ylabel("value (start = 1.0)")
    a.legend(loc="upper left")

    # (0,1) Drawdown curves + the safe agent's 15% circuit-breaker line.
    a = ax[0, 1]
    a.plot(drawdown_series(safe.pv_history) * 100, label="SafeEvolve", color="#1b7837")
    a.plot(drawdown_series(naive.pv_history) * 100, label="Naive", color="#b2182b", alpha=0.9)
    a.axhline(15, ls="--", color="black", lw=1, label="15% breaker")
    a.set_title("Drawdown from running peak")
    a.set_ylabel("drawdown (%)")
    a.legend(loc="upper left")

    # (1,0) True vs estimated regime (encoded as integers), + confidence trace.
    a = ax[1, 0]
    regime_ids = {"trending": 0, "mean_reverting": 1, "high_volatility": 2, "uncertain": 3}
    true_ids = [regime_ids[r] for r in market.regimes[:len(est_regimes)]]
    est_ids = [regime_ids.get(r, 3) for r in est_regimes]
    a.plot(true_ids, label="true regime", color="black", lw=2.4, alpha=0.5)
    a.step(range(len(est_ids)), est_ids, label="detected", color="#2166ac", lw=1, where="mid")
    a.set_yticks(list(regime_ids.values()))
    a.set_yticklabels(list(regime_ids.keys()), fontsize=8)
    a.set_title("Regime detection: true vs BOCPD estimate")
    a.legend(loc="upper right", fontsize=8)

    # (1,1) Rolling regime-detection accuracy.
    a = ax[1, 1]
    a.plot(roll_acc * 100, color="#762a83")
    a.axhline(100 / 3, ls=":", color="gray", label="random (33%)")
    a.set_ylim(0, 100)
    a.set_title("Rolling regime-detection accuracy (50-step window)")
    a.set_ylabel("accuracy (%)")
    a.legend(loc="lower right", fontsize=8)

    # (2,0) Evolution events: allowed vs blocked, over confidence.
    a = ax[2, 0]
    conf = [e["confidence"] for e in evolver.events]
    ev_t = [e["t"] for e in evolver.events]
    allow_t = [e["t"] for e in evolver.events if e["status"] == "allowed"]
    block_t = [e["t"] for e in evolver.events if e["status"] == "blocked"]
    a.scatter(allow_t, [1] * len(allow_t), color="#1b7837", label="evolution ALLOWED", zorder=3)
    a.scatter(block_t, [0] * len(block_t), color="#b2182b", marker="x",
              s=60, label="evolution BLOCKED", zorder=3)
    a.set_yticks([0, 1]); a.set_yticklabels(["blocked", "allowed"])
    a.set_ylim(-0.5, 1.5)
    a.set_title(f"Self-evolution gate: {evolver.n_allowed} allowed / {evolver.n_blocked} blocked")
    a.set_xlim(0, T)
    a.legend(loc="center right", fontsize=8)

    # (2,1) Final scorecard as a text panel (kept in-figure so the PNG is self-contained).
    a = ax[2, 1]; a.axis("off")
    lines = [
        f"{'metric':<26}{'SafeEvolve':>12}{'Naive':>12}",
        "-" * 50,
        f"{'total return':<26}{summary['safe']['ret']:>11.1%}{summary['naive']['ret']:>12.1%}",
        f"{'max drawdown':<26}{summary['safe']['mdd']:>11.1%}{summary['naive']['mdd']:>12.1%}",
        f"{'Sharpe (annualized)':<26}{summary['safe']['sharpe']:>12.2f}{summary['naive']['sharpe']:>12.2f}",
        f"{'tail events survived':<26}{summary['safe']['surv']:>12}{summary['naive']['surv']:>12}",
        f"{'(of total tail events)':<26}{summary['n_tail']:>12}{summary['n_tail']:>12}",
        "-" * 50,
        f"{'safe-mode halts':<26}{summary['safe']['halts']:>12}{'n/a':>12}",
        f"{'evolutions allowed':<26}{evolver.n_allowed:>12}{'n/a':>12}",
        f"{'evolutions blocked':<26}{evolver.n_blocked:>12}{'n/a':>12}",
        f"{'regime acc (overall)':<26}{summary['overall_acc']:>11.1%}{'n/a':>12}",
    ]
    a.text(0.0, 0.98, "\n".join(lines), family="monospace", fontsize=11,
           va="top", ha="left")
    a.set_title("Results summary", loc="left")

    for row in ax:
        for a in row:
            a.set_xlabel("step")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
