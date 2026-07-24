"""
Entry point. Runs SafeEvolve vs the naive baseline on one
shared non-stationary market stream, prints an audit-friendly summary, and writes a
single comparison figure to results/comparison.png.

Run:  python eval/run_experiment.py

Timing convention (no look-ahead): at step t the agents see information through
returns[0..t] and choose an exposure; the P&L they are graded on is returns[t+1],
which is realized only after the decision is locked in.
"""

from __future__ import annotations

import os
import sys

# Make `python eval/run_experiment.py` work from anywhere by putting the project
# root (the parent of this file's directory) on the import path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from env.market import MarketSimulator, MarketConfig
from agent.regime_detector import BOCPDRegimeDetector, DetectorConfig
from agent.safe_agent import SafeAgent, SafetyConfig
from agent.evolution import Evolver, EvolutionConfig
from baselines.naive_agent import NaiveAgent
from eval import metrics
from eval.plots import build_figure

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def compute_features(returns: np.ndarray, prices: np.ndarray, t: int) -> dict:
    """Causal features from data available at step t (returns[0..t] only).

    mom        -- 5-step mean return (momentum signal for the trend policy)
    recent_vol -- 20-step volatility (fast risk gauge, feeds the vol-spike gate)
    avg_vol    -- 60-step volatility (slow baseline the spike is measured against)
    price_dev  -- price gap vs its 20-step moving average (mean-reversion signal)
    """
    r = returns[max(0, t - 4): t + 1]
    mom = float(r.mean())
    recent = returns[max(0, t - 19): t + 1]
    recent_vol = float(recent.std(ddof=1)) if recent.size > 1 else 0.0
    base = returns[max(0, t - 59): t + 1]
    avg_vol = float(base.std(ddof=1)) if base.size > 1 else 0.0
    ma_window = prices[max(0, t - 19): t + 1]
    price_dev = float(prices[t] / ma_window.mean() - 1.0)
    return {"mom": mom, "recent_vol": recent_vol, "avg_vol": avg_vol, "price_dev": price_dev}


def run() -> None:
    # Reviewers can resample the whole market with e.g. SAFEEVOLVE_SEED=101.
    cfg = MarketConfig()
    if os.environ.get("SAFEEVOLVE_SEED"):
        cfg = MarketConfig(seed=int(os.environ["SAFEEVOLVE_SEED"]))
    market = MarketSimulator(cfg)
    detector = BOCPDRegimeDetector(DetectorConfig())
    safety_cfg = SafetyConfig()
    safe = SafeAgent(safety_cfg)
    naive = NaiveAgent()
    evolver = Evolver(EvolutionConfig(conf_thresh=safety_cfg.conf_thresh))

    returns, prices = market.returns, market.prices
    T = market.cfg.n_steps
    est_regimes: list[str] = []
    halts = 0
    prev_safe_mode = False

    for t in range(T - 1):
        est = detector.update(returns[t])           # detector sees returns[0..t]
        feats = compute_features(returns, prices, t)
        safe.decide(t, feats, est.regime, est.confidence)
        naive.decide(t, feats)

        r_next = returns[t + 1]                      # graded on the NEXT return
        pnl_safe = safe.settle(r_next)
        naive.settle(r_next)

        # Track safe-mode entries (a distinct, auditable safety action).
        if safe.safe_mode and not prev_safe_mode:
            halts += 1
        prev_safe_mode = safe.safe_mode

        # Feed evolution with the regime the agent ACTED under, then attempt a gated
        # update. Low-confidence steps are tagged "uncertain" and never learned from.
        effective = est.regime if est.confidence >= safety_cfg.conf_thresh else "uncertain"
        evolver.record(effective, pnl_safe)
        evolver.maybe_evolve(t, safe, est.confidence)

        est_regimes.append(est.regime)

    _report(market, safe, naive, evolver, est_regimes, halts)


def _report(market, safe, naive, evolver, est_regimes, halts) -> None:
    true_labels = list(market.regimes[:len(est_regimes)])
    roll_acc = metrics.rolling_accuracy(true_labels, est_regimes, window=50)
    overall_acc = float(np.mean([t == e for t, e in zip(true_labels, est_regimes)]))
    s_surv, n_tail = metrics.tail_survival(np.array(safe.pv_history), market.tail_events)
    n_surv, _ = metrics.tail_survival(np.array(naive.pv_history), market.tail_events)

    summary = {
        "n_tail": n_tail,
        "overall_acc": overall_acc,
        "safe": {
            "ret": metrics.total_return(safe.pv_history),
            "mdd": metrics.max_drawdown(np.array(safe.pv_history)),
            "sharpe": metrics.sharpe(np.array(safe.pv_history)),
            "surv": s_surv, "halts": halts,
        },
        "naive": {
            "ret": metrics.total_return(naive.pv_history),
            "mdd": metrics.max_drawdown(np.array(naive.pv_history)),
            "sharpe": metrics.sharpe(np.array(naive.pv_history)),
            "surv": n_surv,
        },
    }

    # --- stdout summary table -------------------------------------------------
    print("\n" + "=" * 64)
    print("SafeEvolve experiment")
    print(market.summary())
    print("=" * 64)
    hdr = f"{'metric':<28}{'SafeEvolve':>14}{'Naive baseline':>16}"
    print(hdr)
    print("-" * 64)
    rows = [
        ("total return", f"{summary['safe']['ret']:.1%}", f"{summary['naive']['ret']:.1%}"),
        ("max drawdown", f"{summary['safe']['mdd']:.1%}", f"{summary['naive']['mdd']:.1%}"),
        ("Sharpe (annualized)", f"{summary['safe']['sharpe']:.2f}", f"{summary['naive']['sharpe']:.2f}"),
        ("tail events survived", f"{s_surv}/{n_tail}", f"{n_surv}/{n_tail}"),
        ("safe-mode halts", str(halts), "n/a"),
        ("evolutions allowed", str(evolver.n_allowed), "n/a"),
        ("evolutions blocked", str(evolver.n_blocked), "n/a"),
        ("regime accuracy (overall)", f"{overall_acc:.1%}", "n/a"),
    ]
    for name, a, b in rows:
        print(f"{name:<28}{a:>14}{b:>16}")
    print("-" * 64)

    # A couple of sampled evolution decisions so the gate is visible in the log too.
    print("\nSample self-evolution decisions (safety gate in action):")
    shown = 0
    for ev in evolver.events:
        if shown >= 6:
            break
        tag = "ALLOW" if ev["status"] == "allowed" else "BLOCK"
        detail = ev["reason"] if ev["status"] == "blocked" else ", ".join(
            f"{k}:{v['old']}->{v['new']}" for k, v in ev["changes"].items()) or "no eligible regime"
        print(f"  t={ev['t']:>4} [{tag}] conf={ev['confidence']:.2f}  {detail}")
        shown += 1

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Export the full audit trail as CSV so a researcher can inspect the reason string
    # behind every single action and every evolution decision, step by step.
    pd.DataFrame(safe.log).to_csv(os.path.join(RESULTS_DIR, "safe_agent_log.csv"), index=False)
    pd.DataFrame(evolver.events).to_csv(os.path.join(RESULTS_DIR, "evolution_log.csv"), index=False)
    out_path = os.path.join(RESULTS_DIR, "comparison.png")
    build_figure(out_path, market, safe, naive, evolver, est_regimes, roll_acc, summary)
    print(f"\nFigure written to {out_path}")
    print(f"Audit logs written to {RESULTS_DIR}/safe_agent_log.csv, evolution_log.csv\n")


if __name__ == "__main__":
    run()
