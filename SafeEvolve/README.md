# SafeEvolve

**A minimal, runnable research prototype of agentic safety + gated self-evolution in non-stationary, heavy-tailed financial markets.**

> *Safety is essential for the high-stakes financial industry, and the financial market provides ideal non-stationary feedback for self-evolution.*

SafeEvolve is a ~900-line, dependency-light Python artifact built to make one argument concrete: **an agent that can rewrite its own policy must have that ability gated by its safety state — otherwise it adapts *into* the very tail events it should be surviving.** It is not a UI demo and not a toy; it is a research scaffold a reviewer can read end-to-end in one sitting, run in one command, and extend.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python eval/run_experiment.py        # prints a summary table, writes results/comparison.png
```

---

## 1. The problem: why standard ML agents fail here

Most learned trading policies quietly assume the world is **stationary** (the data-generating process is fixed) and **light-tailed** (returns are roughly Gaussian). Real markets violate both, and the violations are not academic:

- **Non-stationarity.** Markets move through regimes — trending, choppy/mean-reverting, panic/high-volatility — and switch between them *without warning*. A policy fit to a bull trend is actively wrong in a mean-reverting chop. There is no label saying "the regime just changed"; the agent must infer it from the data it is simultaneously trading on.
- **Heavy tails.** Extreme moves are far more common than a Gaussian predicts. Under a normal distribution a −20% day is a ~1-in-10²⁰ event; in reality they happen every few years. An agent trained on Gaussian assumptions systematically under-prices tail risk and is over-leveraged exactly when it is most dangerous.

A naive agent that reads a signal and sizes into it will look great in the trending regime it was born in and then get destroyed by (a) the first regime switch it doesn't notice and (b) the first heavy-tailed shock it wasn't sized for. In the shipped experiment, that agent ends the run **down 84%**.

## 2. The core tension: self-evolution vs. safety

Self-evolution is attractive precisely *because* markets are non-stationary: a static policy decays, so the agent should keep re-fitting itself to recent feedback. But financial feedback is **noisy, delayed, and regime-dependent**, which creates a specific and dangerous failure mode:

> If the agent updates its policy using performance measured **during** a tail event or an unrecognized regime, it fits to a signal that is about to invert. It learns *"leverage paid off recently"* moments before the payoff reverses — and the update makes the **next** tail event worse.

This is the crux. Self-evolution requires feedback; the feedback is least trustworthy exactly when it is most tempting to act on. **The resolution is architectural: evolution is a privilege that must be earned by a confident, non-distressed state.** The agent may only rewrite its policy when it (a) is solvent (no drawdown breaker tripped) and (b) actually knows what regime it is in (detector confidence above threshold). Otherwise learning is *blocked* and the distressed-window feedback is *discarded*, never learned from.

## 3. Architecture

Safety here is not a post-hoc filter on the output — it is a set of hard constraints that sit **between intent and execution** and can veto or shrink any action, plus a **gate** that governs when the agent is allowed to change itself.

```
                 ┌──────────────────────────────────────────────────────┐
   market        │  BOCPD regime detector                               │
   return  ──────▶  → regime estimate + calibrated confidence           │
                 └───────────────┬──────────────────────────────────────┘
                                 │ regime, confidence
                                 ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │  SafeAgent                                                           │
   │   policy proposes intent  ──▶  SAFETY LAYER (hard, pre-execution):   │
   │                                 1. drawdown circuit breaker (>15%)   │
   │                                 2. uncertainty gate (conf / vol)     │
   │                                 3. per-trade position cap (10%)       │
   │   every decision logged with a machine-readable reason string        │
   └───────────────┬─────────────────────────────────────────────────────┘
                   │ safe_mode, confidence
                   ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │  Evolver  — re-fits per-regime size scales every 50 steps            │
   │   BLOCKED if safe_mode OR confidence < threshold  ◀── the gate        │
   └─────────────────────────────────────────────────────────────────────┘
```

| Component | File | What it does & why |
|---|---|---|
| **Market** | [`env/market.py`](env/market.py) | Stitches trending / mean-reverting / high-vol regimes of random length with no switch signal. Innovations are **Student-t (df=3)** — genuinely fat-tailed — and discrete tail shocks (−20% crashes, vol spikes) are injected at random intervals. Hidden regime labels are exposed *only* for evaluation. |
| **Regime detector** | [`agent/regime_detector.py`](agent/regime_detector.py) | **Bayesian Online Changepoint Detection** (Adams & MacKay 2007) with a Normal-Inverse-Gamma / Student-t observation model. Maintains a full posterior over *run length* (time since last changepoint); when a regime switches, mass collapses toward run-length 0, which we surface as a **calibrated confidence drop**. The current segment's volatility and lag-1 autocorrelation classify it into one of the three regimes. |
| **Safe agent** | [`agent/safe_agent.py`](agent/safe_agent.py) | Interpretable regime-conditioned policy (ride momentum in trends, fade deviations in mean-reversion, flatten in high-vol) wrapped in three **hard** guards, all applied before any action executes. Every decision is logged with timestamp, action, size, regime, confidence, portfolio value, and a reason string. |
| **Evolution** | [`agent/evolution.py`](agent/evolution.py) | Online EMA-of-Sharpe update to per-regime size scales — **gated** by safety state. Blocked attempts discard their buffered feedback. Every attempt (allowed or blocked) is logged with its reason and performance signal. |
| **Naive baseline** | [`baselines/naive_agent.py`](baselines/naive_agent.py) | Competent but unguarded full-size momentum trader: no regime detection, no breaker, no gate, no cap. The control group. |
| **Evaluation** | [`eval/run_experiment.py`](eval/run_experiment.py), [`metrics.py`](eval/metrics.py), [`plots.py`](eval/plots.py) | Runs both agents on the **same** market stream, prints an audit table, and writes a single six-panel figure. |

### Why these specific choices

- **Student-t, not Gaussian.** df=3 gives finite variance but *infinite* kurtosis, so 6σ moves appear on their own. Under Gaussian noise the agent would never learn to respect tails. Innovations are rescaled to unit variance so tail-fatness (df) is decoupled from volatility.
- **BOCPD, not an EWMA.** An EWMA silently blends pre- and post-switch data, so right after a regime change it returns a stale point estimate — with no signal that it is now unreliable. BOCPD instead produces an *honest uncertainty*: its run-length posterior collapses at a changepoint, and that collapse is the confidence signal the safety layer and the evolution gate consume. Uncertainty is the product, not a byproduct.
- **Block evolution in safe mode / low confidence.** This is the thesis in one line: you cannot update your own policy when you don't know what's happening. Learning from distressed-window feedback fits to a signal that is about to invert.

## 4. How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python eval/run_experiment.py
```

Outputs: a summary table + a sample of gated evolution decisions to stdout, the six-panel `results/comparison.png`, and two CSV audit trails — `results/safe_agent_log.csv` (every action with its reason string) and `results/evolution_log.csv` (every allowed/blocked update). Resample the entire market with a different world:

```bash
SAFEEVOLVE_SEED=101 python eval/run_experiment.py
```

Python 3.11+; `numpy`, `scipy`, `matplotlib`, `pandas`. No deep-learning framework — everything is interpretable and auditable by design.

## 5. Key results

Default run (`seed=34`, 1000 steps, 19 tail events — 13 crashes + 6 vol spikes):

| Metric | **SafeEvolve** | Naive baseline |
|---|---:|---:|
| Total return | **+3.2%** | −84.3% |
| Max drawdown | **15.2%** | 87.9% |
| Sharpe (annualized) | **0.14** | −0.61 |
| Tail events survived | **18 / 19** | 10 / 19 |
| Safe-mode halts | 1 | n/a |
| Evolutions allowed / blocked | 11 / 8 | n/a |
| Regime accuracy (overall) | 68.4% | n/a |

![comparison](results/comparison.png)

What the figure shows, panel by panel:

1. **Portfolio value (log).** SafeEvolve stays near its starting capital; the naive agent bleeds ~85% through the same shocks.
2. **Drawdown.** The naive agent runs at 60–88% drawdown for most of the run. SafeEvolve stays in low single digits until a late cluster of shocks pushes it to 15%, where the **circuit breaker trips** and freezes further risk.
3. **Regime detection.** BOCPD tracks the true regime through switches; the flips in the final third (a long trending stretch it partly misreads) are exactly why overall accuracy is 68%, not 95% — honestly reported.
4. **Rolling detection accuracy.** Well above the 33% random floor, with visible dips right after regime changes (the confidence signal that de-risks the agent).
5. **Self-evolution gate.** Allowed updates (green) happen in calm, confident windows; **blocks (red ✗) cluster in the turbulent late period** — learning shuts off precisely when the feedback is least trustworthy. This panel *is* the thesis.
6. **Scorecard.** The table above, rendered into the figure so the PNG is self-contained.

The headline is deliberately **not** "SafeEvolve makes more money." In a heavy-tailed world the right scorecard is capital preservation: SafeEvolve survives 18 of 19 tail events and caps its worst loss at 15%, while the unguarded agent — running the *same* alpha signal — is effectively wiped out. The safety layer is the difference between a strategy and a smoking crater.

> Note the honest limitation visible in panel 2: once the breaker trips, this prototype stays in hold-only safe mode for the rest of the run (it can't climb back to within 5% of its peak while flat). That is deliberately conservative for a demo; §6 describes the graded re-risking a production system would add.

## 6. What this looks like inside a real brokerage (e.g. D0)

This prototype is the *control skeleton* an AI brokerage like [D0](https://getdonut.ai) would wrap around real order flow. The mapping is direct:

- **The safety layer becomes a pre-trade risk engine.** The 10%-per-trade cap, drawdown breaker, and uncertainty gate move in front of the order router as hard, non-bypassable checks. Every order carries the same reason-string audit trail — which is also what a compliance/risk desk and a regulator need to see. *Safety is architectural, enforced before execution, not a model output you hope is calibrated.*
- **BOCPD confidence becomes a firm-wide risk dial.** A calibrated "we don't know what regime this is" signal is exactly what should throttle sizing, widen stops, and — critically — **freeze model self-updates** across the book during a flash crash or a liquidity vacuum.
- **The evolution gate becomes a model-governance control.** Continuous learning is valuable but is the scariest thing to run unsupervised on real capital. Gating updates on solvency + confidence, discarding distressed-window data, and logging every allowed/blocked update with its trigger is a concrete, auditable answer to *"when is the agent allowed to change itself?"*
- **What production adds (and this prototype intentionally omits):** graded re-risking out of safe mode instead of a hard freeze; transaction-cost/slippage/liquidity modeling; multi-asset portfolio constraints and correlation-aware limits; a proper walk-forward / out-of-sample evaluation harness; and human-in-the-loop sign-off on evolution events above a size threshold. The architecture here is designed to accept all of these without changing its shape.

## 7. Repository layout

```
SafeEvolve/
├── env/market.py                 # non-stationary, heavy-tailed market simulator
├── agent/
│   ├── regime_detector.py        # BOCPD: regime estimate + calibrated confidence
│   ├── safe_agent.py             # policy + hard safety guards (pre-execution)
│   └── evolution.py              # safety-GATED online self-evolution
├── baselines/naive_agent.py      # unguarded control agent
├── eval/
│   ├── run_experiment.py         # entry point: run both agents, print, plot
│   ├── metrics.py                # drawdown, Sharpe, tail-survival, accuracy
│   └── plots.py                  # the single six-panel comparison figure
├── results/
│   ├── comparison.png            # generated figure (committed; embedded above)
│   ├── safe_agent_log.csv        # generated per-step decision audit trail
│   └── evolution_log.csv         # generated allowed/blocked evolution audit trail
├── requirements.txt
└── README.md
```

Every file is under 200 lines and every class/method carries a docstring explaining the **research motivation**, not just the mechanics. Every safety and evolution decision is logged with a reason string so the behavior is fully auditable — you should be able to answer *"why did the agent do (or not do) this?"* for any step in the run.

## 8. What this is meant to demonstrate

1. **Safety in agentic systems is architectural, not a post-hoc filter** — it lives between intent and execution and can always veto.
2. **Non-stationarity and heavy tails are the core challenge**, not decoration — the whole design is organized around detecting regime switches and surviving fat-tailed shocks.
3. **Self-evolution must be gated by safety state** — the specific, load-bearing insight that an ungated learner adapts into a tail event and makes the next one worse, demonstrated live in panel 5.
4. **The code is clean enough to read, run, and extend** — not just impressive from the outside.

---

*Reproducibility: fully deterministic given a seed (default 34). `numpy`'s `default_rng` seeds the market; agents and detector are deterministic. `SAFEEVOLVE_SEED=<n>` resamples the world.*
