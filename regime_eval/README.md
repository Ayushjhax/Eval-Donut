# Regime-Aware Evaluation for a Trading Agent

A small, self-contained harness that argues one point:

> **In non-stationary, heavy-tailed markets, a single backtest number is not just noisy — it is invalid. An agent that knows *when its own evaluation has gone stale* is safer than one that just optimizes harder.**

Built as a demo for [**Donut Labs**](https://donut.xyz) — agentic safety and self-evolution in non-stationary, heavy-tailed financial markets. It makes Donut's core tension legible: *"making numbers that make sense AND making numbers higher"* — eval validity and optimization, held at the same time, while the distribution keeps shifting.

The engine is a set of typed, docstringed, importable modules under `regime_eval/`. The demo surface is a single notebook, [`notebooks/main.ipynb`](notebooks/main.ipynb), that runs top-to-bottom against a cached CSV with **no network or API keys required**.

![Price by regime with distribution-shift onsets](assets/price_regimes_shift.png)

---

## The argument in five steps

1. **Segment the market into regimes.** A 3-state `GaussianHMM` on `[log_return, realized_vol_20d]` splits SOL/USDT history into `trending`, `mean-reverting`, and `high-vol`. Regimes are **auto-labelled from their own statistics** — no hardcoded `state → name` map — and independently cross-checked with PELT change-point detection (`ruptures`): **~72% of HMM regime transitions land within 7 days of a PELT breakpoint.**

2. **Evaluate per regime, never aggregate.** The same momentum strategy is scored *inside each regime* with tail-aware metrics. This table is the whole point:

   | regime | Sharpe *(ref only)* | CVaR&nbsp;95 | max drawdown | tail ratio |
   |---|---:|---:|---:|---:|
   | trending | +0.63 | −11.0% | −81% | 1.07 |
   | mean-reverting | +0.41 | −7.1% | −71% | 1.10 |
   | **high-vol** | **+0.98** | **−22.3%** | **−90%** | 1.25 |
   | *ALL (aggregate)* | *+0.63* | *−12.9%* | *−85%* | *1.14* |

   The aggregate Sharpe (+0.63) hides a **2.4× spread** across regimes. Worse: **Sharpe crowns `high-vol` the *best* regime (+0.98)** while its CVaR is **3.1× the tail loss** of the calm regime. *An optimizer maximizing Sharpe would lever straight into the regime most likely to blow it up.* That is "numbers higher" fighting "numbers that make sense."

3. **Measure the tails, not the average.** Sharpe assumes thin tails; these don't. We use **CVaR / expected shortfall** (the mean loss inside the worst 5%), **max drawdown as a rolling *distribution*** (not one number), and a **tail ratio** (upside tail ÷ downside tail). Sharpe is retained everywhere *only* as a labelled reference — never a decision variable.

4. **Detect distribution shift — "is my eval still valid?"** A rolling 30-day `scipy.stats.ks_2samp` test compares recent returns to the regime the eval is anchored to (the thinnest-tailed, most benign regime, where a backtest's numbers can be trusted). Severity is the normalized KS statistic; the shift **fires** when the live distribution has drifted out of distribution.

5. **Close the loop with self-evolution.** When a shift fires the agent shrinks its own exposure (`SHRINK_FACTOR`); when severity falls back below `RECOVERY_THRESHOLD` it restores it. Every self-modification is logged in a structured, auditable record.

   > **detect shift → eval is now invalid → reduce exposure → re-evaluate → restore when back in-distribution**

---

## The four notebook outputs

| # | Output | What it shows |
|---|---|---|
| 1 | **Price timeline coloured by regime + shift markers** | where each regime lives, and when the eval went stale ([above](#regime-aware-evaluation-for-a-trading-agent)) |
| 2 | **Per-regime performance table** (styled) | the divergence the aggregate number hides |
| 3 | **Overlaid return distributions with tail annotations** | the heavy left tail of `high-vol`, exposed on a log-density axis |
| 4 | **Self-modification event timeline** | when the agent changed its own behaviour, and why |

<p align="center">
  <img src="assets/return_distributions.png" width="49%" alt="Return distributions by regime">
  <img src="assets/evolution_timeline.png" width="49%" alt="Self-modification timeline">
</p>

### Did the safety loop help — or did it just deleverage?

The honest test compares the adaptive agent against **uniform deleveraging at the same average exposure (0.86×)**:

| strategy | Sharpe *(ref)* | CVaR 95 | max drawdown | terminal wealth |
|---|---:|---:|---:|---:|
| baseline (1.00×) | +0.64 | −12.9% | −85% | 73.3× |
| uniform (0.86×) | +0.64 | −11.1% | −81% | 40.2× |
| **adaptive (safety loop)** | +0.63 | **−10.4%** | **−77%** | 31.9× |

At **equal average exposure**, the targeted agent posts a smaller CVaR and shallower drawdown than blunt deleveraging — the shift detector *earns its keep* by cutting risk exactly where the eval is invalid. The safety loop is **not free**: it gives up compounded upside. But it buys tail safety at **no cost to risk-adjusted return (Sharpe)** and more efficiently than deleveraging blindly.

---

## Quickstart

```bash
cd regime_eval
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run the demo end-to-end (headless)
jupyter nbconvert --to notebook --execute --inplace notebooks/main.ipynb
# ...or open it interactively
jupyter lab notebooks/main.ipynb
```

First run fetches SOL/USDT daily OHLCV from Binance (via `ccxt`) and caches it to `data/cache/`. Every subsequent run — including the notebook — reads the CSV and needs no network.

---

## Project structure

```
regime_eval/
├── config.py            # every tunable in one place; paths derived, none hardcoded
├── data/                # price fetching + CSV caching (ccxt → CryptoCompare fallback)
│   └── fetch.py
├── models/              # GaussianHMM regime detector + auto-labelling + PELT cross-check
│   └── regime.py
├── strategy/            # baseline momentum strategy (causal, 1-day lag)
│   └── momentum.py
├── eval/                # per-regime heavy-tail metrics + KS distribution-shift detector
│   ├── metrics.py
│   └── shift.py
├── evolution/           # self-evolution stub: the safety loop + structured event log
│   └── self_evolve.py
├── notebooks/
│   └── main.ipynb       # the demo surface — runs top to bottom
├── requirements.txt
└── README.md
```

Each component is its own importable module with type hints and docstrings on all public functions:

```python
from regime_eval.data import load_price_data
from regime_eval.models import RegimeDetector
from regime_eval.strategy import strategy_returns
from regime_eval.eval import per_regime_metrics, run_shift_detection
from regime_eval.evolution import run_self_evolution

prices  = load_price_data()
regimes = RegimeDetector().fit(prices).regime_series()
```

---

## Configuration

All tunables live in [`config.py`](config.py):

| name | default | meaning |
|---|---|---|
| `SYMBOL` | `SOL/USDT` | market symbol (USDT — deeper Binance history than USDC) |
| `TIMEFRAME` | `1d` | candle timeframe |
| `HMM_STATES` | `3` | number of regimes |
| `MOMENTUM_WINDOW` | `20` | momentum lookback (periods) |
| `SHIFT_DETECTION_WINDOW` | `30` | rolling window for the KS test |
| `SHRINK_FACTOR` | `0.5` | size multiplier when a shift fires |
| `RECOVERY_THRESHOLD` | `0.2` | severity below which size is restored |
| `CVAR_LEVEL` | `0.95` | confidence level for CVaR / expected shortfall |

---

## Design notes & honest caveats

- **The regime labels are a descriptive overlay, not a tradeable signal.** The HMM is fit on the full sample and its state assignment uses information from the whole sequence, so a regime label at time *t* is not known causally at *t*. That is fine here — regimes are used to *evaluate* validity, not to trade. The momentum strategy itself is strictly causal (positions lag the signal by one day).
- **Self-evolution is a stub, deliberately.** The response to a shift is one legible control law (halve size, restore on recovery). The point is the *shape* of the loop — an agent reacting to the invalidation of its own eval — not a production risk model.
- **`high-vol` posts the highest mean return *and* the highest variance.** That is not a bug; it is the seed of the whole problem. Crypto's largest single-day moves cluster in the high-vol regime, which is exactly why a tail-blind metric (Sharpe) rates it best and a tail-aware one (CVaR) rates it worst.
- **Data source.** Binance via `ccxt` is primary. The CryptoCompare fallback now requires an API key (`CRYPTOCOMPARE_API_KEY`); it is attempted only if `ccxt` fails, and the CSV cache makes both irrelevant after the first successful fetch.

---

*Engine: `regime_eval/` — typed, docstringed, importable modules. Demo surface: `notebooks/main.ipynb`.*
