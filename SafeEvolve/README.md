# SafeEvolve

A simulated trading agent whose every action passes through three hard safety constraints, and whose self-updates are blocked whenever it is in drawdown or unsure which regime it is in.
It runs against a synthetic market that switches regimes without warning and carries Student-t (df=3) tails, and is compared to an unguarded full-size momentum trader on the same return path.
The claim: the guards cost some upside and buy a large reduction in drawdown and tail exposure.

## Result

One run, seed 34, 1,000 steps, 19 tail events (13 crashes, 6 volatility spikes). Both agents see the identical return path and identical P&L mechanics.

| Metric | SafeEvolve | Naive baseline |
| --- | ---: | ---: |
| Total return | +3.2% | -84.3% |
| Max drawdown | 15.2% | 87.9% |
| Sharpe (annualized) | 0.14 | -0.61 |
| Tail events survived | 18 / 19 | 10 / 19 |
| Safe-mode halts | 1 | n/a |
| Evolution steps allowed / blocked | 11 / 8 | n/a |
| Regime detection accuracy | 68.4% | n/a |

The naive agent is not a strawman — full-size momentum is a real strategy — but it holds maximum exposure through every shock and ends down 84.3% with an 87.9% drawdown, while the guarded agent's drawdown stops at 15.2% because the circuit breaker halts trading at 15%. "Survived" means the portfolio lost no more than 10% over the five steps following a shock, measured from just before it hit, and the guarded agent clears that bar on 8 more of the 19 shocks than the naive one does — which is what being small or flat going into a shock buys.

![](results/comparison.png)

## Reproduce

From a fresh clone. All commands are run from inside the `SafeEvolve` directory.

Linux / macOS:

```bash
git clone <repo-url> Donut
cd Donut/SafeEvolve
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python eval/run_experiment.py
```

Windows PowerShell:

```powershell
git clone <repo-url> Donut
cd Donut\SafeEvolve
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python eval/run_experiment.py
```

The differences are `python3` vs `python` for creating the virtual environment, the activation script path, and the path separator in `cd`. The run took 1.6 to 2.0 seconds on the machine it was tested on, prints the table above, overwrites `results/comparison.png` — byte-identically, since the market is seeded — and writes the per-step audit logs `results/safe_agent_log.csv` and `results/evolution_log.csv`.

To resample the entire market on a different seed — `SAFEEVOLVE_SEED=42` gives a run where the agent loses 13.2% instead of gaining 3.2%, on a path with 18 tail events instead of 19:

```bash
SAFEEVOLVE_SEED=42 python eval/run_experiment.py          # Linux / macOS
```

```powershell
$env:SAFEEVOLVE_SEED=42; python eval/run_experiment.py    # Windows PowerShell
```

A seeded run overwrites `results/comparison.png` like any other, so re-run without the variable set to restore the figure this README embeds.

## How it works

- The market (`env/market.py`) stitches together regimes of 60-160 steps drawn from `{trending, mean_reverting, high_volatility}`, each an AR(1) process with its own drift, autocorrelation sign and volatility, so every regime leaves a statistical signature but no switch is announced. Innovations are Student-t with 3 degrees of freedom: finite variance, infinite kurtosis. Crashes and volatility spikes are injected separately at 1.5% per step so individual tail events can be counted and scored.
- Regime detection (`agent/regime_detector.py`) is Bayesian online changepoint detection (Adams and MacKay, 2007) with a Normal-Inverse-Gamma prior, so the posterior predictive is itself Student-t. The chosen output is not the point estimate but the confidence: the probability that the run length is at least 8, which collapses right after a switch. An EWMA would blend across a changepoint and give a point estimate with no honest uncertainty attached. Classification then uses only data since the estimated changepoint: volatility above 0.020 is `high_volatility`, otherwise lag-1 autocorrelation below -0.12 is `mean_reverting`, else `trending`.
- Three guards sit between intent and execution (`agent/safe_agent.py`). A drawdown breaker halts trading above 15% below peak and only resumes within 5% of peak. An uncertainty gate reacts to regime confidence under 0.60 or to 20-step volatility above twice the 60-step baseline. A per-trade cap allows no single step to move more than 10% of the book. Every decision is stamped with which guards fired and why.
- Those reason strings make the run auditable, and they are how the counts above were checked. Of 999 steps: the breaker held the agent flat on 171, the 10% cap bound on 213, the low-confidence branch fired on 135, the volatility-spike branch fired on 0, and 497 were nominal.
- The policy is regime-conditioned and intentionally plain: ride momentum in a trend, fade the deviation from the 20-step mean in mean-reversion, flatten in high volatility. Confidence below 0.60 reroutes sizing to an `uncertain` profile whose signal and size scale are both zero, so the agent goes flat rather than guessing — which also means the gate's additional 0.5 haircut is redundant on that path and would only bite on a volatility spike.
- Self-evolution (`agent/evolution.py`) nudges the per-regime size scales every 50 steps using an EMA of realized per-regime Sharpe, bounded by `tanh` and clipped to [0.05, 1.00]. Over the 11 allowed updates the scales moved very little — trending 0.800 to 0.774, mean_reverting 0.600 to 0.622 — which is a bounded update rule behaving as designed rather than a null result.
- The gate is the point of that module: an update is refused outright if the agent is in safe mode or if confidence is below 0.60, and the buffered performance from that window is discarded rather than deferred, because returns earned while distressed or misclassified are exactly the samples that would teach the wrong lesson. Of 19 attempts, 8 were refused: 5 for low confidence and 3 for safe mode. Two of those three came at confidence above 0.95, so the solvency condition blocks updates that the confidence condition would have waved straight through.
- Timing is strictly causal: at step *t* the agents see returns through *t*, choose exposure, and are graded on return *t+1*. Both agents pay 5 basis points per unit of turnover.

## Why this matters for a constrained-autonomy stack

Donut's D0 stack puts the Constraint Layer outside the reasoning path, and Lesson 1 gives the reason: "the moment a boundary enters the reasoning space as text, it becomes another object the model can optimize around" — so a separate engine returns a coarse verdict and "the model never sees the threshold table, the internal state machine, or the precise profile logic that produced that verdict." SafeEvolve is that separation in miniature. The policy proposes a target exposure and never reads `SafetyConfig`; the guards clamp, halve, or zero it afterwards and stamp a machine-readable reason on the result, which is the same shape as a coarse verdict over a proposed action. It also targets the layer above: D0's Closed-Loop Evolution feeds verified outcomes back into "policy tuning, release gates, and personalization updates," and the failure mode this prototype is built around is that a feedback loop which ingests outcomes indiscriminately will tune on data gathered while the system was distressed or misclassifying its environment. Gating the update on solvency and confidence, and dropping the buffer when the gate refuses, is a concrete answer to which verified outcomes should be allowed to change the policy at all.

## Limitations

- The market is synthetic and its generative process is regime-switching AR(1) with Student-t noise, which is close to what the BOCPD detector assumes. The 68.4% detection accuracy is measured against labels the simulator itself produced; nothing here has been run on real prices, and the detector's advantage over an EWMA is argued from its construction rather than demonstrated by an ablation.
- The default seed is chosen in the code so that the run exercises every guard including the drawdown breaker. Across 12 seeds I ran (34, 0, 1, 7, 11, 42, 99, 101, 202, 303, 777, 2024) the drawdown reduction and tail-survival advantage held in 12 of 12, but the naive agent finished ahead on total return in 2 of them (seed 0: +15.0% against -2.0%; seed 777: +199.0% against +12.7%), and SafeEvolve's own return was negative in 4 of 12. The headline seed is the favourable end of that distribution.
- The comparison is unguarded-full-size against guarded-and-small, so it does not separate "the guards fired at the right moments" from "the agent simply traded smaller." There is no constant-low-exposure control, which is the obvious next test and the one that would decide whether the guards are doing timing work or only sizing work. Part of the safety surface also did no work here: the volatility-spike branch never fired, and because the policy already returns a zero signal in `high_volatility` and `uncertain`, the size scales for those two regimes are multiplied by zero and cannot affect anything — including the ones the evolver keeps updating.
- Nothing here shows the evolution gate improved the outcome. The run reports 11 allowed and 8 blocked updates and the reason for each, but there is no ungated-evolution arm to compare against, so "blocking updates during distress is safer" remains an argument about the construction of the loop, not a measured result.
