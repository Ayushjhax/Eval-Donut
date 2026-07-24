# Eval-Donut

Two small research prototypes on the same question: how an autonomous trading agent should be evaluated, and constrained, when the market it trades keeps changing distribution.
Each folder is self-contained, runs in seconds from a fresh clone, and its README reports only numbers produced by running the code in it.

- [regime_eval](regime_eval/) — Splits six years of real SOL/USDT data into three regimes and shows that one aggregate Sharpe ratio ranks the highest-tail-risk regime as the best one.
- [SafeEvolve](SafeEvolve/) — A simulated agent with hard safety guards between intent and execution, whose self-updates are refused while it is in drawdown or uncertain about the regime.
