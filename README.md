# Neural Martingale Dual Bounds for Bermudan Options

Pricing Bermudan put options via primal-dual bounds:  
**lower bound** from Longstaff-Schwartz Monte Carlo (LSMC) and  
**upper bound** from a neural martingale trained to minimize the Rogers (2002) dual objective.

---

## Theoretical background

The Bermudan option price satisfies the duality (Rogers 2002):

$$V_0 = \sup_{\tau} \mathbb{E}[e^{-r\tau} h_\tau] = \inf_{M \in \mathcal{M}} \mathbb{E}\!\left[\max_t \bigl(e^{-rt} h_t - M_t\bigr)\right]$$

For any martingale $M$ with $M_0 = 0$, the right side is a valid upper bound.  
The tighter $M$ approximates the Doob martingale of the value process, the tighter the bound.

**Martingale parametrization.** Under GBM the risk-neutral Brownian increment is recoverable from the path:

$$Z_t = \frac{\log(S_{t+1}/S_t) - (r - \frac{1}{2}\sigma^2)dt}{\sigma\sqrt{dt}}$$

Since $\mathbb{E}[Z_t \mid \mathcal{F}_t] = 0$, we parametrize:

$$\Delta M_t = h_\theta(S_t) \cdot Z_t$$

which is a martingale difference **by construction** for any network $h_\theta$.  
Training minimizes $\mathbb{E}[\max_t(e^{-rt}h_t - M_t)]$ directly — no regularization needed.

---

## Methods

| Method | Type | Description |
|---|---|---|
| LSMC | Lower bound | Longstaff-Schwartz backward induction with polynomial basis |
| Doob dual | Upper bound | Doob martingale built from LSMC continuation values |
| Scaled dual | Upper bound | Doob martingale scaled by optimal scalar α |
| Neural dual | Upper bound | Stochastic integral martingale, h(S_t)·Z_t, trained end-to-end |
| Binomial tree | Reference | Exact backward induction on recombining tree |

---

## Results

ATM Bermudan put — S₀=100, K=100, r=0.05, σ=0.2, T=1, n\_steps=10, N=8000 paths.

| Method | Lower | Upper | ±95% CI | Gap |
|---|---|---|---|---|
| Binomial (ref) | — | 6.0043 | — | — |
| LSMC | 5.8782 | — | — | — |
| Doob dual | 5.8782 | 19.7717 | 0.4415 | 13.8935 |
| Scaled dual | 5.8782 | 8.5867 | 0.1239 | 2.7086 |
| **Neural dual** | **5.8782** | **6.4702** | **0.0401** | **0.5921** |

The neural upper bound brackets the binomial reference price and reduces the dual gap by **95%** relative to the raw Doob bound.

---

## Project structure

```
src/bermudan/
    simulate.py          GBM path simulation
    payoff.py            put / call payoff functions
    basis.py             normalized polynomial basis for LSMC regression
    lsmc.py              Longstaff-Schwartz backward induction
    binomial.py          binomial tree pricer (reference price)
    dual.py              Doob, scaled, and neural dual upper bounds
    neural_martingale.py stochastic integral martingale (main method)
    neural_martingale_modelfree.py  (f,g) parametrization — model-free variant

tests/
    test_martingale.py         LSMC and Doob bound unit tests
    test_neural_dual_bounds.py full comparison table with binomial reference
```

---

## How to run

```bash
# install dependencies
pip install numpy torch scipy pytest

# run all tests (prints the comparison table)
pytest tests/ -v -s
```

---

## References

- Rogers, L.C.G. (2002). *Monte Carlo valuation of American options.* Mathematical Finance.
- Longstaff, F.A. & Schwartz, E.S. (2001). *Valuing American options by simulation.* Review of Financial Studies.
- Becker, S., Cheridito, P. & Jentzen, A. (2019). *Deep optimal stopping.* Journal of Machine Learning Research.

---

## Future work

- Multi-asset extension: max-call option in d = 2, 5, 10 dimensions
- Model-free variant: replace stochastic integral with (f, g) parametrization that works on any path distribution
- Convergence study: dual gap as a function of N paths and network capacity
