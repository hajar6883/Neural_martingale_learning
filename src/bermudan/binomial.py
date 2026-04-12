import numpy as np


def binomial_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
) -> float:
    """
    Bermudan/American put price via backward induction on a binomial tree.

    With n_steps matching your LSMC, this gives a reference price on the same
    exercise grid. Use n_steps=1000+ for a near-exact American price.

    Returns:
        float: option price at t=0
    """
    dt = T / n_steps
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1.0 / u
    p  = (np.exp(r * dt) - d) / (u - d)   # risk-neutral up probability
    disc = np.exp(-r * dt)

    # terminal node prices and payoffs
    j = np.arange(n_steps + 1)                         
    S_T = S0 * u**j * d**(n_steps - j)
    V = np.maximum(K - S_T, 0.0)                      
    
    # backward induction
    for i in range(n_steps - 1, -1, -1):
        j = np.arange(i + 1)
        S = S0 * u**j * d**(i - j)
        hold     = disc * (p * V[1:i+2] + (1 - p) * V[0:i+1])
        exercise = np.maximum(K - S, 0.0)
        V = np.maximum(hold, exercise)

    return float(V[0])
