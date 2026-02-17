

import numpy as np 
from typing import Callable
from bermudan.lsmc import lsmc_price
from bermudan.payoff import put_payoff


def compute_martingale(paths: np.ndarray, payoff_fct: Callable[[np.ndarray, float], np.ndarray], K: float, r: float, T: float) -> np.ndarray:

    """
    Compute Doob martingale associated with the discounted Bermudan value process.
    Returns:
        np.ndarray: Martingale process, shape (n_paths, n_exec_times)
    """

    _, _, continuation_values, value_process = lsmc_price(paths, K, r, T, payoff_fct)
    print("value_process mean:", np.mean(value_process))
    print("continuation mean:", np.mean(continuation_values))
    n_paths, n_exec_times = value_process.shape

    
    doob_increments = np.zeros((n_paths, n_exec_times))
    doob_increments[:,0]= 0 
    dt = T / (n_exec_times - 1)
    discount = np.exp(-r * dt)



    for t in range(n_exec_times-1):
        doob_increments[:, t+1] = (discount * value_process[:, t+1] - continuation_values[:, t]) # realized_term (discounted to time-t units to be comparable with continuation ) - conditional expectation
    
    martingale = np.cumsum(doob_increments, axis=1)

    return martingale

def compute_upper_bound(
        paths: np.ndarray,
        payoff_fct: Callable[[np.ndarray, float], np.ndarray],
        K: float,
        r: float,
        T: float
    ) -> float:
    """
    Compute dual upper bound price using martingale correction.

    """

    martingale = compute_martingale(paths, payoff_fct, K, r, T)

    payoff = payoff_fct(paths, K)
    _, n_exec_times = paths.shape

    dt = T / (n_exec_times - 1)
    discount_factors = np.exp(-r * dt * np.arange(n_exec_times))

    discounted_payoff = payoff * discount_factors

    dual_process = discounted_payoff - martingale

    pathwise_max = np.max(dual_process, axis=1)

    upper_bound = np.mean(pathwise_max)

    return upper_bound





