

import numpy as np 
from typing import Callable
from bermudan.lsmc import lsmc_price
from scipy.optimize import minimize_scalar
import logging
from bermudan.neural_martingale import train_neural_martingale, construct_neural_martingale

logger = logging.getLogger(__name__)


def ci(samples: np.ndarray) -> float:
    """95% confidence interval half-width: 1.96 * std / sqrt(N)."""
    return 1.96 * np.std(samples) / np.sqrt(len(samples))

def compute_martingale(paths: np.ndarray, payoff_fct: Callable[[np.ndarray, float], np.ndarray], K: float, r: float, T: float) -> np.ndarray:

    """
    Compute Doob martingale associated with the discounted Bermudan value process.
    Returns:
        np.ndarray: Martingale process, shape (n_paths, n_exec_times)
    """

    _, _, continuation_values, value_process = lsmc_price(paths, K, r, T, payoff_fct)

    logger.debug(
        "Martingale input stats | value_mean=%.6f continuation_mean=%.6f",
        np.mean(value_process),
        np.mean(continuation_values)
        )

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
    ) -> tuple:
    """
    Compute dual upper bound price using martingale correction.

    Returns:
        (estimate, half_width): mean and 95% CI half-width
    """

    martingale = compute_martingale(paths, payoff_fct, K, r, T)

    payoff = payoff_fct(paths, K)
    _, n_exec_times = paths.shape

    dt = T / (n_exec_times - 1)
    discount_factors = np.exp(-r * dt * np.arange(n_exec_times))

    discounted_payoff = payoff * discount_factors
    dual_process = discounted_payoff - martingale
    pathwise_max = np.max(dual_process, axis=1)

    return float(np.mean(pathwise_max)), float(ci(pathwise_max))


def compute_upper_bound_with_scaling(
        paths: np.ndarray,
        payoff_fct: Callable[[np.ndarray, float], np.ndarray],
        K: float,
        r: float,
        T: float
    ) -> tuple:
    """
    Returns:
        (estimate, half_width): mean and 95% CI half-width at optimal alpha
    """

    martingale = compute_martingale(paths, payoff_fct, K, r, T)
    payoff = payoff_fct(paths, K)
    _, n_exec_times = paths.shape

    dt = T / (n_exec_times - 1)
    discount_factors = np.exp(-r * dt * np.arange(n_exec_times))
    discounted_payoff = payoff * discount_factors

    def pathwise_max_at(alpha):
        return np.max(discounted_payoff - alpha * martingale, axis=1)

    opt = minimize_scalar(
        lambda a: np.mean(pathwise_max_at(a)),
        bounds=(0.0, 5.0),
        method='bounded'
    )

    logger.debug(
        "Martingale input stats | init alpha=%.6f optimal alpha=%.6f",
        1.0, opt.x
    )

    samples = pathwise_max_at(opt.x)
    return float(np.mean(samples)), float(ci(samples))



def compute_upper_bound_neural(
            train_paths: np.ndarray,
            test_paths: np.ndarray,
            payoff_fct: Callable[[np.ndarray, float], np.ndarray],
            K: float,
            r: float,
            T: float,
            device: str = "cpu",
        ) -> tuple:
    """
    Returns:
        (estimate, half_width): mean and 95% CI half-width
    """
    f_net, g_net = train_neural_martingale(
        train_paths, payoff_fct, K, r, T,
        n_epochs=30, batch_size=2048, lr=1e-3, lam=1e-3, device=device
    )

    martingale = construct_neural_martingale(test_paths, f_net, g_net, K)

    payoff = payoff_fct(test_paths, K)
    _, n_exec_times = test_paths.shape
    dt = T / (n_exec_times - 1)
    discount_factors = np.exp(-r * dt * np.arange(n_exec_times))
    discounted_payoff = payoff * discount_factors

    samples = np.max(discounted_payoff - martingale, axis=1)
    return float(np.mean(samples)), float(ci(samples))








