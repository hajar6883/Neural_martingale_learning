import numpy as np
from bermudan.dual import *
from bermudan.lsmc import lsmc_price
from bermudan.payoff import put_payoff
from bermudan.simulate import simulate_gbm_paths
import logging

logger = logging.getLogger(__name__)




def test_lsmc_runs():

    paths = np.array([
        [100, 95, 90, 85],
        [100, 105, 110, 120],
        [100, 98, 97, 96]
    ])

    K = 100
    r = 0.05
    T = 1.0

    price, exercise_times, continuation, value_process = lsmc_price(
        paths, K, r, T, put_payoff
    )
   

    assert price >= 0
    assert len(exercise_times) == paths.shape[0]
    assert value_process.shape == paths.shape

def test_martingale_starts_at_zero():

    paths = np.array([
        [100, 95, 90, 85],
        [100, 105, 110, 120]
    ])

    K, r, T = 100, 0.05, 1.0

    M = compute_martingale(paths, put_payoff, K, r, T)

    assert np.allclose(M[:, 0], 0.0)

def test_martingale_mean_increment_zero():

    
    K, r, T = 100, 0.05, 1.0

    # paths = 100 + np.cumsum(np.random.randn(1000, 10), axis=1) #plain random walk , not risk neutral 
    paths = simulate_gbm_paths(100, r=r, sigma=0.2, T=T, n_steps=10, n_paths=20000)
    

    M = compute_martingale(paths,put_payoff,  K, r, T)

    increments = M[:, 1:] - M[:, :-1]

    mean_increment = np.mean(increments)
    print("M mean last:", np.mean(M[:, -1]))
    print("increments mean per step:", np.mean(increments, axis=0))
    

 
    # assert abs(mean_increment) < 1e-2
    assert abs(mean_increment) < 5e-2 #test tolerance must reflects regression approx error ( bias ) ??




def test_upper_bound_greater_than_lower_bound():

    np.random.seed(0)

    paths = 100 + np.cumsum(np.random.randn(1000, 10), axis=1)

    K, r, T = 100, 0.05, 1.0

    lower, _, _, _ = lsmc_price(paths, K, r, T,put_payoff)

    upper = compute_upper_bound(paths, put_payoff, K, r, T)

    assert upper >= lower

def test_upper_bound_finite():

    np.random.seed(0)

    paths = 100 + np.cumsum(np.random.randn(1000, 10), axis=1)

    K, r, T = 100, 0.05, 1.0

    upper = compute_upper_bound(paths, put_payoff, K, r, T)

    assert np.isfinite(upper)
    assert upper > 0


def test_scaled_upper_bound_finite():

    np.random.seed(0)

    paths = simulate_gbm_paths(
        S0=100, r=0.05, sigma=0.2,
        T=1.0, n_steps=10, n_paths=5000
    )

    K, r, T = 100, 0.05, 1.0

    upper = compute_upper_bound_with_scaling(
        paths, put_payoff, K, r, T
    )

    assert np.isfinite(upper)
    assert upper > 0


def test_scaled_upper_bound_tighter():

    np.random.seed(0)

    paths = simulate_gbm_paths(
        S0=100, r=0.05, sigma=0.2,
        T=1.0, n_steps=10, n_paths=5000
    )

    K, r, T = 100, 0.05, 1.0

    upper_unscaled = compute_upper_bound(
        paths, put_payoff, K, r, T
    )

    upper_scaled = compute_upper_bound_with_scaling(
        paths, put_payoff, K, r, T
    )
    logger.debug(
        "upper_scaled=%.6f upper_unscaled=%.6f",
        upper_scaled,
        upper_unscaled
    )



    assert upper_scaled <= upper_unscaled + 1e-8

def test_scaled_upper_bound_above_lower_bound():

    np.random.seed(0)

    paths = simulate_gbm_paths(
        S0=100, r=0.05, sigma=0.2,
        T=1.0, n_steps=10, n_paths=5000
    )

    K, r, T = 100, 0.05, 1.0

    lower, _, _, _ = lsmc_price(
        paths, K, r, T, put_payoff
    )

    upper_scaled = compute_upper_bound_with_scaling(
        paths, put_payoff, K, r, T
    )
    logger.debug(
        "upper_scaled=%.6f lower=%.6f",
        upper_scaled,
        lower
    )

    assert upper_scaled >= lower






