import numpy as np
import torch
import pytest
import logging

from bermudan.simulate import simulate_gbm_paths
from bermudan.payoff import put_payoff
from bermudan.lsmc import lsmc_price
from bermudan.binomial import binomial_price

from bermudan.dual import (
    compute_upper_bound,
    compute_upper_bound_with_scaling,
)

from bermudan.neural_martingale import (
    train_neural_martingale,
    construct_neural_martingale,
)



def compute_upper_bound_from_martingale(paths, martingale, payoff_fct, K, r, T):

    payoff = payoff_fct(paths, K)
    _, n_times = paths.shape

    dt = T / (n_times - 1)
    discount = np.exp(-r * dt * np.arange(n_times))

    discounted_payoff = payoff * discount
    dual = discounted_payoff - martingale

    return np.mean(np.max(dual, axis=1))


@pytest.mark.parametrize("device", ["cpu"])
def test_neural_martingale_dual_bounds(device):

    

    seed = 12345

    np.random.seed(seed)
    torch.manual_seed(seed)

    
    S0 = 100
    r = 0.05
    sigma = 0.2
    T = 1.0

    K = 100

    n_steps = 10
    n_train = 8000
    n_test = 8000


    train_paths = simulate_gbm_paths(
        S0=S0,
        r=r,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_paths=n_train,
    )

    test_paths = simulate_gbm_paths(
        S0=S0,
        r=r,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_paths=n_test,
    )


    binomial_ref = binomial_price(S0, K, r, sigma, T, n_steps)

    lower_bound, _, _, _ = lsmc_price(
        test_paths,
        K,
        r,
        T,
        put_payoff,
    )

    

    upper_doob, hw_doob = compute_upper_bound(
        test_paths,
        put_payoff,
        K,
        r,
        T,
    )

    upper_scaled, hw_scaled = compute_upper_bound_with_scaling(
        test_paths,
        put_payoff,
        K,
        r,
        T,
    )

    f_net, g_net = train_neural_martingale(
        train_paths,
        put_payoff,
        K,
        r,
        T,
        n_epochs=50,
        batch_size=2048,
        lr=1e-3,
        lam=1e-3,
        device=device,
    )

    martingale_nn = construct_neural_martingale(
        test_paths,
        f_net,
        g_net,
        K,
    )

    upper_nn = compute_upper_bound_from_martingale(
        test_paths,
        martingale_nn,
        put_payoff,
        K,
        r,
        T,
    )
    hw_nn = 1.96 * np.std(np.max(
        put_payoff(test_paths, K) * np.exp(-r * (T / (test_paths.shape[1] - 1)) * np.arange(test_paths.shape[1])) - martingale_nn,
        axis=1
    )) / np.sqrt(len(test_paths))

    #  COMPARISON TABLE
    gap_doob   = upper_doob   - lower_bound
    gap_scaled = upper_scaled - lower_bound
    gap_nn     = upper_nn     - lower_bound

    print("\n")
    print("=" * 70)
    print("DUAL BOUND COMPARISON")
    print("=" * 70)
    print(f"{'Method':<25}{'Lower':>10}{'Upper':>10}{'±95% CI':>10}{'Gap':>10}")
    print("-" * 70)
    print(f"{'Binomial (ref)':<25}{binomial_ref:>10.4f}{'-':>10}{'-':>10}{'-':>10}")
    print(f"{'LSMC':<25}{lower_bound:>10.4f}{'-':>10}{'-':>10}{'-':>10}")
    print(f"{'Doob dual':<25}{lower_bound:>10.4f}{upper_doob:>10.4f}{hw_doob:>10.4f}{gap_doob:>10.4f}")
    print(f"{'Scaled dual':<25}{lower_bound:>10.4f}{upper_scaled:>10.4f}{hw_scaled:>10.4f}{gap_scaled:>10.4f}")
    print(f"{'Neural dual':<25}{lower_bound:>10.4f}{upper_nn:>10.4f}{hw_nn:>10.4f}{gap_nn:>10.4f}")
    print("=" * 70)

    # ASSERTIONS
    assert upper_doob   >= lower_bound - 1e-6
    assert upper_scaled >= lower_bound - 1e-6
    assert upper_nn     >= lower_bound - 1e-6
    assert upper_nn     <= upper_doob + 0.5
    assert gap_nn < 2.0