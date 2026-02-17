import numpy as np
from bermudan.lsmc import lsmc_price

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
        paths, K, r, T
    )

    assert price >= 0
    assert len(exercise_times) == paths.shape[0]
    assert value_process.shape == paths.shape
