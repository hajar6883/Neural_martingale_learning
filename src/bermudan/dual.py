

import numpy as np 
from lsmc import lsmc_price

def compute_martingale(paths, K, r, T):

    price, _, continuation_values, value_process = lsmc_price(paths, K, r, T)

    _, n_exec_times = value_process.shape
    
    doob_increments = np.zeros (n_exec_times)
    doob_increments[0]= 0 


    for t in range(1, n_exec_times):
        doob_increments[:, t+1] = (value_process[:, t+1] - continuation_values[:, t])

    martingale = np.cumsum(doob_increments, axis=1)

    return martingale


# def dual_upper_bound(paths, martingale, payoff)
