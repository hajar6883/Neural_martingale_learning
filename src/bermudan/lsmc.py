import numpy as np 
from bermudan.payoff import put_payoff
from bermudan.basis import polynomial_basis
from typing import Callable

#backward induction
# regression
# exercise decision
# cashflow tracking

def lsmc_price(
        paths: np.ndarray,
        K: float,
        r: float,
        T: float, 
        payoff_fct : Callable
    ):
    """
    Returns:
        price
        exercise_times
        value_process ( discounted bermudan value)
        continuation_values
    """
    n_paths , n_exec_times = paths.shape
    immediate_payoff = np.zeros((n_paths , n_exec_times))
    value_process = np.zeros((n_paths , n_exec_times))
    continuation_values = np.zeros((n_paths, n_exec_times))

    

    immediate_payoff = payoff_fct(paths, K)

    value_process[:,-1] = immediate_payoff[:,-1]

    dt = T / (n_exec_times - 1)
    discount = np.exp(-r * dt)

    exercise_times = np.full(n_paths, -1)


    for time in range(n_exec_times -2, -1, -1):

        # mask = immediate_payoff[:,time] > 0
        # ITM_paths = paths[:,time][mask] 

        # if len(ITM_paths) == 0:
        #     continuation_values[:, time] = 0.0 # must be also set 

        #     # Exercise is impossible because immediate payoff is 0 
        #     # so option value must be the continuation value, 
        #     # but since there are no ITM paths, you cannot run regression to estimate continuation 
        #     # So instead, you use the already known value from the next time step
        #     value_process[:,time] = discount * value_process[:,time+1] 
        #     continue

        # X =  polynomial_basis(ITM_paths) # only regress on ITM paths for variance reduction purposes
        # Y = discount * value_process[:,time+1][mask]   # continuation target Y_t = disc(V_t+1) in dt
        X = polynomial_basis(paths[:, time], K=K)
        Y = discount * value_process[:,time+1]
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]

        continuation = X @ beta

        continuation_values[:, time] = continuation
        exercise = immediate_payoff[:,time] > continuation # decision_rule 

        value_process[:,time] = np.where(
            exercise,
            immediate_payoff[:,time],
            discount * value_process[:,time+1] #otherwise actual realized discounted future value (not continuation)
        )
        new_ex = exercise & (exercise_times == -1)
        exercise_times[new_ex] = time

    exercise_times[exercise_times == -1] = n_exec_times - 1

    price = np.mean(value_process[:,0])
    

    return price, exercise_times, continuation_values, value_process





   

    
        







