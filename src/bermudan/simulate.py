import numpy as np 

def simulate_gbm_paths(
   
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,

    n_paths: int,
    seed: int = 0
) -> np.ndarray:
    """
    Returns array shape: (n_paths, n_steps+1)
    """
    

    dt = T/ n_steps
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_paths, n_steps))
    S = np.zeros((n_paths, n_steps+1))

    S[:, 0]= S0
    for t in range(1, n_steps+1):
        S[:,t] = S[:, t-1]* np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[:,t-1])

    return S

