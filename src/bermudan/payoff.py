import numpy as np

def put_payoff(S: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(K - S, 0.0)
def call_payoff(S: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(S-K, 0.0)
