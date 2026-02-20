import logging
logging.basicConfig(level=logging.DEBUG, 
                    format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
                    )


from bermudan.dual import compute_martingale
import numpy as np

# dummy inputs
paths = np.random.randn(1000, 10)

def payoff(x, K):
    return np.maximum(x - K, 0)

K = 100
r = 0.01
T = 1.0

martingale = compute_martingale(paths, payoff, K, r, T)