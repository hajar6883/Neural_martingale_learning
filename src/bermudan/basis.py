import numpy as np

def polynomial_basis(S: np.ndarray, K: float = 1.0, degree: int = 2) -> np.ndarray:
    """
    Input shape: (n,)
    Output shape: (n, degree+1) — columns [1, S/K, (S/K)^2, ...]

    Normalizing by K keeps all columns O(1), which avoids ill-conditioning
    in the lstsq solve when S is large (e.g. S=100 -> S^2=10000 without normalization).
    """
    S_norm = S / K
    return np.column_stack([S_norm**deg for deg in range(degree + 1)])

