import numpy as np 

def polynomial_basis(S: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Input shape: (n,)x
    Output shape: (n, degree+1)
    Example columns: [1, S, S^2]#default , ([1, S, S^2, S^3] #sometimes useful 
    """
    out = np.column_stack([S**deg for deg in range(degree + 1)])

    return out
