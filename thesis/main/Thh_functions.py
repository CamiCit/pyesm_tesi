"""
Thh_functions.py 
Building a matrix to model phenomena with a linear change through time, 
each column containing a power series from the top 1 to the bottom (cooling_factor)^n_rows
"""

from typing import Iterable
import numpy as np
import pandas as pd
import cvxpy as cp

def power_tril(
        dimension: int, 
        power_factor: cp.Parameter,
) -> np.array:
    """
    Generate a square matrix with values in the lower triangular region
    (ones in the diagonal) and zeros elsewhere. Each column contains a power series, 
    multiplying the cell above by the cooling factor.

    Parameters:
        dimension (int): The size of the square matrix.
        cooling_factor:

    Returns:
        np.ndarray: A square matrix of size 'dimension x dimension' with 
            values in the lower triangular region (ones on the diagonal) and zeros elsewhere.

    Raises:
        ValueError: If passed dimension is not greater than zero.
        TypeError: If passed dimension is not an integer.
    """
    #warnings

    #Extract values from cvxpy parameters
    pf: np.ndarray = power_factor.value #pf defined for storage technologies, one equation for each storage tech 
    dim: np.ndarray = dimension.value
   
    matrix = np.zeros((dim, dim))

    for i in range(dim):
        for j in range(i + 1):
            if i == j:
                matrix[i, j] = 1
            else:
                matrix[i, j] = matrix[i - 1, j] * pf #how to extract the cf for the right tech?

    return matrix