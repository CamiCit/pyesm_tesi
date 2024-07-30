"""
special_tril.py 
Building a matrix to create lower triangular matrixes...
"""
import numpy as np
import pandas as pd
import cvxpy as cp


def special_tril(
        dimension: cp.Parameter, 
        matr_type: str,
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
    # Warnings
    if not isinstance(dimension, cp.Parameter):
        raise TypeError("dimension must be a cvxpy.Parameter")
    if not isinstance(power_factor, cp.Parameter):
        raise TypeError("power_factor must be a cvxpy.Parameter")
    
    #Extract values from cvxpy parameters
    pf: np.ndarray = power_factor.value #pf defined for storage technologies, one equation for each storage tech 
    dim = int(dimension.value)  # Extract the integer value of the dimension
    
    # Initialize the matrix
    matrix = np.zeros((dim, dim))

    for i in range(dim):
        for j in range(i + 1):
            if i == j:
                matrix[i, j] = 1
            else:
                matrix[i, j] = matrix[i - 1, j] * pf

    return matrix