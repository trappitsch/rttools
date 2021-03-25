"""Helper functions and utilities to be used and re-used in multiple areas."""

import numpy as np


def kron_delta(ind1, ind2):
    """Calculate Kronecker-delta for variables i,j.

    Compare two indexes and return 0 if the same, otherwise 1. If an ndarray is given,
    return an ndarray comparing each index individually.

    :param ind1: Index(es)
    :type ind1: int, ndarray<int>
    :param ind2: Index(es)
    :type ind2: int, ndarray<int>

    :return: 1 if ind 1 is identical to ind2, otherwise 0
    :rtype: int, ndarray<int>

    :raises ValueError: The input indexes have different shape.
    """
    if np.shape(ind1) != np.shape(ind2):
        raise ValueError("The inputs must have the same shape.")

    if np.shape(ind1) == ():  # don't have arrays
        return 1 if ind1 == ind2 else 0
    else:
        ret_arr = np.zeros_like(ind1)
        ret_arr[np.where(ind1 == ind2)] = 1
        return ret_arr
