"""Helper functions and utilities to be used and re-used in multiple areas.

Import as:

```python
from rttools import utils
```
"""

import decimal
from typing import Any

import numpy as np

from rttools import ureg


def assume_units(value: Any, unit: ureg.Quantity) -> Any:
    """Take a vale with or without unit and return it with a unit.

    If a value is given without units, assume the given units.
    If the value is given with units, return it.

    :param value: Value to be converted
    :param unit: Unit to be assumed

    :return: Value with assumed units
    """
    if isinstance(value, ureg.Quantity):
        return value
    else:
        return value * unit


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


def reduce_decimal(num: Any, prec: int = 2) -> str:
    """Return a given number with precision or cut trailing zeros if possible.

    :param num: Number given, anything that can be turned into a decimal.
    :param prec: given precision.

    :return: Number with at most the given precision.
    """
    dec = decimal.Decimal(f"{num:.{prec}f}").normalize()
    return str(dec)
