"""Test for routiens in utils.py."""

import pytest
import numpy as np

import rttools.utils as utils
from rttools import ureg


def test_assume_units():
    """If no units are given, assume default standard units."""
    assert utils.assume_units(3, ureg.mW) == 3 * ureg.mW
    assert utils.assume_units(3 * ureg.W, ureg.mW) == 3000 * ureg.mW
    assert utils.assume_units(60 * ureg.min, ureg.s) == 1 * ureg.h


@pytest.mark.parametrize("in_arr", [np.array([1, 2, 3, 4]), [1, 2, 3, 4], (1, 2, 3, 4)])
def test_assume_units_array(in_arr):
    """Assume units for an array of values."""
    unit = ureg.mW
    out = utils.assume_units(in_arr, unit)
    for item in out:
        assert isinstance(item, ureg.Quantity)
        assert item.units == unit


def test_kron_delta():
    """Ensure that 1 is returned if two indexes are the same, zero otherwise."""
    assert utils.kron_delta(1, 1) == 1
    assert utils.kron_delta(1, 2) == 0


def test_kron_delta_array():
    """Kronecker delta return for an array of indexes."""
    in1 = np.array([1, 2, 3, 4])
    in2 = np.array([1, 2, 4, 4])
    in2alt = [1, 2, 4, 4]  # as list
    out_exp = np.array([1, 1, 0, 1])
    np.testing.assert_equal(utils.kron_delta(in1, in2), out_exp)
    np.testing.assert_equal(utils.kron_delta(in1, in2alt), out_exp)


def test_kron_delta_value_error():
    """Raise ValueError if shapes mismatch."""
    arr1 = np.array([1, 2])
    arr2 = np.array([1])
    num = 3
    with pytest.raises(ValueError) as err_info:
        utils.kron_delta(num, arr1)
    err_msg = err_info.value.args[0]
    assert err_msg == "The inputs must have the same shape."

    with pytest.raises(ValueError) as err_info:
        utils.kron_delta(arr1, arr2)
    err_msg = err_info.value.args[0]
    assert err_msg == "The inputs must have the same shape."


@pytest.mark.parametrize(
    "value", [[(3.0, 2), "3"], [(5.1264, 3), "5.126"], [(3.0102, 3), "3.01"]]
)
def test_reduce_decimal(value):
    """Test reducing numbers with parameterized values."""
    num, prec = value[0]
    expected = value[1]
    assert utils.reduce_decimal(num, prec) == expected
