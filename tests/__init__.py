# Some helper functions

from rttools import ureg


def assert_equal_unitful(exp: ureg.Quantity, act: ureg.Quantity):
    """Compare to quantities for equality.

    :param exp: Expected quantity.
    :param act: Actual quantity.
    """
    act = act.to(exp.units)
    assert exp.magnitude == act.magnitude
    assert exp.units == act.units
