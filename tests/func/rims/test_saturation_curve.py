# Tests for saturation curve module - these are simply run-through tests!:w


import pytest
import numpy as np

import rttools.rims.saturation_curve as sc
from rttools import ureg

XDAT = np.array([27821, 22257, 13911, 6955, 2782, 31994, 25039, 18084, 0])
YDAT = np.array([1.0, 0.8535, 0.5161, 0.4816, 0.3486, 1.0358, 0.9622, 0.7882, 0.0001])
YERR = np.array([0.007, 0.0066, 0.0054, 0.0054, 0.0047, 0.0083, 0.0082, 0.0076, 0.0003])


@pytest.mark.parametrize("units", [["mW", None], [None, None]])
def test_saturation_curve_unitless_no_err(units):
    """Create saturation curve plot with unitless data and no error bars."""
    xunit, yunit = units
    _ = sc.saturation_curve(
        XDAT, YDAT, xunit=xunit, yunit=yunit, xlabel="Powdiance", fit=False
    )


def test_saturation_curve_unitful_no_err():
    """Create saturation curve plot with unitful data and no error bars."""
    xdat = XDAT * ureg.W / ureg.cm**2
    ydat = YDAT * ureg.counts
    _ = sc.saturation_curve(xdat, ydat)


def test_saturation_curve_yerr():
    """Plot a saturation curve with error bars on yaxis only."""
    xdat = XDAT * ureg.W / ureg.cm**2
    ydat = np.stack([YDAT, YERR])
    _ = sc.saturation_curve(xdat, ydat)


def test_saturation_curve():
    """Plot saturation curve with errors in both axes."""
    xdat = np.stack([XDAT, XDAT / 10])
    ydat = np.stack([YDAT, YERR])
    _ = sc.saturation_curve(xdat, ydat, xunit="mW", yunit="count")
