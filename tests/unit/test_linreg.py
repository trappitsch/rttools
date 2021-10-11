"""Unit tests for linear fitting routine - mahon.py."""

import numpy as np

from rttools.linreg import Mahon

# MAHON TEST DATA FROM PAPER #
MAHON_XDAT = np.array([0.037, 0.035, 0.032, 0.040, 0.013, 0.038, 0.042, 0.030])
MAHON_YDAT = np.array(
    [0.00080, 0.00084, 0.00100, 0.00085, 0.00270, 0.00071, 0.00043, 0.00160]
)
MAHON_XDAT_UNC = 0.03 * MAHON_XDAT
MAHON_YDAT_UNC = 0.1 * MAHON_YDAT

MAHON_RHO = 0.7071  # for correlated analysis

# Results for mahon: Slope, 1sig, Y intercept, 1sig, X intercept, 1sig, MSWD
MAHON_RES_UNCORR = np.array(
    [-0.07638, 0.00952, 0.003641, 0.000349, 0.04767, 0.00153, 1.73]
)
MAHON_RES_CORR = np.array(
    [-0.07752, 0.01027, 0.003684, 0.000375, 0.04753, 0.00169, 1.04]
)


def test_mahon_calculate_uncorr():
    """Linear regressison with uncorrelated Mahon test dataset."""
    reg = Mahon()
    reg.calculate(MAHON_XDAT, MAHON_YDAT, MAHON_XDAT_UNC, MAHON_YDAT_UNC)

    values_calculated = np.array(
        [
            reg.slope,
            reg.slopeunc,
            reg.yinter,
            reg.yinterunc,
            reg.xinter,
            reg.xinterunc,
        ]
    )
    values_expected = MAHON_RES_UNCORR[:-1]

    np.testing.assert_allclose(values_calculated, values_expected, atol=1e-5)
    np.testing.assert_approx_equal(reg.mswd, MAHON_RES_UNCORR[-1], significant=2)


def test_mahon_calculate_corr():
    """Linear regressison with correlated Mahon test dataset."""
    reg = Mahon()
    reg.calculate(
        MAHON_XDAT, MAHON_YDAT, MAHON_XDAT_UNC, MAHON_YDAT_UNC, p_corr=MAHON_RHO
    )

    values_calculated = np.array(
        [
            reg.slope,
            reg.slopeunc,
            reg.yinter,
            reg.yinterunc,
            reg.xinter,
            reg.xinterunc,
        ]
    )
    values_expected = MAHON_RES_CORR[:-1]

    np.testing.assert_allclose(values_calculated, values_expected, atol=1e-5)
    np.testing.assert_approx_equal(reg.mswd, MAHON_RES_CORR[-1], significant=2)


def test_mahon_calculate_with_ci_end_result():
    """Ensure that Mahon with CI has the same class result as `calculate` at end."""
    reg = Mahon()
    reg.calculate(
        MAHON_XDAT, MAHON_YDAT, MAHON_XDAT_UNC, MAHON_YDAT_UNC, p_corr=MAHON_RHO
    )
    reg_ci = Mahon()
    reg_ci.calculate_with_ci(
        MAHON_XDAT, MAHON_YDAT, MAHON_XDAT_UNC, MAHON_YDAT_UNC, p_corr=MAHON_RHO
    )
    assert reg_ci.slope == reg.slope
    assert reg_ci.slopeunc == reg.slopeunc
    assert reg_ci.yinter == reg.yinter
    assert reg_ci.yinterunc == reg.yinterunc
    assert reg_ci.xinter == reg.xinter
    assert reg_ci.xinterunc == reg.xinterunc
    assert reg_ci.mswd == reg.mswd
