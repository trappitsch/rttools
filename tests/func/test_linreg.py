"""Function tests for linear fitting routine - mahon.py."""

import pytest
import numpy as np

from rttools.linreg import Mahon, Stephan

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

# results from Stephan & Trappitsch (2022) for fixed intercept Mahon Data
STEPHAN_FIX_PT = np.array([0.01, 0.003])
STEPHAN_RES_FIXED = np.array(
    [-0.080842, 0.002222, 0.0038084, 0.0000222, np.nan, np.nan, 0.911]
)


def test_mahon_calculate_with_ci_setup():
    """Assure setup and some sanity checks."""
    bins = 10
    reg = Mahon()
    reg.calculate_with_ci(
        MAHON_XDAT, MAHON_YDAT, MAHON_XDAT_UNC, MAHON_YDAT_UNC, bins=bins
    )
    assert reg.ci_yax_pos is not None
    assert reg.ci_yax_neg is not None

    ci_xax_expected = np.linspace(np.min(MAHON_XDAT), np.max(MAHON_XDAT), bins)
    np.testing.assert_equal(reg.ci_xax, ci_xax_expected)


def test_stephan_fit_correlated():
    """Test Stephan fit with correlated Mahon data."""
    reg = Stephan(MAHON_XDAT, MAHON_YDAT, MAHON_XDAT_UNC, MAHON_YDAT_UNC, rho=MAHON_RHO)
    slope_exp = MAHON_RES_CORR[0:2]
    intercept_exp = MAHON_RES_CORR[2:4]
    mswd_exp = MAHON_RES_CORR[6]

    slope_rec = np.array(reg.slope)
    intercept_rec = np.array(reg.intercept)
    mswd_rec = reg.mswd

    assert slope_exp == pytest.approx(slope_rec, abs=1e-5)
    assert intercept_exp == pytest.approx(intercept_rec, abs=1e-5)
    assert mswd_exp == pytest.approx(mswd_rec, abs=0.01)


def test_stephan_fit_uncorrelated():
    """Test Stephan fit with uncorrelated Mahon data."""
    reg = Stephan(MAHON_XDAT, MAHON_YDAT, MAHON_XDAT_UNC, MAHON_YDAT_UNC)
    slope_exp = MAHON_RES_UNCORR[0:2]
    intercept_exp = MAHON_RES_UNCORR[2:4]
    mswd_exp = MAHON_RES_UNCORR[6]

    slope_rec = np.array(reg.slope)
    intercept_rec = np.array(reg.intercept)
    mswd_rec = reg.mswd

    assert slope_exp == pytest.approx(slope_rec, abs=1e-5)
    assert intercept_exp == pytest.approx(intercept_rec, abs=1e-5)
    assert mswd_exp == pytest.approx(mswd_rec, abs=0.01)


def test_stephan_fit_correlated_fixed():
    """Test Stephan regression with Mahon Data and Fixed intercept."""
    reg = Stephan(
        MAHON_XDAT,
        MAHON_YDAT,
        MAHON_XDAT_UNC,
        MAHON_YDAT_UNC,
        rho=MAHON_RHO,
        fixpt=STEPHAN_FIX_PT,
    )
    slope_exp = STEPHAN_RES_FIXED[0:2]
    intercept_exp = STEPHAN_RES_FIXED[2:4]
    mswd_exp = STEPHAN_RES_FIXED[6]

    slope_rec = np.array(reg.slope)
    intercept_rec = np.array(reg.intercept)
    mswd_rec = reg.mswd

    assert slope_exp == pytest.approx(slope_rec, abs=1e-6)
    assert intercept_exp == pytest.approx(intercept_rec, abs=1e-6)
    assert mswd_exp == pytest.approx(mswd_rec, abs=0.01)
