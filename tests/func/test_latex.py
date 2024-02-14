"""Tests for latex.py routines."""

import pytest

import rttools.latex as latex


@pytest.mark.parametrize(
    "value",
    [
        [(0.0312, 0.001, 2), "$(3.12 \\pm 0.10) \\times 10^{-2}$"],
        [(0.004256854, 0.004256854, 2), "$(4.26 \\pm 4.26) \\times 10^{-3}$"],
        [(305.3, 2, 4), "$(3.0530 \\pm 0.0200) \\times 10^{2}$"],
        [(3.14, 0.1555, 1), "$3.1 \\pm 0.2$"],
    ],
)
def test_error_formatting(value):
    """Format value plus uncertainty in LaTeX proper way."""
    val, unc, prec = value[0]
    expected = value[1]
    assert latex.error_formatting(val, unc, prec) == expected


@pytest.mark.parametrize(
    "value",
    [
        [(3.12, 2), "$3.12 \\times 10^{0}$"],
        [(0.004256854, 2), "$4.26 \\times 10^{-3}$"],
        [(0.0042, 2), "$4.20 \\times 10^{-3}$"],
    ],
)
def test_exp_notation(value):
    """Turn number into exponential notation LaTeX string."""
    num, prec = value[0]
    expected = value[1]
    assert latex.exp_notation(num, prec) == expected


def test_delta_iso():
    """Check if delta notation is returned properly."""
    iso1 = "Si-30"
    iso2 = "Si-28"
    exp_short = "$\\delta{^{30}}\\mathrm{Si}_{28}$ (‰)"
    exp_full = "$\\delta({^{30}}\\mathrm{Si}/{^{28}}\\mathrm{Si})$ (‰)"
    assert latex.delta_iso(iso1, iso2) == exp_short  # default
    assert latex.delta_iso(iso1, iso2, full=True) == exp_full


def test_delta_iso_alternative_writing():
    """Check if delta notation is returned properly."""
    iso1 = "Si30"
    iso2 = "28Si"
    exp_short = "$\\delta{^{30}}\\mathrm{Si}_{28}$ (‰)"
    exp_full = "$\\delta({^{30}}\\mathrm{Si}/{^{28}}\\mathrm{Si})$ (‰)"
    assert latex.delta_iso(iso1, iso2) == exp_short  # default
    assert latex.delta_iso(iso1, iso2, full=True) == exp_full


def test_isotope_transformer():
    """Transform notation of isotopes back and forth, e.g., `46Ti` <-> `Ti-46`."""
    not1 = "46Ti"
    not2 = "Ti-46"
    assert latex.iso_transformer(not1) == not2
    assert latex.iso_transformer(not2) == not1


def test_ratio_iso():
    """Check if isotope ratio is returned properly."""
    iso1 = "Si-30"
    iso2 = "Si-28"
    exp = "${^{30}}\\mathrm{Si}/{^{28}}\\mathrm{Si}$"
    assert latex.ratio_iso(iso1, iso2) == exp


def test_split_iso():
    """Split isotope given as string into element name and mass number."""
    iso = "Si-28"
    assert latex.split_iso(iso) == ("Si", 28)


def test_split_iso_other_notation():
    """Accept different notation `46Ti` for isotope splitter."""
    iso = "28Si"
    assert latex.split_iso(iso) == ("Si", 28)
