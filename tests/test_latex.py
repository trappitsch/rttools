"""Tests for latex.py routines."""

import rttools.latex as latex


def test_delta_iso():
    """Check if delta notation is returned properly."""
    iso1 = "Si-30"
    iso2 = "Si-28"
    exp_short = "$\\delta{^{30}}\\mathrm{Si}_{28}$"
    exp_full = "$\\delta({^{30}}\\mathrm{Si}/{^{28}}\\mathrm{Si})$"
    assert latex.delta_iso(iso1, iso2) == exp_short  # default
    assert latex.delta_iso(iso1, iso2, full=True) == exp_full


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
