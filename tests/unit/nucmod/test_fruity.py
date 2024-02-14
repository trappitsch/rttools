"""Tests to read Maria Lugaro's files."""


import pandas as pd
import pytest

from rttools.nucmod import fruity


@pytest.fixture
def fruity_file(models_path):
    """Return fruity sample data file as a Path object."""
    return models_path.joinpath("fruity.txt")


def test_drop_a_z(fruity_file):
    """Drop column with mass and proton number from dataframe."""
    data = fruity.reader(fruity_file)
    data_exp = data.drop(columns=["A", "Z"])
    pd.testing.assert_frame_equal(fruity.drop_a_z(data), data_exp)


def test_get_delta_isoratio(fruity_file):
    """Get delta isotope ratio for two isotopes."""
    iso1 = "Si-29"
    iso2 = "30Si"
    delta = fruity.get_delta_isoratio(fruity_file, iso1, iso2)
    assert isinstance(delta, pd.Series)


def test_fruity_reader(fruity_file):
    """Read a simple file."""
    data = fruity.reader(fruity_file)
    assert isinstance(data, pd.DataFrame)
