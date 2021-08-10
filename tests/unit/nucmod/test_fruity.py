"""Tests to read Maria Lugaro's files."""

from pathlib import Path

import pandas as pd
import pytest

from rttools.nucmod import fruity


@pytest.fixture
def fruity_file(models_path):
    """Return fruity sample data file as a Path object."""
    return models_path.joinpath("fruity.txt")


# @pytest.mark.parametrize(
#     "val",
#     [
#         ["m3p5z03pmz1m3ST", (3.5, 3 / 1.4, 1e-3, True)],
#         ["m3p5z03pmz1m3.dat", (3.5, 3 / 1.4, 1e-3, False)],
#         ["m4z03pmz5m4", (4, 3 / 1.4, 5e-4, False)],
#     ],
# )
# def test_lugaro_label_parser(val):
#     """Parse a label."""
#     label = val[0]
#     expected = val[1]
#     assert lugaro.label_parser(label) == expected


def test_drop_a_z(fruity_file):
    """Drop column with mass and proton number from dataframe."""
    data = fruity.reader(fruity_file)
    data_exp = data.drop(columns=["A", "Z"])
    pd.testing.assert_frame_equal(fruity.drop_a_z(data), data_exp)


def drop_a_z_empty_df():
    """Return the same dataframe as passed in if "A", "Z" non existent."""
    df = pd.DataFrame()
    pd.testing.assert_frame_equal(fruity.drop_a_z(df), df)


def test_get_delta_isoratio(fruity_file):
    """Get delta isotope ratio for two isotopes."""
    iso1 = "Si-29"
    iso2 = "30Si"
    delta = fruity.get_delta_isoratio(fruity_file, iso1, iso2)
    assert isinstance(delta, pd.Series)


@pytest.mark.parametrize("iso", ["28Si", "Si28", "Si-28"])
def test_make_fruity_isoname(iso):
    """Turn a isotope name into FRUITY specific notation."""
    assert fruity.make_fruity_isoname(iso) == "Si28"


def test_fruity_reader(fruity_file):
    """Read a simple file."""
    data = fruity.reader(fruity_file)
    assert isinstance(data, pd.DataFrame)
