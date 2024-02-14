"""Function tests for FRUITY reader."""

import pandas as pd
import pytest

from rttools.nucmod import fruity


def drop_a_z_empty_df():
    """Return the same dataframe as passed in if "A", "Z" non existent."""
    df = pd.DataFrame()
    pd.testing.assert_frame_equal(fruity.drop_a_z(df), df)


@pytest.mark.parametrize(
    "val",
    [
        ["isotopi_m2p0z3m3_T00_20210810_17102.txt", (2.0, 3e-3 / 1.4e-2, 0.0, "ext")],
        ["isotopi_m2p0z3m3_T00_20210810_17102", (2.0, 3e-3 / 1.4e-2, 0.0, "ext")],
        ["isotopi_m5p0z1m3_060_20210810_17102", (5.0, 1e-3 / 1.4e-2, 60, "std")],
        ["isotopi_m1p3zsun_000_20210810_17102", (1.3, 1, 0.0, "std")],
    ],
)
def test_fruity_label_parser(val):
    """Parse a label."""
    label = val[0]
    expected = val[1]
    assert fruity.label_parser(label) == expected


@pytest.mark.parametrize("iso", ["28Si", "Si28", "Si-28"])
def test_make_fruity_isoname(iso):
    """Turn a isotope name into FRUITY specific notation."""
    assert fruity.make_fruity_isoname(iso) == "Si28"


def test_make_fruity_isoname_long():
    """Long isotope names have the first number cut off to have 4 bytes."""
    iso = "Ce142"
    iso_exp = "Ce42"
    assert fruity.make_fruity_isoname(iso) == iso_exp
