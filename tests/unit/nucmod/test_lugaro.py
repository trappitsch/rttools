"""Tests to read Maria Lugaro's files."""

from pathlib import Path

import pandas as pd
import pytest

from rttools.nucmod import lugaro


@pytest.fixture
def lugaro_file(models_path):
    """Return Lugaro sample data file as a Path object."""
    return models_path.joinpath("lugaro.dat")


@pytest.mark.parametrize(
    "val",
    [
        ["m3p5z03pmz1m3ST", (3.5, 3 / 1.4, 1e-3, True)],
        ["m3p5z03pmz1m3.dat", (3.5, 3 / 1.4, 1e-3, False)],
        ["m4z03pmz5m4", (4, 3 / 1.4, 5e-4, False)],
    ],
)
def test_lugaro_label_parser(val):
    """Parse a label."""
    label = val[0]
    expected = val[1]
    assert lugaro.label_parser(label) == expected


def test_lugaro_reader(lugaro_file):
    """Read a simple file."""
    data = lugaro.reader(lugaro_file)
    assert isinstance(data, pd.DataFrame)


def test_lugaro_reader_cols(lugaro_file):
    """Read and set columns for Lugaro data file."""
    cols = ["CO_ratio", "d92Mo", "d46Ti"]
    data = lugaro.reader(lugaro_file, cols=cols)
    cols_exp = cols.copy()
    cols_exp.insert(0, "model number")  # add model number
    assert list(data.columns) == cols_exp
