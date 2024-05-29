"""Tests to read Maria Lugaro's files."""

import pandas as pd
import pytest

from rttools.nucmod import lugaro


@pytest.fixture
def lugaro_file(models_path):
    """Return Lugaro sample data file as a Path object."""
    return models_path.joinpath("lugaro.dat")


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
