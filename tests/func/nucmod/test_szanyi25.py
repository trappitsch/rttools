"""Function tests for Balzazs 2025 reader."""

from hypothesis import given, strategies as st
import pytest

from rttools.nucmod import szanyi25 as sza


def test_lugaro_label_parser():
    """Get closest model for some edge cases."""
    fname = sza.get_closest_model(2.5, 0.0105)
    assert fname.name == "srf_m3z014_set2.dat"

    fname = sza.get_closest_model(0, 0)
    assert fname.name == "srf_m2z014_set2.dat"

    fname = sza.get_closest_model(10, 1, 0)
    assert fname.name == "srf_m4z030_set0.dat"

@given(
    mass=st.floats(min_value=-0.5, max_value=5.0),
    metallicity=st.floats(min_value=-0.0001, max_value=0.07),
    rate_set=st.integers(min_value=-2, max_value=4)
)
def test_get_closest_model(mass, metallicity, rate_set):
    """Just tests that all random models actually exist (assert in function)."""
    _ = sza.get_closest_model(mass, metallicity, rate_set)

def test_szanyi25_read_model():
    """Test reading a model file."""
    reader = sza.Szanyi25Reader(3.0, 0.014, 1)

    assert reader.mass == 3.0
    assert reader.y == 0.28
    assert reader.z == 0.014
    assert reader.rate_set == 1

    assert reader._data is not None

    assert "TP #1" in reader._data.columns.tolist()
    assert "o16" in reader._data.index.tolist()

def test_szanyi25_co_ratio():
    """Ensure a reasonable value comes back."""
    mass = 3.0
    metallicity = 0.014
    rate_set = 1

    reader = sza.Szanyi25Reader(mass, metallicity, rate_set)
    co_ratio = reader.get_co_ratio()

    assert len(co_ratio) == 29
    assert all(val >= 0. for val in co_ratio)
    pytest.approx(co_ratio.iloc[2], 0.4700536385838516)  # specific value check

def test_szanyi25_delta_mo():
    """Calculate delta values for Mo isotopes."""
    mass = 3.0
    metallicity = 0.014
    rate_set = 1

    nominator = "Mo-94"
    denominator = "Mo-96"

    reader = sza.Szanyi25Reader(mass, metallicity, rate_set)
    delta_mo = reader.get_delta(nominator, denominator)

    pytest.approx(delta_mo[0], 20.998722635556135)  # specific value check
