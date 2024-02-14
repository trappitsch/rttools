"""Function tests for lugaro reader."""


import pytest

from rttools.nucmod import lugaro


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
