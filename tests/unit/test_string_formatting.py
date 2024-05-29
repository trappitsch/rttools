"""Test the overarching string formatting functions."""

import pytest

from rttools import StringFmt


def test_identity():
    """Check that the same string goes in as comes out, if the type is the same."""
    string = r"$^{3}P_{33}$"
    string_type = StringFmt.Type.latex

    cls = StringFmt(string, string_type)
    assert cls.latex == string


@pytest.mark.parametrize(
    "cases",
    [
        [r"$^{3}P_{33}$", "<sup>3</sup>P<sub>33</sub>"],
        [r"^{3}P_{33}", "<sup>3</sup>P<sub>33</sub>"],
        [r"^3P_9", "<sup>3</sup>P<sub>9</sub>"],
        [r"^3P{\circ}", "<sup>3</sup>P°"],
        ["^3P{\\circ}", "<sup>3</sup>P°"],
        [r"5 \times 10^{7}", "5 × 10<sup>7</sup>"],
        [r"5\,\times 10^{7}", "5 × 10<sup>7</sup>"],
    ],
)
def test_html(cases):
    """Check for correct LaTeX to html conversion."""
    string_latex = cases[0]
    string_type = StringFmt.Type.latex

    cls = StringFmt(string_latex, string_type)
    assert cls.html == cases[1]
