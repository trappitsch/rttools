"""Unit tests for Peirce Outlier Rejection."""

import pytest
import numpy as np

from rttools import peirce


def test_peirce_ross_2003_example():
    """Run example in Ross (2003) through Peirce criterion."""
    data = np.array([101.2, 90, 99, 102, 103, 100.2, 89, 98.1, 101.5, 102])
    outliers_exp = np.array([89, 90])
    avg_exp = 100.9
    std_exp = 1.66

    avg_rec, std_rec, outliers_rec = peirce.reject_outliers(data)

    assert pytest.approx(avg_rec, avg_exp)
    assert pytest.approx(std_rec, std_exp)
    np.testing.assert_equal(outliers_rec, outliers_exp)
