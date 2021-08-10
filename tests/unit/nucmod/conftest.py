"""Fixtures for nucleosynthesis model routines."""

from pathlib import Path

import pytest


@pytest.fixture
def models_path(request):
    """Provides the path to the `models` folder.

    :return: Path to the folder
    :rtype: Path
    """
    curr = Path(request.fspath).parents[0]
    return Path(curr).joinpath("models").absolute()
