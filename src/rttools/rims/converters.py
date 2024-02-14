"""Converters for RIMS work.

This module contains converters that are conventient when working with RIMS data,
etc.

Import as:

```python
from rttools.rims import converters
```
"""

from typing import Any, Union

import numpy as np
from numpy.typing import ArrayLike

from rttools import ureg
from rttools import utils as ut


def power_after_window(power: Any, passes: int = 1, transmittance: float = 0.96) -> Any:
    """Calculate power after a laser passes through a window.

    Going through a window will reduce the power by transmittance**2 per pass.
    This is because of the following:

                S1      S2
        laser   |       |
        --------|-------|--------
                |       |
                 window

    As the simplified drawing shows, the window has an entrance and an exit surface,
    at which the laser loses in each case `transmittance` of its total power.
    Surfaces are labeled S1 and S2.

    :param power: Initial power
    :param passes: Number of passes through a window.
    :param transmittance: Transmittance of the window, defaults to 0.96.

    :return: Power after the `passes` passes through the window.
    """
    return power * transmittance ** (2 * passes)


def power_to_irradiance(
    power: Union[float, ureg.Quantity],
    rep_rate: Union[float, ureg.Quantity],
    beam_size: Union[list, tuple, ArrayLike, Union[float, ureg.Quantity]],
    beam_dt: Union[float, ureg.Quantity],
) -> ureg.Quantity:
    """Calculate irradiance from beam power and characteristics.

    Take the users beam parameters and calculate the irradiance for saturation curves.
    All parameters can be passed as floats or as pint quantities. Standard units as
    given below are assumed if no quantities are provided.

    :param power: Beam power (defaults to mW)
    :param rep_rate: Repetition rate of laser (defaults to Hz)
    :param beam_size: Tuple of beam diameters or float of diameters if equal
        (defaults to mm for individual entries).
    :param beam_dt: Pulse width (defaults to ns)

    :return: Unitful irradiance (W/cm^2)

    :Example:

        >>> power = 811 * ureg.mW
        >>> rep_rate = 1000 * ureg.Hz
        >>> beam_size = (1.5, 1.6) * ureg.mm
        >>> beam_dt = 1.355e-8 * ureg.s
        >>> power_to_irradiance(power, rep_rate, beam_size, beam_dt)
        <Quantity(3175268.36, 'watt / centimeter ** 2')>

    """
    power = ut.assume_units(power, ureg.mW)
    rep_rate = ut.assume_units(rep_rate, ureg.Hz)
    try:
        _ = beam_size[1]  # check if iterable
        beam_size = ut.assume_units(beam_size, ureg.mm)
    except TypeError:  # not iterable
        if isinstance(beam_size, ureg.Quantity):
            beam_size = beam_size.to(ureg.mm).magnitude
        beam_size = ut.assume_units([beam_size, beam_size], ureg.mm)
    beam_dt = ut.assume_units(beam_dt, ureg.ns)

    beam_area = beam_size[0] / 2 * beam_size[1] / 2 * np.pi

    irr = power / (rep_rate * beam_area * beam_dt)
    return irr.to(ureg.W / ureg.cm**2)
