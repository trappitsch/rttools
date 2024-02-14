# Functional tests for RIMS converters

import numpy as np

import rttools.rims.converters as cvt
from rttools import ureg
from tests import assert_equal_unitful


def test_power_to_irradiance():
    """Convert mW to irradiance."""

    pow_mw = 811 * ureg.mW
    rep_rate = 1000 * ureg.Hz
    beam_size = (1.5, 1.6) * ureg.mm
    beam_dt = 1.355e-8 * ureg.s

    beam_area = beam_size[0] / 2 * beam_size[1] / 2 * np.pi

    irradiance_exp = pow_mw / (rep_rate * beam_area * beam_dt)
    print(irradiance_exp)

    assert_equal_unitful(
        cvt.power_to_irradiance(pow_mw, rep_rate, beam_size, beam_dt), irradiance_exp
    )

    # test unitless
    assert_equal_unitful(
        cvt.power_to_irradiance(
            pow_mw.magnitude,
            rep_rate.magnitude,
            beam_size.magnitude,
            beam_dt.to(ureg.ns).magnitude,
        ),
        irradiance_exp,
    )


def test_power_to_irradiance_one_beamsize():
    """Convert mW to irradiance with only one beam size."""
    pow_mw = 811 * ureg.mW
    rep_rate = 1000 * ureg.Hz
    beam_size = 1.5 * ureg.mm
    beam_dt = 1.355e-8 * ureg.s

    beam_area = beam_size / 2 * beam_size / 2 * np.pi

    irradiance_exp = pow_mw / (rep_rate * beam_area * beam_dt)

    assert_equal_unitful(
        cvt.power_to_irradiance(pow_mw, rep_rate, beam_size, beam_dt), irradiance_exp
    )

    # test unitless
    assert_equal_unitful(
        cvt.power_to_irradiance(
            pow_mw.magnitude,
            rep_rate.magnitude,
            beam_size.magnitude,
            beam_dt.to(ureg.ns).magnitude,
        ),
        irradiance_exp,
    )
