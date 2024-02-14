"""Tools for LaTeX formatting

This module contains functions for formatting numbers, isotopes, etc.
It can be imported as `ltx` as following:

```python
from rttools import latex
```
"""

import decimal
from typing import Tuple, Union

from iniabu import ini


def error_formatting(value: float, unc: float, prec: int) -> str:
    """Take a value and its uncertainty and express is as a formatted LaTeX string.

    Scientific notation is assumed. As an example, if value is 0.0002153 and uncertainty
    is 0.00002, at a precision of 3 the formatted string would be:
    $(2.15 \\pm 0.20) \\times 10^{-4}$

    :param value: Value to be given
    :param unc: Uncertainty of the value
    :param prec: Significant digits.

    :return: LaTeX formatted string, see example above.
    """
    value_exp_not = f"{value:E}"
    exponent = int(value_exp_not.split("E")[1])
    value_str = f"{value*10**(-exponent):.{prec}f}"
    unc_str = f"{unc*10**(-exponent):.{prec}f}"
    if exponent != 0:
        ret_str = f"$({value_str} \\pm {unc_str}) \\times 10^{{{exponent}}}$"
    else:
        ret_str = f"${value_str} \\pm {unc_str}$"
    return ret_str


def exp_notation(num: float, prec: int = 2) -> str:
    """Take a number and return it in LaTeX exponential notation.

    :param num: Number itself.
    :param prec: Precision.

    :return: LaTeX formatted string.
    """
    num_str = f"{num:.{prec}E}"
    e_index = num_str.find("E")
    value_str = decimal.Decimal(num_str[:e_index]).normalize()
    zeros_required = prec - len(str(value_str).split(".")[1])  # might be cut off!
    exp_str = decimal.Decimal(num_str[e_index + 1 :]).normalize()
    ret_str = f"${value_str}{'0'*zeros_required} \\times 10^{{{exp_str}}}$"
    return ret_str


def delta_iso(iso1: str, iso2: str, full=False) -> str:
    """Return LaTeX formatted string for delta notation of two isotopes.

    Full label is, e.g., d(30Si/28Si). Short version is d30Si28.

    Note: For short version (default) of the return it is assumed that all the
        both elements are the same.

    :param iso1: Nominator isotope, e.g., "Si-30"
    :type iso1: str
    :param iso2: Denominator isotope, e.g., "Si-28"
    :type iso2: str
    :param full: Do you want a full label? Otherwise, short version used.

    :return: LaTeX formatted label for delta value. Unit not included.
    :rtype: str
    """
    iso1 = ini.iso[iso1].name
    iso2 = ini.iso[iso2].name
    ele1, aa1 = split_iso(iso1)
    ele2, aa2 = split_iso(iso2)
    if full:
        ret_val = (
            f"$\\delta({{^{{{aa1}}}}}\\mathrm{{{ele1}}}"
            f"/{{^{{{aa2}}}}}\\mathrm{{{ele2}}})$ (‰)"
        )
    else:
        ret_val = f"$\\delta{{^{{{aa1}}}}}\\mathrm{{{ele1}}}_{{{aa2}}}$ (‰)"
    return ret_val


def iso_transformer(iso: str):
    """Transform isotope from `46Ti` notation to `Ti-46` notation and vice verse.

    Transformation direction is automatically determined by input.

    :param iso: Isotope as string

    :return: iso, but in transformed notation
    """
    if "-" in iso:
        iso = iso.split("-")
        return f"{iso[1]}{iso[0]}"
    else:
        index_to = None
        for it, number in enumerate(iso):
            try:
                int(number)
            except ValueError:
                index_to = it
                break
        return f"{iso[index_to:]}-{iso[:index_to]}"


def ratio_iso(iso1: str, iso2: str) -> str:
    """Return LaTeX formatted string for ratio of two isotopes.

    :param iso1: Nominator isotope, e.g., "Si-30"
    :type iso1: str
    :param iso2: Denominator isotope, e.g., "Si-28"

    :return: LaTeX formatted label for isototope ratio.
    :rtype: str
    """
    ele1, aa1 = split_iso(iso1)
    ele2, aa2 = split_iso(iso2)
    ret_val = f"${{^{{{aa1}}}}}\\mathrm{{{ele1}}}/{{^{{{aa2}}}}}\\mathrm{{{ele2}}}$"
    return ret_val


def split_iso(iso: str) -> Tuple[str, Union[int, str]]:
    """Split isotope string into element name and mass number.

    :param iso: Isotope name, e.g., "Si-28"
    :type iso: str

    :return: Isotope name, mass number (as int if possible)
    """
    # transform to correct format if necessary
    if "-" not in iso:
        iso = iso_transformer(iso)

    ele, aa = iso.split("-")
    try:
        aa = int(aa)
    except ValueError:
        pass
    return ele, aa
