"""Tools for LaTeX formatting, e.g., of isotope strings, etc."""
import decimal
from typing import Tuple, Union


def exp_notation(num: float, prec: int = 2) -> str:
    """Take a number and return it in LaTeX exponential notation.

    :param num: Number itself.
    :param prec: Precision.

    :return: LaTeX formatted string.
    """
    num_str = f"{num:.{prec}E}"
    e_index = num_str.find("E")
    value_str = decimal.Decimal(num_str[:e_index]).normalize()
    exp_str = decimal.Decimal(num_str[e_index + 1 :]).normalize()
    ret_str = f"${value_str} \\times 10^{{{exp_str}}}$"
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
    if not "-" in iso:
        iso = iso_transformer(iso)

    ele, aa = iso.split("-")
    try:
        aa = int(aa)
    except ValueError:
        pass
    return ele, aa
