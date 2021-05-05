"""Tools for LaTeX formatting, e.g., of isotope strings, etc."""

from typing import Tuple


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
            f"/{{^{{{aa2}}}}}\\mathrm{{{ele2}}})$"
        )
    else:
        ret_val = f"$\\delta{{^{{{aa1}}}}}\\mathrm{{{ele1}}}_{{{aa2}}}$"
    return ret_val


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


def split_iso(iso: str) -> Tuple[str, int]:
    """Split isotope string into element name and mass number.

    :param iso: Isotope name, e.g., "Si-28"
    :type iso: str

    :return: Isotope name, mass number
    :rtype: Tuple[str, int]
    """
    ele, aa = iso.split("-")
    return ele, int(aa)
