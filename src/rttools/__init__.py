"""Tools for my own research projects - an assortment."""

from rttools.string_formatting import StringFmt

from pint import UnitRegistry

ureg = UnitRegistry()


__all__ = ["StringFmt", "ureg"]
