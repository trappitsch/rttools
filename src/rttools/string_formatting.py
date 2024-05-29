"""StringFmt class to format strings for different types of output."""

from enum import Enum
import re


class StringFmt:
    """Class to format strings for different types of output.

    This is a simple string formatter and takes for now LaTeX strings and turns them
    into HTML strings. Only sub and superscript, and a few symbols are currently
    included.
    """

    class Type(Enum):
        """Enum for the different types of string formatting."""

        latex = 0

    def __init__(self, string: str, string_type: Type):
        """Initialize the class with a string and a type."""
        if string_type != self.Type.latex:
            raise NotImplementedError(
                "String formatting is currently only implemented for LaTeX strings."
            )

        self._string = string
        self._string_type = string_type

    @property
    def html(self):
        string = self._string

        # replace unescaped $ with nothing
        string = re.sub(r"(?<!\\)\$", "", string)

        # replace superscript in between { } (unescaped) with <sup>...</sup>
        string = re.sub(r"(?<!\\)\^\{((.*?)|\w)\}", r"<sup>\1</sup>", string)

        # replace single character superscript without { } and unescaped with <sup>...</sup>
        string = re.sub(r"(?<!\\)\^(\w)", r"<sup>\1</sup>", string)

        # replace subscript in between { } (unescaped) with <sub>...</sub>
        string = re.sub(r"(?<!\\)_\{((.*?)|\w)\}", r"<sub>\1</sub>", string)

        # replace single character subscript without { } and unescaped with <sub>...</sub>
        string = re.sub(r"(?<!\\)_(\w)", r"<sub>\1</sub>", string)

        # replace {\\circ} with °
        string = string.replace(r"{\circ}", "°")

        return string

    @property
    def latex(self) -> str:
        """Return the string formatted for LaTeX."""
        return self._string
