"""Handler for FRUITY model files (from http://fruity.oa-teramo.inaf.it/).

This module provides some functions to work with the FRUITY nucleosynthesis models.

Import as:

```python
from rttools.nucmod import fruity
```
"""

from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
from iniabu import ini

from rttools.utils import reduce_decimal


def drop_a_z(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns for A and Z for a dataframe.

    Leaves only columns with compositions over from a FRUITY dataframe and removes
    columns labeled "A" and "Z". If these columns don't exist, no error will be raised.

    :param df: FRUITY dataframe with columns "A" and "Z" present.

    :return: same dataframe with those vectors dropped.
    """
    retval = df.copy()
    columns = ["A", "Z"]
    for col in columns:
        try:
            retval.drop(columns=[col], inplace=True)
        except KeyError:
            pass

    return retval


def get_delta_isoratio(
    fname: Path, iso1: str, iso2: str, norm_self: bool = False
) -> pd.Series:
    """Take a FRUITY file and return delta values for one isotope ratio.

    Delta will be calculated with respect to current iniabu default database. The ratio
    is going to be iso1/iso2. Initial and first dredge up (FDU) columns are returned as
    well.

    :param fname: FRUITY file name.
    :param iso1: Nominator isotope, any `iniabu` isotope name accepted.
    :param iso2: Denominator isotope, any `iniabu` isotope name accepted.
    :param norm_self: Norm to itself, i.e., to the initial composition instead of using
        standard database by `iniabu`.

    :return: Series with delta ratio of the two.
    """
    iso1f = make_fruity_isoname(iso1)
    iso2f = make_fruity_isoname(iso2)

    data = drop_a_z(reader(fname))
    ratios = data.loc[iso1f] / data.loc[iso2f]

    if norm_self:
        delta = (ratios / ratios.iloc[0] - 1.0) * 1000
    else:
        delta = ini.iso_delta(iso1, iso2, ratios)
        # fixme: this should be handled correctly by iniabu, but is not
        delta = pd.Series(delta, index=ratios.index)
    return delta


def label_latex_mass(fname: str, prec: int = 1) -> str:
    """Create a LaTeX label for only mass.

    :param fname: Filename of data file.
    :param prec: Precision of return value (at most).

    :return: LaTeX label.
    """
    m, _, _, _ = label_parser(fname)
    return f"${reduce_decimal(m, prec)}\\,M_{{\\odot}}$"


def label_latex_mass_z(fname: str, prec: int = 2) -> str:
    """Create a LaTeX label for mass and metallicity.

    :param fname: Filename of data file.
    :param prec: Precision of return value (at most).

    :return: LaTeX label.
    """
    m, z, _, _ = label_parser(fname)
    return (
        f"${reduce_decimal(m, prec)}\\,M_{{\\odot}}$, "
        f"${reduce_decimal(z, prec)}\\,Z_{{\\odot}}$"
    )


def label_latex_mass_z_pocket(fname: str, prec: int = 2) -> str:
    """Create a LaTeX label for mass and metallicity.

    :param fname: Filename of data file.
    :param prec: Precision of return value (at most).

    :return: LaTeX label.
    """
    m, z, _, pocket = label_parser(fname)
    return (
        f"${reduce_decimal(m, prec)}\\,M_{{\\odot}}$, "
        f"${reduce_decimal(z, prec)}\\,Z_{{\\odot}}$, "
        f"{pocket}"
    )


def label_latex_mass_z_pocket_rot(fname: str, prec: int = 2) -> str:
    """Create a LaTeX label for mass and metallicity.

    :param fname: Filename of data file.
    :param prec: Precision of return value (at most).

    :return: LaTeX label.
    """
    m, z, rot, pocket = label_parser(fname)
    return (
        f"${reduce_decimal(m, prec)}\\,M_{{\\odot}}$, "
        f"${reduce_decimal(z, prec)}\\,Z_{{\\odot}}$, "
        f"{pocket}, "
        f"IRV: ${reduce_decimal(rot, prec)}\\,$km\\,s$^{{-1}}$"
    )


def label_parser(fname: Union[str, Path]) -> Tuple[float, float, float, str]:
    """Label parser for FRUITY models.

    Take a FRUITY filename and return various parameters as numbers.

    :param fname: File name of the FRUITY file.

    :return: Mass (M_sun), Metallicity (Z_sun), Rotation (km/s), C13 pocket (std / ext)
    """
    if isinstance(fname, Path):
        fname = str(fname.name)

    # remove 'isotopi_'
    fname = fname.replace("isotopi_", "")

    # now break by '_' and use first two
    fname = fname.split("_")[:2]

    # mass and metallicity
    z_start = fname[0].find("z")
    mass = float(fname[0][1:z_start].replace("p", "."))
    z = fname[0][z_start + 1 :].replace("m", "e-")
    if z == "sun":
        z = 1
    else:
        z = float(z) / 1.4e-2

    # pocket and rotation
    if fname[1][0] == "T":
        pocket = "ext"
    else:
        pocket = "std"
    rot = float(fname[1][1:])
    return mass, z, rot, pocket


def make_fruity_isoname(iso: str) -> str:
    """Pass an isotope through `iniabu` and then parse a FRUITY isotope name.

    :param iso: Isotope, e.g., "28Si", "Si-28", "Si28"

    :return: FRUITY name, e.g., "Si28"
    """
    iso = ini.iso[iso].name.split("-")
    iso_fruity = iso[0] + iso[1][-2:]
    return iso_fruity


def plot_mod(
    ax: plt.Axes,
    fname: Path,
    xisos: Tuple[str, str],
    yisos: Tuple[str, str],
    norm_self: bool = False,
    marker: str = "o",
    color: str = "tab:blue",
    label=None,
    linestyle: str = "-",
    **kwargs,
) -> None:
    """Plot FRUITY Model curve onto a plot.

    Note that **kwargs are passed along to both plotting routines. For C/O < 1, the
    marker is set to 'None'.

    :param ax: Matplotlib axis to plot on.
    :param fname: Filename of model data file
    :param xisos: Tuple of two isotopes (nominator, denominator) for x axis.
    :param yisos: Tuple of two isotopes (nominator, denominator) for y axis.
    :param norm_self: Norm to itself, i.e., to the initial composition instead of using
        standard database by `iniabu`.
    :param marker: Marker of the plot, defaults to 'o'
    :param color: Color of the plot, defaults to 'tab:blue'
    :param linestyle: Linestyle of the plot, defaults to '-'

    :return: Nothing
    """
    drop_cols = ["INI", "FDU"]
    xdat_all = get_delta_isoratio(fname, xisos[0], xisos[1], norm_self=norm_self).drop(
        columns=drop_cols
    )
    ydat_all = get_delta_isoratio(fname, yisos[0], yisos[1], norm_self=norm_self).drop(
        columns=drop_cols
    )
    co_ratio = drop_a_z(reader(fname).loc["CO_ratio"]).drop(columns=drop_cols)

    crich_mask = co_ratio >= 1

    # plot all
    ax.plot(
        xdat_all, ydat_all, marker="None", linestyle=linestyle, color=color, **kwargs
    )
    # plot c-rich
    ax.plot(
        xdat_all[crich_mask],
        ydat_all[crich_mask],
        marker=marker,
        color=color,
        linestyle="None",
        label=label,
        **kwargs,
    )


def reader(fname: Path) -> pd.DataFrame:
    """Read in a FRUITY file and return values as DataFrame."""
    data = pd.read_csv(fname, sep=r" ", index_col=0, skipinitialspace=True)

    # drop nan columns
    data = data.dropna(axis=1, how="all")

    # set the column labels straight
    col_labels = ["A", "Z", "INI", "FDU"]  # the first four, isotope is index
    co_ratio_cols = ["INI", "FDU"]  # calculate CO ratios for these plus TDUs
    for it in range(data.shape[1] - len(col_labels)):
        tdu = f"TDU_{it+1}"
        col_labels.append(tdu)
        co_ratio_cols.append(tdu)
    data.columns = col_labels

    # create a new series with C/O ratio called CO_ratio
    co_ratio = data.loc[["C12", "C13"], co_ratio_cols].sum(axis=0) / data.loc[
        ["O16", "O17", "O18"], co_ratio_cols
    ].sum(axis=0)

    data.loc["CO_ratio"] = co_ratio

    return data
