"""Model reader for Maria Lugaro's AGB star files."""

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from rttools.latex import exp_notation
from rttools.utils import reduce_decimal


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


def label_latex_mass_z_pmz(fname: str, prec: int = 2) -> str:
    """Create a LaTeX label for mass, metallicity, and PMZ.

    :param fname: Filename of data file.
    :param prec: Precision of return value (at most).

    :return: LaTeX label.
    """
    m, z, pmz, st = label_parser(fname)

    return (
        f"${reduce_decimal(m, prec)}\\,M_{{\\odot}}$, "
        f"${reduce_decimal(z, prec)}\\,Z_{{\\odot}}$, "
        f"PMZ: {exp_notation(pmz, prec)}$\\,M_{{\\odot}}$"
    )


def label_parser(fname: str) -> Tuple[float, float, float, bool]:
    """Parse a filename and return properties of star.

    :param fname: File name (with or without ".dat")

    :return: Mass (M_sun), Metallicity (Z_sun), PMZ, st-case
    """
    if isinstance(fname, Path):
        fname = str(fname.name)

    mass_start = fname.find("m")
    z_start = fname.find("z", mass_start)
    pmz_start = fname.find("pmz", z_start)

    st = False
    if "ST" in fname or "st" in fname:
        st = True

    # remove .dat and ST
    fname = fname.replace(".dat", "")
    fname = fname.replace("ST", "")
    fname = fname.replace("st", "")

    # parse data
    mass = float(fname[mass_start + 1 : z_start].replace("p", "."))
    z = fname[z_start + 1 : pmz_start]
    if len(z) == 2:
        z = f"{z}0"
    z = float(z) / 14
    pmz = float(fname[pmz_start + 3 :].replace("m", "e-"))

    return mass, z, pmz, st


def plot_mod(
    ax: plt.Axes,
    fname: Path,
    xdat_col: int,
    ydat_col: int,
    co_ratio_col: int = 0,
    marker: str = "o",
    color: str = "tab:blue",
    label=None,
    linestyle: str = "-",
    **kwargs,
) -> None:
    """Plot Lugaro Model curve onto a plot.

    Note that **kwargs are passed along to both plotting routines. For C/O < 1, the
    marker is set to 'None'.

    :param ax: Matplotlib axis to plot on.
    :param fname: Filename of model data file
    :param xdat_col: Number of column to plot on x-axis
    :param ydat_col: Number of column to plot on y-axis
    :param co_ratio_col: Number of column for CO ratio (defaults to 0)
    :param marker: Marker of the plot, defaults to 'o'
    :param color: Color of the plot, defaults to 'tab:blue'
    :param linestyle: Linestyle of the plot, defaults to '-'

    :return: Nothing
    """
    data = reader(fname)

    xdat_all = data[f"{xdat_col}"]
    ydat_all = data[f"{ydat_col}"]
    co_ratio = data[f"{co_ratio_col}"]

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


def reader(fname: Path, cols: List[str] = None) -> pd.DataFrame:
    """Read a Lugaro ABG star file and return it.

    In these files, the first column is generally the model number. Subsequent columns
    are labeled with integers, starting at zero.
    In my experience, column 0 is generally the C/O ratio. Subsequent columns are
    delta-values of specific isotope ratios.

    :param fname: Filename of file to read
    :param cols: Optional, headers for the columns starting with integers.

    :return: Data Frame of the whole file, with columns as headers if given.
    """
    data = pd.read_csv(fname, sep="\t")

    if cols is not None:
        cols_new = cols.copy()
        cols_new.insert(0, "model number")
        data.columns = cols_new

    return data
