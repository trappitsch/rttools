"""Handler for FRUITY model files (from http://fruity.oa-teramo.inaf.it/)."""

from pathlib import Path
from typing import Tuple

from iniabu import ini
import matplotlib.pyplot as plt
import pandas as pd


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


def get_delta_isoratio(fname: Path, iso1: str, iso2: str) -> pd.Series:
    """Take a FRUITY file and return delta values for one isotope ratio.

    Delta will be calculated with respect to current iniabu default database. The ratio
    is going to be iso1/iso2. Initial and first dredge up (FDU) columns are returned as
    well.

    :param fname: FRUITY file name.
    :iso1: Nominator isotope, any `iniabu` isotope name accepted.
    :iso2: Denominator isotope, any `iniabu` isotope name accepted.

    :return: Series with delta ratio of the two.
    """
    iso1 = make_fruity_isoname(iso1)
    iso2 = make_fruity_isoname(iso2)

    data = drop_a_z(reader(fname))
    ratios = data.loc[iso1] / data.loc[iso2]
    delta = ini.iso_delta(iso1, iso2, ratios)
    # fixme: this should be handled correctly by iniabu, but is not
    delta = pd.Series(delta, index=ratios.index)
    return delta


def make_fruity_isoname(iso: str) -> str:
    """Pass an isotope through `iniabu` and then parse a FRUITY isotope name.

    :param iso: Isotope, e.g., "28Si", "Si-28", "Si28"

    :return: FRUITY name, e.g., "Si28"
    """
    iso = ini.iso[iso].name
    return iso.replace("-", "")


def plot_mod(
    ax: plt.Axes,
    fname: Path,
    xisos: Tuple[str, str],
    yisos: Tuple[str, str],
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
    :param marker: Marker of the plot, defaults to 'o'
    :param color: Color of the plot, defaults to 'tab:blue'
    :param linestyle: Linestyle of the plot, defaults to '-'

    :return: Nothing
    """
    drop_cols = ["INI", "FDU"]
    xdat_all = get_delta_isoratio(fname, xisos[0], xisos[1]).drop(columns=drop_cols)
    ydat_all = get_delta_isoratio(fname, yisos[0], yisos[1]).drop(columns=drop_cols)
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
    data = pd.read_csv(fname, sep=r"\s\s+", index_col=0)  # RegEx to find spaces

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
