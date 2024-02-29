"""Plot some good-looking saturation curves from measurements.

Import as:

```python
from rttools.rims import saturation_curve
```
"""

from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from rttools import latex, ureg


def saturation_curve(
    xdata: np.ndarray,
    ydata: np.ndarray,
    xunit: Union[ureg.Quantity, str] = None,
    yunit: Union[ureg.Quantity, str] = None,
    xlabel: str = None,
    ylabel: str = "Signal",
    fit: bool = True,
    darkmode: bool = False,
    title: str = None,
) -> plt.Figure:
    """Plot a saturation curve from data.

    If fitting is desired, a Letokhov saturation curve is fitted to the data and plotted.
    The fit parameters are printed on the plot.
    The fit is done with the `scipy.optimize.curve_fit` function and only considers
    uncertainties in the y-axis data, if given.
    The fit function is `n = ni + nmax * (1 - exp(-x / isat))`.

    :param xdata: x-axis data. Can contain two columns for error bars.
    :param ydata: y-axis data. Can contain two columns for error bars.
    :param xunit: Unit for x-axis data. If `xdata` is unitful, this is ignored.
    :param yunit: Unit for y-axis data. If `ydata` is unitful, this is ignored.
    :param xlabel: Label for x-axis. If None, try to infer from xunit.
        Infers "Power" if xunit is in W or "Irradiance" if xunit is in W/cm^2.
    :param fit: If True, fit a Letokhov saturation curve to the data, plot it,
        and mark up with fit parameters.
    :param darkmode: If True, use darkmode for the plot.
    :param title: Title for the plot

    :return: Matplotlib axis
    """
    if isinstance(xdata, ureg.Quantity):
        xunit = xdata.units
        xdata = xdata.magnitude
    elif isinstance(xunit, str):
        xunit = ureg.Unit(xunit)

    if isinstance(ydata, ureg.Quantity):
        yunit = ydata.units
        ydata = ydata.magnitude
    elif isinstance(yunit, str):
        yunit = ureg.Unit(yunit)

    # latex units if not none
    xunit_ltx = None
    yunit_ltx = None
    if xunit is not None:
        xunit_ltx = f"{xunit:P~}"
    if yunit is not None:
        yunit_ltx = f"{yunit:P~}"

    # infer xlabel if not given
    if xunit is not None:
        if xlabel is None:
            if xunit.compatible_units() == ureg.W.compatible_units():
                xlabel = "Power"
            elif xunit.compatible_units() == (ureg.W / ureg.cm**2).compatible_units():
                xlabel = "Irradiance"

    # split xdata and ydata into xdata, xerr and ydata, yerr
    if xdata.ndim == 2:
        xerr = xdata[1]
        xdata = xdata[0]
    else:
        xerr = None
    if ydata.ndim == 2:
        yerr = ydata[1]
        ydata = ydata[0]
    else:
        yerr = None

    if darkmode:
        plt.style.use("dark_background")
        col_blue = "lightblue"
        col_red = "lightsalmon"
    else:
        plt.style.use("default")
        col_blue = "tab:blue"
        col_red = "darkred"

    # create the figure
    fig, ax = plt.subplots(1, 1)

    # plot the data
    ax.errorbar(
        xdata,
        ydata,
        xerr=xerr,
        yerr=yerr,
        marker="o",
        label="Data",
        linestyle="None",
        linewidth=0.5,
        color=col_blue,
    )

    # Axes labels
    if xlabel:
        if xunit_ltx:
            ax.set_xlabel(f"{xlabel} ({xunit_ltx})")
        else:
            ax.set_xlabel(f"{xlabel}")

    if ylabel:
        if yunit_ltx is not None:
            ax.set_ylabel(f"{ylabel} ({yunit_ltx})")
        else:
            ax.set_ylabel(f"{ylabel} (arb)")

    # fit a curve if desired
    if fit:
        # take an initial guess from the data
        initial_guess = [0, 1, np.max(xdata) / 2]

        popt, pcov = curve_fit(
            _letokhov,
            xdata,
            ydata,
            p0=initial_guess,
            sigma=yerr,
        )
        ni, nmax, isat = popt
        xfit = np.linspace(xdata.min(), xdata.max(), 1000)
        yfit = _letokhov(xfit, ni, nmax, isat)

        ax.plot(xfit, yfit, label="Fit", color=col_red, linestyle="-")
        ax.legend(loc="upper left", framealpha=1)

        fit_string = "Fit parameters:\n"
        fit_string += (
            f"I$_\\mathrm{{sat}}$ = {latex.exp_notation(isat, 2)} {xunit_ltx}\n"
        )
        fit_string += f"N$_\\mathrm{{max}}$ = {nmax:.2f}"

        ax.text(
            0.96,
            0.05,
            fit_string,
            color=col_red,
            alpha=1,
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            ma="left",
            bbox=dict(boxstyle="round", facecolor="None", alpha=0.7, edgecolor=col_red),
        )

    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig


def _letokhov(x, ni, nmax, isat):
    """Letokhov saturation curve."""
    return ni + nmax * (1 - np.exp(-x / isat))
