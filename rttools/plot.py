"""Some tools to make plotting with MPL easier."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def corr_error_bars(
    ax: mpl.axes.Axes,
    xdata: np.ndarray,
    ydata: np.ndarray,
    xerr: np.ndarray,
    yerr: np.ndarray,
    rho: np.ndarray,
    linestyle: str = "-",
    linewidth: float = 0.5,
    marker: str = "None",
    color: str = "tab:blue",
    **kwargs
) -> None:
    """Plot correlated error bars on a given axes.

    Further arguments to specify plot can be passed as **kwargs, which will be passed
    to the `matplotlib.pyplot.plot` routine.

    Currently only tested with some Mo data from Stephan et al. (2019) and compared
    in Inkscape.

    :param ax: Matplotlib axes to plot on
    :param xdata: data for x values
    :param ydata: data for y values
    :param xerr: uncertainty for x values
    :param yerr: uncertainty for y values
    :param rho: correlation coefficient
    :param linestyle: Matplotlib linestyle, defaults to '-'
    :param linewidth: Width of the error bar line, defaults to 0.5
    :param marker: Matplotlib Marker, defaults to 'None'. Note: These would be the end
        markers of the line, not a marker in the middle!
    :param color: Matplotlib color, defaults to 'tab:blue'

    :return: None
    """
    # loop through all data points
    for it, xpos in enumerate(xdata):
        ypos = ydata[it]
        # get sig_x, sig_y, and calculate sig_xy of correlation matrix
        sig_x = xerr[it]
        sig_y = yerr[it]
        sig_xy = rho[it] * sig_x * sig_y
        # trace and determinat of covariance matrix
        tr_cov = sig_x ** 2 + sig_y ** 2
        det_cov = sig_x ** 2 * sig_y ** 2 - sig_xy ** 2
        # calculate the eigenvalues
        lam1 = tr_cov / 2 + np.sqrt((tr_cov / 2) ** 2 - det_cov)
        lam2 = tr_cov / 2 - np.sqrt((tr_cov / 2) ** 2 - det_cov)

        # calculate the rotation of the error bars
        phi_x = np.arctan((lam1 - sig_x ** 2) / sig_xy)
        phi_y = phi_x + np.pi / 2

        # calculate delta x and delta y for x error bar:
        dx_x = np.sqrt(lam1) * np.cos(phi_x)
        dy_x = np.sqrt(lam1) * np.sin(phi_x)
        # for y error bar
        dx_y = np.sqrt(lam2) * np.cos(phi_y)
        dy_y = np.sqrt(lam2) * np.sin(phi_y)

        # plot the x error bar
        xdat_x = [xpos - dx_x, xpos + dx_x]
        ydat_x = [ypos - dy_x, ypos + dy_x]
        ax.plot(
            xdat_x,
            ydat_x,
            marker=marker,
            linestyle=linestyle,
            color=color,
            linewidth=linewidth,
            **kwargs
        )

        # plot the y error bar
        xdat_y = [xpos - dx_y, xpos + dx_y]
        ydat_y = [ypos - dy_y, ypos + dy_y]
        ax.plot(
            xdat_y,
            ydat_y,
            marker=marker,
            linestyle=linestyle,
            color=color,
            linewidth=linewidth,
            **kwargs
        )
