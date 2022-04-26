"""Linear fits with uncertainties in both axes according to Mahon et al. (1996)"""
from typing import Tuple, Union
import warnings

import numpy as np
from numpy.polynomial import Polynomial
import scipy.stats as stats

from .utils import kron_delta


class Mahon:
    def __init__(self):
        warnings.warn(
            "This linear regression method is depreciated. Please use "
            "`Stephan` regression instead of `Mahon`."
        )
        # ### Loaded from calc
        # fixed intercept
        self.afx = None

        # create variables for holding the data
        self.xdat, self.xunc = None, None
        self.ydat, self.yunc = None, None
        self.p = None

        # ### Other init stuff for later
        # some variables to have for later
        self.slope = None
        self.xinter = None
        self.yinter = None
        self.slopeunc = None
        self.yinterunc = None
        self.xinterunc = None
        self.mswd = None
        self.fname = None

        # some variables to remember
        self.xbar = None
        self.ybar = None

        # confidence intervals - if calculated
        self.ci_xax = None  # x axis values where yax ci calculated
        self.ci_yax_pos = None  # y axis confidence interval positive
        self.ci_yax_neg = None  # y axis confidence interval negative

    def calculate(self, xdat, ydat, xunc, yunc, p_corr=None, afx=None, mc=False):
        """
        Data, coefficients and intercept. All errors must be 1 sigma!!!
        :param xdat:        <np.array>   x data
        :param ydat:        <np.array>   y data
        :param xunc:        <np.array>   x uncertainty, 1 sigma
        :param yunc:        <np.array>   y uncertainty, 1 sigma
        :param p_corr:      <np.array>   correlation coefficient, None for uncorrelated
        :param afx:         <np.float>   x intercept if fixed, otherwise None
        :param mc:          <bool>       If true, don't calculate uncertainties or x intercept
        :return:
        """
        # set data
        self.xdat, self.xunc = xdat, xunc
        self.ydat, self.yunc = ydat, yunc
        self.p = p_corr
        self.afx = afx

        # if correlation is zero or just one number, then set p_corr accordingly
        if p_corr is None:
            self.p = np.zeros(len(xdat))
        elif not isinstance(p_corr, np.ndarray):
            self.p = np.zeros(len(xdat)) + p_corr

        # fixed intercept checking:
        if self.afx is not None:
            try:
                self.afx = float(self.afx)
            except ValueError:
                print("Please define the fixed intercept as a float.")
                return
            # add a point the the input data that has 1e12 times smaller errors than the smallest error in the system
            xdat = np.zeros(len(self.xdat) + 1)
            ydat = np.zeros(len(self.ydat) + 1)
            xunc = np.zeros(len(self.xunc) + 1)
            yunc = np.zeros(len(self.yunc) + 1)
            p = np.zeros(len(self.p) + 1)
            # find the minimum error to add
            errintercept = (
                np.min(np.array([np.min(self.xunc), np.min(self.yunc)])) / 1.0e18
            )
            # now add the point to the new array and then add all the existing data
            xdat[0] = 0.0
            ydat[0] = self.afx
            xunc[0] = errintercept
            yunc[0] = errintercept
            p[0] = 0.0
            for it in range(len(self.xdat)):
                xdat[it + 1] = self.xdat[it]
                ydat[it + 1] = self.ydat[it]
                xunc[it + 1] = self.xunc[it]
                yunc[it + 1] = self.yunc[it]
                p[it + 1] = self.p[it]
            # now write back
            self.xdat = xdat
            self.ydat = ydat
            self.xunc = xunc
            self.yunc = yunc
            self.p = p
            # now calculate
            self.calcparams()
            if not mc:
                self.calcunc()
                self.calcunc(calcxintunc=True)
        else:
            # run the calculation with the loaded x and y data
            self.calcparams()
            if not mc:
                self.calcunc()
                self.calcunc(calcxintunc=True)

        # calculate the MSWD
        self.calcmswd()

    def calculate_with_ci(
        self,
        xdat,
        ydat,
        xunc,
        yunc,
        p_corr=None,
        afx=None,
        p_conf=0.95,
        bins=100,
        xlims=None,
    ):
        """Regular linear regression, however, also calculated confidence intervals/

        Data, coefficients and intercept. All errors must be 1 sigma!!! Note: the last
        calculation that is done here is the same as when running the regular
        `calculate` routine, however, here the CI values are also populated.

        :param xdat:        <np.array>   x data
        :param ydat:        <np.array>   y data
        :param xunc:        <np.array>   x uncertainty, 1 sigma
        :param yunc:        <np.array>   y uncertainty, 1 sigma
        :param p_corr:      <np.array>   correlation coefficient, None for uncorrelated
        :param afx:         <float>      x intercept if fixed, otherwise None
        :param p_conf:      <float>      which confidence interval? by default 95%
        :param bins:        <int>        How many steps for CI? defaults to 100
        :param xlims:       <np.array>   Limits for x axis to calculate CI in. Defaults
            to None, which will take the minimum and maximum of the axis.
        :return:
        """
        if xlims is None:
            xax_ci = np.linspace(np.min(xdat), np.max(xdat), bins)
        else:
            xax_ci = np.linspace(xlims[0], xlims[1], bins)

        yax_ci = np.zeros(bins)
        for it, deltax in enumerate(xax_ci):
            xdat_tmp = xdat - deltax
            self.calculate(xdat_tmp, ydat, xunc, yunc, p_corr=p_corr, afx=afx, mc=False)
            yax_ci[it] = self.yinterunc

        # now create the confidence interval that we need for double tailed distribution
        zfac = stats.t.ppf(1 - (1 - p_conf) / 2.0, len(xdat) - 2)
        yax_ci *= zfac

        # finally: calculate it with the real dataset for right data in class at end
        self.calculate(xdat, ydat, xunc, yunc, p_corr=p_corr, afx=afx, mc=False)

        # set ci values
        self.ci_xax = xax_ci
        self.ci_yax_pos = xax_ci * self.slope + self.yinter + yax_ci
        self.ci_yax_neg = xax_ci * self.slope + self.yinter - yax_ci

    def clear(self):
        self.xdat, self.xunc = None, None
        self.ydat, self.yunc = None, None
        self.p = None
        self.afx = None

    def calcparams(self):
        """
        Calculate the parameters, both intercepts as well as the slope of the regression
        :return:
        """
        bcalc = 1  # initial guess for slope
        bold = 0  # to compare later

        # read in selfs
        xdat = self.xdat
        xunc = self.xunc
        ydat = self.ydat
        yunc = self.yunc

        # run thorough the while loop
        whilecounter = 0
        whilecountermax = 1e5
        while (
            np.abs((bold - bcalc) / bcalc) > 1.0e-10 and whilecounter < whilecountermax
        ):
            whilecounter += 1
            # prep for while loop, start before this line and compare bold to bcalc
            bold = bcalc
            # calculate xbar and ybar
            xbar = 0
            ybar = 0
            weightsum = 0
            for it in range(len(xdat)):
                wi = self.calc_wi(xunc[it], yunc[it], bcalc, self.p[it])
                xbar += xdat[it] * wi
                ybar += ydat[it] * wi
                weightsum += wi
            xbar /= weightsum
            ybar /= weightsum

            # now calculate b
            btop = 0  # top sum
            bbot = 0  # bottom sum

            for it in range(len(xdat)):
                xi = xdat[it]
                yi = ydat[it]
                sxi = xunc[it]
                syi = yunc[it]
                pit = self.p[it]
                wi = self.calc_wi(sxi, syi, bcalc, pit)
                ui = xi - xbar
                vi = yi - ybar
                # add to sums
                btop += (
                    wi**2.0
                    * vi
                    * (ui * syi**2.0 + bcalc * vi * sxi**2.0 - pit * vi * sxi * syi)
                )
                bbot += (
                    wi**2.0
                    * ui
                    * (
                        ui * syi**2.0
                        + bcalc * vi * sxi**2.0
                        - bcalc * pit * ui * sxi * syi
                    )
                )

            # new bcalc
            bcalc = btop / bbot

        # error message if whilecounter timed out
        if whilecounter == whilecountermax:
            print(
                "Convergence warning",
                "Warning! Your calculation might not have converged "
                + "properly. The difference between the last calculated slope  and the current slope is: "
                + str(np.abs((bold - bcalc) / bcalc))
                + " You can ignore this "
                + "Message if this is an acceptable "
                "convergence for you.",
            )

        # now that slope is determined, calculate the y intercept
        self.yinter = ybar - bcalc * xbar

        # now done, so write back slope
        self.slope = bcalc

        # calculate x intercept
        self.xinter = -self.yinter / self.slope

        # write back xbar and ybar
        self.xbar = xbar
        self.ybar = ybar

    def calcunc(self, calcxintunc=False):
        """
        Calculates the uncertainty of the slope and y
        :param calcxintunc: If it needs to calculate the x uncertainty, then set this to true
        :return:
        """
        if calcxintunc:
            # read in selfs
            # since this is for x uncertainty, simply switch it x and y.
            xdat = self.ydat
            xunc = self.yunc
            ydat = self.xdat
            yunc = self.xunc
            xbar = self.ybar
            ybar = self.xbar
            b = 1.0 / self.slope
        else:
            # read in selfs
            xdat = self.xdat
            xunc = self.xunc
            ydat = self.ydat
            yunc = self.yunc
            xbar = self.xbar
            ybar = self.ybar
            b = self.slope

        # let us first calculate the derivatives
        # dell theta / dell b (dthdb) calculation
        sum1 = 0.0
        sum2 = 0.0
        for it in range(len(xdat)):
            xi = xdat[it]
            yi = ydat[it]
            sxi = xunc[it]
            syi = yunc[it]
            pit = self.p[it]
            wi = self.calc_wi(xunc[it], yunc[it], b, pit)
            ui = xi - xbar
            vi = yi - ybar
            sxyi = pit * sxi * syi
            sum1 += wi**2.0 * (
                2 * b * (ui * vi * sxi**2.0 - ui**2.0 * sxyi)
                + (ui**2.0 * syi**2.0 - vi**2 * sxi**2.0)
            )
            sum2 += (
                wi**3.0
                * (sxyi - b * sxi**2.0)
                * (
                    b**2.0 * (ui * vi * sxi**2 - ui**2 * sxyi)
                    + b * (ui**2 * syi**2 - vi**2 * sxi**2)
                    - (ui * vi * syi**2 - vi**2 * sxyi)
                )
            )
        dthdb = sum1 + 4.0 * sum2

        # calculate the sum of all weights
        wksum = 0.0
        for it in range(len(xdat)):
            wksum += self.calc_wi(xunc[it], yunc[it], b, self.p[it])

        # now calculate sigasq and sigbsq
        sigasq = 0.0
        sigbsq = 0.0
        for it in range(len(xdat)):
            sxi = xunc[it]
            syi = yunc[it]
            pit = self.p[it]
            wi = self.calc_wi(sxi, syi, b, pit)
            sxyi = pit * sxi * syi

            # calculate dell theta / dell xi and dell theta / dell yi
            dthdxi = 0.0
            dthdyi = 0.0
            for jt in range(len(xdat)):
                xj = xdat[jt]
                yj = ydat[jt]
                sxj = xunc[jt]
                syj = yunc[jt]
                pjt = self.p[jt]
                wj = self.calc_wi(sxj, syj, b, pjt)
                uj = xj - xbar
                vj = yj - ybar
                sxyj = pjt * sxj * syj
                # add to dthdxi and dthdyi
                dthdxi += (
                    wj**2.0
                    * (kron_delta(it, jt) - wi / wksum)
                    * (
                        b**2 * (vj * sxj**2 - 2 * uj * sxyj)
                        + 2 * b * uj * syj**2
                        - vj * syj**2
                    )
                )
                # correct equation! not equal to equation 21 in Mahon (1996)
                dthdyi += (
                    wj**2.0
                    * (kron_delta(it, jt) - wi / wksum)
                    * (
                        b**2 * uj * sxj**2
                        + 2 * vj * sxyj
                        - 2 * b * vj * sxj**2.0
                        - uj * syj**2
                    )
                )

            # now calculate dell a / dell xi and dell a / dell yi
            dadxi = -b * wi / wksum - xbar * dthdxi / dthdb
            dadyi = wi / wksum - xbar * dthdyi / dthdb

            # now finally add to sigasq and sigbsq
            sigbsq += (
                dthdxi**2.0 * sxi**2.0
                + dthdyi**2.0 * syi**2.0
                + 2 * sxyi * dthdxi * dthdyi
            )
            sigasq += (
                dadxi**2.0 * sxi**2.0
                + dadyi**2.0 * syi**2.0
                + 2 * sxyi * dadxi * dadyi
            )

        # now divide sigbsq
        sigbsq /= dthdb**2.0

        # now write slope uncertainty and y intercept uncertainty back to class
        if calcxintunc:
            self.xinterunc = np.sqrt(sigasq)
        else:
            self.yinterunc = np.sqrt(sigasq)
            self.slopeunc = np.sqrt(sigbsq)

    def calcmswd(self):
        xdat, ydat, xunc, yunc = self.xdat, self.ydat, self.xunc, self.yunc
        mswd = 0.0
        for it in range(len(xdat)):
            xi = xdat[it]
            yi = ydat[it]
            sxi = xunc[it]
            syi = yunc[it]
            pit = self.p[it]
            wi = self.calc_wi(sxi, syi, self.slope, pit)
            mswd += wi * (yi - self.slope * xi - self.yinter) ** 2.0

        # now divide by degrees of freedom minus 2, since 2 fixed parameters
        mswd /= len(xdat) - 2.0
        self.mswd = mswd

    def calc_wi(self, sx, sy, b, p):
        return 1.0 / (sy**2 + b**2 * sx**2 - 2 * b * p * sx * sy)


class Stephan:
    """Stephan and Trappitsch (2022) linear regression.

    Follows the linear regression lined out in Stephan and Trappitsch (2022).
    Todo: add DOI of the paper, once published.
    Todo: add Example of usage
    """

    def __init__(
        self,
        xdat: np.ndarray,
        sigx: np.ndarray,
        ydat: np.ndarray,
        sigy: np.ndarray,
        rho: Union[float, np.ndarray] = None,
        fixpt: np.ndarray = None,
        autocalc=True,
        **kwargs,
    ):
        """Initialize the class.

        :param xdat: X data.
        :param ydat: Y data.
        :param sigx: 1 sigma uncertainty of x data.
        :param sigy: 1 sigma uncertainty of y data.
        :param rho: Correlation between x and y data, defaults to no correlation.
        :param fixpt: Fixed point through which regression needs to go.
        :param autocalc: Automatically calculate the regression and print params.
        :param kwargs: Additional keyword arguments:
            "iter_max": Maximum iteration limit for slope (default 1e6)
            "reg_limit": Regression limit for slope (default: 1e-6)

        raises ValueError: Fix point is of the wrong shape.
        """
        self.xdat = np.array(xdat)
        self.sigx = np.array(sigx)
        self.ydat = np.array(ydat)
        self.sigy = np.array(sigy)
        if rho is None:
            self.rho = np.zeros_like(self.xdat)
        else:
            self.rho = np.array(rho)

        # calculate correlated uncertainty sigxy
        self.sigxy = self.rho * self.sigx * self.sigy

        if fixpt is not None:
            fixpt = np.array(fixpt)
            if fixpt.shape != (2,):
                raise ValueError("Fix point must be of the form [x_fix, yfix].")
        self.fix_pt = fixpt

        # Initialize the slope and intercept
        self._slope = None
        self._slope_unc = None
        self._intercept = None
        self._intercept_unc = None
        self._chi_squared = None
        self._mswd = None

        # keyword arguments
        if "regression_limit" in kwargs:
            self.reg_limit = kwargs["regression_limit"]
        else:
            self.reg_limit = 1e-6
        if "iter_max" in kwargs:
            self.iter_max = kwargs["iter_max"]
        else:
            self.iter_max = 1e6

        # helper variables
        self.xbar = None
        self.ybar = None
        self.weights = None

        if autocalc:
            self.calculate()

    @property
    def chi_squared(self) -> float:
        """Return chi_squared of the regression."""
        return self._chi_squared

    @property
    def intercept(self) -> Tuple[float, float]:
        """Return intercept and its 1 sigma uncertainty."""
        return self._intercept, self._intercept_unc

    @property
    def mswd(self) -> float:
        """Return MSWD of the regression."""
        return self._mswd

    @property
    def parameters(self) -> np.ndarray:
        """Return all parameters of the linear regression.

        :return: slope, slope_uncertainty, intercept, intercept_uncertainty, MSWD
        """
        return np.array(
            [
                self._slope,
                self._slope_unc,
                self._intercept,
                self._intercept_unc,
                self._mswd,
            ]
        )

    @property
    def slope(self) -> Tuple[float, float]:
        """Return slope and its 1 sigma uncertainty."""
        return self._slope, self._slope_unc

    def calculate(self):
        """Do the linear regression and save the parameters in the class variables."""
        self.slope_initial_guess()
        self.slope_calculation()
        self._intercept = self.ybar - self._slope * self.xbar
        self.unc_calculation()
        self.goodness_of_fit()

    def goodness_of_fit(self):
        """Calculate goodness of fit parameters chi-squared and MSWD."""
        chi_sq = np.sum(
            self.weights * (self.ydat - self._slope * self.xdat - self._intercept) ** 2
        )
        self._chi_squared = chi_sq

        dof = len(self.xdat) - 2 if self.fix_pt is None else len(self.xdat) - 1
        self._mswd = chi_sq / dof

    def slope_calculation(self):
        """Iterate the slope until it fits."""

        def calc_weights(b: float):
            """Calculate weights and return them."""
            return 1 / (self.sigy**2 + b**2 * self.sigx**2 - 2 * b * self.sigxy)

        def calc_xbar(weights: np.ndarray):
            """Calculate x bar and return it.

            :param weights: Weights.
            """
            if self.fix_pt is None:
                return np.sum(weights * self.xdat) / np.sum(weights)
            else:
                return self.fix_pt[0]

        def calc_ybar(weights: np.ndarray):
            """Calculate y bar and return it.

            :param weights: Weights.
            """
            if self.fix_pt is None:
                return np.sum(weights * self.ydat) / np.sum(weights)
            else:
                return self.fix_pt[1]

        def iterate_b(b_old):
            """Do one iteration step with the slope and return the new value."""
            b = b_old
            weights = calc_weights(b)
            u_all = self.xdat - calc_xbar(weights)
            v_all = self.ydat - calc_ybar(weights)
            b_new = np.sum(
                weights**2
                * v_all
                * (
                    u_all * self.sigy**2
                    + b * v_all * self.sigx**2
                    - v_all * self.sigxy
                )
            ) / np.sum(
                weights**2
                * u_all
                * (
                    u_all * self.sigy**2
                    + b * v_all * self.sigx**2
                    - b * u_all * self.sigxy
                )
            )
            return b_new

        # iterate until solution is found
        iter_cnt = 0
        b_old = self._slope
        b_new = iterate_b(b_old)
        while np.abs(b_old - b_new) > self.reg_limit and iter_cnt <= self.iter_max:
            b_old = b_new
            b_new = iterate_b(b_old)
            iter_cnt += 1

        if iter_cnt == self.iter_max:
            warnings.warn(
                f"Iteration count for slope optimization hit the limt at "
                f"{self.iter_max}. The current difference between the old and new "
                f"slope is {np.abs(b_old - b_new)}"
            )

        self._slope = b_new
        self.weights = calc_weights(b_new)
        self.xbar = calc_xbar(self.weights)
        self.ybar = calc_ybar(self.weights)

    def slope_initial_guess(self):
        """Calculate an initial guess of the slope without uncertainties and save it."""
        polyfit = Polynomial.fit(self.xdat, self.ydat, deg=1)
        self._slope = polyfit.convert().coef[1]

    def unc_calculation(self):
        """Calculate uncertainties for slope and intercept with no fixed point."""
        # helper variables
        sigx = self.sigx
        sigy = self.sigy
        sigxy = self.sigxy
        b = self._slope
        weights = self.weights
        xbar = self.xbar
        ybar = self.ybar
        u_all = self.xdat - xbar
        v_all = self.ydat - ybar

        sum_weights = np.sum(weights)

        # d(\theta) / db
        dthdb = np.sum(
            weights**2
            * (
                2 * b * (u_all * v_all * sigx**2 - u_all**2 * sigxy)
                + (u_all**2 * sigy**2 - v_all**2 * sigx**2)
            )
        ) + 4 * np.sum(
            weights**3
            * (sigxy - b * sigx**2)
            * (
                b**2 * (u_all * v_all * sigx**2 - u_all**2 * sigxy)
                + b * (u_all**2 * sigy**2 - v_all**2 * sigx**2)
                - (u_all * v_all * sigy**2 - v_all**2 * sigxy)
            )
        )

        # d(\theta) / dxi
        def calc_dtheta_dxi(it: int):
            """Calculate partial derivative d(\theta)/dxi.

            :param ind: Index where the $i$ is at.
            """
            if self.fix_pt is None:
                sum_all = 0.0
                for jt, wj in enumerate(weights):
                    kron = kron_delta(it, jt)
                    sum_all += (
                        wj**2
                        * (kron - weights[it] / sum_weights)
                        * (
                            b**2 * v_all[jt] * sigx[jt] ** 2
                            - b**2 * 2 * u_all[jt] * sigxy[jt]
                            + 2 * b * u_all[jt] * sigy[jt] ** 2
                            - v_all[jt] * sigy[jt] ** 2
                        )
                    )
                return sum_all
            else:
                return weights[it] ** 2 * (
                    b**2 * v_all[it] * sigx[it] ** 2
                    - b**2 * 2 * u_all[it] * sigxy[it]
                    + 2 * b * u_all[it] * sigy[it] ** 2
                    - v_all[it] * sigy[it] ** 2
                )

        # d(\theta) / dyi
        def calc_dtheta_dyi(it: int):
            """Calculate partial derivative d(\theta)/dyi.

            :param ind: Index where the $i$ is at.
            """
            if self.fix_pt is None:
                sum_all = 0.0
                for jt, wj in enumerate(weights):
                    kron = kron_delta(it, jt)
                    sum_all += (
                        wj**2
                        * (kron - weights[it] / sum_weights)
                        * (
                            b**2 * u_all[jt] * sigx[jt] ** 2
                            - 2 * b * v_all[jt] * sigx[jt] ** 2
                            - u_all[jt] * sigy[jt] ** 2
                            + 2 * v_all[jt] * sigxy[jt]
                        )
                    )
                return sum_all
            else:
                return weights[it] ** 2 * (
                    b**2 * u_all[it] * sigx[it] ** 2
                    - 2 * b * v_all[it] * sigx[it] ** 2
                    - u_all[it] * sigy[it] ** 2
                    + 2 * v_all[it] * sigxy[it]
                )

        # da / dxi
        def calc_da_dxi(it: int):
            """Calculate partial derivative da/dxi.

            :param ind: Index where the $i$ is at.
            """
            if self.fix_pt is None:
                return (
                    -b * weights[it] / sum_weights - xbar * calc_dtheta_dxi(it) / dthdb
                )
            else:
                return -xbar * calc_dtheta_dxi(it) / dthdb

        # da / dyi
        def calc_da_dyi(it: int):
            """Calculate partial derivative da/dyi.

            :param ind: Index where the $i$ is at.
            """
            if self.fix_pt is None:
                return weights[it] / sum_weights - xbar * calc_dtheta_dyi(it) / dthdb
            else:
                return -xbar * calc_dtheta_dyi(it) / dthdb

        # calculate uncertainty for slope
        sigb_sq = 0.0
        for it, sigxi in enumerate(sigx):
            sigyi = sigy[it]
            sigxyi = sigxy[it]
            dtheta_dxi = calc_dtheta_dxi(it)
            dtheta_dyi = calc_dtheta_dyi(it)
            sigb_sq += (
                dtheta_dxi**2 * sigxi**2
                + dtheta_dyi**2 * sigyi**2
                + 2 * sigxyi * dtheta_dxi * dtheta_dyi
            )
        sigb_sq /= dthdb**2
        self._slope_unc = np.sqrt(sigb_sq)

        if self.fix_pt is None:
            siga_sq = 0.0
            for it, sigxi in enumerate(sigx):
                sigyi = sigy[it]
                sigxyi = sigxy[it]
                da_dxi = calc_da_dxi(it)
                da_dyi = calc_da_dyi(it)
                siga_sq += (
                    da_dxi**2 * sigxi**2
                    + da_dyi**2 * sigyi**2
                    + 2 * sigxyi * da_dxi * da_dyi
                )
        else:
            siga_sq = self.fix_pt[0] ** 2 * sigb_sq
        self._intercept_unc = np.sqrt(siga_sq)
