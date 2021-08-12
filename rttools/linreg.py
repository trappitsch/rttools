"""Linear fits with uncertainties in both axes according to Mahon et al. (1996)"""

import numpy as np

from .utils import kron_delta


class Mahon:
    def __init__(self):
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

        # if correlation is zero, then set p_corr accordingly
        if p_corr is None:
            self.p = np.zeros(len(xdat))

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
                    wi ** 2.0
                    * vi
                    * (ui * syi ** 2.0 + bcalc * vi * sxi ** 2.0 - pit * vi * sxi * syi)
                )
                bbot += (
                    wi ** 2.0
                    * ui
                    * (
                        ui * syi ** 2.0
                        + bcalc * vi * sxi ** 2.0
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
            sum1 += wi ** 2.0 * (
                2 * b * (ui * vi * sxi ** 2.0 - ui ** 2.0 * sxyi)
                + (ui ** 2.0 * syi ** 2.0 - vi ** 2 * sxi ** 2.0)
            )
            sum2 += (
                wi ** 3.0
                * (sxyi - b * sxi ** 2.0)
                * (
                    b ** 2.0 * (ui * vi * sxi ** 2 - ui ** 2 * sxyi)
                    + b * (ui ** 2 * syi ** 2 - vi ** 2 * sxi ** 2)
                    - (ui * vi * syi ** 2 - vi ** 2 * sxyi)
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
                    wj ** 2.0
                    * (kron_delta(it, jt) - wi / wksum)
                    * (
                        b ** 2 * (vj * sxj ** 2 - 2 * uj * sxyj)
                        + 2 * b * uj * syj ** 2
                        - vj * syj ** 2
                    )
                )
                # correct equation! not equal to equation 21 in Mahon (1996)
                dthdyi += (
                    wj ** 2.0
                    * (kron_delta(it, jt) - wi / wksum)
                    * (
                        b ** 2 * uj * sxj ** 2
                        + 2 * vj * sxyj
                        - 2 * b * vj * sxj ** 2.0
                        - uj * syj ** 2
                    )
                )

            # now calculate dell a / dell xi and dell a / dell yi
            dadxi = -b * wi / wksum - xbar * dthdxi / dthdb
            dadyi = wi / wksum - xbar * dthdyi / dthdb

            # now finally add to sigasq and sigbsq
            sigbsq += (
                dthdxi ** 2.0 * sxi ** 2.0
                + dthdyi ** 2.0 * syi ** 2.0
                + 2 * sxyi * dthdxi * dthdyi
            )
            sigasq += (
                dadxi ** 2.0 * sxi ** 2.0
                + dadyi ** 2.0 * syi ** 2.0
                + 2 * sxyi * dadxi * dadyi
            )

        # now divide sigbsq
        sigbsq /= dthdb ** 2.0

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
        return 1.0 / (sy ** 2 + b ** 2 * sx ** 2 - 2 * b * p * sx * sy)
