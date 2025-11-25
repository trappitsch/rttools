"""
Model reader for Szanyiet al. (2025) models.

The models are available to download from here: https://zenodo.org/records/14981333
Set 2 contains the latest nuclear reaction rates.
"""

from pathlib import Path

from iniabu import ini
import matplotlib.pyplot as plt
import pandas as pd

MODEL_DIR = Path(__file__).parent.joinpath("data/szanyi25models")


def get_closest_model(mass: float, metallicity: float, rate_set: int = 2) -> Path:
    """Get the closest model file for a given mass, metallicity, and set.

    :param mass: Stellar mass in solar masses.
    :param metallicity: Stellar metallicity.
    :param rate_set: Model set number (0, 1, or 2), defaults to 2 (if not given or invalid).

    :return: Path to the closest model file.
    """
    if rate_set not in [0, 1, 2]:
        rate_set = 2

    if mass < 2.5:
        mass_str = "m2"
    elif mass < 3.5:
        mass_str = "m3"
    else:
        mass_str = "m4"

    if mass_str == "m2":
        metall_str = "z014"  # only one available
    else:  # now we have 0.007, 0.014, 0.03
        if metallicity < 0.0105:
            metall_str = "z007"
        elif metallicity < 0.022:
            metall_str = "z014"
        else:
            metall_str = "z030"

    fname = MODEL_DIR.joinpath(f"srf_{mass_str}{metall_str}_set{rate_set}.dat")
    assert fname.exists(), f"Model file {fname} does not exist."

    return fname


class Szanyi25Reader:
    """Reader for Szanyi et al. (2025) nucleosynthesis models.

    Load a given mass, metallicity, and rate set model and provide methods to access the data.
    This routine will load the closest available model. Please see the example on how to check which model was loaded. 

    Example usage:
    ```python
    from rttools.nucmod.szanyi25 import Szanyi25Reader

    model = Szanyi25Reader(mass=3.0, metallicity=0.014, rate_set=2)

    # print mass, metallicity and rate set of the loaded model
    print(f"Loaded model: mass={model.mass}, z={model.z}, rate_set={model.rate_set}")
    ```
    """

    def __init__(self, mass: float, metallicity: float, rate_set: int = 2):
        """Initialize the reader with the closest model file.

        :param mass: Stellar mass in solar masses.
        :param metallicity: Stellar metallicity.
        :param rate_set: Model set number (0, 1, or 2), defaults to 2 (if not given or invalid).
        """
        self._model_file = get_closest_model(mass, metallicity, rate_set)

        self._mass = None
        self._y = None
        self._z = None
        self._rate_set = None
        self._data = None

        self._tp_cols = None

        self._read_model()

    def _read_model(self):
        """Read in the model data from the file and fill out
        - _mass: Stellar mass (float)
        - _y: Helium mass fraction (float)
        - _z: Metallicity (float)
        - _rate_set: Rate set number (int)
        - _data: Nucleosynthesis data (pandas DataFrame)
        """
        # Read in the first line and parse mass, z, y, and set
        with open(self._model_file, "r") as f:
            hdr = f.readline().strip()

        hdr = hdr.split(",")
        self._mass = float(hdr[0].split("=")[1].strip())
        self._z = float(hdr[1].split("=")[1].strip())
        self._y = float(hdr[2].split("=")[1].strip())
        self._rate_set = float(hdr[3].strip().split(" ")[1].strip())

        # read the data into a pandas DataFrame
        self._data = pd.read_csv(
            self._model_file, skiprows=2, delimiter="\t", index_col=0
        )

        self._tp_cols = self._data.columns[4:]

    def get_co_ratio(self) -> pd.Series:
        """Get the C/O ratio for all thermal pulses (as number ratio)."""
        c12 = self._data.loc["c12"][self._tp_cols]
        c13 = self._data.loc["c13"][self._tp_cols]
        o16 = self._data.loc["o16"][self._tp_cols]
        o17 = self._data.loc["o17"][self._tp_cols]
        o18 = self._data.loc["o18"][self._tp_cols]

        c_total = c12 + c13
        o_total = o16 + o17 + o18

        co_ratio = (c_total / ini.ele["C"].mass) / (o_total / ini.ele["O"].mass)
        return co_ratio

    def get_delta(self, nominator: str, denominator: str) -> pd.Series:
        """Get the delta value for the two isotopes across all thermal pulses."""
        nomin = ini.iso[nominator]
        denom = ini.iso[denominator]

        ind_nomin = "".join(nomin.name.split("-")).lower()
        ind_denom = "".join(denom.name.split("-")).lower()

        ratio = (self._data.loc[ind_nomin][self._tp_cols] / self._data.loc[ind_denom][self._tp_cols])

        delta = ini.iso_delta(nominator, denominator, ratio, mass_fraction=True)
        return delta
    def plot_mod(
        self,
        ax: plt.Axes,
        xisos: [str, str],
        yisos: [str, str],
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
        :param xisos: Two-element list of isotopes for x-axis (nominator, denominator).
        :param yisos: Two-element list of isotopes for y-axis (nominator, denominator).
        :param marker: Marker of the plot, defaults to 'o'
        :param color: Color of the plot, defaults to 'tab:blue'
        :param linestyle: Linestyle of the plot, defaults to '-'

        :return: Nothing
        """
        xdat_all = self.get_delta(*xisos)
        ydat_all = self.get_delta(*yisos)
        co_ratio = self.get_co_ratio()

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

    @property
    def mass(self) -> float:
        """Stellar mass in solar masses."""
        return self._mass

    @property
    def y(self) -> float:
        """Helium mass fraction."""
        return self._y

    @property
    def z(self) -> float:
        """Metallicity."""
        return self._z

    @property
    def rate_set(self) -> int:
        """Nuclear reaction rate set number."""
        return self._rate_set
