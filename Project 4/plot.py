import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

textfile_prefix = "txtfiles/"
image_prefix = "images/"

class IsingSamples:
    def __init__(self, filename, burn_in_filename, plotColor):
        self.filename = textfile_prefix + filename
        self.plotColor = plotColor

        self.N, self.MC_cycles = np.loadtxt(self.filename, max_rows = 1)
        self.L = int(np.sqrt(self.N))

        self.burn_in_filename = textfile_prefix + burn_in_filename
        self.load_burn_in_data()

        self.temperatures = np.loadtxt(self.filename, skiprows = 1, max_rows = 1)
        self.beta = 1/self.temperatures

        self.data = np.loadtxt(self.filename, skiprows = 2)

        # The expectation values of E, E^2, |M| and M^2 from samples as functions of temperature
        self.E, self.E_square, self.M, self.M_square = self.data

        self.eps = self.E/self.N
        self.m = self.M/self.N

        self.eps_square = self.E_square/self.N**2
        self.m_square = self.M_square/self.N**2

        # Expectation value of C_V from samples
        self.C_V = 1/self.N*1/self.temperatures**2*(self.E_square - self.E**2)

        # Expectation value of chi from samples
        self.chi = 1/self.N*1/self.temperatures*(self.M_square - self.M**2)


        self.quantities = {"E": self.E, "M": self.M, "E2": self.E_square, "M2": self.M_square,
                           "eps": self.eps, "m": self.m, "eps2": self.eps_square, "m2": self.m_square,
                           "C_V": self.C_V, "chi": self.chi}

        self.quantity_name_to_label = {"E": "$E/J$", "M": "$M$", "E2": "$E^2/J^2$", "M2": "$M^2$",
                                       "eps": r"$\langle\epsilon\rangle/J$", "m": r"$\langle|m|\rangle$",
                                       "eps2": "$\langle\epsilon^2\rangle/J^2$", "m2": "$\langle m^2\rangle$",
                                       "C_V": "$C_V/k_B$", "chi": "$\chi$"}

        if self.N == 4:
            self.analytical()

    def load_burn_in_data(self):
        """
        This function loads the data from the simulations where we wrote the energy and magnetization from every
        Monte Carlo cycle into a file.
        """
        _, self.burn_in_T = np.loadtxt(self.burn_in_filename, max_rows = 1)
        self.burn_in_data = np.loadtxt(self.burn_in_filename, skiprows = 1)

        # Values for E and M after every cycle
        self.burn_in_E, self.burn_in_M = self.burn_in_data

        self.burn_in_eps, self.burn_in_m = self.burn_in_E/self.N, self.burn_in_M/self.N

        self.n_cycles = np.arange(1, len(self.burn_in_E) + 1, 1)

        # The expectation values for epsilon and m after every cycle
        self.eps_expect = np.cumsum(self.burn_in_eps)/self.n_cycles
        self.m_expect = np.cumsum(self.burn_in_m)/self.n_cycles

        self.burn_in_quantities = {"eps": self.eps_expect, "m" : self.m_expect,
                                   "burn_in_eps": self.burn_in_eps,
                                   "burn_in_m": self.burn_in_m}

    def analytical(self):
        """
        Only called for N = 4, for which we have analytical results.
        """
        self.Z = 2*np.exp(8*self.beta) + 2*np.exp(-8*self.beta) + 12

        self.analytical_E = -16/self.Z*(np.exp(8*self.beta) - np.exp(-8*self.beta))
        self.analytical_E_square = 128/self.Z*(np.exp(8*self.beta) + np.exp(-8*self.beta))

        self.analytical_M = 8/self.Z*(np.exp(8*self.beta) + 2)
        self.analytical_M_square = 32/self.Z*(np.exp(8*self.beta) + 1)

        self.analytical_eps = self.analytical_E/self.N
        self.analytical_m = self.analytical_M/self.N

        self.analytical_C_V = 1/self.N*1/self.temperatures**2*(self.analytical_E_square - self.analytical_E**2)

        self.analytical_chi = 1/self.N*1/self.temperatures*(self.analytical_M_square - self.analytical_M**2)

        self.analytical_quantities = {"E": self.analytical_E, "E2": self.analytical_E_square,
                                      "M": self.analytical_M, "M2": self.analytical_M_square,
                                      "eps": self.analytical_eps, "m": self.analytical_m,
                                      "C_V": self.analytical_C_V, "chi": self.analytical_chi}

    def plot(self, quantity_name, ax, smooth = False):
        """
        Plots a quantity indicated by quantity_name as a function of temperature.
        smooth is only used when smoothing C_V using the Savitzky-Golay filter
        """

        quantity = self.quantities[quantity_name]

        if smooth:
            quantity = savgol_filter(quantity, 51, 4)

        # For the 2x2 lattice we want the lables to be the number of MC cycles used.
        # Otherwise we want L for the label.

        if self.N == 4:
            label = r"$10^{:.0f}$ MC cycles".format(np.log10(self.MC_cycles))
        else:
            label = "$L = {}$".format(self.L)

        ax.plot(self.temperatures, quantity, self.plotColor, label = r"{}".format(label))

        # Only have the analytical solution for N = 4 and only want to plot it once
        if self.N == 4 and self.MC_cycles == 10000:
            ax.plot(self.temperatures, self.analytical_quantities[quantity_name], "k--", label = "Analytical")

        return quantity

    def plot_burn_in(self, quantity_name, ax):
        """
        Plots the quantity indicated by quantity_name as a function of # MC cycles
        """
        quantity = self.burn_in_quantities[quantity_name]

        # When L = 20 we want to plot the burn-in times for ordered and unordered
        # initial configurations together. So labels should "Ordered" and "Unordered".
        if self.N == 400:
            # Just takes the label "Ordered" or "Unordered" from the filename.
            # Probably the most beautiful line of code you'll ever see.
            label = self.burn_in_filename.replace("txtfiles/burn_in20x20", "").replace(".txt", "")[5:].title()
        else:
            label = r"$L = {}$".format(self.L)

        ax.plot(self.n_cycles, quantity, self.plotColor, label = r"{}".format(label))

    def plot_histogram(self, quantity_name, n_bins, burn_in_time, figName):
        """
        Plots a histogram for the quantity indicated by quantity_name.
        burn_in_time indicates how many of the initial values should be ignored.
        """
        prob_quantities = {"eps": "$p(\epsilon; T)$ [$k_B/J$]", "m": "$p(m; T)$"}
        quant = {"eps": "$\epsilon$ [$J/k_B$]", "m": "$m$"}
        fig, ax = plt.subplots()

        quantity = self.burn_in_quantities["burn_in_" + quantity_name]

        variance = np.var(quantity)
        print("Variance for T = {}: {}".format(self.burn_in_T, variance))

        ax.hist(quantity[burn_in_time: ], bins = n_bins, density = True)
        xlabel = quant[quantity_name]
        ylabel = prob_quantities[quantity_name]
        ax.set_xlabel(r"{}".format(xlabel), fontsize = 12)
        ax.set_ylabel(r"{}".format(ylabel), fontsize = 12)
        ax.tick_params("both", labelsize = 12)

        fig.savefig(image_prefix + figName, bbox_inches = "tight")
        plt.close()

def plotQuantity(quantity_name, isingSamples, figName = "", smooth = False, findMaxima = False):
    """
    Plots a quantity for all sets of samples in isingSamples as functions of temperature.
    """
    fig, ax = plt.subplots(figsize = (5, 4))
    ax.grid()

    if findMaxima:
        maxima = np.zeros(len(isingSamples))

    for i, isingSample in enumerate(isingSamples):
        quantity = isingSample.plot(quantity_name, ax, smooth = smooth)

        if findMaxima:
            max_index = np.where(quantity - np.max(quantity) == 0)[0][0]
            maxima[i] = isingSample.temperatures[max_index]

    ylabel = isingSamples[0].quantity_name_to_label[quantity_name]
    ax.set_xlabel(r"$T$ [$J/k_B$]", fontsize = 12)
    ax.set_ylabel(r"{}".format(ylabel), fontsize = 12)
    ax.tick_params("both", labelsize = 12)
    ax.legend(fontsize = 12)

    if figName == "":
        fig.savefig(image_prefix + quantity_name + ".pdf", bbox_inches = "tight")
    else:
        fig.savefig(image_prefix + figName, bbox_inches = "tight")

    if findMaxima:
        return maxima

    plt.close()

def plotQuantityBurnIn(quantity_name, isingSamples, figName = ""):
    """
    Plots a quantity for all sets of samples in isingSamples as functions of # MC cycles
    """
    fig, ax = plt.subplots(figsize = (5, 4))
    ax.grid()

    for isingSample in isingSamples:
        isingSample.plot_burn_in(quantity_name, ax)

    ylabel = isingSamples[0].quantity_name_to_label[quantity_name]
    ax.set_xlabel("# Monte Carlo cycles", fontsize = 12)
    ax.set_ylabel(r"{}".format(ylabel), fontsize = 12)
    ax.tick_params("both", labelsize = 12)

    if len(isingSamples) > 1:
        ax.legend(fontsize = 12)

    if figName == "":
        fig.savefig(image_prefix + quantity_name + ".pdf", bbox_inches = "tight")
    else:
        fig.savefig(image_prefix + figName, bbox_inches = "tight")

    plt.close()

"""
Plotting the approximated expectation values versus the analytical ones for
100, 1000 and 10000 MC cycles for the 2x2 lattice.
"""

ising2_100 = IsingSamples("2x2_100.txt", "burn_in2x2_2_4_unordered.txt", "b-")
ising2_1000 = IsingSamples("2x2_1000.txt", "burn_in2x2_2_4_unordered.txt", "g-")
ising2_10000 = IsingSamples("2x2_10000.txt", "burn_in2x2_2_4_unordered.txt", "r-")
isingSamples = [ising2_100, ising2_1000, ising2_10000]

plotQuantity("eps", isingSamples, figName = "eps_2x2_comparison.pdf")
plotQuantity("m", isingSamples, figName = "m_2x2_comparison.pdf")
plotQuantity("C_V", isingSamples, figName = "C_V_2x2_comparison.pdf")
plotQuantity("chi", isingSamples, figName = "chi_2x2_comparison.pdf")

"""
Plotting the expectation value for epsilon and m as functions of the number of
MC cycles performed for the 2x2 lattice. This is only done for the case where we
use 10000 MC cycles in total. Used to find burn-in time.
"""

fig, ax = plt.subplots()

plotQuantityBurnIn("eps", [ising2_10000], figName = "burn_in2x2_10000_eps.pdf")
plotQuantityBurnIn("m", [ising2_10000], figName = "burn_in2x2_10000_m.pdf")

"""
The same burn-in stuff as above, but for 20x20. And here we use T = 1.0 and T = 2.4,
as well as trying both an unordered and ordered initial configuration.
"""

# Ordered and unordered for T = 1.0
ising20_1_0_ordered = IsingSamples("20x20_1000.txt", "burn_in20x20_1_0_ordered.txt", "b-")
ising20_1_0_unordered = IsingSamples("20x20_1000.txt", "burn_in20x20_1_0_unordered.txt", "r-")

# ---------||-------------- T = 2.4
ising20_2_4_ordered = IsingSamples("20x20_1000.txt", "burn_in20x20_2_4_ordered.txt", "b-")
ising20_2_4_unordered = IsingSamples("20x20_1000.txt", "burn_in20x20_2_4_unordered.txt", "r-")

plotQuantityBurnIn("eps", [ising20_1_0_unordered, ising20_1_0_ordered], "burn_in20_1_0_eps.pdf")
plotQuantityBurnIn("eps", [ising20_2_4_unordered, ising20_2_4_ordered], "burn_in20_2_4_eps.pdf")

plotQuantityBurnIn("m", [ising20_1_0_unordered, ising20_1_0_ordered], "burn_in20_1_0_m.pdf")
plotQuantityBurnIn("m", [ising20_2_4_unordered, ising20_2_4_ordered], "burn_in20_2_4_m.pdf")

"""
Plotting histograms for the distribution of eps for the 20x20 lattice for both
T = 1.0 and T = 2.4. We do this using the results from the ordered initial config.
"""

ising20_1_0_unordered.plot_histogram("eps", 150, 5000, "eps_hist_20_1_0_unordered.pdf")
ising20_2_4_unordered.plot_histogram("eps", 100, 40000, "eps_hist_20_2_4_unordered.pdf")

"""
Plotting all quantities for the 40x40, 60x60, 80x80 and 100x100 runs.
"""

ising100 = IsingSamples("100x100_1000000.txt", "burn_in100x100_2_4_unordered.txt", "b-")
ising80 = IsingSamples("80x80_1000000.txt", "burn_in80x80_2_4_unordered.txt", "g-")
ising60 = IsingSamples("60x60_1000000.txt", "burn_in60x60_2_4_unordered.txt", "r-")
ising40 = IsingSamples("40x40_1000000.txt", "burn_in40x40_2_4_unordered.txt", "y-")

isingSamples = [ising40, ising60, ising80, ising100]

plotQuantity("eps", isingSamples)
plotQuantity("m", isingSamples)
plotQuantity("C_V", isingSamples)
plotQuantity("chi", isingSamples)

# Smoothing the C_V - curves and fitting the T_C's with aL^(-1) + b
maxima_temps = plotQuantity("C_V", isingSamples, figName = "C_V_smoothed.pdf", smooth = True, findMaxima = True)
print(maxima_temps)
L_values = np.array([isingSample.L for isingSample in isingSamples])

f = lambda L, a, b: a/L + b

params, _ = curve_fit(f, L_values, maxima_temps)
a, b = params

fig, ax = plt.subplots(figsize = (5, 4))
ax.plot(L_values, maxima_temps, "b-o", label = "From MCMC runs")
higher_res_L = np.linspace(40, 100, 100)
ax.plot(higher_res_L, f(higher_res_L, a, b), "k--", label = r"Fitted: {:.3f}/L + {:.3f}".format(a, b))
ax.grid()
ax.set_xlabel(r"$L$", fontsize = 12)
ax.set_ylabel(r"$T_C$", fontsize = 12)
ax.legend(fontsize = 12)
ax.tick_params("both", labelsize = 12)
fig.savefig(image_prefix + "T_C.pdf", bbox_inches = "tight")
plt.close()

plotQuantityBurnIn("m", isingSamples, "burn_in_m_lattices.pdf")
plotQuantityBurnIn("eps", isingSamples, "burn_in_eps_lattices.pdf")
