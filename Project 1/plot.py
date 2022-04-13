import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
This program does the plotting for all the problems
"""

def plot(x_arrays, y_arrays, labels, plotName, xlabel = "", ylabel = "", xlog = False, ylog = False,
         newFig = True, figAx = [], linePlot = True, saveFig = True):
    """
    General function for plotting

    x_arrays - 2d list containing x-values for different curves
    y_arrays - 2d list contaning y-values for the same curves
    labels   - label for each curve
    plotName - filename when saving plot
    xlabel   - label along x-axis
    ylabel   - label along y-axis
    xlog     - whether to plot x in log-scale or not
    ylog     - ------||------- y ------||-----------
    newFig   - whether we are creating a new figure
    figAx    - contains fig and ax if we are using an already existing figure
    linePlot - whether to plot lines or points
    saveFig  - whether to save figure now or not
    """
    colors = sns.color_palette("bright")
    if newFig:
        # If newFig is True, create new fig and ax objects
        fig, ax = plt.subplots()

    else:
        # If not, use the fig and ax given as input parameters
        fig, ax = figAx

    for i in range(len(x_arrays)):
        x, y = x_arrays[i], y_arrays[i]
        # x and y for curve number i

        if linePlot:
            ax.plot(x, y, color = colors[i], label = labels[i])
        else:
            ax.plot(x, y, "bo")

    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    if linePlot:
        ax.legend()
    if saveFig:
        fig.savefig(plotName)

def analytical_u(x):
    return 1 - (1 - np.exp(-10))*x - np.exp(-10*x)

def plotFile(numerical_files, plotName, analytical_file = "analytical.txt"):
    # Loading the analytical data from files
    x_analytical, u = np.loadtxt(analytical_file, usecols = (0, 1), unpack = True)

    x_numerical = []
    v_numerical = []

    for filename in numerical_files:
        x, v = np.loadtxt(filename, usecols = (0, 1), unpack = True)
        x_numerical.append(x)
        v_numerical.append(v)

    fig, axes = plt.subplots(2, 2, figsize = (8, 8))
    axes = axes.flatten()
    labels = ["Numerical", "Analytical"]

    for i, ax in enumerate(axes):
        plot([x_numerical[i], x_analytical], [v_numerical[i], u], labels, "v_solution.pdf", r"$x$", r"$v(x)$",
             figAx = [fig, ax], newFig = False, saveFig = False)

        ax.set_title("n = {}".format(len(x_numerical[i])))

    fig.subplots_adjust(hspace = 0.5)
    fig.savefig(plotName)

def plotErrors(filenames):
    """
    filenames is a list of the filenames of the solutions for different n
    """
    max_epsilons = []
    n_points = []

    x_arrays = []
    abs_errors = []
    epsilons = []

    for i, filename in enumerate(filenames):
        x, v = np.loadtxt(filename, usecols = (0, 1), unpack = True)

        # Excluding the first and last points here because these are given by the
        # boundary conditions, so the errors at these points will always be zero
        x = x[1: -1]
        v = v[1: -1]

        u = analytical_u(x)

        # Number of grid points (+2 because I earlier excluded the endpoints)
        n = len(x) + 2
        n_points.append(n)

        abs_error = np.abs(u - v)
        epsilon = np.abs(abs_error/u)

        x_arrays.append(x)
        abs_errors.append(abs_error)
        epsilons.append(epsilon)

        max_epsilons.append(np.max(epsilon))

    labels = ["n = {}".format(n) for n in n_points]
    plot(x_arrays[:4], abs_errors[:4], labels[:4], "absErrors.pdf", xlabel = r"$x$", ylabel = r"$\Delta_i$", ylog = True)
    plot(x_arrays[:4], epsilons[:4], labels[:4], "relErrors.pdf", xlabel = r"$x$", ylabel = r"$\epsilon_i$", ylog = True)

    return n_points, max_epsilons

def timing(generalFile, specialFile):
    timeGeneral = np.loadtxt(generalFile)
    timeSpecial = np.loadtxt(specialFile)

    n = timeGeneral[0]

    avgGeneral = np.mean(timeGeneral[1:], axis = 0)
    avgSpecial = np.mean(timeSpecial[1:], axis = 0)

    labels = ["General algorithm", "Special algorithm"]
    plot([n, n], [avgGeneral, avgSpecial], labels, "timing.pdf", xlabel = r"$n$", ylabel = "Run time (s)",
         xlog = True, ylog = True)

# Problem 2
x, u = np.loadtxt("analytical.txt", usecols = (0, 1), unpack = True)
plot([x], [u], ["Analytical solution"], "analyticalSolution.pdf", r"$x$", r"$u(x)$")

# Problem 7 and 9
n_points = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
generalFiles = ["v_solution{}.txt".format(n) for n in n_points]
specialFiles = ["v_solution{}_special.txt".format(n) for n in n_points]

# Don't want to plot for n = 100 000, 1 000 000, 10 000 000, hence the [:4]
plotFile(generalFiles[:4], "v_solution.pdf")
plotFile(specialFiles[:4], "v_solution_special.pdf")

# Problem 8
n_points, max_epsilons = plotErrors(generalFiles)
print("n \t\t\t epsilon")
for n, epsilon in zip(n_points, max_epsilons):
    print("{} \t\t\t {:.2e}".format(n, epsilon))

plot([n_points], [max_epsilons], [], "max_epsilons.pdf", xlabel = r"$n$", ylabel = r"max$(\epsilon_i)$",
     xlog = True, ylog = True, linePlot = False)

# Problem 10
timing("timeGeneral.txt", "timeSpecial.txt")
