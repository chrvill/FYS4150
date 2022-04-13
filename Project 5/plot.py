import numpy as np
import matplotlib.pyplot as plt
import pyarma as pa
from matplotlib import animation
import matplotlib

dt = 2.5e-5
h = 0.005

textfile_prefix = "txtfiles/"
image_prefix = "images/"
anim_prefix = "animations/"

class Data:
    def __init__(self, filename):
        self.filename = filename
        self.U = pa.cx_cube()

        path = textfile_prefix + filename
        self.U.load(path, pa.arma_binary)

        U_conj = pa.conj(self.U)    # Complex conjugate of self.U

        # Swapping axes 1 and 2 because the x- and y-axes are swapped in the U-matrix
        self.prob = np.swapaxes(pa.real(self.U @ U_conj), axis1 = 1, axis2 = 2)

        self.initialize_animation()

    def initialize_animation(self):
        """
        Creates the figure and everything needed to generate the animation later.
        """
        fontsize = 15
        self.fig = plt.figure()
        self.ax = plt.gca()

        #norm = matplotlib.cm.colors.Normalize(vmin = 0.0, vmax = np.max(np.sqrt(self.prob[pa.single_slice, 0])))
        norm = matplotlib.cm.colors.Normalize(vmin = 0.0, vmax = np.max(np.sqrt(self.prob[0])))

        # Plot the first frame
        self.img = self.ax.imshow(np.sqrt(self.prob[0]), extent = [0, 1, 0, 1], cmap = plt.get_cmap("viridis"), norm = norm)

        # Axis labels
        self.ax.set_xlabel(r"$x$", fontsize = fontsize)
        self.ax.set_ylabel(r"$y$", fontsize = fontsize)
        self.ax.tick_params("both", labelsize = fontsize)

        # Add a colourbar
        cbar = self.fig.colorbar(self.img, ax = self.ax)
        cbar.set_label(r"$\sqrt{p(x, y; t)}$", fontsize = fontsize)
        cbar.ax.tick_params(labelsize = fontsize)

        self.time_txt = self.ax.text(0.95, 0.95, "t = {:.3e}".format(0), color = "white",
                                horizontalalignment = "right", verticalalignment = "top", fontsize = fontsize)

    def plot_error(self):
        """
        Plots the error in the total probability as a function of
        simulation time.
        """
        fontsize = 15
        fig, ax = plt.subplots()
        error = [np.sum(self.prob[i]) - 1 for i in range(len(self.prob))]
        timesteps = [dt*i for i in range(len(self.prob))]

        ax.plot(timesteps, error, "b-")
        ax.set_xlabel(r"$t$", fontsize = fontsize)
        ax.set_ylabel(r"$\sum_{ij} p_{ij} - 1$", fontsize = fontsize)
        ax.tick_params("both", labelsize = fontsize)
        fig.savefig(image_prefix + self.filename.replace(".txt", "_error.pdf"), bbox_inches = "tight")

    def animate(self, i):
        """
        The function that is called by FuncAnimation to
        generate an image for each timestep.

        i is the index for which to plot the colormap
        The function returns the image.
        """
        norm = matplotlib.cm.colors.Normalize(vmin = 0.0, vmax = np.max(np.sqrt(self.prob[i])))
        self.img.set_norm(norm)

        # Update z data
        self.img.set_data(np.sqrt(self.prob[i]))

        # Update the time label
        current_time = i * dt
        self.time_txt.set_text("t = {:.3e}".format(current_time))

        return self.img

    def generate_animation(self):
        """
        Calls all the necessary stuff to actually generate the animation.
        """
        writergif = animation.PillowWriter(fps = 30)

        anim = animation.FuncAnimation(self.fig, self.animate, interval = 10, frames = np.arange(0, len(self.prob), 2),
                                       repeat = False, blit = 0)

        gif_name = anim_prefix + self.filename.replace(".txt", ".gif")
        anim.save(gif_name, writer = writergif)
        plt.close()

    def generate_colormap(self, t, fig, ax, type = "prob"):
        """
        Creates a colormap of the desired quantity. type specifies
        which quantity should be plotted. And t specifies the time for
        which to plot the colormap.

        fig and ax are given to be able to plot different colormaps in the same
        figure.
        """
        fontsize = 16
        t_index = int(t/dt)

        if type == "prob":
            z = np.sqrt(self.prob)
            label = r"$\sqrt{p(x, y; t)}$"
        elif type == "real":
            z = np.swapaxes(pa.real(self.U), axis1 = 1, axis2 = 2)
            label = r"Re$\{u(x, y; t)\}$"
        elif type == "imag":
            z = np.swapaxes(pa.imag(self.U), axis1 = 1, axis2 = 2)
            label = r"Im$\{u(x, y; t)\}$"

        norm = matplotlib.cm.colors.Normalize(vmin = 0.0, vmax = np.max(z[t_index]))

        img = ax.imshow(z[t_index], extent = [0, 1, 0, 1], cmap = plt.get_cmap("viridis"), norm = norm)

        cbar = fig.colorbar(img, ax = ax)
        cbar.set_label(label, fontsize = 11)
        cbar.ax.tick_params(labelsize = 11)
        plt.close()

    def wall_pattern(self, color_linestyle):
        """
        Plots the probability distribution along the wall.
        color_linestyle is a string which specifies what
        color and linestyle to use for the curve
        """
        p_xy = np.copy(self.prob[-1])
        x = np.arange(h, 1, h)
        i = np.argmin(np.abs(x - 0.8))

        y = np.copy(x)

        p_y = p_xy[:, i]
        p_y /= np.sum(p_y)

        fig, ax = plt.subplots()
        ax.plot(y, p_y, color_style)
        ax.set_xlabel(r"$y$", fontsize = 15)
        ax.set_ylabel(r"$p(y|x = 0.8; t = 0.002)$", fontsize = 15)
        ax.tick_params("both", labelsize = 15)
        fig.savefig(image_prefix + self.filename.replace(".txt", ".pdf"), bbox_inches = "tight")
        plt.close()

no_slit = Data("no_slit.txt")
no_slit.generate_animation()
no_slit.plot_error()

double_slit_initial = Data("double_slit_0_008.txt")
double_slit_initial.generate_animation()
double_slit_initial.plot_error()

double_slit = Data("double_slit_0_002.txt")
double_slit.generate_animation()
double_slit.wall_pattern("b-")

single_slit = Data("single_slit.txt")
single_slit.generate_animation()
single_slit.wall_pattern("r-")

triple_slit = Data("triple_slit.txt")
triple_slit.generate_animation()
triple_slit.wall_pattern("g-")

triple_slit_smaller = Data("triple_slit_smaller_slits.txt")
triple_slit_smaller.wall_pattern("g-")

times = [0, 0.001, 0.002]

fig, ax = plt.subplots(3, 1, figsize = (8, 10))
ax = ax.flatten()

# Creating the colormaps for the three desired times.

for i, t in enumerate(times):
    double_slit.generate_colormap(t, fig, ax[i])

    fig.tight_layout()
    fig.savefig(image_prefix + "colormap_prob.pdf", bbox_inches = "tight")

fig, ax = plt.subplots(3, 1, figsize = (8, 10))
ax = ax.flatten()

for i, t in enumerate(times):
    double_slit.generate_colormap(t, fig, ax[i], "real")

    fig.tight_layout()
    fig.savefig(image_prefix + "colormap_real.pdf", bbox_inches = "tight")

fig, ax = plt.subplots(3, 1, figsize = (8, 10))
ax = ax.flatten()

for i, t in enumerate(times):
    double_slit.generate_colormap(t, fig, ax[i], "imag")

    fig.tight_layout()
    fig.savefig(image_prefix + "colormap_imag.pdf", bbox_inches = "tight")
