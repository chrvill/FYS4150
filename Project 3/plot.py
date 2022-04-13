import numpy as np
import matplotlib.pyplot as plt

# Calculating coefficients and frequencies for the analytical solution
q, m, B0, V0, d = np.loadtxt("txt/constants.txt")
omega_0 = q*B0/m
omega_z2 = 2*q*V0/(m*d**2)

omega_plus = 1/2*(omega_0 + np.sqrt(omega_0**2 - 2*omega_z2))
omega_minus = 1/2*(omega_0 - np.sqrt(omega_0**2 - 2*omega_z2))

print("omega_plus = {}".format(omega_plus))
print("omega_minus = {}".format(omega_minus))
print("omega_z = {}".format(np.sqrt(omega_z2)))

print((omega_minus + omega_plus + np.sqrt(omega_z2))/0.44)

A_minus = -(1 + omega_plus)/(omega_minus - omega_plus)
A_plus = (1 + omega_minus)/(omega_minus - omega_plus)

def readfile(filename, skiprows = 0):
    """
    Reads a data file and returns its contents.
    skiprows is the number of rows to skip at the beginning
    of the data file. Is used for files where t_max and dt are
    written at the top.
    """
    data0 = np.loadtxt(filename, skiprows = skiprows)

    # Number of timesteps
    n = int(len(data0)/3)

    # len(data[0]) doesn't work if data is one-dimensional
    n_particles = 1 if data0.ndim == 1 else len(data0[0])

    data = np.zeros((n_particles, n, 3))

    # When writing out arma::mat to file in C++ it doesn't
    # add any brackets to separate matrices. So np.loadtxt() doesn't
    # know how to separate positions (or velocities) from different timesteps.
    # So this ugly code does that manually.
    for j in range(n_particles):
        for i in range(n):
            if data0.ndim == 1:
                data[j, i] = data0[3*i: 3*i + 3]
            else:
                data[j, i] = data0[:, j][3*i:3*i + 3]

    if skiprows == 0:
        return data
    else:
        dt, t_max = np.loadtxt(filename, max_rows = 1)
        return data, dt, t_max

"""
Class for reading and storing the data from simulations. It reads in a
file containing the positions of all particles at all timesteps,
and possible a file for the velocity of all particles at all timesteps
one_particle is a parameter just telling whether the simulation contains
one or more particles. It is used for checking whether to plot an analytical sol.
"""
class SimulationData:
    def __init__(self, position_filename, velocity_filename = None, one_particle = False):
        self.position_filename = position_filename

        self.positions, self.dt, self.t_max = readfile(position_filename, skiprows = 1)
        self.x, self.y, self.z = np.transpose(self.positions)

        self.t = np.linspace(0, self.t_max, len(self.x[:, 0]))

        if velocity_filename != None:
            self.velocities = readfile(velocity_filename)
            self.vx, self.vy, self.vz = np.transpose(self.velocities)

        # Just for labels on the plots
        self.micrometer = r"$\mu$m"
        self.microsecond = r"$\mu$s"
        self.micro_m_per_s = r"$\mu$m/$\mu$s"

        self.one_particle = one_particle
        # Errors can only be calculated when analytical sol. exists.
        if self.one_particle:
            self.compute_errors()

    def plotMotion2d(self, particle_index, color_linestyle = "b-"):
        """
        Plots the motion of particle number particle_index in the xy-plane
        """
        x = self.x[:, particle_index]
        y = self.y[:, particle_index]

        fig, ax = plt.subplots()

        # Plotting analytical sol. against the numerical one
        # if there is one particle
        if self.one_particle:
            ax.plot(x, y, color_linestyle, label = "Numerical")
            x_exact, y_exact, z_exact = self.analytical(self.t)
            ax.plot(x_exact, y_exact, "r--", label = "Analytical")
        else:
            ax.plot(x, y, color_linestyle, label = "Particle {}".format(particle_index + 1))

        ax.legend()

        ax.set_xlabel(r"$x$ [%s]" % self.micrometer)
        ax.set_ylabel(r"$y$ [%s]" % self.micrometer)
        ax.set_aspect("equal")
        # Switching folder name and file-type
        if self.one_particle:
            image_filename = self.position_filename.replace("txt/", "images/").replace(".txt", ".pdf")
        else:
            image_filename = self.position_filename.replace("txt/", "images/").replace(".txt",
                                                                                       "_{}.pdf".format(particle_index))

        fig.savefig(image_filename, bbox_inches = "tight")
        # Matplotlib whined that I opened too many figures, so I just manually close every figure after saving
        plt.close()

    def plotZ(self, color_linestyle = "b-"):
        """
        Plots the z-coordinate of the particle (only used when
        we have one particle) as a function of time.
        """
        z = self.z[:, 0]

        fig, ax = plt.subplots()

        if self.one_particle:
            ax.plot(self.t, z, color_linestyle, label = "Numerical")
            x, y, z_exact = self.analytical(self.t)
            ax.plot(self.t, z_exact, "r--", label = "Analytical")
        else:
            ax.plot(self.t, z, color_linestyle, label = "Particle 1")

        ax.set_xlabel(r"$t$ [{}]".format(self.microsecond), fontsize = 13)
        ax.set_ylabel(r"$z$ [{}]".format(self.micrometer), fontsize = 13)
        ax.legend(fontsize = 13)
        ax.tick_params(axis = "both", labelsize = 13)

        # Creates filename
        if self.one_particle:
            image_filename = self.position_filename.replace("txt/one_particle_positions", "images/z(t)").replace(".txt", ".pdf")
        else:
            image_filename = self.position_filename.replace("txt/", "images/z(t)_").replace("_r.txt", ".pdf")

        fig.savefig(image_filename, bbox_inches = "tight")
        plt.close()

    def plotPhaseSpace(self, particle_index, axis = "x", color_linestyle = "b-"):
        """
        Plots phase space for particle given by particle_index. Either (x, v_x), (y, v_y) or (z, v_z) depending on axis
        """
        if axis == "x":
            r = self.x[:, particle_index]
            v = self.vx[:, particle_index]
        elif axis == "y":
            r = self.y[:, particle_index]
            v = self.vy[:, particle_index]
        else:
            r = self.z[:, particle_index]
            v = self.vz[:, particle_index]

        fig, ax = plt.subplots()
        ax.plot(r, v, color_linestyle, label = "Particle {}".format(particle_index + 1))
        ax.set_xlabel(r"$%s$ [%s]" % (axis, self.micrometer))
        ax.set_ylabel(r"$%s$ [%s]" % ("v_%s" % axis, self.micro_m_per_s))
        ax.legend()

        position_file = self.position_filename.replace("txt/", "").replace("_r.txt", ".pdf")
        image_filename = "images/phase_{}_{}_{}".format(axis, particle_index, position_file)
        fig.savefig(image_filename, bbox_inches = "tight")
        plt.close()

    def plot3d(self, particle_index, color_linestyle = "b-"):
        """
        Plots 3d motion of particle given by particle_index
        """
        x = self.x[:, particle_index]
        y = self.y[:, particle_index]
        z = self.z[:, particle_index]

        fig = plt.figure()
        ax = plt.axes(projection = "3d")
        ax.plot(x, y, z, color_linestyle, label = "Particle {}".format(particle_index + 1))
        ax.set_xlabel(r"$x$ [{}]".format(self.micrometer))
        ax.set_ylabel(r"$y$ [{}]".format(self.micrometer))
        ax.set_zlabel(r"$z$ [{}]".format(self.micrometer))
        ax.legend()

        position_file = self.position_filename.replace("txt/", "").replace("_r.txt", ".pdf")
        image_filename = "images/3d_{}_{}".format(particle_index, position_file)
        fig.savefig(image_filename, bbox_inches = "tight")
        plt.close()

    def analytical(self, t):
        """
        Computes the analytical solution. Of course
        only used to compare with single-particle simulations
        """
        x = A_plus*np.cos(omega_plus*self.t) + A_minus*np.cos(omega_minus*self.t)
        y = -A_plus*np.sin(omega_plus*self.t) + -A_minus*np.sin(omega_minus*self.t)
        z = np.cos(np.sqrt(omega_z2)*self.t)

        return x, y, z

    def compute_errors(self):
        """
        Calculates the errors between the analytical and numerical
        solutions. Again only for single-particle simulations.
        """
        x_exact, y_exact, z_exact = self.analytical(self.t)
        r_exact = np.sqrt(x_exact**2 + y_exact**2 + z_exact**2)

        # self.x[:, 0] is the x-values of the first (and only) particle
        diff_x = x_exact - self.x[:, 0]
        diff_y = y_exact - self.y[:, 0]
        diff_z = z_exact - self.z[:, 0]

        diff_r = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)

        self.abs_error = diff_r
        self.relative_error = diff_r/r_exact

# Plots 2d motion and z(t) for the single-particle case, from the simulation using RK4
data_oneParticle_RK4 = SimulationData("txt/one_particle_positions_RK4.txt", one_particle = True)
data_oneParticle_RK4.plotMotion2d(0)
data_oneParticle_RK4.plotZ()

# Plots 2d motion and z(t) for the single-particle case, from the simulation using Euler
data_oneParticle_Euler = SimulationData("txt/one_particle_positions_Euler.txt", one_particle = True)
data_oneParticle_Euler.plotMotion2d(0)
data_oneParticle_Euler.plotZ()

# PLots 2d motion, phase space (x, v_x) and 3d motion for both particles in the two-particle simulation
# without interactions. ALso plots (z, v_z) phase space and z(t) for the first particle.
data_no_interactions = SimulationData("txt/no_interactions_r.txt", "txt/no_interactions_v.txt")
data_no_interactions.plotMotion2d(0)
data_no_interactions.plotMotion2d(1, color_linestyle = "g-")
data_no_interactions.plotZ()
data_no_interactions.plotPhaseSpace(0, "x")
data_no_interactions.plotPhaseSpace(1, "x")
data_no_interactions.plotPhaseSpace(0, "z")
data_no_interactions.plot3d(0)
data_no_interactions.plot3d(1, "g-")

# PLots 2d motion and 3d motion for both particles in the two-particle simulation
# with interactions. ALso plots (x, v_x) and (z, v_z) phase spaces and z(t) for the first particle.
data_interactions = SimulationData("txt/interactions_r.txt", "txt/interactions_v.txt")
data_interactions.plotMotion2d(0)
data_interactions.plotMotion2d(1, color_linestyle = "g-")
data_interactions.plotZ()
data_interactions.plotPhaseSpace(0, "x")
data_interactions.plotPhaseSpace(0, "z")
data_interactions.plot3d(0)
data_interactions.plot3d(1, "g-")

class VaryingTimestep:
    def __init__(self, filename):
        self.t_max = np.loadtxt(filename, max_rows = 1)
        self.timesteps = np.loadtxt(filename, skiprows = 1, max_rows = 2)
        self.positions = np.loadtxt(filename, skiprows = 3)

        self.t = np.array([np.linspace(0, self.t_max, len(positions)) for positions in self.positions])


def plotErrors(textfiles, euler_or_RK4):
    """
    Plots the relative errors of the Euler and RK4 algorithms for 5
    different values of dt over a time interval. Then returns the
    error convergence rates of the Euler and RK4 algorithms.
    The parameter euler_or_RK4 is just a string that is used in the image-name.
    """
    errors = [SimulationData(filename, one_particle = True) for filename in textfiles]

    fig, ax = plt.subplots(figsize = (6, 7))

    for i in range(len(errors)):
        error = errors[i]
        ax.plot(error.t, error.relative_error, label = r"$h$ = {} $\mu$s".format(error.dt))


    ax.legend(fontsize = 13)
    ax.tick_params(axis = "both", labelsize = 13)
    ax.set_xlabel(r"$t$ [$\mu$s]", fontsize = 13)
    ax.set_ylabel(r"$\epsilon_i$", fontsize = 13)
    ax.set_yscale("log")
    fig.savefig("images/rel_errors_h_{}.pdf".format(euler_or_RK4), bbox_inches = "tight")
    plt.close()

    max_abs_errors = [np.max(np.abs(error.abs_error)) for error in errors]
    convergence_rate = 0

    for i in range(1, len(errors)):
        convergence_rate += np.log10(max_abs_errors[i]/max_abs_errors[i - 1])/np.log10(errors[i].dt/errors[i - 1].dt)

    return 1/4*convergence_rate

txt_files_RK4 = ["txt/varying_dt_0.001_RK4.txt", "txt/varying_dt_0.005_RK4.txt", "txt/varying_dt_0.01_RK4.txt",
             "txt/varying_dt_0.05_RK4.txt", "txt/varying_dt_0.1_RK4.txt"]

txt_files_Euler = ["txt/varying_dt_0.001_Euler.txt", "txt/varying_dt_0.005_Euler.txt", "txt/varying_dt_0.01_Euler.txt",
             "txt/varying_dt_0.05_Euler.txt", "txt/varying_dt_0.1_Euler.txt"]

# Calculating, plotting and printing the error-related stuff
converg_rate_RK4 = plotErrors(txt_files_RK4, "RK4")
converg_rate_Euler = plotErrors(txt_files_Euler, "Euler")

print("r_err for RK4: {:.2}".format(converg_rate_RK4))
print("r_err for Euler: {:.2}".format(converg_rate_Euler))

# Plots the number of particles left in the trap after the simulations for the
# low omega_V-resolution case where omega_V is in [0.2, 2.5] MHz
fig, ax = plt.subplots()
f_values = np.loadtxt("txt/particles_inside.txt", max_rows = 1)
omega_V_values = np.loadtxt("txt/particles_inside.txt", skiprows = 2, max_rows = 2)

particles_inside = np.loadtxt("txt/particles_inside.txt", skiprows = 4)

for i in range(len(f_values)):
    ax.plot(omega_V_values, particles_inside[i], label = r"$f$ = {}".format(f_values[i]))

ax.legend()
ax.set_xlabel(r"$\omega_V$ [MHz]")
ax.set_ylabel(r"$N_{inside}/N_0$")
fig.savefig("images/particles_inside.pdf", bbox_inches = "tight")
plt.close()


def read_particles_inside(filename):
    """
    Reads the files containing particle numbers inside the trap after
    simulations with externally applied time-varying electric field.
    """
    f_values = np.loadtxt(filename, max_rows = 1)
    omega_V_values = np.loadtxt(filename, skiprows = 1, max_rows = 2)

    particles_inside = np.loadtxt(filename, skiprows = 3)

    return f_values, omega_V_values, particles_inside

def plot_particles_inside(filename, N_0):
    """
    Plots the number of particles as a function of omega_V for each of the
    5 runs given in filename. N_0 is the initial number of particles inside the trap.
    """
    f_values, omega_V, particles_inside = read_particles_inside(filename)

    fig, ax = plt.subplots(figsize = (5, 5))

    particles_inside_mean = np.mean(particles_inside, axis = 1)

    for i in range(len(particles_inside[0])):
        ax.plot(omega_V, particles_inside[:, i]/N_0, label = "Run #{}".format(i + 1))

    ax.plot(omega_V, particles_inside_mean/N_0, "k--", label = "Mean of runs")

    ax.legend()
    ax.set_xlabel(r"$\omega_V$ [MHz]")
    ax.set_ylabel(r"$N_{inside}/N_0$")

    # Ugly code for creating title and filename
    title = "Interactions off" if "no" in filename else "Interactions on"
    image_filename = filename.replace("txt/", "images/").replace(".txt", ".pdf")

    ax.set_title(title)
    fig.savefig(image_filename, bbox_inches = "tight")
    plt.close()

    # Returns so that I can plot the means together
    return omega_V, particles_inside_mean

# Number of particles at the beginning of each simulation
N_0 = 100

omega_V_no_int, mean_no_int = plot_particles_inside("txt/particles_inside_zoomed_no_int.txt", N_0)
omega_V_int, mean_int = plot_particles_inside("txt/particles_inside_zoomed_int.txt", N_0)

fig, ax = plt.subplots()
ax.plot(omega_V_no_int, mean_no_int/N_0, "b-", label = "No interactions")
ax.plot(omega_V_int, mean_int/N_0, "g-", label = "Interactions")
ax.legend()
ax.set_xlabel(r"$\omega_V$ [MHz]")
ax.set_ylabel(r"$N_{inside}/N_0$")
fig.savefig("images/particles_inside_means.pdf", bbox_inches = "tight")
plt.close()
