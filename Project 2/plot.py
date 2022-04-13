import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

textfile_prefix = "textFiles/"
image_prefix = "images/"

class EigenData:
    def __init__(self, eigenvals_filename, eigenvecs_filename):
        a_d_eigenvalues = np.loadtxt(eigenvals_filename)
        self.a, self.d = a_d_eigenvalues[:2]
        self.eigenvalues = a_d_eigenvalues[2:]
        # Transposing here because this means self.eigenvectors[i]
        # is the i-th eigenvector, instead of having to do self.eigenvectors[:, i]
        self.eigenvectors = np.transpose(np.loadtxt(eigenvecs_filename))
        self.N = len(self.eigenvalues)

    def plotEigenVectors(self, m):
        """
        Plots the m eigenvectors with the largest eigenvalues
        """
        # Total number of points, including the endpoints x = 0 and x = L
        n = len(self.eigenvectors[0]) + 2
        x = np.linspace(0, 1, n)

        # Extending the eigenvectors array to include the endpoints for
        # each eigenvector
        y = np.zeros((n - 2, n))
        y[:, 1:-1] = self.eigenvectors

        abs_eigvals = np.abs(self.eigenvalues)

        indices = np.argpartition(abs_eigvals, -self.N)[-self.N:]
        sorted_indices = indices[np.argsort(abs_eigvals[indices])]

        sorted_eigenvalues = self.eigenvalues[sorted_indices]
        sorted_eigenvectors = y[sorted_indices]

        fig, axes = plt.subplots(2, 2, figsize = (12, 10))
        axes = axes.flatten()
        plt.suptitle(r"$N =$ {}".format(n - 2), fontsize = 17)

        for i in range(m):
            ax = axes[i]
            # Plotting the m eigenvectors with the smallest
            # corresponding eigenvalues
            ax.grid()
            ax.set_xlabel(r"$\hat{x} = x/L$", fontsize = 17)
            ax.set_ylabel(r"$u(\hat{x})$", fontsize = 17)
            ax.plot(x, sorted_eigenvectors[i], label = "Numerical")

            ax.set_title(r"$\lambda \approx$ {:.2f}".format(sorted_eigenvalues[i]), fontsize = 17)
            ax.tick_params(axis = "both", labelsize = 15)

            analytical = np.zeros_like(x)
            #print(self.analytical_eigenvecs(), "sgllskg\n")
            analytical[1: -1] = self.analytical_eigenvecs()[:, i]
            ax.plot(x, analytical, label = "Analytical")

            ax.legend(fontsize = 17)
            #fig.savefig(imageFilename + "{:.2f}".format(sorted_eigenvalues[i]) + ".pdf")
        fig.delaxes(axes[-1])
        fig.tight_layout()
        fig.savefig("{}eigenvectors_numerical_N{}.pdf".format(image_prefix, self.N))

    def analytical_eigenvals(self, printLambdas = False):
        """
        Calculates the eigenvalues from the analytical formula
        """
        lambdas = np.array([self.d + 2*self.a*np.cos(i*np.pi/(self.N + 1)) for i in range(1, self.N + 1)])

        # If we want to print the eigenvalues
        if printLambdas:
            print("Analytical eigenvalues: \n", np.round(lambdas, 4), "\n")

        return lambdas

    def analytical_eigenvecs(self, printVectors = False):
        """
        Calculates the eigenvectors from the analytical formula
        """
        vecs = np.array([[np.sin(j*i*np.pi/(self.N + 1)) for j in range(1, self.N + 1)] for i in range(1, self.N + 1)])

        # Normalizing
        vecs /= np.array([np.linalg.norm(vec) for vec in vecs])

        # If we want to print the eigenvectors
        if printVectors:
            print("Analytical eigenvectors: \n", np.round(vecs, 4), "\n")

        return vecs

    def plotTransformationNumbers(self, filename, dense = False):
        """
        Plots the number of similarity transformations performed
        as a function of N
        """
        N, transfNumb = np.loadtxt(filename, usecols = (0, 1), unpack = True)
        fig, ax = plt.subplots()
        ax.plot(N, transfNumb, "b-", label = "Numerical result")
        ax.grid()

        # Quadratic function that will be fitted to the curve
        f = lambda x, a, b: a*x**b

        # Best fit parameters
        fitted_parameters = curve_fit(f, N, transfNumb)
        a, b = fitted_parameters[0]

        # Just an array of more N values than in the numerical
        # calculations, because the fitted curve looks smoother then
        smooth_N = np.linspace(np.min(N), np.max(N), 100)

        # Plotting the fitted curve together with the numerical curve
        ax.plot(smooth_N, f(smooth_N, a, b), "r-",
                label = r"Fitted curve ($a\cdot N^b$ for $a \approx$ {:.2f}, $b \approx$ {:.2f})".format(a, b))
        ax.legend()
        ax.set_xlabel(r"$N$")
        ax.set_ylabel("Number of similarity transformations")

        # Just a string that is added onto the image filename if the
        # matrix we've considered was dense
        denseSuffix = "Dense" if dense else "Sparse"

        ax.set_title("Number of similarity transformations as function of N ({} matrix)".format(denseSuffix))

        fig.savefig("{}transformationNumbers{}.pdf".format(image_prefix, denseSuffix))

N = [10, 100]

# PLotting three eigenvectors for N = 10 and N = 100
for i in N:
    eigen = EigenData("{}eigenvals{}.txt".format(textfile_prefix, i), "{}eigenvecs{}.txt".format(textfile_prefix, i))
    eigen.plotEigenVectors(m = 3)

# Plotting the number of transformations as function of N for a
# tridiagonal and a dense matrix
eigen.plotTransformationNumbers("{}numTransfSparse.txt".format(textfile_prefix))
eigen.plotTransformationNumbers("{}numTransfDense.txt".format(textfile_prefix), dense = True)   

# Printing the analytical eigenvalues and eigenvectors for N = 6
eigen6 = EigenData("{}eigenvals6.txt".format(textfile_prefix), "{}eigenvecs6.txt".format(textfile_prefix))
eigen6.analytical_eigenvals(True)
eigen6.analytical_eigenvecs(True)
