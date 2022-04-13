#ifndef SYSTEM_MATRIX_HPP
#define SYSTEM_MATRIX_HPP

#include <armadillo>

class SystemMatrix
{
private:
  int M;
  double h;
  double dt;
  arma::mat V;

  arma::sp_cx_mat A;
  arma::sp_cx_mat B;

public:
  SystemMatrix(double h_, double dt_);

  // Method for calculating k from i and j
  // Takes in integers i and j and returns integer k
  int k_from_ij(int i, int j) const;

  // Method that generates the matrices A and B from vectors a and b, and complex number r
  // Is only called when initializing the system.
  void generate_matrices(std::complex<double> r, arma::cx_vec a, arma::cx_vec b);

  // Method for generating the vectors a and b that go into A and B
  // Takes in a complex number r and calculates the vectors a and b
  // which are the diagonals of A and B.
  // The function returns a matrix where the two columns correspond to a and b
  arma::cx_mat generate_a_b(std::complex<double> r) const;

  // Initializes the system, meaning it generates A and B and the potential
  // wall_thickness specifies the thickness of the wall, wall_x_pos specifies the
  // position of the left edge of the wall, slit_height gives the height of each slit.
  // slit_separation is the height of the wall section between the slits, n_slits is the number
  // of slits in the wall and v0 is the value of the potential at the wall
  void initialize(double wall_thickness, double wall_x_pos, double slit_height,
                  double slit_separation, double n_slits, double v0 = 1e10);

  // Calculates u^(n + 1) from u^n using arma::spsolve
  // The argument u is a vector of the current wavefunction
  // Returns u in the next timestep as a vector
  arma::cx_vec calc_new_u(const arma::cx_vec& u) const;

  // Method for converting a U matrix to a column vector.
  // This is used when initializing, because we give a matrix U
  // and the algorithm for solving the PDE uses a vector u.
  // The argument u is a matrix containing the wavefunction at each point in space,
  // and the function returns a vector where the columns of u are stacked after each other.
  arma::cx_vec u_mat_to_vec(const arma::cx_mat& u) const;

  // The inverse of the previous method. Converts u from a
  // column vector to a matrix. This is used when storing the
  // new u, because we store the wavefunction as slices of a arma::cx_cube
  // The argument u is a vector, and the funtion returns the wavefunction
  // as a matrix.
  arma::cx_mat u_vec_to_mat(const arma::cx_vec& u) const;

  // Generates the U matrix containing the initial Gaussian wave packet.
  // The U matrix only consists of u at the interior points.
  // (x_c, y_c) is the center position of the wave packet. sigma_x and sigma_y give the width of the
  // wave packet. p_x and p_y are to momenta of the wave packet in the x- and y-directions.
  arma::cx_vec initialize_gauss(double x_c, double y_c, double sigma_x, double sigma_y, double p_x, double p_y) const;

  // Method for initializing the potential.
  // The arguments to this function mean the same as those in the initialize()-function
  void initialize_V(double wall_thickness, double wall_x_pos, double slit_height, double slit_separation,
                    double n_slits, double v0);

  // Method for simulating the time-evolution of the system over a given time T
  // T is the total simulation time, while the other arguments are the same as in initialize_gauss()
  arma::cx_cube simulate(double T, double x_c, double y_c, double sigma_x, double sigma_y,
                         double p_x, double p_y) const;
};

#endif
