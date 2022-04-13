#include "system_matrix.hpp"
#include <iostream>
#include <cmath>

SystemMatrix::SystemMatrix(double h_, double dt_):
  h{h_}, dt{dt_}
{
  M = 1.0/h + 1;

  int length{(M - 2)*(M - 2)};

  A = arma::sp_cx_mat(length, length);
  B = arma::sp_cx_mat(length, length);
}

int SystemMatrix::k_from_ij(int i, int j) const
{
  return (i - 1) + (j - 1)*(M - 2);
}

void SystemMatrix::generate_matrices(std::complex<double> r, arma::cx_vec a, arma::cx_vec b)
{
  int length{(M - 2)*(M - 2)};

  A.diag(0) = a;
  B.diag(0) = b;

  // Filling the sub- and superdiagonals
  A.diag(1).fill(-r);
  A.diag(-1).fill(-r);

  B.diag(1).fill(r);
  B.diag(-1).fill(r);

  // Filling the (M - 2) and -(M - 2) diagonals
  A.diag(M - 2).fill(-r);
  A.diag(-(M - 2)).fill(-r);

  B.diag(M - 2).fill(r);
  B.diag(-(M - 2)).fill(r);

  // Correcting for the elemnts along the super- and sub-diagonals
  // that should be zero.
  for (int j{0}; j < length; ++j)
  {
    if ((j + 1) % (M - 2) == 0 && j != (length - 1))
    {
      A(j, j + 1) = 0;
      A(j + 1, j) = 0;

      B(j, j + 1) = 0;
      B(j + 1, j) = 0;
    }
  }
}

arma::cx_mat SystemMatrix::generate_a_b(std::complex<double> r) const
{
  arma::cx_mat a_b((M - 2)*(M - 2), 2, arma::fill::zeros);

  for (int j{1}; j <= M - 2; ++j)
  {
    for (int i{1}; i <= M - 2; ++i)
    {
      int k{k_from_ij(i, j)};

      a_b(k, 0) = 1.0 + 4.0*r + std::complex<double>(0, 1.0)*dt/2.0*V(i, j);
      a_b(k, 1) = 1.0 - 4.0*r - std::complex<double>(0, 1.0)*dt/2.0*V(i, j);
    }
  }

  return a_b;
}

void SystemMatrix::initialize(double wall_thickness, double wall_x_pos, double slit_height,
                                 double slit_separation, double n_slits, double v0)
{
  initialize_V(wall_thickness, wall_x_pos, slit_height, slit_separation, n_slits, v0);

  std::complex<double> r = std::complex<double>(0, 1.0)*dt/(2.0*h*h);

  arma::cx_mat a_b{generate_a_b(r)};

  arma::cx_vec a = a_b.col(0);
  arma::cx_vec b = a_b.col(1);

  generate_matrices(r, a, b);
}

arma::cx_vec SystemMatrix::calc_new_u(const arma::cx_vec& u) const
{
  arma::cx_vec b = B*u;

  arma::cx_vec u_new = arma::spsolve(A, b, "superlu");

  return u_new;
}

arma::cx_vec SystemMatrix::u_mat_to_vec(const arma::cx_mat& u) const
{
  arma::cx_vec u_vec((M - 2)*(M - 2), arma::fill::zeros);

  int col_len{M - 2};

  for (int i{0}; i < M - 2; ++i)
  {
    u_vec.subvec(col_len*i, col_len*(i + 1) - 1) = u.col(i);
  }

  return u_vec;
}

arma::cx_mat SystemMatrix::u_vec_to_mat(const arma::cx_vec& u) const
{
  arma::cx_mat u_mat(M - 2, M - 2, arma::fill::zeros);

  int col_len{M - 2};

  for (int i{0}; i < u_mat.n_cols; ++i)
  {
    u_mat.col(i) = u.subvec(col_len*i, col_len*(i + 1) - 1);
  }

  return u_mat;
}

arma::cx_vec SystemMatrix::initialize_gauss(double x_c, double y_c, double sigma_x, double sigma_y,
                                            double p_x, double p_y) const
{
  arma::mat x(M - 2, M - 2, arma::fill::zeros);
  arma::mat y(M - 2, M - 2, arma::fill::zeros);

  // Doing separate loops over x and y because in principle there
  // could be different number of points in each direction.

  for (int j{0}; j < M - 2; ++j)
  {
    y.row(j) = arma::trans(arma::linspace(h, 1 - h, M - 2));
  }

  for (int i{0}; i < M - 2; ++i)
  {
    x.col(i) = arma::linspace(h, 1 - h, M - 2);
  }

  std::complex<double> i_imag(0, 1.0);

  arma::cx_mat u_mat = arma::exp(-(x - x_c) % (x - x_c)/(2*sigma_x*sigma_x)
                                 -(y - y_c) % (y - y_c)/(2*sigma_y*sigma_y)
                                 +i_imag*p_x*(x - x_c) + i_imag*p_y*(y - y_c));

  arma::cx_vec u_vec = u_mat_to_vec(u_mat);
  arma::cx_vec u_vec_conj = arma::conj(u_vec);

  double norm_const = std::sqrt(arma::sum(u_vec % u_vec_conj).real());

  return u_vec/norm_const;
}

void SystemMatrix::initialize_V(double wall_thickness, double wall_x_pos, double slit_height,
                                double slit_separation, double n_slits, double v0)
{
  int wall_x_index = wall_x_pos/h;

  // Width of wall counted in number of indices
  int wall_dx_index = wall_thickness/h;

  // Height of slit counted in number of indices
  int slit_dy_index = slit_height/h;

  int slit_sep_index = slit_separation/h;

  // y-coord. of the top of the top slit. We want the same
  // height below the bottom slit as above the top slit.
  double top_slit_y = 1.0/2.0 - n_slits/2.0*slit_height - (n_slits - 1.0)/2.0*slit_separation;
  int top_slit_index = top_slit_y*1.0/h;

  V = arma::mat(M, M, arma::fill::zeros);

  for (int i{wall_x_index}; i < wall_x_index + wall_dx_index; ++i)
  {
    // Initially filling in the whole wall
    V.submat(i, 0, i, M - 1).fill(v0);

    for (int j{0}; j < n_slits; ++j)
    {
      int slit_top = top_slit_index + j*slit_dy_index + j*slit_sep_index;
      int slit_bottom = slit_top + slit_dy_index - 1;

      V.submat(i, slit_top, i, slit_bottom).fill(0);
    }
  }
}

arma::cx_cube SystemMatrix::simulate(double T, double x_c, double y_c, double sigma_x, double sigma_y,
                                     double p_x, double p_y) const
{
  int n_timesteps = T/dt;

  arma::cx_vec u_i = initialize_gauss(x_c, y_c, sigma_x, sigma_y, p_x, p_y);

  arma::cx_cube U(M - 2, M - 2, n_timesteps + 1, arma::fill::zeros);

  U.slice(0) = u_vec_to_mat(u_i);

  for (int i{1}; i <= n_timesteps; ++i)
  {
    arma::cx_mat temp_u = calc_new_u(u_i);
    U.slice(i) = u_vec_to_mat(temp_u);

    u_i = temp_u;
  }

  return U;
}
