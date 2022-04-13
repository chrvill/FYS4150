#include "symmetric_matrix.hpp"
#include <cmath>
#include <iostream>

double SymmetricMatrix::max_offdiag_symmetric(int& k, int& l) const
{
  // The lower triangular part of A
  arma::mat lower_triangular{arma::trimatl(A)};
  // The diagonal part of A
  arma::mat diagonal{arma::diagmat(A)};

  // Lower triangular part without the main diagonal
  arma::mat U{lower_triangular - diagonal};

  // Same as U, just with absolute values
  arma::mat abs_U{arma::abs(U)};
  // Maximum (in abs.value) element in U
  double max{abs_U.max()};

  // Looping through each row of U
  for (int i{0}; i < U.n_rows; ++i)
  {
    // Looping through the columns, but only until
    // j = i - 1, since U is a lower triangular
    // matrix with no main diagonal
    for (int j{0}; j < i; ++j)
    {
      // Where the maximum is reached
      if (abs_U(i, j) == max)
      {
        // The row- and column-indices are stored in k and l
        k = i;
        l = j;

        // And we stop the loop here, jumping to the code after the for loop over i
        goto loop_over;
      }
    }
  }
  loop_over:

  // Returning element in U with maximum absolute value
  return abs_U(k, l);
}

void SymmetricMatrix::jacobi_rotate(arma::mat& R, int k, int l)
{
  double a_ll{A(l, l)};
  double a_kk{A(k, k)};
  double a_kl{A(k, l)};

  double tau{(a_ll - a_kk)/(2*a_kl)};

  double t;

  if (tau > 0)
  {
    t = 1.0/(tau + std::sqrt(1 + tau*tau));
  }
  else
  {
    t = -1.0/(-tau + std::sqrt(1 + tau*tau));
  }

  double c{1.0/(std::sqrt(1 + t*t))};
  double s{c*t};

  A(k, k) = a_kk*c*c - 2*a_kl*c*s + a_ll*s*s;
  A(l, l) = a_ll*c*c + 2*a_kl*c*s + a_kk*s*s;
  A(k, l) = 0;
  A(l, k) = 0;

  for (int i{0}; i < N; ++i)
  {
    if (i != l && i != k)
    {
      double a_ik{A(i, k)};

      A(i, k) = a_ik*c - A(i, l)*s;
      A(k, i) = A(i, k);

      A(i, l) = A(i, l)*c + a_ik*s;
      A(l, i) = A(i, l);
    }

    double r_ik{R(i, k)};
    double r_il{R(i, l)};

    R(i, k) = r_ik*c - r_il*s;
    R(i, l) = r_il*c + r_ik*s;
  }
}

void SymmetricMatrix::jacobi_eigensolver(double eps, arma::vec& eigenvalues, arma::mat& eigenvectors,
                                         const int maxiter, int& iterations, bool& converged)
{
  int k{};
  int l{};

  arma::mat R;
  R.eye(size(A));

  double max{max_offdiag_symmetric(k, l)};

  while (max > eps)
  {
    jacobi_rotate(R, k, l);

    max = max_offdiag_symmetric(k, l);
    ++iterations;
  }

  //std::cout << A << "\n";
  for (int i{0}; i < A.n_rows; ++i)
  {
    R.col(i) = arma::normalise(R.col(i));
  }

  eigenvectors = R;
  eigenvalues = A.diag();
}

void SymmetricMatrix::fill_tridiag(double a, double d)
{
  for (int i{0}; i < N; ++i)
  {
    A(i, i) = d;

    if (i < (N - 1))
    {
      A(i, i + 1) = a;
    }

    if (i > 0)
    {
      A(i, i - 1) = a;
    }
  }
}

void SymmetricMatrix::set_matrix(arma::mat A_matrix)
{
  A = A_matrix;
}
