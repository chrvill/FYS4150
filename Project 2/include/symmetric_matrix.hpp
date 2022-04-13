#ifndef SYMMETRIC_MATRIX_HPP
#define SYMMETRIC_MATRIX_HPP

#include <armadillo>

class SymmetricMatrix
{
private:
  int N;
  arma::mat A;

public:
  SymmetricMatrix(int N_rows, arma::mat A_matrix): N{N_rows}, A(A_matrix) {}

  // Some function declarations
  double max_offdiag_symmetric(int& k, int& l) const;

  void jacobi_rotate(arma::mat& R, int k, int l);

  void jacobi_eigensolver(double eps, arma::vec& eigenvalues, arma::mat& eigenvectors,
                          const int maxiter, int& iterations, bool& converged);

  void fill_tridiag(double a, double d);

  void set_matrix(arma::mat A_matrix);

};

#endif
