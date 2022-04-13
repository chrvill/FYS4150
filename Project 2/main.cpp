#include <armadillo>
#include <iostream>
#include <vector>
#include <string>
#include "symmetric_matrix.hpp"

void run_max_offdiag_symmetric_test()
{
  arma::mat A(4, 4, arma::fill::zeros);
  A(0, 0) = 1;
  A(3, 0) = 0.5;
  A(1, 1) = 1;
  A(2, 1) = -0.7;
  A(2, 2) = 1;
  A(1, 2) = -0.7;
  A(0, 3) = 0.5;
  A(3, 3) = 1;

  SymmetricMatrix A_{4, A};

  std::cout << A << "\n";

  int k;
  int l;

  double max{A_.max_offdiag_symmetric(k, l)};
  std::cout << "Max element: " << max << "\n";
  std::cout << "Max off diagonal element at indices: (" << k << ", " << l << ")" << "\n";
}

int solveAndWrite(int N, bool tridiag = true, bool writeToFile = true, double eps = 1e-8, int maxiter = 1000)
{

  arma::mat A_matrix(N, N, arma::fill::zeros);
  SymmetricMatrix A(N, A_matrix);

  double h{1.0/(N + 1)};
  double a{-1.0/(h*h)};
  double d{2.0/(h*h)};

  if (tridiag)
  {
    A.fill_tridiag(a, d);
  }

  else
  {
    A_matrix.randn();
    A_matrix = arma::symmatu(A_matrix);
    A.set_matrix(A_matrix);
  }

  arma::vec eigenvalues;
  arma::mat eigenvectors;

  int iterations{0};
  bool converged{false};

  A.jacobi_eigensolver(eps, eigenvalues, eigenvectors, maxiter, iterations, converged);

  if (writeToFile)
  {
    std::string eigenvalsFilename{"textFiles/eigenvals" + std::to_string(N) + ".txt"};
    std::string eigenvecsFilename{"textFiles/eigenvecs" + std::to_string(N) + ".txt"};

    std::ofstream eigenvalsOutf{eigenvalsFilename};
    std::ofstream eigenvecsOutf{eigenvecsFilename};

    // Writing the values of a and d to the eigenvalues-file
    // so they can be used in the Python program
    eigenvalsOutf << a << "\n" << d << "\n";

    eigenvalsOutf << eigenvalues << "\n";
    eigenvecsOutf << eigenvectors << "\n";
  }

  return iterations;
}

int main()
{
  run_max_offdiag_symmetric_test();
  // A vector that will be filled with the number of
  // similarity transformations necessary to diagonalize
  // the matrix, for different values of N
  std::vector<int> numTransf{{6, 10, 15, 20, 30, 40, 50, 70, 100}};

  std::ofstream numTransfOutf{"textFiles/numTransfSparse.txt"};
  std::ofstream numTransfDenseOutf{"textFiles/numTransfDense.txt"};

  for (int N: numTransf)
  {
    // Making sure I only save the eigenvectors and eigenvalues
    // for N = 6, 10 and 100. The other values of N are only used to
    // make plot of the number of similarity transformations as function of N
    bool writeToFile = (N == 6 || N == 10 || N == 100) ? true : false;
    int iterations{solveAndWrite(N, true, writeToFile)};
    numTransfOutf << N << "\t" << iterations << "\n";

    int iterationsDense{solveAndWrite(N, false, false)};
    numTransfDenseOutf << N << "\t" << iterationsDense << "\n";
  }

  return 0;
}
