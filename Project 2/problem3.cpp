#include <iostream>
#include <armadillo>
#include <vector>
#include <fstream>

void fill_tridiag(arma::mat& A, double a, double d)
{
  int n_rows{static_cast<int>(A.n_rows)};

  for (int i{0}; i < n_rows; ++i)
  {
    A(i, i) = d;

    if (i < (n_rows - 1))
    {
      A(i, i + 1) = a;
    }

    if (i > 0)
    {
      A(i, i - 1) = a;
    }
  }
}

int main()
{
  int N{6};

  arma::mat A(N, N, arma::fill::zeros);

  double h{1.0/(N + 1)};

  double a{-1.0/(h*h)};
  double d{2/(h*h)};

  fill_tridiag(A, a, d);

  arma::vec eigenvals;
  arma::mat eigenvecs;

  arma::eig_sym(eigenvals, eigenvecs, A);

  //std::cout << A << "\n";

  std::cout << "Eigenvalues from armadillo: \n" << eigenvals << "\n";
  std::cout << "Eigenvectors from armadillo: \n" << eigenvecs << "\n";

  std::ofstream outf_eigenvals{"textFiles/eigenvalsAnalytical.txt"};
  outf_eigenvals << a << "\n" << d << "\n";
  outf_eigenvals << eigenvals << "\n";

  std::ofstream outf_eigenvecs{"textFiles/eigenvecsAnalytical.txt"};
  outf_eigenvecs << eigenvecs << "\n";

  return 0;
}
