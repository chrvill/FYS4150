#include "system_matrix.hpp"
#include <complex>
#include <armadillo>
#include <iostream>

int main()
{
  double h{0.005};
  double dt{2.5e-5};

  SystemMatrix sysmat{h, dt};

  arma::cx_cube U;

  // The no-slit setup
  sysmat.initialize(0.02, 0.5, 0.05, 0.05, 0, 0);
  U = sysmat.simulate(0.008, 0.25, 0.5, 0.05, 0.05, 200, 0);
  U.save("txtfiles/no_slit.txt", arma::arma_binary);

  // The double slit setup with T = 0.008
  sysmat.initialize(0.02, 0.5, 0.05, 0.05, 2);
  U = sysmat.simulate(0.008, 0.25, 0.5, 0.05, 0.1, 200, 0);
  U.save("txtfiles/double_slit_0_008.txt", arma::arma_binary);

  // The double slit setup with T = 0.002
  sysmat.initialize(0.02, 0.5, 0.05, 0.05, 2);
  U = sysmat.simulate(0.002, 0.25, 0.5, 0.05, 0.2, 200, 0);
  U.save("txtfiles/double_slit_0_002.txt", arma::arma_binary);

  // The single slit setup
  sysmat.initialize(0.02, 0.5, 0.05, 0.05, 1);
  U = sysmat.simulate(0.002, 0.25, 0.5, 0.05, 0.2, 200, 0);
  U.save("txtfiles/single_slit.txt", arma::arma_binary);

  // The triple slit setup
  sysmat.initialize(0.02, 0.5, 0.05, 0.05, 3);
  U = sysmat.simulate(0.002, 0.25, 0.5, 0.05, 0.2, 200, 0);
  U.save("txtfiles/triple_slit.txt", arma::arma_binary);

  // The triple slit setup where the slits and the wall sections
  // between the slits have height 0.01
  sysmat.initialize(0.02, 0.5, 0.01, 0.01, 4);
  U = sysmat.simulate(0.002, 0.25, 0.5, 0.05, 0.2, 200, 0);
  U.save("txtfiles/triple_slit_smaller_slits.txt", arma::arma_binary);

  return 0;
}
