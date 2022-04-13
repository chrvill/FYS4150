#include "MCMC.hpp"
#include <iostream>
#include <fstream>
#include <string>

// The function collects samples for a range of different temperatures.
// It then calculates the approx. expectation values of E, |M|, E^2 and M^2,
// excluding the burn-in time from these estimates. These exp. values are
// written to file.
//
// min_temp and max_temp are the ranges of temperature to collect samples for
// N_temps is the number of temperature values
// L is the number of spins in each direction
// n_cycles is the number of Monte Carlo cycles to perform for each temperature
// burn_in_stop is the index (in terms of number of cycles) at which burn-in stops. So burn_in_stop = 100
// means burn_in stops at 100 cycles (this is unrealistically low of course)
// seed is the seed that is fed into the RNG
void varying_temperature(double min_temp, double max_temp, int N_temps, int L, int n_cycles,
                         int burn_in_stop, int seed, bool parallelize)
{
  arma::vec temperatures{arma::linspace(min_temp, max_temp, N_temps)};

  std::string filename{"txtfiles/" + std::to_string(L) + "x" + std::to_string(L) +
                       "_" + std::to_string(n_cycles) + ".txt"};

  std::ofstream outf{filename};
  outf << L*L << "\t" << n_cycles << "\n";
  outf << arma::trans(temperatures) << "\n";

  arma::vec E_values{arma::vec(N_temps, arma::fill::zeros)};
  arma::vec M_values{arma::vec(N_temps, arma::fill::zeros)};
  arma::vec E2_values{arma::vec(N_temps, arma::fill::zeros)};
  arma::vec M2_values{arma::vec(N_temps, arma::fill::zeros)};

  // Just a way to be able to specify whether or not to parallelize the temperature loop.
  if (parallelize)
  {
    omp_set_num_threads(8);
  }
  else
  {
    omp_set_num_threads(1);
  }

  #pragma omp parallel for
  for (int j = 0; j < N_temps; ++j)
  {
    double E{0};
    double E_square{0};
    double M{0};
    double M_square{0};

    double T{temperatures(j)};

    MCMC mcmc{L, T, seed, false};
    arma::mat E_M = mcmc.generate_samples(n_cycles);

    // Only storing the samples after burn-in
    arma::vec E_i = E_M.col(0).subvec(burn_in_stop, n_cycles - 1);
    arma::vec M_i = E_M.col(1).subvec(burn_in_stop, n_cycles - 1);

    E += arma::accu(E_i);
    M += arma::accu(arma::abs(M_i));

    E_square += arma::accu(E_i % E_i);
    M_square += arma::accu(M_i % M_i);

    int length_E{static_cast<int>(E_i.n_elem)};

    E_values(j) = E/length_E;
    E2_values(j) = E_square/length_E;
    M_values(j) = M/length_E;
    M2_values(j) = M_square/length_E;

  }

    outf << arma::trans(E_values) << "\n";
    outf << arma::trans(E2_values) << "\n";
    outf << arma::trans(M_values) << "\n";
    outf << arma::trans(M2_values) << "\n";
}
