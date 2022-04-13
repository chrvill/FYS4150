#include "MCMC.hpp"
#include <iostream>
#include <fstream>
#include <string>

// The function collects samples for a fixed temperature. It is used to
// estimate the burn-in time. So it saves the energy and magnetiation
// in each sample configuration. These are then written to file.
//
// T is temperature.
// L is number of spins along each axis.
// n_cycles is the number of Monte Carlo cycles to perform.
// seed is a seed that is fed into the RNG.
// ordered specifies whether the initial spin_state should
// be ordered or unordered (see MCMC.hpp for clarification)
void fixed_temperature(double T, int L, int n_cycles, int seed, bool ordered)
{
  std::string burn_in_filename;

  std::string T_string{(T - 2.4 == 0)? "2_4": "1_0"};

  if (ordered)
  {
    burn_in_filename = "txtfiles/burn_in" + std::to_string(L) + "x" + std::to_string(L) + "_" +
                        T_string + "_ordered.txt";
  }
  else
  {
    burn_in_filename = "txtfiles/burn_in" + std::to_string(L) + "x" + std::to_string(L) + "_" +
                        T_string + "_unordered.txt";
  }

  std::ofstream outf_burn_in{burn_in_filename};
  outf_burn_in << L*L << "\t" << T << "\n";

  arma::vec E_every_cycle{arma::vec(n_cycles, arma::fill::zeros)};
  arma::vec M_every_cycle{arma::vec(n_cycles, arma::fill::zeros)};


  MCMC mcmc{L, T, seed, ordered};
  arma::mat E_M = mcmc.generate_samples(n_cycles);

  E_every_cycle = E_M.col(0);
  M_every_cycle = arma::abs(E_M.col(1));

  outf_burn_in << arma::trans(E_every_cycle) << "\n";
  outf_burn_in << arma::trans(M_every_cycle) << "\n";
}
