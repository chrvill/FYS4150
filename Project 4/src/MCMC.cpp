#include "MCMC.hpp"
#include <random>
#include <iostream>

MCMC::MCMC(int L_, double temperature, int seed, bool ordered)
{
  L = L_;
  T = temperature;
  E = 0; // Initializing the energy
  M = 0; // Initializing the magnetization

  int_dist = std::uniform_int_distribution<int>(0, L - 1);
  real_dist = std::uniform_real_distribution<double>(0, 1);
  generator.seed(seed);

  std::uniform_int_distribution<int> init_int_dist = std::uniform_int_distribution<int>(0, 1);

  if (ordered)
  {
    spin_state = arma::mat(L, L, arma::fill::ones);
  }

  else
  {
    spin_state = arma::mat(L, L, arma::fill::zeros);

    for (int j{0}; j < L; ++j)
    {
      for (int i{0}; i < L; ++i)
      {
        // Insanely ugly way of filling it, but I didn't have time to
        // think of a clever way. This fills position (i, j) with either -1 or 1
        spin_state(j, i) = std::pow(3, init_int_dist(generator)) - 2;
      }
    }
  }

  compute_initial_energy();
  compute_initial_magnetization();

  beta = 1.0/T;

  // The delta_E of a spin flip. Say we flip spin s_ij. Then
  // the first value in each bracket is the net spin of ij's neighbors
  // multiplied by s_ij. So positive if neighbors have net same spin
  // as ij and negative otherwise.
  flip_energies = {
      {4, 8},
      {2, 4},
      {0, 0},
      {-2, -4},
      {-4, -8}
  };

  // Boltzmann factors corresponding to a spin flip.
  boltzmann_factors = {
      {4, std::exp(-8*beta)},
      {2, std::exp(-4*beta)},
      {0, 1},
      {-2, std::exp(4*beta)},
      {-4, std::exp(8*beta)}
  };
}

void MCMC::compute_initial_energy()
{
  E = 0;

  for (int j{0}; j < L; ++j)
  {
    for (int i{0}; i < L; ++i)
    {
      double spin_ji{spin_state(j, i)};
      
      // Thanks to Morten's lecture notes for this idea!
      E -= spin_ji*(spin_state((L + j - 1) % L, i) + spin_state(j, (i + 1) % L));
    }
  }
}

void MCMC::compute_initial_magnetization()
{
  M = arma::accu(spin_state);
}

void MCMC::flip_single_spin()
{
  int row_index{int_dist(generator)};
  int col_index{int_dist(generator)};

  int spin_ji = spin_state(row_index, col_index);

  // Sum of spins of all four neighbors
  int neighbors_spins = spin_state(row_index, (L + col_index - 1) % L) + spin_state(row_index, (col_index + 1) % L) +
                        spin_state((L + row_index - 1) % L, col_index) + spin_state((row_index + 1) % L, col_index);

  neighbors_spins *= spin_ji;

  double r{real_dist(generator)};

  // This is the p(s_new)/p(s_old) used in the accept/reject step
  double prob_ratio{boltzmann_factors[neighbors_spins]};

  if (r < prob_ratio)
  {
    spin_state(row_index, col_index) *= -1;

    E += flip_energies[neighbors_spins];
    M -= 2*spin_ji;
  }
}

arma::mat MCMC::generate_samples(int num_cycles)
{
  arma::vec E_vec{arma::vec(num_cycles, arma::fill::zeros)};
  arma::vec M_vec{arma::vec(num_cycles, arma::fill::zeros)};

  E_vec(0) = E;
  M_vec(0) = M;

  for (int i{1}; i < num_cycles; ++i)
  {
    for (int j{0}; j < L*L; ++j)
    {
      flip_single_spin();
    }
    E_vec(i) = E;
    M_vec(i) = M;
  }

  arma::mat E_M{arma::mat(num_cycles, 2, arma::fill::zeros)};

  E_M.col(0) = E_vec;
  E_M.col(1) = M_vec;

  return E_M;
}
