#ifndef MCMC_HPP
#define MCMC_HPP

#include <armadillo>
#include <map>
#include <cmath>

class MCMC
{
private:
  int L;    // Number of spins along each axis
  double T; // Temperature
  // Initializing energy and magnetization
  double E;
  double M;

  // Initializing RNG and probability distributions for integers and doubles
  std::mt19937 generator;
  std::uniform_int_distribution<int> int_dist;
  std::uniform_real_distribution<double> real_dist;

  double beta; // Inverse temperature

  // These are used to calculate the changes in energy after a flip. The index that
  // should be used when accessing an element in the maps are the sum of the magnetizations
  // of the neighboring spins.
  std::map<int, double> flip_energies;
  std::map<int, double> boltzmann_factors;

public:
  // The configuration of spins
  arma::mat spin_state;

  // Constructor. The ordered argument specifies whether the
  // spin_state should be initialized as ordered or unordered.
  // ordered = true  -> all spins initialized pointing up
  // ordered = false -> all spins initialized randomly 
  MCMC(int N_spins, double temperature, int seed, bool ordered);

  // Function for calculating the energy at the beginning.
  // This is not used afterwards, because we can just use the known
  // changes in energy after a flip.
  void compute_initial_energy();

  // Function for calculating the magnetization at the beginning.
  // Same as for energy, this is not used afterwards. Can just
  // use known changes in magnetization after a flip.
  void compute_initial_magnetization();

  // Attempts a spin flip. If the flip is accepted then the
  // function also updates spin_state and calculates new
  // energy and magnetization
  void flip_single_spin();

  // Performs num_cycles cycles and stores the energy after
  // each sampling. The returned matrix has shape (num_cycles, 2).
  // First column is energy for each sample configuration, second
  // column is magnetization.
  arma::mat generate_samples(int num_cycles);
};

#endif
