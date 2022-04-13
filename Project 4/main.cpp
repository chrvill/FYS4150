#include "MCMC.hpp"
#include <chrono>
#include <iostream>

// Forward declaring functions defined in other files.
void fixed_temperature(double T, int L, int n_cycles, int base_seed, bool ordered);
void varying_temperature(double min_temp, double max_temp, int N_temps, int L, int n_cycles,
                         int burn_in_stop, int base_seed, bool parallelize = true);

int main()
{
  int base_seed{2030289};

  /*
  Code for the 2x2 lattice:
  */
  
  arma::vec n_cycles_2x2{100, 1000, 10000};

  int L{2};
  double T{2.4};

  fixed_temperature(T, L, 10000, base_seed, false);


  for (int n: n_cycles_2x2)
  {
    // We don't remove the burn-in steps here, so just let burn_in_stop = 0.
    varying_temperature(1.0, 20, 100, L, n, 0, base_seed);
  }

  /*
  Code for the 20x20 lattice:
  */

  auto t0 = std::chrono::high_resolution_clock::now();

  L = 20;
  // First running a parallelized temperature loop, then a
  // non-parallelized loops to find speed-up factor. Running two
  // times for each and take average.
  varying_temperature(1.0, 20, 100, L, 1000, 0, base_seed);
  auto t1 = std::chrono::high_resolution_clock::now();

  varying_temperature(1.0, 20, 100, L, 1000, 0, base_seed);
  auto t2 = std::chrono::high_resolution_clock::now();

  varying_temperature(1.0, 20, 100, L, 1000, 0, base_seed, false);
  auto t3 = std::chrono::high_resolution_clock::now();

  varying_temperature(1.0, 20, 100, L, 1000, 0, base_seed, false);
  auto t4 = std::chrono::high_resolution_clock::now();

  double time_parallelized = (std::chrono::duration<double>(t2 - t0).count())/2;

  double time_unparallelized = (std::chrono::duration<double>(t4 - t2).count())/2;

  double speed_up = time_unparallelized/time_parallelized;

  std::cout << "Speed-up factor: " << speed_up << "\n";

  fixed_temperature(T, L, 100000, base_seed, false); // Unordered init. config.
  fixed_temperature(T, L, 100000, base_seed, true);  // Ordered init. config.

  // Changing T and collecting samples for the 20x20 lattice agin.
  T = 1.0;

  fixed_temperature(T, L, 100000, base_seed, false); // Unordered init. config.
  fixed_temperature(T, L, 100000, base_seed, true); // Ordered init. config.

  /*
  Code for the 40, 60, 80 and 100 lattices.
  */

  // Performing 1 000 000 MC cycles
  int n_cycles{1000000};

  double min_temp{2.1};
  double max_temp{2.4};
  int N_temps{100};

  // Using a burn-in time of 200 000 cycles (takes quite a bit of time for M to stabilize)
  int burn_in_stop{static_cast<int>(n_cycles*0.2)};

  arma::vec L_values{40, 60, 80, 100};

  // Temperature used when calculating burn-in time
  double temp_fixed{2.4};

  // How to waste ~15+ hours of your life:
  for (int L: L_values)
  {
    fixed_temperature(temp_fixed, L, n_cycles, base_seed, false);
    varying_temperature(min_temp, max_temp, N_temps, L, n_cycles, burn_in_stop, base_seed);
  }

  return 0;
}
