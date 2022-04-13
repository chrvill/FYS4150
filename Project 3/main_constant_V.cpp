#include "particle.hpp"
#include "penningtrap.hpp"
#include <armadillo>
#include <vector>
#include <iostream>
#include <fstream>

int main()
{
  double B0{9.65e1};
  double V0{9.65e8};
  double q{1};
  double m{40.078}; //Calcium mass
  double d{1e4};
  std::vector<Particle> particles{};

  Particle p1{q, m, {1, 0, 1}, {0, 1, 0}};
  particles.push_back(p1);

  PenningTrap P_RK4{B0, V0, d, particles};

  // Objects for writing the positions and velocities to file.
  std::ofstream outf_RK4{"txt/one_particle_positions_RK4.txt"};
  std::ofstream outf_Euler{"txt/one_particle_positions_Euler.txt"};
  std::ofstream outf_interactions_r{"txt/interactions_r.txt"};
  std::ofstream outf_no_interactions_r{"txt/no_interactions_r.txt"};

  std::ofstream outf_interactions_v{"txt/interactions_v.txt"};
  std::ofstream outf_no_interactions_v{"txt/no_interactions_v.txt"};

  double total_time{100};
  double dt{1e-2};

  int n_timesteps{static_cast<int>(total_time/dt)};

  outf_RK4 << dt << "\t" << total_time << "\n";
  for (int i{0}; i < n_timesteps; ++i)
  {
    outf_RK4 << P_RK4.positions << "\n";
    // Using RK4 algorithm to step forward in time
    P_RK4.evolve_RK4(i*dt, dt);
  }

  PenningTrap P_Euler{B0, V0, d, particles};

  outf_Euler << dt << "\t" << total_time << "\n";
  for (int i{0}; i < n_timesteps; ++i)
  {
    outf_Euler << P_Euler.positions << "\n";
    // Using Euler's algorithm to step forward in time
    P_Euler.evolve_Euler(i*dt, dt);
  }

  // Creating a second particle and adding to the simulation.
  Particle p2{q, m, {-1, 0, 1}, {0, -1, 0}};
  particles.push_back(p2);

  // PenningTrap where we neglect the particle interactions
  PenningTrap P_no_interactions{B0, V0, d, particles};
  P_no_interactions.particle_interactions = false;

  outf_no_interactions_r << dt << "\t" << total_time << "\n";
  for (int i = 0; i < n_timesteps; ++i)
  {
    outf_no_interactions_r << P_no_interactions.positions << "\n";
    outf_no_interactions_v << P_no_interactions.velocities << "\n";
    P_no_interactions.evolve_RK4(i*dt, dt);
  }

  // PenningTrap where we also consider the particle interactions
  PenningTrap P_interactions{B0, V0, d, particles};

  outf_interactions_r << dt << "\t" << total_time << "\n";
  for (int i{0}; i < n_timesteps; ++i)
  {
    outf_interactions_r << P_interactions.positions << "\n";
    outf_interactions_v << P_interactions.velocities << "\n";
    P_interactions.evolve_RK4(i*dt, dt);
  }

  arma::vec timesteps{{1e-3, 5e-3, 1e-2, 5e-2, 1e-1}};

  // Removing the second particle
  particles.pop_back();

  PenningTrap P_varying_dt{B0, V0, d, particles};

  // Defining files where the motion of the particle is stored for the
  // case where I vary dt.
  // Really ugly to define 10 different files, but I couldn't think
  // of a better way to do it at first, and now it's too much work
  // fixing it.

  std::ofstream outf_dt1{"txt/varying_dt_0.001_RK4.txt"};
  std::ofstream outf_dt2{"txt/varying_dt_0.005_RK4.txt"};
  std::ofstream outf_dt3{"txt/varying_dt_0.01_RK4.txt"};
  std::ofstream outf_dt4{"txt/varying_dt_0.05_RK4.txt"};
  std::ofstream outf_dt5{"txt/varying_dt_0.1_RK4.txt"};

  std::ofstream outf_dt6{"txt/varying_dt_0.001_Euler.txt"};
  std::ofstream outf_dt7{"txt/varying_dt_0.005_Euler.txt"};
  std::ofstream outf_dt8{"txt/varying_dt_0.01_Euler.txt"};
  std::ofstream outf_dt9{"txt/varying_dt_0.05_Euler.txt"};
  std::ofstream outf_dt10{"txt/varying_dt_0.1_Euler.txt"};

  std::vector<std::ofstream*> files_RK4{{&outf_dt1, &outf_dt2, &outf_dt3, &outf_dt4, &outf_dt5}};
  std::vector<std::ofstream*> files_Euler{{&outf_dt6, &outf_dt7, &outf_dt8, &outf_dt9, &outf_dt10}};

  // Varying the timestep and simulating the motion of the particle
  // for each case, both using the RK4 and Euler algorithms.
  for (int j{0}; j < timesteps.size(); ++j)
  {
    double dt{timesteps[j]};
    int n_timesteps{static_cast<int>(total_time/dt)};

    std::ofstream* outf_RK4{files_RK4[j]};
    *outf_RK4 << dt << "\t" << total_time << "\n";

    // Simulating for each dt-value using RK4
    for (int i{0}; i < n_timesteps; ++i)
    {
      *outf_RK4 << P_varying_dt.positions << "\n";
      P_varying_dt.evolve_RK4(i*dt, dt);
    }
    // Resetting so we start at the initial conditions again
    P_varying_dt.reset();

    std::ofstream* outf_Euler{files_Euler[j]};
    *outf_Euler << dt << "\t" << total_time << "\n";

    // Simulating for each dt-value using Euler
    for (int i{0}; i < n_timesteps; ++i)
    {
      *outf_Euler << P_varying_dt.positions << "\n";
      P_varying_dt.evolve_Euler(i*dt, dt);
    }
    // Resetting so we start at the initial conditions again
    P_varying_dt.reset();
  }

  std::ofstream constants_outf{"txt/constants.txt"};
  constants_outf << q << "\n" << m << "\n" << B0 << "\n" << V0 << "\n" << d;

  return 0;
}
