#include "particle.hpp"
#include "penningtrap.hpp"
#include <armadillo>
#include <vector>
#include <iostream>
#include <fstream>

int main()
{
  double B0{9.65e1};
  double V0{0.0025*9.65e7};
  double q{1};
  double m{40.078}; //Calcium mass
  double d{0.05e4};

  std::vector<Particle> particles{};

  int n_particles{100};

  arma::arma_rng::set_seed(1034392);
  // Creating particles with random positions and velocities
  for (int i{0}; i < n_particles; ++i)
  {
    arma::vec r{arma::vec(3).randn()*0.1*d};
    arma::vec v{arma::vec(3).randn()*0.1*d};
    Particle p{q, m, r, v};

    particles.push_back(p);
  }

  double total_time{500};
  double dt{1e-2};
  int n_timesteps{static_cast<int>(total_time/dt)};

  arma::vec f_values{{0.1, 0.4, 0.7}};
  // Linearly spaced omega_V values
  arma::vec omega_V_values(arma::regspace(0.2, 0.02, 2.5));


  // Number of particles inside still inside the penning trap after total_time for each
  // combination of f and omega_V.
  arma::mat particles_inside(f_values.size(), omega_V_values.size(), arma::fill::zeros);


  for (int j = 0; j < f_values.size(); ++j)
  {
    // Parallelizing the simulations, since they are completely
    // independent
    #pragma omp parallel for
    for (int i = 0; i < omega_V_values.size(); ++i)
    {
      double f{f_values[j]};
      double omega_V{omega_V_values[i]};

      PenningTrap P{B0, V0, d, particles, f, omega_V};

      P.particle_interactions = false;
      for (int k{0}; k < n_timesteps; ++k)
      {
        P.evolve_RK4(k*dt, dt);
      }
      particles_inside.row(j)[i] = P.particles_inside();
    }
  }

  std::ofstream outf{"txt/particles_inside.txt"};
  //outf << arma::reshape(f_values, 1, f_values.size()) << "\n";
  //outf << arma::reshape(omega_V_values, 1, omega_V_values.size()) << "\n";
  outf << arma::trans(f_values) << "\n";
  outf << arma::trans(omega_V_values) << "\n";
  outf << particles_inside << "\n";

  int n_runs = 5;

  double f = 0.1;
  double delta = 0.05;
  arma::vec omega_V_values_zoomed{arma::linspace(0.44 - delta, 0.44 + delta, 100)};

  arma::mat particles_inside_zoomed(omega_V_values_zoomed.size(), n_runs, arma::fill::zeros);

  for (int j = 0; j < n_runs; ++j)
  {
    std::vector<Particle> particles_zoomed{};

    for (int k = 0; k < n_particles; ++k)
    {
      arma::vec r{arma::vec(3).randn()*0.1*d};
      arma::vec v{arma::vec(3).randn()*0.1*d};
      Particle p{q, m, r, v};

      particles_zoomed.push_back(p);
    }

    arma::vec particles_inside_vec(omega_V_values_zoomed.size(), arma::fill::zeros);

    #pragma omp parallel for
    for (int i = 0; i < omega_V_values_zoomed.size(); ++i)
    {

      double omega_V{omega_V_values_zoomed[i]};
      PenningTrap P{B0, V0, d, particles_zoomed, f, omega_V};
      P.particle_interactions = false;

      for (int k = 0; k < n_timesteps; ++k)
      {
        P.evolve_RK4(k*dt, dt);
      }

      particles_inside_vec[i] = P.particles_inside();
    }
    particles_inside_zoomed.col(j) = particles_inside_vec;
  }

  std::ofstream outf_zoomed_no_int{"txt/particles_inside_zoomed_no_int.txt"};
  outf_zoomed_no_int << f << "\n";
  outf_zoomed_no_int << arma::trans(omega_V_values_zoomed) << "\n";
  outf_zoomed_no_int << particles_inside_zoomed << "\n";

  for (int j = 0; j < n_runs; ++j)
  {
    std::vector<Particle> particles_zoomed{};

    for (int k = 0; k < n_particles; ++k)
    {
      arma::vec r{arma::vec(3).randn()*0.1*d};
      arma::vec v{arma::vec(3).randn()*0.1*d};
      Particle p{q, m, r, v};

      particles_zoomed.push_back(p);
    }

    arma::vec particles_inside_vec(omega_V_values_zoomed.size(), arma::fill::zeros);

    #pragma omp parallel for
    for (int i = 0; i < omega_V_values_zoomed.size(); ++i)
    {

      double omega_V{omega_V_values_zoomed[i]};
      PenningTrap P{B0, V0, d, particles_zoomed, f, omega_V};

      for (int k = 0; k < n_timesteps; ++k)
      {
        P.evolve_RK4(k*dt, dt);
      }

      particles_inside_vec[i] = P.particles_inside();
    }
    particles_inside_zoomed.col(j) = particles_inside_vec;
  }

  std::ofstream outf_zoomed_int{"txt/particles_inside_zoomed_int.txt"};
  outf_zoomed_int << f << "\n";
  outf_zoomed_int << arma::trans(omega_V_values_zoomed) << "\n";
  outf_zoomed_int << particles_inside_zoomed << "\n";

  return 0;
}
