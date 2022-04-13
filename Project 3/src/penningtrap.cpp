#include "penningtrap.hpp"
#include <cmath>

double k_e = 1.38935333e5;

PenningTrap::PenningTrap(double B_0, double V_0, double d_, std::vector<Particle> Particles,
                         double f_, double omegaV):
            B0{B_0}, V0{V_0}, d{d_}, particles{Particles}, f{f_}, omega_V{omegaV}
{
  n_particles = static_cast<int>(particles.size());

  positions = arma::mat(3, n_particles, arma::fill::zeros);
  velocities = arma::mat(3, n_particles, arma::fill::zeros);

  // Each column in the positions matrix is the position
  // of one particle. Likewise for the velocities matrix.
  for (int i{0}; i < n_particles; ++i)
  {
    positions.col(i) = particles[i].r;
    velocities.col(i) = particles[i].v;
  }
}

arma::vec PenningTrap::external_E_field(const arma::vec& r, double t) const
{
  arma::vec E(3, arma::fill::zeros);

  double x{r(0)};
  double y{r(1)};
  double z{r(2)};

  // Time varying potential. f and omega_V are only non-zero
  // when finding resonant-frequencies.
  double V{V0*(1 + f*std::cos(omega_V*t))};

  double constant{V/(d*d)};

  E(0) = constant*x;
  E(1) = constant*y;
  E(2) = -2*constant*z;

  return E;
}

arma::vec PenningTrap::external_B_field(const arma::vec& r, double t) const
{
  arma::vec B(3, arma::fill::zeros);

  B(2) = B0;

  return B;
}

//arma::vec PenningTrap::force_particle(int i, int j, const PenningState& penningstate) const
arma::vec PenningTrap::force_particle(int i, int j, const arma::mat& positions) const
{
  // Positions of particle i and j in the PenningState given by penningstate.
  arma::vec r_i{positions.col(i)};
  arma::vec r_j{positions.col(j)};
  //arma::vec r_i{penningstate.positions.col(i)};
  //arma::vec r_j{penningstate.positions.col(j)};

  double q_i{particles[i].q};
  double q_j{particles[j].q};

  arma::vec j_to_i{r_i - r_j};
  //double distance_squared{arma::dot(j_to_i, j_to_i)};
  double distance{arma::norm(j_to_i)};

  // Force from particle j on particle i
  return k_e*q_i*q_j*(j_to_i)/(std::pow(distance, 3));
  //return k_e*q_i*q_j*j_to_i/(std::pow(distance_squared, 3/2));
}

//arma::vec PenningTrap::total_force_particles(int i, const PenningState& penningstate) const
arma::vec PenningTrap::total_force_particles(int i, const arma::mat& positions) const
{
  arma::vec F{3, arma::fill::zeros};

  #pragma omp parallel for
  for (int j = 0; j < n_particles; ++j)
  {
    if (j != i)
    {
      F += force_particle(i, j, positions);
    }
  }

  return F;
}

//arma::vec PenningTrap::total_force_external(int i, const PenningState& penningstate, double t) const
arma::vec PenningTrap::total_force_external(int i, const arma::mat& positions, const arma::mat& velocities, double t) const
{
  arma::vec F(3, arma::fill::zeros);

  //arma::vec r{penningstate.positions.col(i)};
  arma::vec r{positions.col(i)};

  // Only calculating forces from external fields when the
  // particle is inside the penning trap. Using the square of the length here
  // because sqrt is computationally expensive
  double q{particles[i].q};

    //arma::vec v{penningstate.velocities.col(i)};
  arma::vec v{velocities.col(i)};

  arma::vec E{external_E_field(r, t)};
  arma::vec B{external_B_field(r, t)};

  F += q*E + q*arma::cross(v, B);


  return F;
}

//arma::vec PenningTrap::total_force(int i, const PenningState& penningstate, double t) const
arma::vec PenningTrap::total_force(int i, const arma::mat& positions, const arma::mat& velocities, double t) const
{
  arma::vec F(3, arma::fill::zeros);

  arma::vec r{positions.col(i)};

  double length_squared{arma::dot(r, r)};
  if (length_squared < d*d)
  {
    F += total_force_external(i, positions, velocities, t);

    if (particle_interactions)
    {
      F += total_force_particles(i, positions);
    }
  }
  return F;
}

//arma::mat PenningTrap::acceleration(const PenningState& penningstate, double t) const
arma::mat PenningTrap::acceleration(const arma::mat& positions, const arma::mat& velocities, double t) const
{
  arma::mat accelerations(3, n_particles, arma::fill::zeros);

  for (int i{0}; i < n_particles; ++i)
  {
    double m{particles[i].m};
    //arma::vec F{total_force(i, penningstate, t)};
    arma::vec F{total_force(i, positions, velocities, t)};
    // Acceleration of particle i is column i in the
    // accelerations matrix.
    accelerations.col(i) = 1.0/m*F;
  }
  return accelerations;
}

void PenningTrap::evolve_RK4(double t, double dt)
{
  // Avoids having to do dt/2.0 many different times in the same timestep
  double dt_div_2{dt/2.0};

  // Positions, velocities and accelerations of all particles at the beginning
  // of the timestep.
  arma::mat k1_r{positions};
  arma::mat k1_v{velocities};
  arma::mat k1_a{acceleration(k1_r, k1_v, t)};

  // Positions, velocities and accelerations after using k1_r, k1_v, k1_a to step
  // forward half a timestep.
  arma::mat k2_r{positions + k1_v*dt_div_2};
  arma::mat k2_v{velocities + k1_a*dt_div_2};
  arma::mat k2_a{acceleration(k2_r, k2_v, t + dt_div_2)};

  // Positions, velocities and accelerations after using k2_r, k2_v, k2_a to step
  // forward half a timestep.
  arma::mat k3_r{positions + k2_v*dt_div_2};
  arma::mat k3_v{velocities + k2_a*dt_div_2};
  arma::mat k3_a{acceleration(k3_r, k3_v, t + dt_div_2)};

  // Positions, velocities and accelerations after using k3_r, k3_v, k3_a to step
  // forward a full timestep.
  arma::mat k4_r{positions + k3_v*dt};
  arma::mat k4_v{velocities + k3_a*dt};
  arma::mat k4_a{acceleration(k4_r, k4_v, t + dt)};

  // RK4-weighting of the different derivatives
  arma::mat dvdt{1.0/6*(k1_a + 2*(k2_a + k3_a) + k4_a)};
  arma::mat drdt{1.0/6*(k1_v + 2*(k2_v + k3_v) + k4_v)};

  // Updating the member variables positions and velocities
  positions += drdt*dt;
  velocities += dvdt*dt;
}

void PenningTrap::evolve_Euler(double t, double dt)
{
  arma::mat a{acceleration(positions, velocities, t)};

  positions += velocities*dt;
  velocities += a*dt;
}

void PenningTrap::reset()
{
  for (int i{0}; i < n_particles; ++i)
  {
    positions.col(i) = particles[i].r;
    velocities.col(i) = particles[i].v;
  }
}

int PenningTrap::particles_inside() const
{
  // Counter variable
  int n{0};

  for (int i{0}; i < n_particles; ++i)
  {
    arma::vec r{positions.col(i)};
    if (arma::norm(r) < d)
    {
      ++n;
    }
  }
  return n;
}
