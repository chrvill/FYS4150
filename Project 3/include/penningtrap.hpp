#ifndef PENNINGTRAP_HPP
#define PENNINGTRAP_HPP

#include "particle.hpp"
#include <vector>

class PenningTrap
{
public:
  double B0;
  double V0;
  double d;
  std::vector<Particle> particles;
  int n_particles;

  // Contains the position of each particle. Column i is the
  // position of particle i.
  arma::mat positions;

  // Contains the velocity of each particle. Column i is the
  // velocity of particle i.
  arma::mat velocities;

  bool particle_interactions{true};

  // The amplitude and angular frequency respectively of the
  // external time-dependent potential. They are zero before we
  // start experimenting with this oscillating field.
  double f;
  double omega_V;

  // Constructor
  PenningTrap(double B_0, double V_0, double d_, std::vector<Particle> Particles, double f_ = 0.0, double omegaV = 0.0);

  // Calculates external electric field at a position r at time t
  arma::vec external_E_field(const arma::vec& r, double t) const;

  // Calculates external magnetic field at a position r at time t
  arma::vec external_B_field(const arma::vec& r, double t) const;

  // Calculates the force on particle i from particle j when all the particle positions
  // are given in positions. Column i in positions contains the position of particle i.
  // Positions (and velocities in other functions) can be the positions (and velocities)
  // at an intermediate step in the RK4 algorithm.
  arma::vec force_particle(int i, int j, const arma::mat& positions) const;

  // Calculates the total force on particle i from the external electric and magnetic fields
  // at time t and for particle positions and velocities given in the matrices of the same names.
  // Column i in velocities contains the velocity of particle i.
  arma::vec total_force_external(int i, const arma::mat& positions, const arma::mat& velocities, double t) const;

  // Calculates the total force on particle i from all other particles for a system in
  // with particles positions in the positions matrix
  arma::vec total_force_particles(int i, const arma::mat& positions) const;

  // Calculates the total force on particle i at time t from both other particles and the
  // external EM-field, for particles in the given positions and velocities
  arma::vec total_force(int i, const arma::mat& positions, const arma::mat& velocities, double t) const;

  // Calculates the acceleration of all the particles in the simulation at time t where the
  // particle positions and velocities are given in the argument matrices. The function returns a 3xN matrix,
  // where N is the number of particles. Column i in the matrix contains the acceleration of the i-th particle
  // at the given time and for the given positions and velocities.
  arma::mat acceleration(const arma::mat& positions, const arma::mat& velocities, double t) const;

  // Evolves the system one timestep forward in time using the RK4 algorithm.
  // In the RK4 algorithm we use the derivatives k1, k2, k3 and k4 for the velocity
  // and acceleration. And here I use different PenningStates to calculate this.
  // So for updating velocity, k1 is the acceleration at a PenningState evaluated at
  // the current timestep, k2 is the acceleration at a PenningState evaluated at the
  // current timestep plus half a step forward with k1, and so on.

  // Evolves the system one timestep forward in time using the RK4 algorithm.
  void evolve_RK4(double t, double dt);

  // Evolves the system one timestep forward in time using the Euler algorithm.
  void evolve_Euler(double t, double dt);

  // Resets the positions and velocities of all particles to their initial values.
  // It is used when simulating the same system for different values of the timestep.
  void reset();

  // Returns the number of particles that is inside the penning trap. This is determined
  // by finding the number of particles for which the distance to the origin is smaller than d.
  int particles_inside() const;
};

#endif
