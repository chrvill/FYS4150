#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include <armadillo>

class Particle
{
private:
  double q;
  double m;
  arma::vec v;

public:
  arma::vec r;
  Particle(double charge, double mass, arma::vec position, arma::vec velocity):
          q{charge}, m{mass}, r{position}, v{velocity} {}

  friend class PenningTrap;
};

#endif
