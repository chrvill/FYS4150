#ifndef MATRIXEQ_HPP
#define MATRIXEQ_HPP

#include <vector>
#include <functional>

class MatrixEq
{
private:
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> c;

public:
  MatrixEq(std::vector<double> a_, std::vector<double> b_, std::vector<double> c_):
    a{a_}, b{b_}, c{c_} {}

  // Both functions return the time taken to run the algorithm
  double solveGeneral(const std::function<double(double)>& f, double x_min, double x_max, int n_points, double v_0 = 0.0, double v_last = 0.0);
  double solveSpecial(const std::function<double(double)>& f, double x_min, double x_max, int n_points, double v_0 = 0.0, double v_last = 0.0);

};

#endif
