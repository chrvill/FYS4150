#include "matrixEq.hpp"
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>

double f(double x)
{
  return 100*std::exp(-10*x);
}

int main()
{
  // Upper and lower bounds of x-range
  double x_min{0.0};
  double x_max{1.0};

  // Value of v at upper and lower x-bounds
  double v_0{0.0};
  double v_last{0.0};

  // Vector containing different values for n
  std::vector<int> n_points{{10, 100, 1000, 10000, 100000, 1000000, 10000000}};

  std::ofstream outfGeneral;
  outfGeneral.open("timeGeneral.txt", std::ios_base::app);

  // Solving for each value of n
  for (int n: n_points)
  {
    // Diagonals of matrix
    std::vector<double> a(n - 3, -1);
    std::vector<double> b(n - 2, 2);
    std::vector<double> c(n -3, -1);

    MatrixEq M{a, b, c};

    double duration_seconds{M.solveGeneral(f, x_min, x_max, n)};

    outfGeneral << duration_seconds << "\t\t";
  }
  outfGeneral << "\n";

  // Object that will be used to write time information
  // to file
  std::ofstream outfSpecial;
  // std::ios_base::app allows appending to file instead
  // of overwriting
  outfSpecial.open("timeSpecial.txt", std::ios_base::app);

  MatrixEq M{{}, {}, {}};

  for (int n: n_points)
  {
    double duration_seconds{M.solveSpecial(f, x_min, x_max, n)};
    outfSpecial << duration_seconds << "\t\t";
  }
  outfSpecial << "\n";

  return 0;
}
