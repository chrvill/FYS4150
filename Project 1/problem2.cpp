#include <fstream>
#include <vector>
#include <cmath>

double u(double x)
{
  // Calculates the exact solution
  return 1 - (1 - std::exp(-10))*x - std::exp(-10*x);
}

void writeToFile(std::ofstream& outf, double x, double u)
{
  // Writes x and u out to file
  outf << std::scientific << x << "\t" << std::scientific << u << "\n";
}

int main()
{
  int n_points{1000};
  double dx{1.0/n_points}; // Distance between points

  // Initializing x and u_x
  std::vector<double> x(n_points, 0.0);
  std::vector<double> u_x(n_points, 0.0);

  std::ofstream outf{"analytical.txt"};
  outf.precision(3); // Defining 3 decimal precision in file output

  for (int i{0}; i < n_points; ++i)
  {
    // Calculating the x value and u(x) value of point
    double x_i{dx*i};
    double u_i{u(x_i)};

    x[i] = x_i;
    u_x[i] = u_i;

    writeToFile(outf, x_i, u_i);
  }

  return 0;
}
