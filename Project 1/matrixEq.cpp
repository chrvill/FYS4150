#include "matrixEq.hpp"
#include <fstream>
#include <iostream>
#include <chrono>

double MatrixEq::solveGeneral(const std::function<double(double)>& f, double x_min, double x_max, int n_points, double v_0, double v_last)
{
  // Step size
  double h{(x_max - x_min)/(n_points - 1.0)};

  std::vector<double> x(n_points, 0.0);
  std::vector<double> g(n_points - 2, 0.0);

  // Filling x and g
  for (int i{0}; i < n_points; ++i)
  {
    x[i] = x_min + i*h;

    // g only has values on the interior points
    if (i > 0 && i < (n_points - 1))
    {
      g[i - 1] = h*h*f(x[i]);
    }
  }

  // Adding v_0 and v_last to the first and last elements of g
  g[0] += v_0;
  g.back() += v_last;

  // Number of elements in b
  int b_size{static_cast<int>(b.size())};

  std::vector<double> b_tilde{b};

  // Initializing right hand side
  std::vector<double> g_tilde{g};

  // The solution to the matrix equation
  // Copying b just to get same length as b
  std::vector<double> v{x};

  // Time at beginning of algorithm
  auto t1 = std::chrono::high_resolution_clock::now();

  // From 2nd element of b to the last element
  for (int i{1}; i < (b_size); ++i)
  {
    // a[i - 1] because a is a = [a_2, a_3, ...,]
    // So a_2 in the formula means a[0]
    b_tilde[i] = b[i] - a[i - 1]/b_tilde[i - 1]*c[i - 1];
    g_tilde[i] = g[i] - a[i - 1]/b_tilde[i - 1]*g_tilde[i - 1];
  }

  // Assigning second to last element of v (last interior point)
  v[v.size() - 2] = g_tilde[b_size - 1]/b_tilde[b_size - 1];

  // Filling v, starting from second to last element (which has index b_size - 2)
  for (int i{b_size - 2}; i >= 0; --i)
  {
    v[i + 1] = (g_tilde[i] - c[i]*v[i + 2])/b_tilde[i];
  }

  // Time at end of algorithm
  auto t2 = std::chrono::high_resolution_clock::now();

  // Time taken to run algorithm
  double duration_seconds = std::chrono::duration<double>(t2 - t1).count();

  // Enforcing the boundary conditions
  v[0] = v_0;
  v.back() = v_last;

  // Object used to write the x-values and v-values to file
  std::string outputName{"v_solution" + std::to_string(n_points) + ".txt"};
  std::ofstream outf{outputName};

  for (int i{0}; i < static_cast<int>(v.size()); ++i)
  {
    outf << x[i] << "\t" << v[i] << "\n";
  }

  return duration_seconds;
}

double MatrixEq::solveSpecial(const std::function<double(double)>& f, double x_min, double x_max, int n_points, double v_0, double v_last)
{
  // Step size
  double h{(x_max - x_min)/(n_points - 1.0)};

  std::vector<double> x(n_points, 0.0);
  std::vector<double> g(n_points - 2, 0.0);

  // Filling x and g
  for (int i{0}; i < n_points; ++i)
  {
    x[i] = x_min + i*h;

    // g only has values on the interior points
    if (i > 0 && i < (n_points - 1))
    {
      g[i - 1] = h*h*f(x[i]);
    }
  }

  // Adding v_0 and v_last to the first and last elements of g
  g[0] += v_0;
  g.back() += v_last;

  //int b_size{static_cast<int>(b.size())};
  int b_size{static_cast<int>(x.size() - 2)};

  std::vector<double> g_tilde{g};
  g_tilde[0] = 0.5*g[0];

  std::vector<double> v{x};

  // Time before start of algorithm
  auto t1 = std::chrono::high_resolution_clock::now();

  for (int i{1}; i < (b_size); ++i)
  {
    // Somewhat weirdly written to avoid having to convert i + 1
    // and i + 2 to double
    g_tilde[i] = (i + 1)*(g[i] + g_tilde[i - 1])/(i + 2);
  }

  // Assigning second to last element of v (last interior point)
  v[v.size() - 2] = g_tilde[b_size - 1];

  // Filling v, starting from second to last element (which has index b_size - 2)
  for (int i{b_size - 1}; i > 0; --i)
  {
    // Again written to avoid having to do floating point operations with i
    v[i] = g_tilde[i - 1] + i*v[i + 1]/(i + 1);
  }

  // Time at end of algorithm
  auto t2 = std::chrono::high_resolution_clock::now();

  // Calculating time of algorithm
  double duration_seconds = std::chrono::duration<double>(t2 - t1).count();

  // Enforcing the boundary conditions
  v[0] = v_0;
  v.back() = v_last;

  // Object used to write the x-values and v-values to file
  std::string outputName{"v_solution" + std::to_string(n_points) + "_special.txt"};
  std::ofstream outf{outputName};

  for (int i{0}; i < static_cast<int>(v.size()); ++i)
  {
    outf << x[i] << "\t" << v[i] << "\n";
  }

  return duration_seconds;
}
