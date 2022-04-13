# FYS4150_Project_2

Code for project 2 in FYS4150

include/symmetric_matrix.hpp
-----------------------------
C++ code declaring the SymmetricMatrix class and its methods.

src/symmetric_matrix.cpp
-----------------------------
C++ code defining the methods in the SymmetricMatrix class.

main.cpp
-----------------------------
C++ code running the calculations for all the different matrices we are considering, utilizing the SymmetricMatrix class.

Build command: g++ main.cpp src/symmetric_matrix.cpp -I include -larmadillo -o main.exe

Run command: ./main.exe

This program prints the 6x6 matrix given in the project description along with the maximum off diagonal element and the indices corresponding to this element.

problem3.cpp
-----------------------------
C++ code calling armadillo's eig_sym function to calculate the eigenvalues and eigenvectors of a 6x6 tridiagonal matrix.
Prints the eigenvectors and eigenvalues from armadillo, for N = 6.

Build command: g++ problem3.cpp -larmadillo -o problem3.exe

Run command: ./problem3.exe

plot.py
-----------------------------
Python code for plotting two things:
- three eigenvectors for a 10x10 and 100x100 tridiagonal matrix
- the number of similarity transformations necessary to converge in the Jacobi rotation method, for matrices of different sizes.

Prints the eigenvalues and eigenvectors calculated from the analytical formulas, for N = 6. These printouts can then be compared with the ones from problem3.cpp, and they do indeed agree.

Run command: python3 plot.py

-----------------------------

The images and textFiles directories are just for sorting the images and textFiles into separate folders. The .txt files created by the C++ programs are saved in the textFiles folder and the plots are saved in the images folder. Currently the folders are both filled with the files my code produced because I'm too lazy to delete them. They could just as well be empty folders. I'm not sure if the code would work if the folders didn't exist, since when the code runs the plots are saved using figure.savefig("images/filename.txt"), so I don't know what would happen if the folder didn't already exist. It might just create the folder, but I don't know and can't be bothered trying it out.
