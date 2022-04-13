# FYS4150_Project_3

Code for project 3 in FYS4150

------------------

#### include

Contains particle.hpp and penningtrap.hpp. Both are C++ code. The first declares the Particle class, while the second declares the PenningTrap class and its methods.

------------------

#### src

Contains penningtrap.cpp. C++ code defining the member functions of the PenningTrap class.

------------------

#### main_constant_V.cpp

C++ code for running the simulations where the externally applied electric field is constant.

Build command: g++ main_constant_V.cpp src/penningtrap.cpp -I include -O3 -fopenmp -larmadillo -lgomp -o main_constant_V.exe

Run command: ./main_constant_V.exe

This program outputs all the relevant simulation data to files. It takes a bit of time to run, but is quicker with the optimization and parallelization that is used in the build command. The introduction to parallelization on the course webpage said to use -lomp when linking, but that didn't work for me. I had to use -lgomp. So you may have to change that.

------------------

#### main_varying_V.cpp

C++ code for running the simulations where the externally applied filed varies in time.

Build command: g++ main_varying_V.cpp src/penningtrap.cpp -I include -O3 -fopenmp -larmadillo -lgomp -o main_varying_V.exe

Run command: ./main_varying_V.exe

This program outputs all the relevant simulation data to files. This one takes several hours to run, but all the text files it outputs are given in the txt-folder.

------------------

#### plot.py

Python code or plotting the motion of the single-particle system, two-particle system and the 100-particle system
with time-varying externally applied electric field. All the plots it produces are given in the images-folder, but the program does not take long to run, so it is easy to check that the code works.

------------------
