# FYS4150_Project_4

Code for project 4 in FYS4150

-----------------------
### include 

Contains MCMC.hpp. C++ code declaring the MCMC class with its methods. This is used to generate samples for a given lattice size, temperature etc.

----------------------

### src

Contains MCMC.cpp. C++ code defining the methods of the MCMC class. 

----------------------

### fixed_temp.cpp

C++ code defining a function for generating samples for a given lattice size, number of MC cycles and a fixed temperture. 

----------------------

### vary_temp.cpp

C++ code defining a function for generating samples for a given lattice size, number of MC cycles, but this time for a specified range of temperatures.

----------------------

### main.cpp

C++ code calling the functions from fixed_temp.cpp and vary_temp.cpp in order to sample the lattices we want. 

Build command: g++ main.cpp vary_temp.cpp fixed_temp.cpp src/MCMC.cpp -I include -O3 -fopenmp -larmadillo -lgomp -o main.exe

Run command: ./main.exe

----------------------

### plot.py

Python code for plotting all the expectation values as functions of MC cycles and temperatures. 

Run command: python3 plot.py

----------------------


Did not include the timing of the speed-up factor from parallelizing in the report, because it did not feel like it fit. It is calculated and printed by main.exe, but it felt out of place in the report. 
