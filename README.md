# Steady-State-Project

The steady state recursion module contains tools necessary to describe the dynamics of an open, driven system.

The primary tool is a master equation solver that approximates the solution to the Lindblad master equation with
a steady-state solution. The steady-state is determined via our "adaptive rotating wave approximation", where irrelevant drive
terms have been neglected. It is an iterative perturbative algorithm, which involves solving a set of inhomogeneous linear equations
for each iteration. 

The secondary tool is also a master equation solver that solves the Lindblad ME exactly. This is a standard numerical
differential equation solver. The solution is automatically time averaged.

The use of this module is as follows:
1) create a class instance of type "DrivenOpenSystem" with a specified hamiltonian, dissipation rates, drive coupling, drive frequency
2) call either of the master equation solvers from this class which will output a density matrix in the long-time limit.
