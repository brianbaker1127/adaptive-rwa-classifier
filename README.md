# Adaptive-RWA Solver
The adaptiveRWA_solver.py module contains tools necessary to describe the dynamics of an open, driven system.

## requirements: installation of qutip (http://qutip.org/docs/4.1/installation.html)

The primary tool is a master equation solver that approximates the solution to the Lindblad master equation with
a steady-state solution. The steady-state is determined via our "adaptive rotating wave approximation", where irrelevant drive
terms have been neglected. It is an iterative perturbative algorithm, which involves solving a set of inhomogeneous linear equations
for each iteration.

## use: 
1) create a system Hamiltonian as a quantum object. For example, this could be a 2-level system (qubit) or a simple harmonic oscillator, or even a superconducting circuit coupled to a resonator (transmon + resonator). 
2) create a drive coupling term as a quantum object (the total Hamiltonian is H_system + (V*exp(i\omega_d t) + h.c.), the coupling term is V). 
3) create a table (list of lists or numpy array) of decoherence rates, \gamma_{nm}, where n and m label eigenstates of H_system. 
4) choose a drive frequency, \omega_d. 
5) initialize a driven open system object: driven_open_system = DrivenOpenSystem(H_system, decoherence_rates, drive_coupling, drive_frequency). 
6) solve for the steady-state solution: solution = driven_open_system.recursive_steadystate_solve(v_min, it_max). v_min is a lower bound cutoff matrix element size for V that solver considers (default is 0). it_max is maximum number of iterations solver will loop through (default is 10). 

