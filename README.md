# Adaptive-RWA Solver

## requirements: installation of qutip (http://qutip.org/docs/4.1/installation.html)

The adaptive-rwa solver is a master equation solver that approximates the solution to the Lindblad master equation with
a steady-state solution. The steady-state is determined via our "adaptive rotating wave approximation", where irrelevant drive
terms have been neglected. It is an iterative perturbative algorithm, which involves solving a set of inhomogeneous linear equations
for each iteration.

## use: 
1) create a system Hamiltonian as a quantum object. For example, this could be a 2-level system (qubit) or a simple harmonic oscillator, or even a superconducting circuit coupled to a resonator (transmon + resonator). 
2) create a drive coupling term as a quantum object (the total Hamiltonian is H_system + (V*exp(i\omega_d t) + h.c.), the coupling term is V). Make sure this drive Hamiltonian has the same dimensions as the system Hamiltonian. 
3) create a table (list of lists or numpy array) of decoherence rates, \gamma_{nm}, where n and m label eigenstates of H_system. This should be a D x D table, where D is the Hilbert space size. 
4) choose a drive angular frequency, \omega_d.
5) initialize a driven open system object: driven_open_system = DrivenOpenSystem(system_hamiltonian, decoherence_rates, drive_hamiltonian, angular_frequency). 
6) solve for the steady-state density matrix: solution = driven_open_system.recursive_steadystate_solve(v_min, it_max,neglected_term_info). v_min is a lower bound cutoff matrix element size for each drive term that solver considers (default is 0). it_max is maximum number of iterations solver will loop through (default is 10). neglected_term_info is set to False by default; if set to True, then solution will consist of i) the steady-state density matrix ii) a list of all neglected drive terms (as QObj type) and iii) a measure of the deviation from the exact asympotic solution, calculated from the relevance parameters of these neglected terms. If False, then it will just output the steady-state density matrix

