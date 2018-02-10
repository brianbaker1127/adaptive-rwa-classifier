"""
The 'steady_state_recursion' module provides the routines needed to solve
for the density matrix for an open system with hamiltonian driven by an external 
field with oscillatory frequency, omega. The solving approach is to determine the
rotating frame deemed best suited to solve the steady state master equation in, and
then update that frame recursively until a frame is converged upon. 
"""
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from sc_qubits import sc_qubits2 as qubit  # import the superconducting circuit module

import multiprocessing as mp
import time
from numpy import linalg as la
from functools import partial

from qutip import * ## Import only qutip packages you need (change this)


class DrivenOpenSystem:

    def __init__(self, hamiltonian, decoherence_rates, drive_coupling_operator, drive_strength, frequency):
        self.hamiltonian = hamiltonian
        self.dim = self.hamiltonian.shape[0]
        self.gamma_table = decoherence_rates
        self.v = drive_coupling_operator
        self.omega = frequency
        self.zeta = drive_strength
        eigensystem = self.hamiltonian.eigenstates()
        self.evecs = eigensystem[1]
        self.evals = eigensystem[0]

        self.jump_ops = self.jump_operators(self.gamma_table)
        self.rho0 = steadystate(self.hamiltonian, self.jump_ops)
        self.density_matrix = [[(self.evecs[a].dag() * self.rho0 * self.evecs[b])[0] for b in range(self.dim)] for a in range(self.dim)]
        self.density_matrix = np.asarray(self.density_matrix)

        self.integer_list = [0 for k in range(self.dim)]
        self.drive_term_list = []
        self.rotating_frame_hamiltonian = self.hamiltonian

        self.Gamma = self.transition_rates_table(self.gamma_table)

    # run steady state master equation recursively until the same terms in v are kept
    def recursive_steadystate_solve(self, v_min, it_max):
        
        evecs = self.evecs
        evals = self.evals
        iteration_count = 0
        while iteration_count <= it_max:
            # Construct a rotating frame transformation with corresponding integers, hamiltonian, and drive terms
            self.rotating_frame_hamiltonian, self.integer_list, new_drive_term_list = self.frame_rotate(v_min)
           
            if new_drive_term_list == self.drive_term_list:
                break
            self.drive_term_list = new_drive_term_list[:]

            # Solve the steady-state master equation in chosen rotating frame
            self.rho0 = steadystate(self.rotating_frame_hamiltonian, self.jump_ops)
            # Update density matrix to determine new rotating frame
            self.density_matrix = [[(evecs[a].dag() * self.rho0 * evecs[b])[0] for b in range(self.dim)] for a in range(self.dim)]
            self.density_matrix = np.asarray(self.density_matrix)
            iteration_count += 1
            print(self.integer_list)
        if iteration_count > it_max:
                warnings.warn("Solution may be non-convergent!")

        v_truncated = sum(self.drive_term_list)
        #V1 = sum(results[3])
        print(iteration_count)
        #error_measure = results[4]
        #[density_matrix, V0, V1, error_measure]
        return self.rho0   

    def mesolve_with_time_averaging(self, intial_state, integration_time_list, averaging_time, averaging_start, measurements, args={}, options=None):
        # Construct time dependent hamiltonian:
        pass
        # returns density matrix, measurements, and measurement standard deviations
    def jump_operators(self, gamma_table):

        evecs = self.evecs
        evals = self.evals
       
        A = []
       
        for k in range(self.dim):
            for j in range(self.dim):
                A.append(np.sqrt(abs(gamma_table[j][k])) * evecs[k] * evecs[j].dag())
        
                
        return A

        """
    This auxillary function returns a D^2 x D^2 size diagonal array 
    where the diagonal elements are the total net decoherence rate out of
    an eigenstate of H_0 and into another one: \Gamma = 1/2 * (\Gamma(b) - \Gamma(a)).
    This quantity appears when solving for the first order perturbative correction to 
    the density matrix.
    """
    def transition_rates_table(self, gamma_table):
        
        total_rates =  [0.0 for i in range(self.dim)]
        for index1 in range(self.dim):
            for index2 in range(self.dim):
                total_rates[index1] += gamma_table[index1][index2]
        # Create full rate numpy array:
        Gamma = np.zeros((self.dim**2,self.dim**2))
        n = 0
        for b in range(self.dim):
            for a in range(self.dim):
                Gamma[n][n] = 0.5 * (total_rates[b] - total_rates[a])
                n += 1
        return Gamma

    def calculate_first_order_correction(self, v_min, n, m):
        if n >= m:
            return 0.0
        evecs = self.evecs
        evals = self.evals

        v_nm = (evecs[n].dag()*(self.v*evecs[m]))[0][0][0]
        if abs(v_nm)  <= v_min:
            return 0.0
        
        k = self.integer_list
        rho_0 = np.reshape(self.density_matrix,(self.dim**2,1),order='F')
        H0_super = (1j*liouvillian(self.rotating_frame_hamiltonian)).full()

        V_nm = self.zeta*(evecs[n]*evecs[m].dag()*(evecs[n].dag()*(self.v*evecs[m])))
        L_nm = 1j*liouvillian(V_nm)
        b = np.dot(L_nm.full(),rho_0)
        omega_of_k = (k[n] - k[m] + 1)*self.omega
        A = omega_of_k * identity(self.dim**2) + 1j*self.Gamma + H0_super
        A = A.full()
        rho_correction = la.lstsq(A,b,rcond = 1e-6)[0]
        return la.norm(rho_correction)

    def term_order(self, v_min):
        start = time.time()
        #R = np.zeros((self.dim,self.dim))
        R = [[self.calculate_first_order_correction(v_min, n, m) for m in range(self.dim)] for n in range(self.dim)]
        end = time.time()
        print(end - start)
        R = np.asarray(R)
        # now order terms according to how large their value in R[][] is
        number_of_transitions = int(self.dim*(self.dim-1)/2)
        transition_order = [[] for i in range(number_of_transitions)]
        rank = 0
        for i in range(number_of_transitions):
            R_max = np.where(R == R.max())
            indices = [R_max[0][0], R_max[1][0]]
            transition_order[rank] = [R.max(), indices]
            R[indices[0]][indices[1]] = R[indices[1]][indices[0]] = 0
            rank += 1
        
        # This algorithm assigns an integer to each eigenstate of the hamiltonian based on the ordering from above
        # Initialize Lists:
        states = [[] for i in range(self.dim)]
        # START ALGORITHM
            # initialize first term into a diagram
        first_index = transition_order[0][1]
        diagram_list = [[first_index[0],first_index[1]]]
        states[max(first_index)] = 1
        states[min(first_index)] = 0
            # assign subsequent terms
        for i in range(1,number_of_transitions):
            #print(diagram_list)
            if transition_order[i]==[] or transition_order[i][1] ==[0,0]:
                break
            else:
                index = transition_order[i][1]
                if states[index[0]]==states[index[1]]==[]: 
                    states[max(index)] = 1
                    states[min(index)] = 0
                    # creates a new diagram
                    diagram_list.append([index[0],index[1]])
                elif states[index[0]]==[]:
                    if index[0] > index[1]:
                        states[index[0]] = states[index[1]] + 1
                    else:
                        states[index[0]] = states[index[1]] - 1
                    for k,diagram in enumerate(diagram_list):
                        if index[1] in diagram:
                            diagram_list[k].append(index[0]) # place in same diagram
                            break
                elif states[index[1]]==[]:
                    if index[0] > index[1]:
                        states[index[1]] = states[index[0]] - 1
                    else:
                        states[index[1]] = states[index[0]] + 1
                    for k,diagram in enumerate(diagram_list):
                        if index[0] in diagram:
                            diagram_list[k].append(index[1]) # place in same diagram
                            break
                else:
                    for k,diagram in enumerate(diagram_list):
                        overlap = list(set(index) & set(diagram))
                        if (len(overlap) == 2):
                            # loop closure: can't do anything
                            break
                        elif (len(overlap) == 1):
                            fixed_index = overlap[0]
                            shift_index = list(set(index) - set(diagram))[0]
                            old_integer = states[shift_index]
                            if shift_index > fixed_index:
                                new_integer = states[fixed_index] + 1
                            else:
                                new_integer = states[fixed_index] - 1
                            shift_amount = new_integer - old_integer
                            # shift the whole diagram
                            for j,diagram2 in enumerate(diagram_list):
                                if shift_index in diagram2:
                                    for m,index2 in enumerate(diagram2):
                                        states[index2] = states[index2] + shift_amount
                                    diagram_list[k] = diagram_list[k] + diagram2
                                    diagram_list.pop(j)
                                    break
                            break
                        else:
                            continue
                    continue
        # Buffer for extremely small values of R
        if [] in states:
            for i,integer in enumerate(states):
                if integer == []:
                    states[i] = 0
        #print(transition_order)
        #print(diagram_list)
        #print(states)
         ## END algorithm
        return transition_order, states

    def frame_rotate(self, v_min):
        # Set up 0th order eigenstates and eigenenergies
        evecs = self.evecs
        evals = self.evals
        v = self.v

        ordering, states = self.term_order(v_min)
        
        h_undriven = 0
        for j in range(self.dim):
            h_undriven += (evals[j])*(evecs[j]*evecs[j].dag()) - self.omega * states[j]*(evecs[j]*evecs[j].dag())

        # This while loop sets up the drive term in the rotating frame. 
        h_drive = 0
        error = 0
        number_of_transitions = int(self.dim*(self.dim-1)/2)
        drive_terms = []
        neglected_terms = []
        for j in range(number_of_transitions):
            index = ordering[j][1]
            if (index[0] != index[1]):
                if (states[max(index)] == states[min(index)] + 1):
                    h_drive += self.zeta*(evecs[min(index)]*evecs[max(index)].dag()*(evecs[min(index)].dag()*(v*evecs[max(index)])))
                    drive_terms.append(evecs[min(index)]*evecs[max(index)].dag()*(evecs[min(index)].dag()*(v*evecs[max(index)])))
                else:
                    neglected_terms.append(evecs[min(index)]*evecs[max(index)].dag()*(evecs[min(index)].dag()*(v*evecs[max(index)])))
                    error += ordering[j][0]**2
            else:
                break
        error = np.sqrt(error)

        return h_undriven + h_drive + h_drive.dag(), states, drive_terms



"""---------wrapper function  (optional)---------------------------------------------"""
def recursive_me_solve(frequency, hamiltonian, decoherence_rates, drive_coupling_operator, drive_strength, minimum_matrix_element_size=0.0, iteration_max=10):
    
    our_system = DrivenOpenSystem(hamiltonian,decoherence_rates,drive_coupling_operator,drive_strength,frequency)
    steady_state_density_matrix = our_system.recursively_steadystate_solve(minimum_matrix_element_size,iteration_max)

    """ Here we can calculate observables by taking |Tr(rho*O)| and take that to be the solution """
    return steady_state_density_matrix
    
"""
These next few functions are for the purpose of parallelizing the computation (strongly recommended)
"""
def parallel_map(f, a_list, *args, **kw):
    pool=mp.Pool(*args, **kw)

    result=pool.map(f, a_list)

    try:
        pool.close()
        pool.terminate()
    except:
        pass

    return result

def recursive_me_solve_parallel(hamiltonian, decoherence_rates, drive_coupling_operator, drive_strength, frequency, minimum_matrix_element_size=0.0, iteration_max=10, threads = 24):
    #flux_list = np.linspace(0.35,0.35,1)
    #transmission_data = np.empty(shape=(len(flux_list), len(drive_freq_list)))
    #error_data = np.empty(shape=(len(flux_list), len(drive_freq_list)))
    #transmission_data2 = np.empty(shape=(len(flux_list), len(drive_freq_list)))
    
    start = time.time()
    rho = partial(recursive_me_solve, hamiltonian=hamiltonian, decoherence_rates=decoherence_rates, drive_coupling_operator=drive_coupling_operator, drive_strength=drive_strength, minimum_matrix_element_size=minimum_matrix_element_size, iteration_max=iteration_max)
    solution = parallel_map(rho, frequency, threads)
    print(solution)
    #density_matrix = [col[0] for col in solution]
    #V0 = [col[1] for col in solution]
    #V1 = [col[2] for col in solution]
    #error_measure = [col[3] for col in solution]
    
    #transmission_data[0] = [abs(expect(V0[omega], r))  for omega,r in enumerate(density_matrix)]
    #error_data[0] = [sum(abs(V1[omega][m][0][n]*r[n][0][m])**2 for m in range(15) for n in range(15)) for omega,r in enumerate(density_matrix)]
    #transmission_data2[0] = np.sqrt(transmission_data[0]**2 + error_data[0])
    
    end = time.time()
    computation_time = end - start
    print(computation_time)
    return solution