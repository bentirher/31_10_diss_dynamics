from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
import numpy as np
from qiskit.quantum_info import Operator, Statevector
import math
from qiskit.circuit.classical import expr

# You may have to change these three lines of imports below according to your 
# working directory.
import sys
sys.path.append("..") # This is for the imports from adjacent folders to work
from utils.chain_initial_layout import chain_init_layout

from typing import Union
from qiskit.providers import BackendV2


def get_circuit(n:int, omega_m:list, omega_c:list, g:list, gamma:list, kappa:list, initial_state:list, r:int, backend:BackendV2) -> QuantumCircuit:

    """

    Creates the quantum circuits representing one Trotter step of the Markovian evolution of $n$ emitters
    coupled to $n-1$ cavities, expressed as a function of $t$.

    Parameters
    ----------
    n : int
        Number of qubits in the system's register (i.e., number of emitters).
    omega_m : list
        Transition frequencies of the emitters.
    omega_c : float
        Cavity mode frequency.
    g : list
        Coupling strengths.
    gamma : list
        Spontaneous decay rate of each emitter.
    kappa : list
        Cavity decay rates (only admits one at the moment)
    initial_state : list[str(int)]
        List containing the index of initially-excited system qubits.
    r : int
        Number of circuit repetitions (Trotter steps)
    backend : BackendV1, BackendV2
        Target backend for the circuit. Used to determine the initial layout.

    Returns
    -------
    QuantumCircuit
        The output parametrized circuit.

    Description
    -----------
   
    Examples
    --------
    Typical parameter values for a two-emitter plasmonic chains are $omega_m
    = [1.2, 1.21]$, $\omega_c = 1.1$, $\gamma = [0.8*10**(-6), 0.8*10**(-6)]$,
    $kappa = [0.2]$, $g = [0.03]$.

    """

    # SYSTEM PARAMETERS: In this section, we define the effective variables of the Markovian
    # Master equation.

    delta = [ x - omega_c for x in omega_m ]
    mean_delta = [ 0.5*(omega_m[i] + omega_m[i+1]) - omega_c for i in range(n-1)]
    omega_eff = []
    gamma_eff = []

    for i in range(n):

        if i == 0: # This is the molecule on the very left (only coupled to one cavity)

            omega_eff.append(omega_m[i] + (delta[i]*(g[i]**2))/((0.5*kappa[0])**2 + delta[i]**2))
            gamma_eff.append(gamma[i] + (kappa[0]*(g[i]**2))/((0.5*kappa[0])**2 + delta[i]**2))

        elif i == (n-1): # This is the molecule on the very right (only coupled to one cavity)
            
            j = 2*i
            omega_eff.append(omega_m[i] + (delta[i]*(g[j-1]**2))/((0.5*kappa[0])**2 + delta[i]**2))
            gamma_eff.append(gamma[i] + (kappa[0]*(g[j-1]**2))/((0.5*kappa[0])**2 + delta[i]**2))

        else: # The molecules in the middle have their frequencies and decay rates modified by two adjacent cavities

            j = 2*i
            omega_eff.append(omega_m[i] + (delta[i]*(g[j-1]**2))/((0.5*kappa[0])**2 + delta[i]**2) + (delta[i]*(g[j]**2))/((0.5*kappa[0])**2 + delta[i]**2) )
            gamma_eff.append(gamma[i] + 0.5*(kappa[0]*(g[j-1]**2))/((0.5*kappa[0])**2 + delta[i]**2) + 0.5*(kappa[0]*(g[j]**2))/((0.5*kappa[0])**2 + delta[i]**2))
    
    g_eff = []
    gamma_cross = []

    for i in range(n-1):
        
        j = 2*i
        g_eff.append((g[j]*g[j+1]*(mean_delta[i]))/((kappa[0]/2)**2 + (mean_delta[i])**2))
        gamma_cross.append((g[j]*g[j+1]*(kappa[0]))/((kappa[0]/2)**2 + mean_delta[i]**2))
    
    det = [ omega_eff[i+1] - omega_eff[i] for i in range(n-1) ] 
    lam = [ 0.5*np.sqrt(4*g_eff[i]**2 + det[i]**2) for i in range(n-1) ]
    sen_theta = [ g_eff[i]/np.sqrt(lam[i]*(2*lam[i] + det[i])) for i in range(n-1) ]
    cos_theta = [ np.sqrt((2*lam[i] + det[i])/(4*lam[i])) for i in range(n-1) ]

    gamma_g_minus = [ (gamma_eff[i]*(cos_theta[i]**2) + gamma_eff[i+1]*(sen_theta[i]**2) - 2*gamma_cross[i]*sen_theta[i]*cos_theta[i]) for i in range(n-1) ]
    gamma_g_plus = [ (gamma_eff[i+1]*(cos_theta[i]**2) + gamma_eff[i]*(sen_theta[i]**2) + 2*gamma_cross[i]*sen_theta[i]*cos_theta[i]) for i in range(n-1) ]
    gamma_minus_e = [ (gamma_eff[i+1]*(cos_theta[i]**2) + gamma_eff[i]*(sen_theta[i]**2) - 2*gamma_cross[i]*sen_theta[i]*cos_theta[i]) for i in range(n-1) ]
    gamma_plus_e = [ (gamma_eff[i]*(cos_theta[i]**2) + gamma_eff[i+1]*(sen_theta[i]**2) + 2*gamma_cross[i]*sen_theta[i]*cos_theta[i]) for i in range(n-1) ]


    # REGISTERS DEFINITION

    system = QuantumRegister(n, 'q')
    environment = QuantumRegister(math.trunc(n/2), 'e')  
    classical_bits = ClassicalRegister(n-1, 'c')

    # PARAMETER DEFINITION: We define the only Parameter, delta_t = t/r and express the other
    # rotation angles as functions of t.

    delta_t = Parameter('$t$')/r
    beta = [ omega_eff[j]*delta_t for j in range(n) ]
    alpha = [ g_eff[j]*delta_t for j in range(n-1) ]

    theta_g_minus = [ (((1 - (-delta_t*gamma_g_minus[i]).exp())**(1/2)).arcsin())*2 for i in range(n-1) ] 
    theta_g_plus = [ (((1 - (-delta_t*gamma_g_plus[i]).exp())**(1/2)).arcsin())*2 for i in range(n-1) ]
    theta_minus_e = [ (((1 - (-delta_t*gamma_minus_e[i]).exp())**(1/2)).arcsin())*2 for i in range(n-1) ]
    theta_plus_e = [ (((1 - (-delta_t*gamma_plus_e[i]).exp())**(1/2)).arcsin())*2 for i in range(n-1) ]

    # Initialization circuit

    init = QuantumCircuit(system, environment, classical_bits)

    for q in initial_state:
        init.x(system[int(q)])
    
    # CORE CIRCUIT: 

    qc = QuantumCircuit(system, environment, classical_bits)

    num_labels = (2*(n-1)*r)*(n + math.trunc(n/2) - 2) # at least two qubits are involved in each conditional block, the rest are idle.
    # the first number in parethesis is the amount of IfElse operations in the circuit
    # stretch_labels = [chr(ord('a') + i) for i in range(num_labels)]
    width = len(str(num_labels - 1))
    stretch_labels = [f"A{i:0{width}d}" for i in range(1,num_labels + 1)]

    s_counter = 0
    for l in range(r):

        # Free Hamiltonian evolution

        for i in range(n):

            qc.rz(beta[i], system[i])

        # First interaction layer plus decay layer

        # Before going into this layer, we will define and store the pairwise
        # basis change gates that preceed the decay layer.

        P_gates = []
        P_dag_gates = []

        for i in range(n-1):

            matrix_P = np.array([[1, 0, 0, 0], [0, -sen_theta[i], cos_theta[i], 0], [0, cos_theta[i], sen_theta[i], 0], [0, 0, 0, 1]])
            P_gates.append(Operator(matrix_P).to_instruction())
            P_dag_gates.append(Operator(matrix_P.transpose()).to_instruction())

        counter = 0

        for j in range(0, n-1, 2):

            qc.ryy(alpha[j], qubit1 = system[j], qubit2 = system[j+1])
            qc.rxx(alpha[j], qubit1 = system[j], qubit2 = system[j+1])

            qc.append(P_gates[j], [system[j], system[j+1]])

            # SWAPs

            qc.swap(system[j+1], environment[counter])

            # We need to decompose the Toffoli into simpler gates
            # to insert SWAPs in the middle

            qc.h(system[j+1])
            qc.cx(environment[counter], system[j+1])
            qc.tdg(system[j+1])
            qc.cx(system[j], system[j+1])
            qc.t(system[j+1])
            qc.cx(environment[counter], system[j+1])
            qc.tdg(system[j+1])
            qc.t(environment[counter])
            qc.cx(system[j], system[j+1])
            qc.t(system[j+1])
            qc.h(system[j+1])

            # Mid Toffoli SWAP layer (back to original)

            qc.swap(system[j+1], environment[counter])

            # Second half of the Toffoli

            qc.cx(system[j], system[j+1])
            qc.t(system[j])
            qc.tdg(system[j+1])
            qc.cx(system[j], system[j+1])

            # Measurement
            qc.barrier()
            qc.measure(environment[counter], classical_bits[j])

            # # SWAP layer

            # qc.swap(system[j+1], environment[counter])

            # Checking classically the state of the auxiliary qubit to determine
            # whether the system is in the doubly-excited state or not, since the decay
            # paths are different in each case.

            with qc.if_test((classical_bits[j], 0)) as else_:
                qc.swap(system[j+1], environment[counter])
                qc.cry(theta_g_plus[j], system[j], system[j+1])
                qc.cx(system[j+1], system[j]) 

            with else_:
                qc.swap(system[j+1], environment[counter])
                qc.x(system[j+1]) # To make it go back to 0.
                qc.cry(theta_minus_e[j], system[j], system[j+1])
                qc.cx(system[j+1], system[j])

            used_qbits = [system[j], system[j+1], environment[counter]]
            total_qbits = system[::] + environment[::]
            target_qbits = [item for item in total_qbits if item not in used_qbits]

            for qb in target_qbits:
                s = qc.add_stretch(stretch_labels[s_counter])
                qc.delay(s, qb)
                qc.x(qb)
                qc.delay(expr.mul(s, 2), qb)
                qc.y(qb)
                qc.delay(expr.mul(s, 2), qb)
                qc.rx(-np.pi, qb) # -X
                qc.delay(expr.mul(s, 2), qb)
                qc.ry(-np.pi, qb) # - Y
                qc.delay(s, qb)     
                s_counter = s_counter + 1

            qc.barrier()

            qc.reset(system[j+1])

            qc.barrier()

            with qc.if_test((classical_bits[j], 0)) as else_:
                qc.cry(theta_g_minus[j], environment[counter], system[j+1]) 
                qc.cx(system[j+1], environment[counter])

            with else_:
                qc.cry(theta_plus_e[j], environment[counter], system[j+1])
                qc.cx(system[j+1], environment[counter])

            used_qbits = [system[j+1], environment[counter]]
            total_qbits = system[::] + environment[::]
            target_qbits = [item for item in total_qbits if item not in used_qbits]

            for qb in target_qbits:
                s = qc.add_stretch(stretch_labels[s_counter])
                qc.delay(s, qb)
                qc.x(qb)
                qc.delay(expr.mul(s, 2), qb)
                qc.y(qb)
                qc.delay(expr.mul(s, 2), qb)
                qc.rx(-np.pi, qb) # -X
                qc.delay(expr.mul(s, 2), qb)
                qc.ry(-np.pi, qb) # - Y
                qc.delay(s, qb)       
                s_counter = s_counter + 1         

            qc.barrier()

            qc.reset(system[j+1])

            # SWAP (back to original)
            qc.swap(system[j+1], environment[counter])
            qc.append(P_dag_gates[j], [system[j], system[j+1]])

            counter = counter + 1
        
        if n > 2: # Then we proceed onto the second layer of the interaction and decay,
        # which is esentially the same as the first one but applied to different qubits. 

            counter = 0

            for j in range(1, n-1, 2):

                # SWAP 

                qc.swap(system[j], environment[counter])

                qc.ryy(alpha[j], qubit1 = environment[counter], qubit2 = system[j+1])
                qc.rxx(alpha[j], qubit1 = environment[counter], qubit2 = system[j+1])

                qc.append(P_gates[j], [environment[counter], system[j+1]])

                # SWAPs (back to original) 

                qc.swap(system[j], environment[counter])

                # We need to decompose the Toffoli to insert SWAPs in the middle

                qc.h(environment[counter])
                qc.cx(system[j+1], environment[counter])
                qc.tdg(environment[counter])
                qc.cx(system[j], environment[counter])
                qc.t(environment[counter])
                qc.cx(system[j+1], environment[counter])
                qc.tdg(environment[counter])
                qc.t(system[j+1])
                qc.cx(system[j], environment[counter])
                qc.t(environment[counter])
                qc.h(environment[counter])

                # Mid Toffoli SWAP layer

                qc.swap(system[j], environment[counter])

                # Second half of the Toffoli

                qc.cx(environment[counter], system[j+1])
                qc.t(environment[counter])
                qc.tdg(system[j+1])
                qc.cx(environment[counter], system[j+1])

                # Measurement
                qc.barrier()
                qc.measure(system[j], classical_bits[j])

                # # SWAP layer (back to normal)

                # qc.swap(system[j], environment[counter])

                # Checking classically for correct parameter assignment

                with qc.if_test((classical_bits[j], 0)) as else_:
                    qc.swap(system[j], environment[counter])
                    qc.cry(theta_g_plus[j], system[j], environment[counter])
                    qc.cx(environment[counter], system[j]) 

                with else_:
                    qc.swap(system[j], environment[counter])
                    qc.x(environment[counter]) # To make it go back to 1.
                    qc.cry(theta_minus_e[j], system[j], environment[counter])
                    qc.cx(environment[counter], system[j])

                used_qbits = [system[j], environment[counter]]
                total_qbits = system[::] + environment[::]
                target_qbits = [item for item in total_qbits if item not in used_qbits]
                for qb in target_qbits:
                    s = qc.add_stretch(stretch_labels[s_counter])
                    qc.delay(s, qb)
                    qc.x(qb)
                    qc.delay(expr.mul(s, 2), qb)
                    qc.y(qb)
                    qc.delay(expr.mul(s, 2), qb)
                    qc.rx(-np.pi, qb) # -X
                    qc.delay(expr.mul(s, 2), qb)
                    qc.ry(-np.pi, qb) # - Y
                    qc.delay(s, qb)      
                    s_counter = s_counter + 1         
                
                qc.barrier()         

                qc.reset(environment[counter])

                qc.barrier()

                with qc.if_test((classical_bits[j], 0)) as else_:
                    qc.cry(theta_g_minus[j], system[j+1], environment[counter]) 
                    qc.cx(environment[counter], system[j+1])
                    
                with else_:
                    qc.cry(theta_plus_e[j], system[j+1], environment[counter])
                    qc.cx(environment[counter], system[j+1])

                used_qbits = [system[j+1], environment[counter]]
                total_qbits = system[::] + environment[::]
                target_qbits = [item for item in total_qbits if item not in used_qbits]
                for qb in target_qbits:
                    s = qc.add_stretch(stretch_labels[s_counter])
                    qc.delay(s, qb)
                    qc.x(qb)
                    qc.delay(expr.mul(s, 2), qb)
                    qc.y(qb)
                    qc.delay(expr.mul(s, 2), qb)
                    qc.rx(-np.pi, qb) # -X
                    qc.delay(expr.mul(s, 2), qb)
                    qc.ry(-np.pi, qb) # - Y
                    qc.delay(s, qb)     
                    s_counter = s_counter + 1         
                
                qc.barrier()    

                qc.reset(environment[counter])
                
                # SWAP
                qc.swap(system[j], environment[counter])
                qc.append(P_dag_gates[j], [environment[counter], system[j+1]])

                # And a final SWAP, to return to the original layout

                qc.swap(system[j], environment[counter])
        
                counter = counter + 1

    # Finally, we put the initialization right at the beginning and 
    # add the measurements at the end to make it Sampler-ready!
    
    parametrized_qc = (qc.compose(init, front = True))
    parametrized_qc.measure_all()

    # Getting the custom initial layout (chain-like, alternating pairs of system qubits
    # with their corresponding ancilla) to later feed onto the Sampler.

    init_layout = chain_init_layout(n, backend, system, environment)

    return parametrized_qc, init_layout
