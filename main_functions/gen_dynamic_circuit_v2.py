# Implement two types: with n ancillas and with 1 ancilla per pair.

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import RYGate
import numpy as np
from qiskit.quantum_info import Operator, Statevector
import math


def get_circuit(n:int, omega_m:list, omega_c:list, g:list, gamma:list, kappa:list, initial_state:list, r:int) -> QuantumCircuit:

    """

    Returns
    -------

    Description
    -----------

    Examples
    --------

    """

    # System parameters:

    delta = [ x - omega_c for x in omega_m ]
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
        g_eff.append((0.5*g[j]*g[j+1]*(delta[i] + delta[i+1]))/((kappa[0]/2)**2 + delta[i]*delta[i+1]))
        gamma_cross.append((g[j]*g[j+1]*(kappa[0]))/((kappa[0]/2)**2 + delta[i]*delta[i+1])) 
    
    det = [ omega_eff[i+1] - omega_eff[i] for i in range(n-1) ] 
    lam = [ 0.5*np.sqrt(4*g_eff[i]**2 + det[i]**2) for i in range(n-1) ]
    sen_theta = [ g_eff[i]/np.sqrt(lam[i]*(2*lam[i] + det[i])) for i in range(n-1) ]
    cos_theta = [ np.sqrt((2*lam[i] + det[i])/(4*lam[i])) for i in range(n-1) ]

    gamma_g_minus = [ (gamma_eff[i]*(cos_theta[i]**2) + gamma_eff[i+1]*(sen_theta[i]**2) - 2*gamma_cross[i]*sen_theta[i]*cos_theta[i]) for i in range(n-1) ]
    gamma_g_plus = [ (gamma_eff[i+1]*(cos_theta[i]**2) + gamma_eff[i]*(sen_theta[i]**2) + 2*gamma_cross[i]*sen_theta[i]*cos_theta[i]) for i in range(n-1) ]
    gamma_minus_e = [ (gamma_eff[i+1]*(cos_theta[i]**2) + gamma_eff[i]*(sen_theta[i]**2) - 2*gamma_cross[i]*sen_theta[i]*cos_theta[i]) for i in range(n-1) ]
    gamma_plus_e = [ (gamma_eff[i]*(cos_theta[i]**2) + gamma_eff[i+1]*(sen_theta[i]**2) + 2*gamma_cross[i]*sen_theta[i]*cos_theta[i]) for i in range(n-1) ]


    # Register creation

    system = QuantumRegister(n, 'q')
    environment = QuantumRegister(math.trunc(n/2), 'e')  
    classical_bits = ClassicalRegister(n-1, 'c')

    # Parameter definition

    delta_t = Parameter('$t$')/r
    beta = [ 2*omega_eff[j]*delta_t for j in range(n) ]
    alpha = [ g_eff[j]*delta_t for j in range(n-1) ]

    theta_g_minus = [ (((1 - (-delta_t*gamma_g_minus[i]).exp())**(1/2)).arcsin())*2 for i in range(n-1) ] 
    theta_g_plus = [ (((1 - (-delta_t*gamma_g_plus[i]).exp())**(1/2)).arcsin())*2 for i in range(n-1) ]
    theta_minus_e = [ (((1 - (-delta_t*gamma_minus_e[i]).exp())**(1/2)).arcsin())*2 for i in range(n-1) ]
    theta_plus_e = [ (((1 - (-delta_t*gamma_plus_e[i]).exp())**(1/2)).arcsin())*2 for i in range(n-1) ]

    # Initialization circuit

    init = QuantumCircuit(system, environment, classical_bits)
    initial_qubit_state = initial_state    
    initial_statevector = Statevector(initial_qubit_state) 
    init.initialize(params = initial_statevector, qubits = system, normalize = True)

    qc = QuantumCircuit(system, environment, classical_bits)

    # Free Hamiltonian evolution
    for l in range(r):

        for i in range(n):

            qc.rz(beta[i], system[i])

        # Two-layer interaction circuit

        for j in range(0, n-1, 2):

            qc.ryy(alpha[j], qubit1 = system[j], qubit2 = system[j+1])
            qc.rxx(alpha[j], qubit1 = system[j], qubit2 = system[j+1])

        if n > 2:

            #u2 = QuantumCircuit(system, environment)

            for k in range(1, n-1, 2):

                qc.ryy(alpha[k], qubit1 = system[k], qubit2 = system[k+1])
                qc.rxx(alpha[k], qubit1 = system[k], qubit2 = system[k+1])

        # Decay layer

        # Basis change gates definition. For each pair of qubits, we have a different basis change gate.

        P_gates = []
        P_dag_gates = []

        for i in range(n-1):

            matrix_P = np.array([[1, 0, 0, 0], [0, -sen_theta[i], cos_theta[i], 0], [0, cos_theta[i], sen_theta[i], 0], [0, 0, 0, 1]])
            P_gates.append(Operator(matrix_P).to_instruction())
            P_dag_gates.append(Operator(matrix_P.transpose()).to_instruction())
            
        # Now we apply P to the first layer and do the classical check

        counter = 0

        for j in range(0, n-1, 2):

            qc.append(P_gates[j], [system[j], system[j+1]])
            
            qc.ccx(system[j], system[j+1], environment[counter])

            qc.measure(environment[counter], classical_bits[j])

            # Checking classically for correct parameter assignment

            with qc.if_test((classical_bits[j], 0)) as else_:

                qc.cry(theta_g_plus[j], system[j], environment[counter])
                qc.cx(environment[counter], system[j]) 
                qc.reset(environment[counter])
                qc.cry(theta_g_minus[j], system[j+1], environment[counter]) 
                qc.cx(environment[counter], system[j+1])
                qc.reset(environment[counter])

            with else_:

                qc.x(environment[counter]) # To make it go back to 1.
                qc.cry(theta_minus_e[j], system[j], environment[counter])
                qc.cx(environment[counter], system[j])
                qc.reset(environment[counter])
                qc.cry(theta_plus_e[j], system[j+1], environment[counter])
                qc.cx(environment[counter], system[j+1])
                qc.reset(environment[counter])    

            qc.append(P_dag_gates[j], [system[j], system[j+1]])

            

            counter = counter + 1

        # We do the same for the second layer

        if n > 2:

            counter = 0

            for j in range(1, n-1, 2):
                
                qc.append(P_gates[j], [system[j], system[j+1]])

                qc.ccx(system[j], system[j+1], environment[counter])

                qc.measure(environment[counter], classical_bits[j])

                # Checking classically for correct parameter assignment

                with qc.if_test((classical_bits[j], 0)) as else_:

                    qc.cry(theta_g_plus[j], system[j], environment[counter])
                    qc.cx(environment[counter], system[j]) 
                    qc.reset(environment[counter])
                    qc.cry(theta_g_minus[j], system[j+1], environment[counter]) 
                    qc.cx(environment[counter], system[j+1])
                    qc.reset(environment[counter])

                with else_:

                    qc.x(environment[counter]) # To make it go back to 1.
                    qc.cry(theta_minus_e[j], system[j], environment[counter])
                    qc.cx(environment[counter], system[j])
                    qc.reset(environment[counter])
                    qc.cry(theta_plus_e[j], system[j+1], environment[counter])
                    qc.cx(environment[counter], system[j+1])
                    qc.reset(environment[counter])


                qc.append(P_dag_gates[j], [system[j], system[j+1]])

                qc.reset(environment[counter])

                counter = counter + 1

    # Finally, we put the initialization right at the beginning and 
    # add the measurements for the Sampler implementation
    
    parametrized_qc = (qc.compose(init, front = True))
    parametrized_qc.measure_all()

    return parametrized_qc
