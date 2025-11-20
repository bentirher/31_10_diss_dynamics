from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, Gate, ControlledGate
import numpy as np
from qiskit.quantum_info import Operator, Statevector
from qiskit.circuit.library import RYGate, RYYGate, Initialize


def get_circuit(n:int, omega_m:list, omega_c:list, g:list, gamma:list, kappa:list, initial_state:list, r:int) -> QuantumCircuit:

    """
    Creates the quantum circuit representing the markovian master equation
    in the coupled basis in terms of a parametric t.

    Parameters
    ----------
    n : int
        Number of qubits in the system's register (i.e., number of molecules).
    omega_m : list
        Transition frequencies of the molecules.
    omega_c : float
        Cavity mode frequency.
    g : float
        Coupling strength.
    gamma : list
        Decay constants of each molecule.
    kappa : list
        Cavity decay rates
    initial_state : list
        Initial state of the molecules in the computational basis.
    r : int
        Number of core circuit repetitions (Trotter steps)


    Returns
    -------
    QuantumCircuit
        The output parametrized circuit, t-dependent.

    Description
    -----------
    

    Examples
    --------

    """

    # System parameters:

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
            omega_eff.append(omega_m[i] + 0.5*(delta[i]*(g[j-1]**2))/((0.5*kappa[0])**2 + delta[i]**2) + 0.5*(delta[i]*(g[j]**2))/((0.5*kappa[0])**2 + delta[i]**2) )
            gamma_eff.append(gamma[i] + (0.5)*((kappa[0]*(g[j-1]**2))/((0.5*kappa[0])**2 + delta[i]**2)) + (0.5)*((kappa[0]*(g[j]**2))/((0.5*kappa[0])**2 + delta[i]**2)))

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

    # Register creation

    system = QuantumRegister(n, 'q')
    environment = QuantumRegister(n, 'e')

    # Parameter definition

    delta_t = Parameter('$t$')/r
    beta = [ omega_eff[j]*delta_t for j in range(n) ]
    alpha = [ g_eff[j]*delta_t for j in range(n-1) ]

    theta_g_minus = [ (((1 - (-delta_t*gamma_g_minus[i]).exp())**(1/2)).arcsin())*2 for i in range(n-1) ] 
    theta_g_plus = [ (((1 - (-delta_t*gamma_g_plus[i]).exp())**(1/2)).arcsin())*2 for i in range(n-1) ]
    theta_minus_e = [ (((1 - (-delta_t*gamma_minus_e[i]).exp())**(1/2)).arcsin())*2 for i in range(n-1) ]
    theta_plus_e = [ (((1 - (-delta_t*gamma_plus_e[i]).exp())**(1/2)).arcsin())*2 for i in range(n-1) ]
    
    # Initialization circuit

    init = QuantumCircuit(system, environment)
    for qubit in initial_state:
        init.x(system[int(qubit)])
    # initial_qubit_state = initial_state    
    # initial_statevector = Statevector(initial_qubit_state) 
    # init.initialize(params = initial_statevector, qubits = system, normalize = True)

    qc = QuantumCircuit(system, environment)

    # Free Hamiltonian evolution

    for i in range(n):

        qc.rz(beta[i], system[i])

    # Two-layer interaction circuit

    for j in range(0, n-1, 2):

        qc.ryy(alpha[j], qubit1 = system[j], qubit2 = system[j+1])
        qc.rxx(alpha[j], qubit1 = system[j], qubit2 = system[j+1])

    if n > 2:

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

    # Now we apply Ps to the first layer, as in the interaction circuit and the decay part.

    for j in range(0, n-1, 2):

        qc.append(P_gates[j], [system[j], system[j+1]])

        ccry_plus = RYGate(theta_minus_e[j] - theta_g_plus[j]).control(2)
        ccry_minus = RYGate(theta_plus_e[j] - theta_g_minus[j]).control(2)

        qc.append(ccry_plus, [system[j], system[j+1], environment[j]])
        qc.append(ccry_minus, [system[j], system[j+1], environment[j+1]])
        qc.cry(theta_g_plus[j], system[j], environment[j])
        qc.cry(theta_g_minus[j], system[j+1], environment[j+1])
        qc.cx(environment[j], system[j])
        qc.cx(environment[j+1], system[j+1])

        qc.append(P_dag_gates[j], [system[j], system[j+1]])
        qc.reset([environment[j], environment[j+1]])
    
    # And onto the second layer

    if n > 2:

        for j in range(1, n-1, 2):

            qc.append(P_gates[j], [system[j], system[j+1]])

            ccry_plus = RYGate(theta_minus_e[j] - theta_g_plus[j]).control(2)
            ccry_minus = RYGate(theta_plus_e[j] - theta_g_minus[j]).control(2)

            qc.append(ccry_plus, [system[j], system[j+1], environment[j]])
            qc.append(ccry_minus, [system[j], system[j+1], environment[j+1]])
            qc.cry(theta_g_plus[j], system[j], environment[j])
            qc.cry(theta_g_minus[j], system[j+1], environment[j+1])
            qc.cx(environment[j], system[j])
            qc.cx(environment[j+1], system[j+1])

            qc.append(P_dag_gates[j], [system[j], system[j+1]])
            qc.reset([environment[j], environment[j+1]])


    # Putting everything (except the initialization) together
    # for Trotterization

    trotterized_qc = qc.repeat(r).decompose()

    # Finally, we put the initialization right at the beginning.
    
    parametrized_qc = trotterized_qc.compose(init, front = True)

    return parametrized_qc

