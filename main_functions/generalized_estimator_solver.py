from qiskit import QuantumCircuit
from qiskit_ibm_runtime import EstimatorV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp, Operator

from typing import Union
from qiskit.providers import BackendV2

import numpy as np
from qutip import tensor, qeye, Qobj

# This is an upgraded version of the qiskit_solver function found in qiskit_solver.py. It needs
# to make one call to the Estimator to compute the expectation values of the observables for any time 
# array.

def estimator_solver(qc:QuantumCircuit, n:int, t:list,
                  backend:Union[str, BackendV2], optimization_level:int, options:dict) -> dict:

    """
    Computes the time evolution of the population of all basis states (computational and coupled)
    under free evolution, decay and pairwise interaction.
    
    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit to solve.
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
    t : list
        Simulation time instants.
    backend : Union[str, BackendV1, BackendV2]
        Source of the default values for the basis_gates, inst_map, coupling_map, 
        backend_properties, instruction_durations, timing_constraints, and target
    optimization_level : int
        Level of optimization to generate the PassManager
        (see https://docs.quantum.ibm.com/guides/set-optimization for more info)
    options : dict
        Dictionary containing the specific EstimatorV2 options
        (see https://docs.quantum.ibm.com/guides/runtime-options-overview)


    Returns
    -------
    list
        A list of lists where list[i] corresponds to the expectation values of observable[i]
        computed on the transpiled circuit evaluated at each parameter value fed to the Estimator. 
        In our case, list[i] represents the evolution of the excited state population of qubit i.

    Description
    -----------
    
    Examples

    """
            
    evs = efficient_ev_calculator(n = n,
                                       qc = qc,
                                       t = t,
                                       backend = backend,
                                       optimization_level = optimization_level,
                                       options = options)
    
    return evs



def efficient_ev_calculator(n:int, qc:QuantumCircuit, t:list, backend:Union[str, BackendV2], 
                  optimization_level:int, options:dict) -> dict:
    

    # n = int(qc.num_qubits*0.5)

    pm = generate_preset_pass_manager(backend  = backend, optimization_level = optimization_level)
    transpiled_qc = pm.run(qc)

    observables = get_observables(n)
    transpiled_observables = [x.apply_layout(transpiled_qc.layout) for x in observables.values()]

    reshaped_obs = np.fromiter(transpiled_observables, dtype=object)
    reshaped_obs = reshaped_obs.reshape((len(observables), 1))

    instants = [[x] for x in t]

    estimator = EstimatorV2(mode = backend, options = options)

    job = estimator.run([(transpiled_qc, reshaped_obs, instants)])

    evs = {}
 
    for key in observables.keys():

        evs[key] = job.result()[0].data.evs[int(key)]

    return evs


def get_observables(n:int):

    """
    Returns a list with the observables being measured, 
    which in our case are the populations of the computational and 
    the coupled basis states (i.e., <\Pi_i> where \Pi_i = |i><i|).

    Parameters
    ----------

    n : int
        Number of qubits in the system's register (i.e., number of molecules).
    
    Returns
    -------
    list[SparsePauliOp]
        Observables

    """

    # The observables must be SparsePauliOps, so we are going to create them
    # using the from_operator method and from their matrix representation.

    # Basis definition

    ket_0 = Qobj([[1],[0]])
    ket_1 = Qobj([[0],[1]])

    # Computational basis 

    system_dim = 2**n

    # Observables are defined and stored in a dictionary 

    observables = {}

    sigmam = ket_0*ket_1.dag()
    sp_sm = sigmam.dag()*sigmam

    for i in range(-1, -n-1, -1): # Due to qiskits ordering we have to go backwards

        ops = [qeye(2)]*n
        ops[i] = sp_sm
        sig = tensor(ops)
        sig.dims = [[system_dim], [system_dim]]
        matrix_rep = sig.full()
        observables[str(-i-1)] = SparsePauliOp('I'*n).tensor(SparsePauliOp.from_operator(matrix_rep))

    return observables