from qiskit import QuantumCircuit
from typing import Union
from qiskit.providers import BackendV1, BackendV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.layout import Layout

def find_optimal_layout(shots:int, qc:QuantumCircuit, backend:Union[str, BackendV1, BackendV2], 
                  optimization_level:int) -> Layout:

    """
    Finds the best initial layout for a given backend and optimization level.
    
    Parameters
    ----------
    shots : int
    Number of shots to sample the circuit with.
    qc : QuantumCircuit
    Ideal quantum circuit to transpile.
    backend : Union[str, BackendV1, BackendV2]
    Target of transpilation
    optimization_level : int
    Optimization level of the preset pass manager.

    Returns
    -------
    Layout
    Mapping of virtual Qubit objects in the input circuit to the positions of the physical qubits.
    It is analogous to the initial_layout attribute.

    Description
    -----------
    This function runs a preset_pass_manager with simply optimization_level and backend inputs
    shots times and saves both the transpiled QuantumCircuits as well as their two-qubit depths.
    Then, it finds the minimum in the two-qubit depth list and retrieves the QuantumCircuit where
    it came from. This last circuit is the one whose layout we return, as it is the one which
    least SWAPs.


    """
    two_qubit_depths = []
    transpiled_qcs = []

    for i in range(shots):

        pm = generate_preset_pass_manager(optimization_level = optimization_level, backend = backend)
        trans_qc = pm.run(qc)

        two_qubit_depths.append(trans_qc.depth(lambda instr: len(instr.qubits) > 1))
        transpiled_qcs.append(trans_qc)

    index = two_qubit_depths.index(min(two_qubit_depths)) 
    good_layout = transpiled_qcs[index].layout
    optimal_layout = good_layout.initial_virtual_layout(filter_ancillas=True)

    return optimal_layout
