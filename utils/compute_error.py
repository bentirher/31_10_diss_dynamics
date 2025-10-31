from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService
import numpy as np

def compute_error(backend: QiskitRuntimeService.backend, circuit: QuantumCircuit, num_emitters:int) -> list:
    """Compute accumulated gate and readout errors for a given circuit on a specific backend."""

    # Defining useful variables
    properties = backend.properties()
    qubit_layout = list(circuit.layout.initial_layout.get_physical_bits().keys())[:int(num_emitters + np.floor(num_emitters/2))]

    errors = { str(q) : {    'acc_single_qubit_error' : 0,
                                'acc_two_qubit_error' : 0,
                            'single_qubit_gate_count' : 0,
                                'two_qubit_gate_count' : 0,
                                'acc_readout_error' : 0,
                                 'acc_total_error' : 0 }
                            for q in qubit_layout }

    # Define readout error (only for qubits in qubit_layout) using `properties.readout_error`
    for q in qubit_layout:
        errors[str(q)]['acc_readout_error'] += properties.readout_error(q)

    if "ecr" in backend.configuration().basis_gates: 
        two_qubit_gate = "ecr"
    elif "cz" in backend.configuration().basis_gates: 
        two_qubit_gate = "cz"

    for instruction in circuit.data:
        target_qubits = instruction.qubits
        gate_name = instruction.operation.name
        if len(target_qubits) == 1: # Count and add errors for one qubit gates
            target_qubit = circuit.qubits.index(target_qubits[0])
            errors[str(target_qubit)]['single_qubit_gate_count'] += 1
            if gate_name == 'measure':
                errors[str(target_qubit)]['acc_single_qubit_error'] += 0
            elif gate_name == 'delay':
                errors[str(target_qubit)]['acc_single_qubit_error'] += 0
            elif gate_name == 'reset':
                errors[str(target_qubit)]['acc_single_qubit_error'] += 0
            else:
                errors[str(target_qubit)]['acc_single_qubit_error'] += properties.gate_error(gate=gate_name, qubits=target_qubit)
        elif len(target_qubits) == 2: # Count and add errors for two qubit gates
            target_qubit_pair = ( circuit.qubits.index(target_qubits[0]), circuit.qubits.index(target_qubits[1]))
            for q in target_qubit_pair:
                errors[str(q)]['two_qubit_gate_count'] += 1
                errors[str(q)]['acc_two_qubit_error'] += properties.gate_error(gate=two_qubit_gate, qubits=target_qubit_pair)

    for q in qubit_layout:
        errors[str(q)]['acc_total_error'] = errors[str(q)]['acc_single_qubit_error'] + errors[str(q)]['acc_two_qubit_error']+ errors[str(q)]['acc_readout_error']

    return errors, qubit_layout