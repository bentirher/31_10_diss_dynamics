from qiskit import QuantumCircuit
import numpy as np

def get_circuit_properties(qc:QuantumCircuit) -> dict:

    """
    Retrieves the main properties of a given circuit

    Parameters
    ----------
    qc : QuantumCircuit
    Circuit whose properties are wished to be retrieved.

    Returns
    -------
    dict
        Dictionary with available keys 'width', 'size', 'two qubit gates',
        'depth', 'two qubit depth', 'swaps'.

    Description
    -----------

    """

    properties = {}

    properties['width'] = qc.width()
    properties['size'] = qc.size()
    properties['two qubit gates'] = qc.num_nonlocal_gates()
    properties['depth'] = qc.depth()
    properties['two qubit depth'] = qc.depth(lambda instr: len(instr.qubits) > 1)
    properties['swaps'] = count_swap(qc)

    return properties

def count_swap(qc:QuantumCircuit) -> int:

    """
    Estimates the number of SWAPs in a transpiled QuantumCircuit that contains sx and cz

    Parameters
    ----------
    qc : QuantumCircuit
    Circuit whose SWAPs are wished to be counted.

    Returns
    -------
    int
        Number of SWAPs

    Description
    -----------

    """

    # Initialize the counter

    swap_count = 0
    
    # Loop through the operations in the transpiled circuit

    for i, instr in enumerate(qc.data):

        # Check for the pattern of gates implementing SWAP:

        if i + 7 < len(qc.data):
            if (instr[0].name == 'sx' and
                qc.data[i+1][0].name == 'cz' and
                qc.data[i+2][0].name == 'sx' and
                qc.data[i+3][0].name == 'sx' and
                qc.data[i+4][0].name == 'cz' and
                qc.data[i+5][0].name == 'sx' and
                qc.data[i+6][0].name == 'sx' and
                qc.data[i+7][0].name == 'cz'):
                
                # If the sequence matches, increase the count
                swap_count += 1
                
    return swap_count