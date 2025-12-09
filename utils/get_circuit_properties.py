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

from qiskit.converters import circuit_to_dag
from qiskit.circuit import ControlFlowOp

def two_qubit_depth(qc):
    """
    Compute 'two-qubit depth' of `qc`, recursing into control-flow ops.

    It works by:
      - Converting to DAGCircuit
      - Removing all 1-qubit ops (and other <=1-qubit ops like meas/reset)
      - Using dag.depth(recurse=True), which traverses into IfElse/loops
        and takes the longer branch for if-else, etc.
    """
    dag = circuit_to_dag(qc)

    # Remove all non-control-flow ops that act on <= 1 qubit
    for node in list(dag.op_nodes()):
        op = node.op
        # Keep control-flow ops; DAGCircuit.depth(recurse=True)
        # will look inside their blocks instead of counting them directly.
        if isinstance(op, ControlFlowOp):
            continue
        if op.num_qubits <= 1:
            dag.remove_op_node(node)

    # This will now effectively give you the depth in terms of
    # multi-qubit gates only, recursing into the IfElse / loops.
    return dag.depth(recurse=True)

from qiskit import QuantumCircuit
from qiskit.circuit import ControlFlowOp

def count_two_qubit_gates_recursive(qc: QuantumCircuit) -> int:
    """
    Count all gates that act on >= 2 qubits in a QuantumCircuit,
    recursing into control-flow operations (if_else, while, for, etc.).

    Loops are counted once (i.e. we count the gates in the body once,
    not multiplied by the number of iterations).
    """

    total = 0

    for instr in qc.data:
        op = instr.operation

        # If it's a control-flow op (IfElseOp, ForLoopOp, WhileLoopOp...)
        if isinstance(op, ControlFlowOp):
            # Recurse into each block circuit
            for block in op.blocks:
                total += count_two_qubit_gates_recursive(block)

        else:
            # Regular operation: count if it touches >= 2 qubits
            if op.num_qubits >= 2:
                total += 1

    return total

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