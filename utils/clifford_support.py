from qiskit.circuit.library import RZGate
import numpy as np
import copy
from qiskit.quantum_info import Clifford
from random import randrange, choices
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.controlflow import IfElseOp
from qiskit.circuit import Qubit, ParameterExpression

def transform_isa_qc(isa_qc, random_init, bias, bias_parameter, param_value):
    """
    Creates a Clifford approximation of a circuit, given that it was
    transpiled beforehand (i.e., ISA circuit).
    """
    dag = circuit_to_dag(isa_qc)
    clifford_angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

    def _biased_rz(target_angle, bias_parameter): # beta in [0,1]
        # Compute (circular) distance to each Clifford angle
        distances = [np.abs((x - target_angle + np.pi) % (2*np.pi) - np.pi) for x in clifford_angles]
        # Convert distances to weights depending on bias type
        if bias == 'inverse':
            weights = [1 / (x + 1e-6) for x in distances] # avoid division by zero
        elif bias == 'exponential':
            beta = bias_parameter
            weights = [np.exp(-x*beta) for x in distances]
        elif bias == 'gaussian':
            sigma = bias_parameter
            weights = [np.exp(-(x**2)/(2*(sigma**2))) for x in distances]
        else: # default to uniform
            weights = np.ones_like(distances)

        probabilities = np.array(weights) / (np.array(weights).sum())
        chosen_angle = choices(clifford_angles, weights=probabilities, k=1)[0]
        return RZGate(chosen_angle)
    
    def _random_z():
        return RZGate(clifford_angles[randrange(4)])

    def _is_clifford(gate):
        try:
            Clifford(gate)
            return True
        except:
            return False
    def process_dag(dag, param_value, bias, bias_parameter):
                 
        for node in dag.op_nodes():
            if isinstance(node.op, IfElseOp):  # Handle if-else blocks
                new_blocks = []
                for qc in node.op.blocks:  # Iterate over both true and false branches
                    sub_dag = circuit_to_dag(qc)
                    process_dag(sub_dag, param_value, bias, bias_parameter)  # Recursively process the sub-DAG
                    new_blocks.append(dag_to_circuit(sub_dag))  # Convert back to circuit

                # Reconstruct the IfElseOp with modified blocks
                new_if_else = IfElseOp(node.op.condition, new_blocks[0], new_blocks[1] if len(new_blocks) > 1 else None)
                dag.substitute_node(node, new_if_else, inplace=True)

            elif isinstance(node.op, RZGate) and bias is not None: # Proceed to biased Clifford
                angle = node.op.params[0]
                if isinstance(angle, ParameterExpression):
                    parameter = list(angle.parameters)[0]
                    param_binding = { parameter : param_value }
                    angle = float(angle.bind(param_binding))
                
                new_gate = _biased_rz(angle, bias_parameter)
                dag.substitute_node(node, new_gate, inplace=True)

            else: # Uniformly randomized Cliffordization
                if not _is_clifford(node.op):
                    try:
                        dag.substitute_node(node, _random_z(), inplace=True)  
                    except:
                        continue
        
    process_dag(dag, param_value, bias, bias_parameter)
    transformed_circuit = dag_to_circuit(dag)

    if random_init:
        layout = isa_qc.layout.initial_layout
        for reg in layout.get_registers():
            if reg.name == 'q':
                target_register = reg
                break   
        # Now map these logical qubits to physical qubits after transpilation
        target_qubits = [ layout[Qubit(target_register, i)] for i in range(target_register.size) ]
        num_qubits = np.random.randint(0, len(target_qubits) + 1)  # Random number of qubits
        targets = np.random.choice(target_qubits, size=num_qubits, replace=False)

        init= copy.deepcopy(isa_qc)
        init.data = []

        if targets.size > 0:
            init.x(targets)  # Apply X gates to the selected qubits
        
        transformed_circuit = init.compose(transformed_circuit)  # Prepend initialization          

    return transformed_circuit

def generate_isa_clifford_circuits(isa_qc, random_init = False, bias = None, bias_parameter = None, param_value = None, num_variants = 10):
    """
    Generates multiple randomized Clifford circuits based on the original (ISA circuit)
    with the option of biasing the randomization towards neareast angles.
    """
    return [transform_isa_qc(isa_qc, random_init, bias, bias_parameter, param_value) for _ in range(num_variants)]