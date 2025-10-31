from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate, RZGate, RXXGate, RYYGate, SXGate, SGate, SdgGate, CRYGate
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from numpy.linalg import inv
from qiskit import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2
from qiskit_aer.primitives import SamplerV2 as AerSampler
import math
from qiskit_aer import AerSimulator

from qiskit.quantum_info import Clifford
from random import randrange
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.controlflow import IfElseOp
from qiskit.circuit import Qubit
from sklearn.utils import resample

def bootstrap(ideal_training_data, noisy_training_data, samples, real_data):
    """
    Resamples the dataset, fits a linear model for each resample,
    corrects the data and returns the mean and std of these
    predictions
    """

    predictions = [] # This will be a 'matrix' with size (rows x colums) = 
    # = (samples x len(real_data))

    for _ in range(samples):
        # Resample the Clifford data with replacement
        resampled_noisy, resampled_ideal = resample(noisy_training_data, ideal_training_data)

        model, delta_a, delta_b = train_model(resampled_ideal, resampled_noisy)
        corrected_data = correct_observable(real_data, model)
        predictions.append(corrected_data)


    mean_prediction = np.mean(np.array(predictions), axis=0)
    std_prediction = np.std(np.array(predictions), axis=0)

    return mean_prediction, std_prediction

def correct_observable(real_data, model):
    """
    Corrects the real_data according to a regression model.
    """
    return (real_data - model.intercept_) / model.coef_[0]


def train_model(ideal_training_data, noisy_training_data):
    """
    Trains a linear regression model using the training data. Returns
    the model and the standard deviation of the slope and y-intercept.
    """
    model = LinearRegression()
    X = np.array(ideal_training_data).reshape(-1, 1)  # Reshape for sklearn
    y = noisy_training_data
    model.fit(X, y)
    a, b = model.coef_[0], model.intercept_  # Extract slope & intercept

    noiseless_result_reshaped = np.array(ideal_training_data).reshape(-1, 1)

    # Compute standard errors manually
    X = np.vstack([np.ones(len(ideal_training_data)), ideal_training_data]).T  # Design matrix
    y_pred = model.predict(noiseless_result_reshaped)
    residuals = noisy_training_data - y_pred
    sigma_sq = np.sum(residuals**2) / (len(ideal_training_data) - 2)  # Estimate of variance

    cov_matrix = sigma_sq * inv(X.T @ X)  # Covariance matrix of coefficients
    delta_b, delta_a = np.sqrt(np.diag(cov_matrix))  # Extract std errors

    print(f"Learned correction model: y = ({a:.4f} ± {delta_a:.4f} )x + ({b:.4f} ± {delta_b:.4f})")
    return model, delta_a, delta_b

def get_training_data(circuits, noisy_backend, n):
    """
    Returns the noisy and noiseless excited state population of each qubit
    """
    pm = generate_preset_pass_manager(optimization_level = 2 , backend = noisy_backend)
    # We will transpile the circuit (noisy and ideal) in the same way

    isa_qcs = pm.run(circuits)
    options = {'default_shots' : 10**4}

    #sampler = AerSampler(options=dict(backend_options=dict(method="stabilizer")))
    sampler = SamplerV2(mode = AerSimulator(), options = options)
    job = sampler.run(isa_qcs)
    result = job.result()
    ideal_data = get_populations(n, result)

    # NOISY SIMULATION

    sampler = SamplerV2(mode = noisy_backend, options = options)
    job = sampler.run(isa_qcs)
    result = job.result()
    noisy_data = get_populations(n, result)

    return ideal_data, noisy_data

def get_populations(n, result):
    """
    Measures the excited state population of each qubit from a PubResult
    """
    evs = {str(i) : [] for i in range(n)}

    for k in range(len(result)):
        pub_result = result[k]
        shots = pub_result.data.c.num_shots
        counts = pub_result.data.meas.get_counts()
        states = [key[math.trunc(n/2):] for key in counts.keys()] # Output states
        coeff = [ np.sqrt(counts[key]/shots) for key in counts.keys()] # Normalized coefficients
        eigenvalues = [1, -1] # Z eigenvalues

        for i in range(n):

            evs[str(i)].append(0.5*( 1 - sum([ (coeff[j]**2)*eigenvalues[int(states[j][-i-1])] for j in range(len(states)) ]))) # Fix this
    
    return evs

def generate_clifford_circuits(original_circuit, num_variants=10, random_init=False):
    """
    Generates multiple randomized Clifford circuits based on the original.
    """
    return [transform_circuit(original_circuit, random_init) for _ in range(num_variants)]

def transform_circuit(circuit, random_init=False):
    """
    Creates a Clifford approximation of the given circuit.
    """
    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)

    for inst in circuit.data:

        qargs = inst.qubits
        cargs = inst.clbits
        operation = inst.operation

        if inst.name == 'if_else':
            true_block = QuantumCircuit(*circuit.qregs, *circuit.cregs)
            false_block = QuantumCircuit(*circuit.qregs, *circuit.cregs)
            new_blocks = [true_block, false_block]
            for index, block in enumerate(operation.blocks):
                for sub_inst in block.data:
                    sub_qargs = sub_inst.qubits
                    sub_cargs = sub_inst.clbits
                    sub_operation = sub_inst.operation
                    new_blocks[index].append(make_gate_clifford(sub_operation), sub_qargs, sub_cargs)

            clifford_ifelse = operation.replace_blocks(new_blocks)
            new_circuit.append(clifford_ifelse, circuit.qubits, circuit.clbits)
        
        elif inst.name == 'unitary':
            approx_p = QuantumCircuit(*circuit.qregs, *circuit.cregs)
            approx_p.rx(get_random_clifford_angle(), qargs)
            approx_p.cx(qargs[0], qargs[1])
            approx_p.rx(get_random_clifford_angle(), qargs)
            approx_p.cx(qargs[0], qargs[1])
            approx_p.rx(get_random_clifford_angle(), qargs)
            approx_p.cx(qargs[0], qargs[1])
            approx_p.rx(get_random_clifford_angle(), qargs)
            
            new_circuit.append(approx_p, circuit.qubits, circuit.clbits)

        else:
            new_circuit.append(make_gate_clifford(operation), qargs, cargs)

    if random_init == True:

        system_qubits = len(circuit.qregs[0])
        init = QuantumCircuit(*circuit.qregs, *circuit.cregs)
        num_qubits = np.random.randint(0, system_qubits)
        targets = np.random.choice(np.arange(0, system_qubits), size=num_qubits, replace=False)
        if targets.size > 0:
            init.x(targets)
        
        clifford_circuit = new_circuit.compose(init, front = True)

    else:

        clifford_circuit = new_circuit

    return clifford_circuit

def get_random_clifford_angle():
    """
    Returns a random Clifford-compatible angle.
    """
    return random.choice([0, np.pi/2, -np.pi/2, np.pi, -np.pi, 3*np.pi/2, -3*np.pi/2])

def make_gate_clifford(gate):
    """
    Returns a Clifford-approximated version of a given gate.
    """
    if isinstance(gate, (RXGate, RYGate, RZGate, CRYGate, RXXGate, RYYGate)):
        return gate.__class__(get_random_clifford_angle())
    elif gate.name == 't':
        return SGate()
    elif gate.name == 'tdg':
        return SdgGate()
    return gate  # Keep other gates unchanged

def generate_isa_clifford_circuits(isa_qc, num_variants = 10):
    """
    Generates multiple randomized Clifford circuits based on the original (ISA circuit).
    """
    return [transform_isa_qc(isa_qc) for _ in range(num_variants)]

def transform_isa_qc(isa_qc, random_init = False):
    """
    Creates a Clifford approximation of a circuit, given that it was
    transpiled beforehand (i.e., ISA circuit).
    """
    dag = circuit_to_dag(isa_qc)

    rand_angles = (0, np.pi / 2, np.pi, 3 * np.pi / 2)
    def _random_z():
        return RZGate(rand_angles[randrange(4)])

    def _is_clifford(gate):
        try:
            Clifford(gate)
            return True
        except:
            return False
    def process_dag(dag):
                 
        for node in dag.op_nodes():
            #print(node.op, _is_clifford(node.op))
            if isinstance(node.op, IfElseOp):  # Handle if-else blocks
                new_blocks = []
                for qc in node.op.blocks:  # Iterate over both true and false branches
                    sub_dag = circuit_to_dag(qc)
                    process_dag(sub_dag)  # Recursively process the sub-DAG
                    new_blocks.append(dag_to_circuit(sub_dag))  # Convert back to circuit

                # Reconstruct the IfElseOp with modified blocks
                node.op = IfElseOp(node.op.condition, new_blocks[0], new_blocks[1] if len(new_blocks) > 1 else None)
            else:
                if not _is_clifford(node.op):
                    try:
                        dag.substitute_node(node, _random_z(), inplace=True)
                        # print('Replaced:', node.op)
                    except:
                        pass
        
    process_dag(dag)
    transformed_circuit = dag_to_circuit(dag)

    if random_init:
        layout = isa_qc.layout.initial_layout
        for reg in layout.get_registers():
            if reg.name == 'q':
                target_register = reg
                break   
        # Now map these logical qubits to physical qubits after transpilation
        target_qubits = [ layout[Qubit(target_register, i)] for i in range(target_register.size) ]
        num_qubits = np.random.randint(0, target_qubits + 1)  # Random number of qubits
        targets = np.random.choice(target_qubits, size=num_qubits, replace=False)

        init = QuantumCircuit(*[reg for reg in isa_qc.regs])
        if targets.size > 0:
            init.x(targets)  # Apply X gates to the selected qubits
        
        transformed_circuit = init.compose(transformed_circuit)  # Prepend initialization          

    return transformed_circuit