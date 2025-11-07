from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate, Barrier, Measure, Reset
import numpy as np
from sklearn.linear_model import LinearRegression
from numpy.linalg import inv
from qiskit_ibm_runtime import SamplerV2, SamplerOptions
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
import math
import copy
from qiskit.quantum_info import Clifford
from random import randrange, choices, choice
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.controlflow import IfElseOp
from qiskit.circuit import Qubit, ParameterExpression
from sklearn.metrics import r2_score
from sklearn.utils import resample

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
            # if isinstance(node.op, Barrier): # We have reached the barrier right before
            #     # measure_all() so we must finish the loop before accidentaly removing the final measuremnts
            #     break

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
                    except: # This will only popup for (mid-circuit) measurements, where you cannot directly substitute the op by
                            # an Rz (due to the mismatch between number of qubits/cbits involved)
                        # dag.remove_op_node(node)
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

def replace_if_else(qc, random = False):
    """
    Replaces an IfElseOp by just one of its branches
    (so that it can be run on AerSimulator Stabilizer mode)
    """
    dag = circuit_to_dag(qc)
    for node in dag.op_nodes():
        if isinstance(node.op, IfElseOp): # This should take into account if the two ifelse are affected
            # by the same outcome, then this should choose the same block for both.
            if random == True:
                new_block = choice(node.op.blocks)
            else:
                new_block = node.op.blocks[0]
            new_instr = new_block.to_instruction(label = 'replaced_if_else')
            dag.substitute_node(node, new_instr, inplace=True)
    replaced_qc = dag_to_circuit(dag)
    dec_replaced_qc = replaced_qc.decompose(gates_to_decompose = 'replaced_if_else')
    return dec_replaced_qc

def remove_measurements(qc): # darle una weltita pa remplazar measurement
    dag = circuit_to_dag(qc)
    for node in dag.op_nodes():
        if isinstance(node.op, Barrier):
            break
        elif isinstance(node.op, Measure):
            dag.remove_op_node(node)

    return dag_to_circuit(dag)

def generate_isa_clifford_circuits(isa_qc, random_init = False, bias = None, bias_parameter = None, param_value = None, num_variants = 10):
    """
    Generates multiple randomized Clifford circuits based on the original (ISA circuit)
    with the option of biasing the randomization towards neareast angles.
    """
    return [transform_isa_qc(isa_qc, random_init, bias, bias_parameter, param_value) for _ in range(num_variants)]

def get_populations(n_emitters, result):
    """
    Measures the excited state population of each qubit from a PubResult
    """
    evs = {str(i) : [] for i in range(n_emitters)}
    std_devs = {str(i) : [] for i in range(n_emitters)}

    for k in range(len(result)):
        pub_result = result[k]
        shots = pub_result.data.c.num_shots
        counts = pub_result.data.meas.get_counts()
        states = [key[math.trunc(n_emitters/2):] for key in counts.keys()] # Output states
        coeff = [ np.sqrt(counts[key]/shots) for key in counts.keys()] # Normalized coefficients
        eigenvalues = [1, -1] # Z eigenvalues

        for i in range(n_emitters):

            ev = 0.5*( 1 - sum([ (coeff[j]**2)*eigenvalues[int(states[j][-i-1])] for j in range(len(states)) ]))
            evs[str(i)].append( ev ) # Fix this
            var = ev*( 1 - ev )
            var = max(var, 0.0) #To avoid negative sqrts
            std_devs[str(i)].append(np.sqrt( var / shots )) 
    
    return evs, std_devs

def get_training_data(isa_circuits, noisy_backend, n_emitters):
    """
    Returns the noisy and noiseless excited state population of each qubit
    """
    options = SamplerOptions()
    options.default_shots = 10**4

    #sampler = AerSampler(options=dict(backend_options=dict(method="stabilizer")))
    sampler = SamplerV2(mode = AerSimulator(), options = options)
    job = sampler.run(isa_circuits)
    result = job.result()
    ideal_data, std = get_populations(n_emitters, result)

    # NOISY SIMULATION

    sampler = SamplerV2(mode = noisy_backend, options = options)
    job = sampler.run(isa_circuits)
    result = job.result()
    noisy_data, std = get_populations(n_emitters, result)

    return ideal_data, noisy_data

def train_model(ideal_training_data, noisy_training_data):
    """
    Trains a linear regression model using the training data. Returns
    the model and the standard deviation of the slope and y-intercept.
    """
    # First, we will check the variance of the data since very
    # skewed probability bias can yield little variability in the Clifford
    # circuits (and therefore, unreliable fits).
    var_tol = 1e-6
    if np.var(ideal_training_data) < var_tol:
        print("Warning: Training data has very low variance. Regression may be unstable.")

    model = LinearRegression()
    X = np.array(ideal_training_data).reshape(-1, 1)  # Reshape for sklearn
    y = noisy_training_data
    model.fit(X, y)
    a, b = model.coef_[0], model.intercept_  # Extract slope & intercept

    noiseless_result_reshaped = np.array(ideal_training_data).reshape(-1, 1)

    # Compute standard errors manually
    X = np.vstack([np.ones(len(ideal_training_data)), ideal_training_data]).T  # Design matrix
    y_pred = model.predict(noiseless_result_reshaped)
    r2 = r2_score(y, y_pred)
    residuals = noisy_training_data - y_pred
    sigma_sq = np.sum(residuals**2) / (len(ideal_training_data) - 2)  # Estimate of variance
    # Covariance matrix calculation will be wrapped in a try/except block
    # just in case cov_matrix is singular (i.e., very little variability in training data)
    try:
        cov_matrix = sigma_sq * inv(X.T @ X)  # Covariance matrix of coefficients
        delta_b, delta_a = np.sqrt(np.diag(cov_matrix))  # Extract std errors
    except np.linalg.LinAlgError:
        print('Covariance matrix is singular. Cannot compute deltas')
        delta_a, delta_b = np.nan, np.nan

    print(f"Learned correction model: y = ({a:.4f} ± {delta_a:.4f} )x + ({b:.4f} ± {delta_b:.4f})")
    print(f"$R^2$ score: {r2:.4f}")
    return model, delta_a, delta_b, r2

def correct_observable(real_data, model):
    """
    Corrects the real_data according to a regression model.
    """
    corrected = (real_data - model.intercept_) / model.coef_[0]
    return corrected

# np.clip(corrected, 0, 1)



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

        model, delta_a, delta_b, r2 = train_model(resampled_ideal, resampled_noisy)
        corrected_data = correct_observable(real_data, model)
        predictions.append(corrected_data)


    mean_prediction = np.mean(np.array(predictions), axis=0)
    std_prediction = np.std(np.array(predictions), axis=0)/np.sqrt(samples)

    return mean_prediction, std_prediction