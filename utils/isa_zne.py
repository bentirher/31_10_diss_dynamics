from qiskit.circuit import IfElseOp
from qiskit import QuantumCircuit
from scipy.optimize import linprog
import copy
import numpy as np
import math
import cvxpy as cp
from scipy.optimize import curve_fit

def get_parallel_cz_layers(isa_circuit: QuantumCircuit):
    """
    Computes the number of CZ layers in the circuit.
    
    Parameters:
        isa_circuit (QuantumCircuit) : ISA circuit

    Returns:
        cz_layers (list(list)): A list of lists, where each of
        the sublists contains (idx, q0, q1, tag, parent_if_else_idx, branch_id)
        of the parallel CZ where,
            idx (int) : Position of the instruction in isa_qc.data
                        q0, q1 (Qubit) : Qubits involved in the CZ 
            tag (str) : 'inner' if the CZ is nested in an IfElseOp, 
                        'outer in any other case.
            parent_if_else_idx (int) : Position of the IfElseOp
                                       if tag = 'inner', else is None.
            branch_id (int) : Extra tag to differentiate If blocks
                              from Else for merging later on. 
                              If part of a If block, branch_id = 0. Else, 1.
         
    """
    cz_layers = []
    #layout = isa_circuit.layout.initial_layout
    #layout = isa_circuit.layout.final_layout
    layout = isa_circuit.layout.final_virtual_layout()
    for reg in isa_circuit.layout.initial_layout.get_registers():
        if reg.name == 'q':
            target_register = reg 
    max_gates_per_layer = math.trunc(target_register.size/2)
    
    def place_in_layer(idx, q0, q1, tag, parent_if_else_idx = None, branch_id = None):
        
        for layer in cz_layers:
            # Reject layer if it has a different tag
            if any(g[3] != tag for g in layer):
                continue

            if tag == 'outer':
                max_gap = 170 
                # Get max index of any gate in this layer
                layer_max_idx = max(g[0] for g in layer)

                # Check if current gate is within acceptable depth range
                if idx - layer_max_idx > max_gap:
                    continue  # too far away, stop checking older layers
            else:        
                # For inner layers: ensure branch ID match, and all gates are at same idx
                if any(g[5] != branch_id for g in layer):
                    continue
                existing_idxs = {g[0] for g in layer}
                if existing_idxs and idx not in existing_idxs:
                    continue
            
            if len(layer) >= max_gates_per_layer:
                continue  

            used_qubits = {q for gate in layer for q in gate[1:3]}
            if q0 in used_qubits or q1 in used_qubits:
                continue

            layer.append((idx, q0, q1, tag, parent_if_else_idx, branch_id))
            return
        # No compatible layer found, start a new one
        cz_layers.append([(idx, q0, q1, tag, parent_if_else_idx, branch_id)])

    def process_block(block, tag = 'outer', parent_if_else_idx=None, branch_id=None):
        for idx, instr in enumerate(block.data):
            qargs = instr.qubits
        
            if instr.name == 'cz':
                #q0 = layout[qargs[0]] if tag == 'inner' else block.find_bit(qargs[0]).index
                #q1 = layout[qargs[1]] if tag == 'inner' else block.find_bit(qargs[1]).index
                q0 = isa_circuit.find_bit(qargs[0]).index
                q1 = isa_circuit.find_bit(qargs[1]).index

                place_in_layer(idx, q0, q1, tag, parent_if_else_idx, branch_id)

            elif instr.name == 'ecr':
                q0 = layout[qargs[0]] if tag == 'inner' else block.find_bit(qargs[0]).index
                q1 = layout[qargs[1]] if tag == 'inner' else block.find_bit(qargs[1]).index

                place_in_layer(idx, q0, q1, tag, parent_if_else_idx, branch_id)
                
            elif instr.name == 'if_else':
                # Recursively process each branch at this point
                operation = instr.operation
                blocks = operation.blocks
                for b_idx, branch in enumerate(blocks):
                    process_block(branch, tag = 'inner', parent_if_else_idx=idx, branch_id=b_idx)

    process_block(isa_circuit)
    return cz_layers


def merge_inner_layers(layers):
    """
    Combines True and False (If and Else) branches in
    cz_layers to count as one single layer for folding later on
    
    Parameters:
        layers : A list of lists, where each of
        the sublists contains (idx, q0, q1, tag, parent_if_else_idx, branch_id)

    Returns:
        merged_layers (list(list)): A list of lists, where each of
        the sublists contains (idx, q0, q1, tag, parent_if_else_idx, branch_id)
        of the parallel CZ, with If and Else blocks merged, where,
            idx (int) : Position of the instruction in isa_qc.data
                        q0, q1 (Qubit) : Qubits involved in the CZ 
            tag (str) : 'inner' if the CZ is nested in an IfElseOp, 
                        'outer in any other case.
            parent_if_else_idx (int) : Position of the IfElseOp
                                       if tag = 'inner', else is None.
            branch_id (int) : Extra tag to differentiate If blocks
                              from Else for merging later on. 
                              If part of a If block, branch_id = 0. Else, 1.
         
    """

    merged_layers = []
    used = set()

    for i, layer in enumerate(layers):
        if i in used:
            continue

        current_layer = list(layer)
        if not all(g[3] == 'inner' for g in current_layer):
            merged_layers.append(current_layer)
            continue

        # Use (q0, q1, tag, parent_idx) as key — skip branch_id (g[5])
        current_keys = set((g[1], g[2], g[3], g[4]) for g in layer)
        current_branch_ids = set(g[5] for g in layer)

        for j in range(i + 1, len(layers)):
            if j in used:
                continue

            other_layer = layers[j]
            if not all(g[3] == 'inner' for g in other_layer):
                continue

            other_keys = set((g[1], g[2], g[3], g[4]) for g in other_layer)
            other_branch_ids = set(g[5] for g in other_layer)

            # Same keys, but different branch_id sets (e.g., merging true and false branches)
            if current_keys == other_keys and current_branch_ids.isdisjoint(other_branch_ids):
                current_layer += other_layer
                current_branch_ids.update(other_branch_ids)
                used.add(j)

        merged_layers.append(current_layer)

    return merged_layers

def fold_cz_layers_by_layer(circuit: QuantumCircuit, n: int, s: int, folding_method='left', layer_indices = None, track_subset = False) -> QuantumCircuit:
    """
    Performs gate folding of CZ gates on a given ISA circuit.
    
    Parameters:
        circuit (QuantumCircuit) : ISA circuit to fold
        n (int) : Number of foldings
        s (int) : Size of the subset of indices to fold S.
                  Layers in S are folded n + 1 times, while
                  layers outside are folded n times.
        folding_method (str) : Strategy to follow to construct
                               the subset S. Admits 'left', 'right',
                               'random' or 'custom'.
        layer_indices (list) : Indices of layers to fold. Only specified
                               if folding_method = 'custom'. Must be consistent
                               with s (i.e., len(layer_indices) = s)
        track_subset (bool) : Whether to keep track of the layers
                              being folded. If set to True, this function
                              returns an extra output.

    Returns:
        folded_qc (QuantumCircuit) : The folded circuit
        fold_indices (set) : Set of indices of layers being folded.
                            Only output if track_subset is set to True.
         
    """

    #layout = circuit.layout.initial_layout
    #layout = circuit.layout.final_layout
    layout = circuit.layout.final_virtual_layout()
    cz_layers = merge_inner_layers(get_parallel_cz_layers(circuit))
    num_layers = len(cz_layers)
    
    if s > num_layers:
        raise ValueError("s is greater than number of CZ layers")

    # Select layers to fold
    if folding_method == 'random':
        fold_indices = set(np.random.choice(range(num_layers), size=s, replace=False))
    elif folding_method == 'left':
        fold_indices = set(range(s))
    elif folding_method == 'right':
        fold_indices = set(range(num_layers - s, num_layers))
    elif folding_method == 'custom':
        if len(layer_indices) == s:
            fold_indices = layer_indices
        else:
            raise ValueError(f"Size of layer indices to fold must match size of subset S (s)")
    else:
        raise ValueError(f"Unsupported folding method: {folding_method}")
    # Identify exactly which CZs we want to fold based on layer
    cz_to_fold = set()
    for i in fold_indices:
        for (idx, q0, q1, tag, parent_if_else_idx, branch_id) in cz_layers[i]:
            cz_to_fold.add((idx, q0, q1, tag, parent_if_else_idx, branch_id))

    def fold_qc(block: QuantumCircuit, tag = 'outer', parent_if_else_idx = None, branch_id = None) -> QuantumCircuit:
        new_block = copy.deepcopy(block)
        new_block.data = []

        for idx, inst in enumerate(block.data):
            qargs = inst.qubits
            cargs = inst.clbits
            operation = inst.operation
            
            if inst.name == 'cz':
                # q0 = layout[qargs[0]] if tag == 'inner' else block.find_bit(qargs[0]).index
                # q1 = layout[qargs[1]] if tag == 'inner' else block.find_bit(qargs[1]).index
                q0 = circuit.find_bit(qargs[0]).index
                q1 = circuit.find_bit(qargs[1]).index

                new_block.cz(qargs[0], qargs[1])

                if (idx, q0, q1, tag, parent_if_else_idx, branch_id) in cz_to_fold:
                    fold_times = n + 1
                else:
                    fold_times = n

                for _ in range(fold_times):
                    new_block.cz(qargs[0], qargs[1])
                    new_block.cz(qargs[0], qargs[1])
                    
            elif inst.name == 'ecr':
                q0 = layout[qargs[0]] if tag == 'inner' else block.find_bit(qargs[0]).index
                q1 = layout[qargs[1]] if tag == 'inner' else block.find_bit(qargs[1]).index

                new_block.ecr(qargs[0], qargs[1])

                if (idx, q0, q1, tag, parent_if_else_idx, branch_id) in cz_to_fold:
                    fold_times = n + 1
                else:
                    fold_times = n

                for _ in range(fold_times):
                    new_block.ecr(qargs[0], qargs[1])
                    new_block.ecr(qargs[0], qargs[1])

            elif inst.name == 'if_else':
                new_blocks = []
                for br_idx, branch in enumerate(operation.blocks):
                    folded_branch = fold_qc(branch, tag = 'inner', parent_if_else_idx = idx, branch_id = br_idx)
                    new_blocks.append(folded_branch)

                # Now append the corrected IfElse
                new_ifelse = IfElseOp(operation.condition, *new_blocks)
                new_block.append(new_ifelse, qargs, cargs)
            else:
                new_block.append(inst, qargs, cargs)

        return new_block

    if track_subset == True:
        return fold_qc(circuit), fold_indices
    else:
        return fold_qc(circuit)

def compute_folding_parameters(isa_qc: QuantumCircuit, lambdas: list):
    """
    Computes the folding parameters corresponding to a set of lambda values
    
    Parameters:
        isa_qc (QuantumCircuit) : ISA circuit to fold
        lambdas (list) : Noise amplification values

    Returns:
        n (int) : Number of foldings
        s (int) : Size of the subset of indices to fold S.
                  Layers in S are folded n + 1 times, while
                  layers outside are folded n times.
         
    """
    n = []
    s = []
    d = len(merge_inner_layers(get_parallel_cz_layers(isa_qc)))

    for l in lambdas:
        k = round(d*(l - 1)/2)

        if k == 0:
           k = 1
        
        n.append(np.floor_divide(k,d))
        s.append(np.mod(k,d))    

    return n, s



def extrapolate_to_zero_noise(lambdas: list, data: list, type: str, order : int = 1, weights_minimization : str = 'none', retrieve_error : bool = False, error_data : list = None):
    """
    Performs an extrapolation to estimate <O>^* from noisy data.
    
    Parameters:
        lambdas (list): Noise amplification values
        data (list): Corresponding noisy data values
        type (str): Type of extrapolation to use on the noise-amplified
                        data. Supported types are 'linear', 'expo',
                        'poly_degree_2' or 'richardson'.
        order (int) : Order of the Richardson extrapolation.
        weights_minimization (str) : 'none', 'l1' for L1-norm minimization and 'l2' for L2-norm minimization
    
    Returns:
        extrapolated_result (float): Extrapolated zero-noise value
    """

    def linear_fit(x, a, b):
        return a*x + b

    def expo_fit(x, a, b):
        return a*np.exp(-x*b)

    def poly_degree_2_fit(x, a, b):
        return a*(x**2) + b
    
    def regular_richardson(lambdas: list, data: list, order: int, retrieve_error : bool, error_data : list):
        n = len(lambdas)
        if n < order + 1:
            raise ValueError(f"Need at least {order + 1} lambda points for a order-{order} Richardson extrapolation.")
        
        # Start with the original data array
        extrapolations = list(data)
        variances = [err**2 for err in error_data] if retrieve_error else None

        # Richardson extrapolation 
        for k in range(1, order + 1):
            new_vals = []
            new_vars = []
            for i in range(len(extrapolations) - 1):
                lambda_ratio = lambdas[i + k] / lambdas[i]
                denom = lambda_ratio**k - 1
                coeff_a = lambda_ratio**k / denom
                coeff_b = -1 / denom
                val = coeff_a * extrapolations[i] + coeff_b * extrapolations[i + 1]
                new_vals.append(val)

                var = (coeff_a**2) * variances[i] + (coeff_b**2) * variances[i + 1]
                new_vars.append(var)

            extrapolations = new_vals
            variances = new_vars
        
        final_extrapolation = sum(extrapolations)/len(extrapolations)
        final_variance = sum(np.sqrt(variances))/len(variances)

        return final_extrapolation, final_variance

    def constrained_richardson(lambdas: list, data: list, order: int, weights_minimization : str, retrieve_error : bool, error_data : list):
        n = len(lambdas)
        if n <= order + 1:
            raise ValueError(f"Constraints can be uniquely determined without minimization. Please try regular Richardson instead")
        
        gamma = cp.Variable(n)
        
        # Constraints: sum gamma_i = 1, sum gamma_i * lambda_i^k = 0 for k = 1, ..., order
        constraints = [cp.sum(gamma) == 1]
        for k in range(1, order + 1):
            constraints.append(cp.sum(cp.multiply(gamma, np.power(lambdas, k))) == 0)
        
        # Objective: minimize L1 or L2 norm of gamma
        if weights_minimization == 'l1':
            objective = cp.Minimize(cp.norm(gamma, 1))
        elif weights_minimization == 'l2':
            objective = cp.Minimize(cp.norm(gamma, 2))
            
        prob = cp.Problem(objective, constraints)
        prob.solve()

        if retrieve_error == True:
            extrapolation_error = np.sqrt(np.dot(error_data, (gamma.value)**2))
        else:
            extrapolation_error = None

        return float(np.dot(data, gamma.value)), extrapolation_error
    
    if type == 'richardson':
        if weights_minimization == 'none':
            extrapolated_result, extrapolation_error = regular_richardson(lambdas, data, order, retrieve_error, error_data)
        elif weights_minimization == 'l1' or weights_minimization == 'l2':
            extrapolated_result, extrapolation_error = constrained_richardson(lambdas, data, order, weights_minimization, retrieve_error, error_data)    
        else:
            raise ValueError(f"Unsupported weights_minimization '{weights_minimization}'. Choose from 'none', 'l1', 'l2'.")
    else:
        fit_functions = {
        'linear': linear_fit,
        'expo': expo_fit,
        'poly_degree_2': poly_degree_2_fit
    }
        if type not in fit_functions:
            raise ValueError(f"Unsupported fit_type '{type}'. Choose from {list(fit_functions.keys())}.")

        fit_fn = fit_functions[type]
        params, _ = curve_fit(fit_fn, lambdas, data)
        a, b = params
        extrapolated_result = fit_fn(0, a, b)

        if retrieve_error == True:
            params, _ = curve_fit(fit_fn, lambdas, error_data)
            a, b = params
            extrapolation_error = fit_fn(0, a, b)
        else:
            extrapolation_error = None
   
    return extrapolated_result, extrapolation_error


def get_populations(n_emitters, result):
    """
    Measures the excited state population of each qubit from a PUBResult

    Parameters:
        n_emitters (int): Number of quantum emitters (system qubits)
        result (PUBResult): Output result from a Sampler job

    Returns:
        dict: Dictionary with keys '0', '1', ..., 'n-1' (labeling each qubit)
             and values, a list containing the expectation value of the population of that 
             qubit for each time instant (number of instants = len(result)).
    """
    
    evs = {str(i) : [] for i in range(n_emitters)}

    for k in range(len(result)):
        pub_result = result[k]
        shots = pub_result.data.c.num_shots
        counts = pub_result.data.meas.get_counts()
        states = [key[math.trunc(n_emitters/2):] for key in counts.keys()] # Output states
        coeff = [ np.sqrt(counts[key]/shots) for key in counts.keys()] # Normalized coefficients
        eigenvalues = [1, -1] # Z eigenvalues

        for i in range(n_emitters):

            evs[str(i)].append(0.5*( 1 - sum([ (coeff[j]**2)*eigenvalues[int(states[j][-i-1])] for j in range(len(states)) ]))) # Fix this
    
    return evs