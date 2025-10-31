from qiskit import QuantumCircuit
import numpy as np
from utils.get_circuit_properties import get_circuit_properties
import random

def fold_cz(isa_circuit, rep):

    new_circuit = QuantumCircuit(*isa_circuit.qregs, *isa_circuit.cregs)
    
    for instruction in isa_circuit.data:
        operation = instruction.operation
        qubits = instruction.qubits
        clbits = instruction.clbits 
        
        if operation.name == 'cz':  # Replace each CZ with 2*n + 1 CZs

            new_circuit.cz(qubits[0], qubits[1]) # This is the original

            for _ in range(rep): # We insert then n identities
                new_circuit.cz(qubits[0], qubits[1])
                new_circuit.cz(qubits[0], qubits[1])
        else:
            new_circuit.append(operation, qubits, clbits)  # Append other operations
    
    return new_circuit

def fold_cx(circuit, rep):
    # Preserve original quantum + classical registers
    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    
    for instruction in circuit.data:
        operation = instruction.operation
        qubits = instruction.qubits
        clbits = instruction.clbits  # Classical bits for measurement
        
        if operation.name == 'cx':  # Replace each CX with 2*rep + 1 CZs

            new_circuit.cx(qubits[0], qubits[1]) # This is the original

            for _ in range(rep): # We insert then rep identities
                new_circuit.cx(qubits[0], qubits[1])
                new_circuit.cx(qubits[0], qubits[1])
        else:
            new_circuit.append(operation, qubits, clbits)  # Append other operations
    
    return new_circuit

def fold_some_cx(circuit, n, s, folding_method):

    ''''
    Performs a partial gate folding of the circuits CX gates, following
    a `folding_method`

    ''' 

    d = get_circuit_properties(circuit)['two qubit depth'] 
    n_emitters = round(circuit.num_qubits/2)
    
    if n_emitters == 2: # This is to account for the two-qubit gate layers inside the IfElse block,
        # which are not considered in the tqd count. Instead, Qiskit considers each block (True and False)
        # of the IfElseOp to contribute 1 unit to the two qubit depth, thats why we are substracting 2.

        d = d -2 + 6 
    
    elif n_emitters > 3:

        d = d -2 + 12

    if folding_method == 'random':

        subset = np.random.choice(np.arange(0, d), size=s, replace=False)

    # Preserve original quantum + classical registers
    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)

    cx_count = 0
    
    for instruction in circuit.data:
        operation = instruction.operation
        qubits = instruction.qubits
        clbits = instruction.clbits  

        if (operation.name == 'cx') and (folding_method == 'left'):  
            
            new_circuit.cx(qubits[0], qubits[1]) # This is the original

            if cx_count < s: # That cx is in S

                for _ in range(n + 1): # We insert then n+1 identities

                    new_circuit.cx(qubits[0], qubits[1])
                    new_circuit.cx(qubits[0], qubits[1])

                cx_count = cx_count + 1 
            
            else: # Then that cx is not in S

                for _ in range(n): # We insert then n identities

                    new_circuit.cx(qubits[0], qubits[1])
                    new_circuit.cx(qubits[0], qubits[1])

        elif (operation.name == 'cx') and (folding_method == 'right'):

            new_circuit.cx(qubits[0], qubits[1]) # This is the original

            if cx_count < (d-s): # That cx is NOT in S

                for _ in range(n): # We insert then n identities

                    new_circuit.cx(qubits[0], qubits[1])
                    new_circuit.cx(qubits[0], qubits[1])

                cx_count = cx_count + 1

            else: # That cx is in S

                for _ in range(n + 1): # We insert then n+1 identities

                    new_circuit.cx(qubits[0], qubits[1])
                    new_circuit.cx(qubits[0], qubits[1])

        elif (operation.name == 'cx') and (folding_method == 'random'):

            new_circuit.cx(qubits[0], qubits[1]) # This is the original

            if (cx_count in subset): # Then this CX is in S

                for _ in range(n + 1): # We insert then n+1 identities

                    new_circuit.cx(qubits[0], qubits[1])
                    new_circuit.cx(qubits[0], qubits[1])
            
            else: # Then the CX is not in S and so will insert just n identities

                for _ in range(n): # We insert then nidentities

                    new_circuit.cx(qubits[0], qubits[1])
                    new_circuit.cx(qubits[0], qubits[1])

            cx_count = cx_count + 1 #This count needs to be updated anyways since,
            # in the random folding case, this is simply counting the CX layers

        elif instruction.name == 'if_else':
            true_block = QuantumCircuit(*circuit.qregs, *circuit.cregs)
            false_block = QuantumCircuit(*circuit.qregs, *circuit.cregs)
            new_blocks = [true_block, false_block]
            for index, block in enumerate(operation.blocks):
                conditional_cx_count = cx_count # This count reset is to ensure that we are folding both blocks in the 
                # same way.
                # new_blocks[index] = fold_some_cx(block, n, s, folding_method) This wont work apparently
                # so i am repeating the same loop as before
                for sub_instruction in block.data:
                    sub_operation = sub_instruction.operation
                    sub_qubits = sub_instruction.qubits
                    sub_clbits = sub_instruction.clbits  

                    if (sub_operation.name == 'cx') and (folding_method == 'left'):  
                        
                        new_blocks[index].cx(sub_qubits[0], sub_qubits[1]) # This is the original

                        if conditional_cx_count < s: # That cx is in S

                            for _ in range(n + 1): # We insert then n+1 identities

                                new_blocks[index].cx(sub_qubits[0], sub_qubits[1])
                                new_blocks[index].cx(sub_qubits[0], sub_qubits[1])

                            conditional_cx_count = conditional_cx_count + 1 
                        
                        else: # Then that cx is not in S

                            for _ in range(n): # We insert then n identities

                                new_blocks[index].cx(sub_qubits[0], sub_qubits[1])
                                new_blocks[index].cx(sub_qubits[0], sub_qubits[1])

                    elif (sub_operation.name == 'cx') and (folding_method == 'right'):

                        new_blocks[index].cx(sub_qubits[0], sub_qubits[1]) # This is the original

                        if conditional_cx_count < (d-s): # That cx is NOT in S

                            for _ in range(n): # We insert then n identities

                                new_blocks[index].cx(sub_qubits[0], sub_qubits[1])
                                new_blocks[index].cx(sub_qubits[0], sub_qubits[1])

                            conditional_cx_count = conditional_cx_count + 1

                        else: # That cx is in S

                            for _ in range(n + 1): # We insert then n+1 identities

                                new_blocks[index].cx(sub_qubits[0], sub_qubits[1])
                                new_blocks[index].cx(sub_qubits[0], sub_qubits[1])

                    elif (sub_operation.name == 'cx') and (folding_method == 'random'):

                        #HERE WE WOULD NEED TO USE THE SAME RANDOM CHOICES ON THE TWO BLOCKS FOR THEM TO BE EQUAL.

                        new_blocks[index].cx(sub_qubits[0], sub_qubits[1]) # This is the original

                        if (conditional_cx_count in subset): # Then this CX is in S

                            for _ in range(n + 1): # We insert then n+1 identities

                                new_blocks[index].cx(sub_qubits[0], sub_qubits[1])
                                new_blocks[index].cx(sub_qubits[0], sub_qubits[1])
                        
                        else: # Then the CX is not in S and so will insert just n identities

                            for _ in range(n): # We insert then nidentities

                                new_blocks[index].cx(sub_qubits[0], sub_qubits[1])
                                new_blocks[index].cx(sub_qubits[0], sub_qubits[1])

                        conditional_cx_count = conditional_cx_count + 1 #This count needs to be updated anyways since,
                        # in the random folding case, this is simply counting the CX layers

                    else:

                        new_blocks[index].append(sub_operation, sub_qubits, sub_clbits)

            cx_count = conditional_cx_count
            new_ifelse = operation.replace_blocks(new_blocks)
            new_circuit.append(new_ifelse, new_blocks[0].qubits, circuit.clbits)

        else:
            new_circuit.append(operation, qubits, clbits)  # Append other operations
    
    return new_circuit

from qiskit.converters import circuit_to_dag
import copy

def fold_some_cz(circuit, n, s, folding_method='left'):
    '''
    Perform partial folding of CZ **layers**.
    Folds s = |S| CZ layers (d = two-qubit depth) (n+1) times, 
    and layers outside S are folded n times.

    Supports left/right/random strategies for bulding the S subset.
    '''

    def fold_layer(qc, cz_ops, fold_times):
        for _ in range(fold_times):
            for op in cz_ops:
                qc.cz(op.qargs[0], op.qargs[1])
            for op in reversed(cz_ops):
                qc.cz(op.qargs[0], op.qargs[1])

    def get_cz_layers(dag):
        cz_layers = 0
        for layer in dag.layers():
            ops = layer['graph'].op_nodes()
            if is_cz_layer(ops):
                cz_layers += 1
            if contains_if_else(ops):
                ifelse = [op for op in ops if op.name == 'if_else']
                dag_block = circuit_to_dag(ifelse[0].op.blocks[0])
                cz_layers += get_cz_layers(dag_block)

        return cz_layers

    def is_cz_layer(ops):
        return True if 'cz' in [op.name for op in ops] else False

    def contains_if_else(ops):
        return True if 'if_else' in [op.name for op in ops] else False

    def fold_block(block, n, cz_layer_idx, subset):
        dag = circuit_to_dag(block)
        new_block = copy.deepcopy(block)
        new_block.data = []

        for layer in dag.layers():
            ops = layer['graph'].op_nodes()
            if (is_cz_layer(ops) == True):
                cz_layer_idx += 1
            else: 
                cz_layer_idx += 0

            cz_ops = [op for op in ops if op.name == 'cz']
            other_ops = [op for op in ops if op.name != 'cz']

            if cz_ops:
                fold_times = (n + 1) if cz_layer_idx in subset else n
                # Add original
                for op in cz_ops:
                    new_block.cz(op.qargs[0], op.qargs[1])
                # Add folding
                fold_layer(new_block, cz_ops, fold_times)

            for op in other_ops:
                new_block.append(op.op, op.qargs, op.cargs)
        post_condit_cz_layer_idx = cz_layer_idx
        return new_block, post_condit_cz_layer_idx

    ### MAIN LOOP ###
    dag = circuit_to_dag(circuit)
    d = get_cz_layers(dag)
    print(f'The number of CZ layers is {d}')

    all_indices = list(range(d))

    # Choose which layer indices will be folded (n+1 times)
    if folding_method == 'random':
        subset = set(np.random.choice(all_indices, size=s, replace=False))
    elif folding_method == 'left':
        subset = set(all_indices[:s])
    elif folding_method == 'right':
        subset = set(all_indices[-s:])
    else:
        raise ValueError("Invalid folding method")

    new_circ = QuantumCircuit(*circuit.qregs, *circuit.cregs)

    cz_layer_idx = -1
    for layer in dag.layers():
        ops = layer['graph'].op_nodes()

        if (is_cz_layer(ops) == True):
                cz_layer_idx += 1
        else: 
                cz_layer_idx += 0

        # Separate CZ and other ops
        cz_ops = [op for op in ops if op.name == 'cz']
        other_ops = [op for op in ops if op.name != 'cz']

        if cz_ops:
            # Append original CZs
            for op in cz_ops:
                new_circ.cz(op.qargs[0], op.qargs[1])

            fold_times = (n + 1) if cz_layer_idx in subset else n
            fold_layer(new_circ, cz_ops, fold_times)

        for op in other_ops:
            # Look if there is a conditional block in the layer
            if op.name == 'if_else':
                blocks = op.op.blocks
                folded_blocks = []
                for b in blocks:
                    folded_b, post_condit_cz_layer_idx = fold_block(b, n, cz_layer_idx, subset)
                    folded_blocks.append(folded_b)

                new_ifelse = op.op.replace_blocks(folded_blocks)
                new_circ.append(new_ifelse, op.qargs, op.cargs)
                cz_layer_idx = post_condit_cz_layer_idx
            else:
                new_circ.append(op.op, op.qargs, op.cargs)
        print(cz_layer_idx, ops)

    return new_circ

def fold_one_layer(circuit, layer_index, n):

    ''''
    Performs a single-layer folding of the `index` layer.

    ''' 

    d = get_circuit_properties(circuit)['two qubit depth'] 
    n_emitters = round(circuit.num_qubits/2)
    
    if n_emitters == 2: # This is to account for the two-qubit gate layers inside the IfElse block,
        # which are not considered in the tqd count. Instead, Qiskit considers each block (True and False)
        # of the IfElseOp to contribute 1 unit to the two qubit depth, thats why we are substracting 2.

        d = d -2 + 6 
    
    elif n_emitters > 3:

        d = d -2 + 12

    # Preserve original quantum + classical registers
    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)

    cx_tracking = 0
    
    for instruction in circuit.data:
        operation = instruction.operation
        qubits = instruction.qubits
        clbits = instruction.clbits  

        if (operation.name == 'cx'): 
            
            new_circuit.cx(qubits[0], qubits[1]) # This is the original

            if (layer_index == cx_tracking):  
                for _ in range(n): # We insert then n identities
                    new_circuit.cx(qubits[0], qubits[1])
                    new_circuit.cx(qubits[0], qubits[1])

            cx_tracking = cx_tracking + 1

        elif instruction.name == 'if_else':
            true_block = QuantumCircuit(*circuit.qregs, *circuit.cregs)
            false_block = QuantumCircuit(*circuit.qregs, *circuit.cregs)
            new_blocks = [true_block, false_block]

            for idx, block in enumerate(operation.blocks):
                conditional_cx_count = cx_tracking # This count reset is to ensure that we are folding both blocks in the 
                # same way.
                for sub_instruction in block.data:
                    sub_operation = sub_instruction.operation
                    sub_qubits = sub_instruction.qubits
                    sub_clbits = sub_instruction.clbits  

                    if (sub_operation.name == 'cx'):  
                        
                        new_blocks[idx].cx(sub_qubits[0], sub_qubits[1]) # This is the original

                        if (layer_index == conditional_cx_count): 

                            for _ in range(n):
                                new_blocks[idx].cx(sub_qubits[0], sub_qubits[1])
                                new_blocks[idx].cx(sub_qubits[0], sub_qubits[1])

                        conditional_cx_count = conditional_cx_count + 1
        
                    else:

                        new_blocks[idx].append(sub_operation, sub_qubits, sub_clbits)

            cx_tracking = conditional_cx_count
            new_ifelse = operation.replace_blocks(new_blocks)
            new_circuit.append(new_ifelse, new_blocks[0].qubits, circuit.clbits)

        else:
            new_circuit.append(operation, qubits, clbits)  # Append other operations
    
    return new_circuit

def replace_cx_with_cp(circuit, rep):
    # Preserve original quantum + classical registers
    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    
    for instruction in circuit.data:
        operation = instruction.operation
        qubits = instruction.qubits
        clbits = instruction.clbits  # Classical bits for measurement
        
        if operation.name == 'cx':  # Replace each CX with 2*rep CPs
            new_circuit.h(qubits[1])

            for _ in range(rep): 
                new_circuit.cp(np.pi/rep, qubits[0], qubits[1])

            new_circuit.h(qubits[1])
        else:
            new_circuit.append(operation, qubits, clbits)  # Append other operations
    
    return new_circuit
