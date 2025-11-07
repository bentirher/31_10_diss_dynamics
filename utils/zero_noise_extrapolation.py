from qiskit.circuit import IfElseOp
from qiskit import QuantumCircuit
import copy
import numpy as np

def get_parallel_cx_layers(circuit: QuantumCircuit):
    cx_layers = []
    
    def place_in_layer(idx, q0, q1, tag, parent_if_else_idx = None, branch_id = None):
        
        for layer in cx_layers:
            # Reject layer if it has a different tag
            if any(g[3] != tag for g in layer):
                continue

            if tag == 'outer':
                max_gap = 150 
                # Get max index of any gate in this layer
                layer_max_idx = max(g[0] for g in layer)

                # Check if current gate is within acceptable depth range
                if idx - layer_max_idx > max_gap:
                    break  # too far away, stop checking older layers
            else:        
                # For inner layers: ensure parent idx and branch ID match, and all gates are at same idx
                if any(g[5] != branch_id for g in layer):
                    continue
                existing_idxs = {g[0] for g in layer}
                if existing_idxs and idx not in existing_idxs:
                    continue

            used_qubits = {q for gate in layer for q in gate[1:3]}
            if q0 not in used_qubits and q1 not in used_qubits:
                layer.append((idx, q0, q1, tag, parent_if_else_idx, branch_id))
                return
        # No compatible layer found, start a new one
        cx_layers.append([(idx, q0, q1, tag, parent_if_else_idx, branch_id)])

    def process_block(block, tag = 'outer', parent_if_else_idx=None, branch_id=None):
        for idx, instr in enumerate(block.data):
            qargs = instr.qubits
        
            if instr.name == 'cx':
                q0 = block.find_bit(qargs[0]).index
                q1 = block.find_bit(qargs[1]).index

                place_in_layer(idx, q0, q1, tag, parent_if_else_idx, branch_id)
                
            elif instr.name == 'if_else':
                # Recursively process each branch at this point
                operation = instr.operation
                blocks = operation.blocks
                for b_idx, branch in enumerate(blocks):
                    process_block(branch, tag = 'inner', parent_if_else_idx=idx, branch_id=b_idx)

    process_block(circuit)
    return cx_layers


def merge_inner_layers(layers):
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


def fold_cx_layers_by_layer(circuit: QuantumCircuit, n: int, s: int, folding_method='left') -> QuantumCircuit:
    
    cx_layers = merge_inner_layers(get_parallel_cx_layers(circuit))
    num_layers = len(cx_layers)
    print(cx_layers)

    if s > num_layers:
        raise ValueError("s is greater than number of CX layers")

    # Select layers to fold
    if folding_method == 'random':
        fold_indices = set(np.random.choice(range(num_layers), size=s, replace=False))
    elif folding_method == 'left':
        fold_indices = set(range(s))
    elif folding_method == 'right':
        fold_indices = set(range(num_layers - s, num_layers))
    else:
        raise ValueError(f"Unsupported folding method: {folding_method}")
    
    # Identify exactly which CXs we want to fold based on layer
    cx_to_fold = set()
    for i in fold_indices:
        for (idx, q0, q1, tag, parent_if_else_idx, branch_id) in cx_layers[i]:
            cx_to_fold.add((idx, q0, q1, tag, parent_if_else_idx, branch_id))
    print(cx_to_fold)     

    def fold_qc(block: QuantumCircuit, tag = 'outer', parent_if_else_idx = None, branch_id = None) -> QuantumCircuit:
        new_block = copy.deepcopy(block)
        new_block.data = []

        for idx, inst in enumerate(block.data):
            qargs = inst.qubits
            cargs = inst.clbits
            operation = inst.operation
            
            if inst.name == 'cx':
                q0 = block.find_bit(qargs[0]).index
                q1 = block.find_bit(qargs[1]).index

                new_block.cx(qargs[0], qargs[1])

                if (idx, q0, q1, tag, parent_if_else_idx, branch_id) in cx_to_fold:
                    fold_times = n + 1
                else:
                    fold_times = n
                    
                for _ in range(fold_times):
                    new_block.cx(qargs[0], qargs[1])
                    new_block.cx(qargs[0], qargs[1])
                    

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

    return fold_qc(circuit)