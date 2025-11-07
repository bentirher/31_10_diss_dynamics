from qiskit import QuantumCircuit
import numpy as np
import copy

def pauli_twirling(circuit:QuantumCircuit, sample:int):
    # First we will define a dictionary with the pairs of sandwiching gates
    # such that when a CZ is sandwhiched between key : value,
    # the resulting twirled gate is equivalent to a CZ.
    conjugate_gates = {
        ('I', 'I'): ('I', 'I'),
        ('I', 'X'): ('Z', 'X'),
        ('I', 'Y'): ('Z', 'Y'),
        ('I', 'Z'): ('I', 'Z'),

        ('X', 'I'): ('X', 'Z'),
        ('X', 'X'): ('Y', 'Y'),
        ('X', 'Y'): ('Y', 'X'),
        ('X', 'Z'): ('X', 'I'),

        ('Y', 'I'): ('Y', 'Z'),
        ('Y', 'X'): ('X', 'Y'),
        ('Y', 'Y'): ('X', 'X'),
        ('Y', 'Z'): ('Y', 'I'),

        ('Z', 'I'): ('Z', 'I'),
        ('Z', 'X'): ('I', 'X'),
        ('Z', 'Y'): ('I', 'Y'),
        ('Z', 'Z'): ('Z', 'Z'),
                }

     

    # We will also pre-define one-qubit circuits, each containing a different Pauli
    # gate, so that composition is easier later on
    pauli_gates = {
        'I' : QuantumCircuit(1),
        'X' : QuantumCircuit(1),
        'Y' : QuantumCircuit(1),
        'Z' : QuantumCircuit(1),

    }

    pauli_gates['X'].x(0)
    pauli_gates['Z'].rz(np.pi, 0)
    pauli_gates['Y'].rz(np.pi, 0) # Y = ZX (up to global phase)
    pauli_gates['Y'].x(0)

    def twirl_block(block: QuantumCircuit):
        new_block = copy.deepcopy(block)
        new_block.data = []    
        
        for instruction in block.data:
            operation = instruction.operation
            qargs = instruction.qubits
            cargs = instruction.clbits 

            if operation.name == 'cz':  

                P_1, P_2 = np.random.choice(['I', 'X', 'Y', 'Z'], size = 2)
                P_1_prime, P_2_prime = conjugate_gates[(P_1, P_2)]

                new_block.compose(pauli_gates[P_1], qargs[0], inplace = True)
                new_block.compose(pauli_gates[P_2], qargs[1], inplace = True)

                new_block.cz(qargs[0], qargs[1])

                new_block.compose(pauli_gates[P_1_prime], qargs[0], inplace = True)
                new_block.compose(pauli_gates[P_2_prime], qargs[1], inplace = True)
            
            elif operation.name == 'if_else': # Do the same twirling as in the parent loop
                new_branches = []

                for index, branch in enumerate(operation.blocks):
                    twirled_branch = twirl_block(branch)
                    new_branches.append(twirled_branch)    

                new_ifelse = operation.replace_blocks(new_branches)
                new_block.append(new_ifelse, qargs, cargs)
            else:
                new_block.append(operation, qargs, cargs)  # Append other operations

        return new_block

    return [twirl_block(circuit) for i in range(sample)]