from qiskit import QuantumCircuit
import numpy as np

def pauli_twirling(circuit:QuantumCircuit, sample:int):
    # First we will define a dictionary with the pairs of sandwiching gates
    # such that when a CX is sandwhiched between key : value,
    # the resulting twirled gate is equivalent to a CX.
    conjugate_gates = {
        ('I', 'I'): ('I', 'I'),
        ('I', 'X'): ('I', 'X'),
        ('I', 'Y'): ('Z', 'Y'),
        ('I', 'Z'): ('Z', 'Z'),
        ('X', 'I'): ('X', 'X'),
        ('X', 'X'): ('X', 'I'),
        ('X', 'Y'): ('Y', 'Z'),
        ('X', 'Z'): ('Y', 'Y'),
        ('Y', 'I'): ('Y', 'X'),
        ('Y', 'X'): ('Y', 'I'),
        ('Y', 'Y'): ('X', 'Z'),
        ('Y', 'Z'): ('X', 'Y'),
        ('Z', 'I'): ('Z', 'I'),
        ('Z', 'X'): ('Z', 'X'),
        ('Z', 'Y'): ('I', 'Y'),
        ('Z', 'Z'): ('I', 'Z')
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
    pauli_gates['Y'].y(0)
    pauli_gates['Z'].z(0)

    pauli_sampled = []

    for i in range(sample):     
        new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
        
        for instruction in circuit.data:
            operation = instruction.operation
            qubits = instruction.qubits
            clbits = instruction.clbits  # Classical bits for measurement
            
            if operation.name == 'cx':  

                P_1, P_2 = np.random.choice(['I', 'X', 'Y', 'Z'], size = 2)
                P_1_prime, P_2_prime = conjugate_gates[(P_1, P_2)]

                new_circuit.compose(pauli_gates[P_1], qubits[0], inplace = True)
                new_circuit.compose(pauli_gates[P_2], qubits[1], inplace = True)

                new_circuit.cx(qubits[0], qubits[1])

                new_circuit.compose(pauli_gates[P_1_prime], qubits[0], inplace = True)
                new_circuit.compose(pauli_gates[P_2_prime], qubits[1], inplace = True)

                print(P_1, P_2, P_1_prime, P_2_prime, qubits[0], qubits[1])
            
            elif operation.name == 'if_else': # Do the same twirling as in the parent loop

                true_block = QuantumCircuit(*circuit.qregs, *circuit.cregs)
                false_block = QuantumCircuit(*circuit.qregs, *circuit.cregs)
                new_blocks = [true_block, false_block]
                for index, block in enumerate(operation.blocks):
                    for sub_instruction in block.data:
                        sub_operation = sub_instruction.operation
                        sub_qubits = sub_instruction.qubits
                        sub_clbits = sub_instruction.clbits  

                        if sub_operation.name == 'cx':  
                            P_1, P_2 = np.random.choice(['I', 'X', 'Y', 'Z'], size = 2)
                            P_1_prime, P_2_prime = conjugate_gates[(P_1, P_2)]

                            new_blocks[index].compose(pauli_gates[P_1], sub_qubits[0], inplace = True)
                            new_blocks[index].compose(pauli_gates[P_2], sub_qubits[1], inplace = True)

                            new_blocks[index].cx(sub_qubits[0], sub_qubits[1])

                            new_blocks[index].compose(pauli_gates[P_1_prime], sub_qubits[0], inplace = True)
                            new_blocks[index].compose(pauli_gates[P_2_prime], sub_qubits[1], inplace = True)

                        else:
                            new_blocks[index].append(sub_operation, sub_qubits, sub_clbits)
                
                new_ifelse = operation.replace_blocks(new_blocks)
                new_circuit.append(new_ifelse, qubits, circuit.clbits)
            else:
                new_circuit.append(operation, qubits, clbits)  # Append other operations

            

        pauli_sampled.append(new_circuit)

    return pauli_sampled
