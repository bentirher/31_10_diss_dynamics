from qiskit.transpiler import Layout
import math
from qiskit import QuantumRegister

from typing import Union
from qiskit.providers import BackendV2

def chain_init_layout(n:int, backend: BackendV2, system:QuantumRegister, environment:QuantumRegister):

    """

    Creates an initial layout of $n$ Markovian emitters using the best
    chain of physical qubits available in the `backend`.

    Parameters
    ----------
    n : int
        Number of qubits in the system's register (i.e., number of emitters).
    backend : BackendV1, BackendV2
        Target backend for the circuit. Used to determine the initial layout.
    system : QuantumRegister
        Register containing the system qubits
    environment: QuantumRegister
        Register containing the ancillary qubits

    Returns
    -------
    Layout

    Description
    -----------
   
    Examples
    --------
    For a Markovian chain of $n=4$, this would return a layout that maps
    the virtual qubits to physical qubits as q_0 - q_1 - e_0 - q_2 - q_3 - e_1.

    """

    total_qubits_needed = n + math.trunc(n/2)

    if backend.name == 'aer_simulator':

        return None # Layout only makes sense for real/fake backend simulations
        
    else:

        if total_qubits_needed <= 4:
            
            physical_qubits = backend.properties().general_qlists[0]['qubits']

        else:

            physical_qubits = backend.properties().general_qlists[total_qubits_needed-4]['qubits']
            
        init_layout = Layout()
        site = 0

        for i in range(0,n,2):

            init_layout.add(system[i], physical_qubits[site])
            site = site + 1

            if i+1<=(n-1):

                init_layout.add(system[i+1], physical_qubits[site])
                site = site + 1

            j = int(i/2)

            if j<(math.trunc(n/2)):

                init_layout.add(environment[j], physical_qubits[site])

            site = site +1
    
        return init_layout