from qiskit import QuantumCircuit, QuantumRegister
from qiskit_ibm_runtime import SamplerV2, Batch, QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from typing import Union
from qiskit.providers import BackendV2

import numpy as np
import math

def sampler_solver(qc:QuantumCircuit, n:int, t:list,
                  backend:Union[str, BackendV2], optimization_level:int, options:dict, initial_layout) -> dict:
    
    version = 'v2+'
    # if qc.width() == 2*(n + math.trunc(n/2)) + (n-1): #Sum of system register, environment, classical bits AND meas in measure_all
    #                                                 # meas register has the same size as the sum of the two quantum registers
    #     version = 'v2+'
    
    # else:

    #     version = 'v1'


    #shots = options['default_shots']
    shots = options.default_shots

    evs = {str(i) : [] for i in range(n)}
    ev_squared = {str(i) : [] for i in range(n)}
    std_devs = {str(i) : [] for i in range(n)}

    pm = generate_preset_pass_manager(backend = backend, optimization_level = optimization_level, initial_layout = initial_layout)#, routing_method='none')
    trans_qc = pm.run(qc)

    #trans_qc = qc
    pubs = [(trans_qc, x) for x in t]

    ############ EXPERIMENTAL ##############

    max_circuits = len(t)

    jobs = []
    
    with Batch(backend=backend):

        sampler = SamplerV2(options = options)
        job = sampler.run(pubs)
        jobs.append(job)

        # for i in range(0, len(t), max_circuits):

        #     if i + max_circuits <= len(t):

        #         job = sampler.run(pubs[i : i + max_circuits])

        #     else:
                
        #         job = sampler.run(pubs[i : len(t)])

        #     jobs.append(job)

    # Classical post-processing

    for j in jobs: 

        result = j.result() 

        # All jobs (except the last one typically) will contain the same number of PUBs so we have to iterate over the number of PUBs
        # which is equal to len(result)

        for k in range(len(result)):

            pub_result = result[k]
            counts = pub_result.data.meas.get_counts()
            #evs = get_spsm_evs(n, counts, shots, evs)
            states = [key[n:] for key in counts.keys()] # Output states
        
            if version == 'v2+':

                states = [key[math.trunc(n/2):] for key in counts.keys()]

            coeff = [ np.sqrt(counts[key]/shots) for key in counts.keys()] # Normalized coefficients
            eigenvalues = [1, -1] # Z eigenvalues
            
            for i in range(n):
                evs[str(i)].append(0.5*( 1 - sum([ (coeff[j]**2)*eigenvalues[int(states[j][-i-1])] for j in range(len(states)) ]))) 
                ev_squared[str(i)].append(0.25*( 1 + sum([ (coeff[j]**2)*(eigenvalues[int(states[j][-i-1])]**2) for j in range(len(states)) ]) - 2*sum([ (coeff[j]**2)*eigenvalues[int(states[j][-i-1])] for j in range(len(states)) ]))) 
                #std_devs[str(i)].append(np.sqrt( ev_squared[str(i)][k] - evs[str(i)][k]**2 )/np.sqrt(shots)) 
                
    return evs#, std_devs


