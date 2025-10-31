import numpy as np
import math

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
