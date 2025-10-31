import sympy as sp
from sympy import Matrix, Inverse
from sympy.physics.quantum import TensorProduct
from sympy import Symbol, sqrt
import numpy as np

def get_kraus(transformed_kets:list[Matrix], basis:str, system_dim:int, ancilla_dim:int, test_CPTP:bool) -> list:

    """
    Computes the Kraus operators of a given quantum channel by its transformation equations
    and prints whether the channel is CPTP or not.
    
    Parameters
    ----------
    transformed_kets : list[Matrix]
    List containing the transformed vectors, ordered in the same way as the basis set.
    basis : str
    Basis choice for the Kraus operators. 'computational', 'coupled'

    Returns
    -------
    list
    A list of the Kraus operators representing the channel

    Description
    -----------
    

    Examples

    """
    #system_dim = 2 # The system (or ancilla) space has dimension equal to the number of elements in its basis set.
    #ancilla_dim = 4

    # Computational basis construction

    system_basis_set = []
    ancilla_basis_set = []

    for i in range(system_dim):

        zeros = sp.zeros(system_dim,1)

        zeros[i] = 1

        system_basis_set.append(zeros)

    for i in range(ancilla_dim):

        zeros = sp.zeros(ancilla_dim,1)

        zeros[i] = 1

        ancilla_basis_set.append(zeros)

    # Isometry matrix V

    # First we will initialize V as a list with whatever

    V = [i for i in range(system_dim)] 

    system_iden = sp.eye(system_dim)
    ancilla_iden = sp.eye(ancilla_dim)

    # Then we will compute the matrix elements of V 

    for i in range(system_dim):

        V[i] = []

        for j in range(system_dim):

            V[i].append(TensorProduct(system_basis_set[i].transpose(), ancilla_iden)*transformed_kets[j])
        
    matrix_V = Matrix(V)

    # And finally the Kraus operators.

    untransformed_kraus = [ TensorProduct(system_iden, ancilla_basis_set[i].transpose())*matrix_V for i in range(ancilla_dim)]

    # Finally, if the basis choice is the coupled one, we will transform the Kraus matrices 
    # through the basis change matrix S

    if basis == 'coupled':

        S = Matrix([[1, 0, 0, 0], [0, -1, 1, 0], [0, 1, 1, 0],[0, 0, 0, 1]])
        kraus = [S.adjoint()*x*S for x in untransformed_kraus]
        
    else:

        kraus = untransformed_kraus

    # We can also check if the channel is CPTP.

    if test_CPTP == True:

        # Complete Positivity: We have to check that \sum_ij K_ij^dag K_ij = I

        elements = [k.transpose()*k for k in kraus]

        cum_sum = sp.zeros(system_dim)

        for e in elements:

            cum_sum = cum_sum + e

        simp_sum = sp.simplify(cum_sum).applyfunc(round) # This is needed to force SymPy to simplify stuff like p_1 - p_1.
        # Additionally, round is included so that 1.0 is turned into 1

        # Trace preserving: We can infer it by checking that <i|i> = 1 for i in transformed_kets (i.e, they are normalized)

        norms = [sp.simplify((i.transpose()*i)[0]) for i in transformed_kets]

        if simp_sum == system_iden and (sum(norms)/system_dim) == 1:

            print('The channel is CPTP')
        
        elif simp_sum == system_iden and (sum(norms)/system_dim) != 1:

            print('The channel is CP but NOT TP')
        
        elif simp_sum != system_iden and (sum(norms)/system_dim) == 1:

            print('The channel is TP but NOT CP')
        
        else:

            print('The channel is neither CP nor TP')


    return kraus

    
def get_unitary(kraus:list[Matrix], basis:str) -> list:

    """
    Computes the unitary evolution operator associated to a quantum channel in
    the markovian approximation.
    
    Parameters
    ----------
    transformed_kets : list[Matrix]
    List containing the transformed vectors, ordered in the same way as the basis set.
    basis : str
    Basis state of the environment. 'computational', 'coupled'

    Returns
    -------
    Matrix

    The Unitary evolution operator

    Description
    -----------
    

    Examples

    """
    system_dim = kraus[0].shape[0]
    
    total_dim = system_dim**2

    basis_kets = []

    for i in range(system_dim):

        zeros = sp.zeros(system_dim,1)

        zeros[i] = 1

        basis_kets.append(zeros)

    elements = [TensorProduct(kraus[i], basis_kets[i]*basis_kets[0].transpose()) for i in range(system_dim)]

    u = sp.zeros(total_dim)

    for e in elements:

        u = u + e

    return u