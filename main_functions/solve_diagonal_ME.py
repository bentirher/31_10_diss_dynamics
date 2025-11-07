import sys
sys.path.append("..") # This is for the imports from adjacent folders to work
from main_functions.ME_solver import solve_master_equation, get_hamiltonian, get_observables
from qutip import basis, Qobj, tensor, qeye, mesolve
import numpy as np

def solve_diag_ME(n:int, omega_m:float, omega_c:float, gamma:list, kappa:list, g:float, t:list, initial_state:list):
    system_dim = 2**n
    qubit_states = []
    for i in range(n):
        if str(i) in initial_state:
            # Qubit i is in the excited state |1>
            qubit_states.append(basis(2, 1))
        else:
            # Qubit i is in the ground state |0>
            qubit_states.append(basis(2, 0))
    rho0 = tensor(qubit_states).unit()
    rho0.dims = [[system_dim], [1]]
    hamiltonian = get_hamiltonian(n, omega_m, omega_c, kappa, g, 'markovian')
    observables = get_observables(n, 'markovian')

    delta = [ x - omega_c for x in omega_m ]
    mean_delta = [ 0.5*(omega_m[i] + omega_m[i+1]) - omega_c for i in range(n-1)]
    omega_eff = []
    g_eff = []
    gamma_eff = []
    gamma_cross = []
    for i in range(n):
        if i==0:
            omega_eff.append(omega_m[i] + (delta[i]*(g[i]**2))/((0.5*kappa[0])**2 + delta[i]**2))
            gamma_eff.append(gamma[i] + (kappa[0]*(g[i]**2))/((0.5*kappa[0])**2 + delta[i]**2))
        elif i == (n-1):
            j = 2*i
            omega_eff.append(omega_m[i] + (delta[i]*(g[j-1]**2))/((0.5*kappa[0])**2 + delta[i]**2))
            gamma_eff.append(gamma[i] + (kappa[0]*(g[j-1]**2))/((0.5*kappa[0])**2 + delta[i]**2))
        else:  
            j = 2*i
            omega_eff.append(omega_m[i] + 0.5*(delta[i]*(g[j-1]**2))/((0.5*kappa[0])**2 + delta[i]**2) + 0.5*(delta[i]*(g[j]**2))/((0.5*kappa[0])**2 + delta[i]**2))
            gamma_eff.append(gamma[i] + (1/2)*(kappa[0]*(g[j-1]**2))/((0.5*kappa[0])**2 + delta[i]**2) + (1/2)*(kappa[0]*(g[j]**2))/((0.5*kappa[0])**2 + delta[i]**2))

    for i in range(0, n-1):
        j = 2*i
        g_eff.append((g[j]*g[j+1]*(mean_delta[i]))/((kappa[0]/2)**2 + (mean_delta[i])**2))
        gamma_cross.append((g[j]*g[j+1]*(kappa[0]))/((kappa[0]/2)**2 + (mean_delta[i])**2))

    det = [ omega_eff[i+1] - omega_eff[i] for i in range(n-1) ] 
    lam = [ 0.5*np.sqrt(4*g_eff[i]**2 + det[i]**2) for i in range(n-1) ]
    sen_theta = [ g_eff[i]/np.sqrt(lam[i]*(2*lam[i] + det[i])) for i in range(n-1) ]
    cos_theta = [ np.sqrt((2*lam[i] + det[i])/(4*lam[i])) for i in range(n-1) ]

    print(gamma_eff)

    gamma_g_minus = [ (gamma_eff[i]*(cos_theta[i]**2) + gamma_eff[i+1]*(sen_theta[i]**2) - 2*gamma_cross[i]*sen_theta[i]*cos_theta[i]) for i in range(n-1) ]
    gamma_g_plus = [ (gamma_eff[i+1]*(cos_theta[i]**2) + gamma_eff[i]*(sen_theta[i]**2) + 2*gamma_cross[i]*sen_theta[i]*cos_theta[i]) for i in range(n-1) ]
    gamma_minus_e = [ (gamma_eff[i+1]*(cos_theta[i]**2) + gamma_eff[i]*(sen_theta[i]**2) - 2*gamma_cross[i]*sen_theta[i]*cos_theta[i]) for i in range(n-1) ]
    gamma_plus_e = [ (gamma_eff[i]*(cos_theta[i]**2) + gamma_eff[i+1]*(sen_theta[i]**2) + 2*gamma_cross[i]*sen_theta[i]*cos_theta[i]) for i in range(n-1) ]

    ket_00 = Qobj([[1],[0],[0],[0]])
    ket_01 = Qobj([[0],[1],[0],[0]])
    ket_10 = Qobj([[0],[0],[1],[0]])
    ket_11 = Qobj([[0],[0],[0],[1]])

    c_ops = []
    for i in range(0, n-1):
        matrix_P = Qobj(np.array([[1, 0, 0, 0], [0, -sen_theta[i], cos_theta[i], 0], [0, cos_theta[i], sen_theta[i], 0], [0, 0, 0, 1]]))
        ops = [qeye(2)]*(n-1)
        ops[i] = matrix_P @ (ket_00 @ ket_01.dag()) @ matrix_P.dag()
        lind = tensor(ops)
        lind.dims = [[system_dim], [system_dim]]
        c_ops.append(np.sqrt(gamma_g_minus[i]) * lind)

        ops = [qeye(2)]*(n-1)
        ops[i] = matrix_P @ (ket_00 @ ket_10.dag()) @ matrix_P.dag()
        lind = tensor(ops)
        lind.dims = [[system_dim], [system_dim]]
        c_ops.append(np.sqrt(gamma_g_plus[i]) * lind)

        ops = [qeye(2)]*(n-1)
        ops[i] = matrix_P @ (ket_01 @ ket_11.dag()) @ matrix_P.dag()
        lind = tensor(ops)
        lind.dims = [[system_dim], [system_dim]]
        c_ops.append(np.sqrt(gamma_minus_e[i]) * lind)

        ops = [qeye(2)]*(n-1)
        ops[i] = matrix_P @ (ket_10 @ ket_11.dag()) @ matrix_P.dag()
        lind = tensor(ops)
        lind.dims = [[system_dim], [system_dim]]
        c_ops.append(np.sqrt(gamma_plus_e[i]) * lind)

    result = mesolve(hamiltonian, rho0, t, c_ops, observables)

    evs = {}
    for key in observables.keys():
        evs[key] = result.expect[int(key)]
            
    return evs