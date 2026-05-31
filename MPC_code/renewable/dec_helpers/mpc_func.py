import cvxpy as cp
import numpy as np

def mpc_func(psihat_0, x_ref, u_ref, N, data):

    dim_m = data['dim_m']

    A_p = data['A_p']
    b_p = data['b_p']
    P_tilde = data['P_tilde']

    H = data['H']
    h = data['h']

    x_ref_hor = np.kron(np.ones((N+1,1)), x_ref)
    u_ref_hor = np.kron(np.ones((N,1)), u_ref)
    u_hor = cp.Variable((dim_m*N, 1))

    Const_P = [A_p @ u_hor <= P_tilde + (b_p @ psihat_0)]
    Constraints = Const_P

    J = cp.quad_form(u_hor, H) + (2 * np.hstack([psihat_0.T, -x_ref_hor.T, -u_ref_hor.T]) @ h @ u_hor)
    Objective = cp.Minimize(J)
    
    prob = cp.Problem(Objective, Constraints)
    prob.solve(solver=cp.GUROBI, verbose=False)

    u_hor_opt = u_hor.value
    u_k = u_hor_opt[0:1]  

    return float(u_k)
