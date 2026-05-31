import cvxpy as cp
import numpy as np

def mpc_func(xnom_0, dhat_0, x_ref, u_ref, N, data):

    dim_n = data['dim_n']
    dim_m = data['dim_m']
    dim_d = data['dim_d']

    H = data['H']
    h = data['h']

    x_ref_hor = np.kron(np.ones((N+1,1)), x_ref)
    u_ref_hor = np.kron(np.ones((N,1)), u_ref)
    dhat_hor = np.kron(np.ones((N,1)), dhat_0) # Or N+1???
    u_hor = cp.Variable((dim_m*N, 1))

    #Input constraint
    Au = data['Au']
    bu = data['bu']
    Const_u = [Au @ u_hor <= bu]

    #State constraint
    Ax = data['Ax']
    bx = data['bx']
    Cx = data['Cx']
    Dx = data['Dx']
    Const_x = [Ax @ u_hor <= bx + (Cx @ xnom_0) + (Dx @ dhat_hor)]

    #Terminal Constraint
    Tcx = data['Tcx']
    Scu = data['Scu']
    Scd = data['Scd']
    x_hor = Tcx @ xnom_0 +  Scu @ u_hor + Scd @ dhat_hor
    x_final = x_hor[dim_n*N:(dim_n*(N+1)), :]
    c_ellipse = data['c_ellipse']
    P_dare = data['P_dare']
    P_dare = cp.psd_wrap(P_dare)
    Const_t = [cp.quad_form(x_final-x_ref, P_dare) <= c_ellipse]

    #Combine constraints
    Constraints = Const_u + Const_x + Const_t

    #Objective
    J = 0.5*cp.quad_form(u_hor, H) + (np.hstack([xnom_0.T, dhat_hor.T, -x_ref_hor.T, -u_ref_hor.T]) @ h @ u_hor)
    Objective = cp.Minimize(J)
    
    prob = cp.Problem(Objective, Constraints)
    prob.solve(solver=cp.GUROBI, verbose=False)

    u_hor_opt = u_hor.value
    u_k = u_hor_opt[0:dim_m].T

    return u_k
