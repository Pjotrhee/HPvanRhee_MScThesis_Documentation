import numpy as np
import cvxpy as cp

def optimal_ref(data, d_hat):

    A = data['A']
    B = data['B']
    C = data['C']
    Bd = data['Bd']
    Dd = data['Dd']

    H = data['H_ref']

    # Dimensions
    dim_n = A.shape[0]
    dim_m = B.shape[1]
    dim_k = C.shape[0]
    dim_f = 1
    dim_p = 1

    y_ref = np.vstack([np.ones((dim_f, 1)), np.zeros((dim_p,1))]) # setpoint (y_sp)
    r_ref = H @ y_ref

    # Decision variables
    x = cp.Variable((dim_n,1))
    u = cp.Variable((dim_m,1))

    # Constraints
    dPmax = data['dPmax']
    dPmin = data['dPmin']
    bx = np.array([[dPmax],[-dPmin]])
    Ax = np.array([[0,0,1,0],[0,0,-1,0]])

    Constraints = [
        (np.eye(dim_n) - A) @ x - B @ u == Bd @ d_hat,
        H @ C @ x == r_ref - H @ Dd @ d_hat,
        Ax @ x <= bx
    ]

    # Objective
    R = np.eye(dim_m)
    J = cp.quad_form(u, R)
    Objective = cp.Minimize(J)
    
    prob = cp.Problem(Objective, Constraints)
    prob.solve(solver=cp.GUROBI)

    return x.value, u.value
