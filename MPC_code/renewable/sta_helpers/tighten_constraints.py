
import cvxpy as cp
import numpy as np

from andes.models.renewable.sta_helpers.pred_matrices import pred_matrices

def tighten_constraints(data):
    dim_n = data["dim_n"]
    dim_m = data["dim_m"]
    dim_k = data["dim_k"]
    dim_ne = data["dim_ne"]

    K = data["K"]
    L = data["L"]
    L_x = data["L_x"]

    A = data["A"]
    Bu = data["Bu"]
    A_ext = data["A_ext"]
    C_ext = data["C_ext"]

    Ak = A + Bu @ K
    Al_ext = A_ext - L @ C_ext

    Ax = data["Ax"]
    bx = data["bx"]
    Au = data["Au"]
    bu = data["bu"]

    Aw_x = data["Aw_x"]
    bw_x = data["bw_x"]
    Aw = data["Aw"]
    bw = data["bw"]
    Av = data["Av"]
    bv = data["bv"]
  
    ### Determine alpha and N
    alpha  = 2
    N = 0
    while alpha >= 0.4:
        N = N + 1
        #1
        w_x = cp.Variable((dim_n, 1))
        AkN = np.linalg.matrix_power(Ak, N)
        Const = [Aw_x @ w_x <= bw_x]

        one = -np.inf
        for i in range(dim_n):
            for s in [1, -1]:
                Obj = cp.Maximize(s * (AkN[i, :] @ w_x))
                prob = cp.Problem(Obj, Const)
                prob.solve(solver=cp.GUROBI, verbose=False)
                one = max(one, prob.value)

        #2
        w_x = cp.Variable((dim_n, 1))
        Const = [Aw_x @ w_x <= bw_x]

        two = -np.inf
        for i in range(dim_n):
            for s in [1, -1]:
                Obj = cp.Maximize(s * w_x[i, 0])
                prob = cp.Problem(Obj, Const)
                prob.solve(solver=cp.GUROBI, verbose=False)
                two = max(two, prob.value)

        #3
        v = cp.Variable((dim_k, 1))
        AlN_ext = np.linalg.matrix_power(Al_ext, N)
        Const = [Av @ v <= bv]

        three = -np.inf
        for i in range(dim_n):
            for s in [1, -1]:
                Obj = cp.Maximize(s * (AlN_ext[i, :] @ L @ v))
                prob = cp.Problem(Obj, Const)
                prob.solve(solver=cp.GUROBI, verbose=False)
                three = max(three, prob.value)

        #4
        v = cp.Variable((dim_k, 1))
        Const = [Av @ v <= bv]

        four = -np.inf
        for i in range(dim_n):
            for s in [1, -1]:
                Obj = cp.Maximize(s * L[i, :] @ v)
                prob = cp.Problem(Obj, Const)
                prob.solve(solver=cp.GUROBI, verbose=False)
                four = max(four, prob.value)
        
        #5
        w_x = cp.Variable((dim_n, 1))
        AkN = np.linalg.matrix_power(Ak, N)
        Const = [Aw_x @ w_x <= bw_x]

        five = -np.inf
        for i in range(dim_m):
            for s in [1, -1]:
                Obj = cp.Maximize(s * (K[i, :] @ AkN @ w_x))
                prob = cp.Problem(Obj, Const)
                prob.solve(solver=cp.GUROBI, verbose=False)
                five = max(five, prob.value)

        #6
        w_x = cp.Variable((dim_n, 1))
        Const = [Aw_x @ w_x <= bw_x]

        six = -np.inf
        for i in range(dim_m):
            for s in [1, -1]:
                Obj = cp.Maximize(s * K[i, :] @ w_x)
                prob = cp.Problem(Obj, Const)
                prob.solve(solver=cp.GUROBI, verbose=False)
                six = max(six, prob.value)

        alpha = np.max([one/two,three/four, five/six])
        
    print(f'N={N} gives alpha={alpha}')

    
    x_tilde_0 = np.zeros((dim_ne,1))

    T_tilde, S_tilde = pred_matrices(Al_ext,np.eye(dim_ne),N)
    Ak_big = None
    for i in range(N):
        Ai = np.linalg.matrix_power(Ak,i)
        if Ak_big is None:
            Ak_big = Ai
        else:
            Ak_big = np.hstack((Ak_big, Ai))
    Al_ext_big = None
    for i in range(N):
        Ai = np.linalg.matrix_power(Al_ext,i)
        if Al_ext_big is None:
            Al_ext_big = Ai
        else:
            Al_ext_big = np.hstack((Al_ext_big, Ai))

    L_x_big = np.kron(np.eye(N),L_x)
    L_big = np.kron(np.eye(N),L)
    L_xC_ext_big = np.kron(np.eye(N),L_x @ C_ext)

    Aw_big = np.kron(np.eye(N),Aw)
    bw_big = np.kron(np.ones((N,1)),bw)
    Av_big = np.kron(np.eye(N),Av)
    bv_big = np.kron(np.ones((N,1)),bv)

    # Tighten state constraints
    for i in range(Ax.shape[0]):
        a = Ax[i, :]
        b = bx[i, :]

        #1
        w_hor = cp.Variable((dim_ne*N, 1))
        v_hor = cp.Variable((dim_k*N,1))
        w_tilde_hor = w_hor - (L_big @ v_hor)
        x_tilde_hor = (T_tilde @ x_tilde_0) + (S_tilde @ w_tilde_hor)
        x_tilde_hor = x_tilde_hor[0:dim_ne*N,:]

        Obj = cp.Maximize(a @ (Ak_big @ ((L_xC_ext_big @ x_tilde_hor) + (L_x_big @ v_hor))))
        Const_w = [Aw_big @ w_hor <= bw_big]
        Const_v = [Av_big @ v_hor <= bv_big]
        Const = Const_w + Const_v
        prob = cp.Problem(Obj, Const)
        prob.solve(solver=cp.GUROBI, verbose=False)
        one = prob.value

        #2
        a_ext = np.hstack([a, 0])

        w_hor = cp.Variable((dim_ne*N, 1))
        v_hor = cp.Variable((dim_k*N,1))
        w_tilde_hor = w_hor - (L_big @ v_hor)
        Obj = cp.Maximize(a_ext @ (Al_ext_big @ w_tilde_hor))
        Const_w = [Aw_big @ w_hor <= bw_big]
        Const_v = [Av_big @ v_hor <= bv_big]
        Const = Const_w + Const_v
        prob = cp.Problem(Obj, Const)
        prob.solve(solver=cp.GUROBI, verbose=False)
        two = prob.value

        Phi_x_N_i = one + two
        bx[i, :] = b - (1/(1 - alpha))*Phi_x_N_i 


    # Tighten input constraints
    for i in range(Au.shape[0]):
        a = Au[i, :]
        b = bu[i, :]

        w_hor = cp.Variable((dim_ne*N, 1))
        v_hor = cp.Variable((dim_k*N,1))
        w_tilde_hor = w_hor - (L_big @ v_hor)
        x_tilde_hor = (T_tilde @ x_tilde_0) + (S_tilde @ w_tilde_hor)
        x_tilde_hor = x_tilde_hor[0:dim_ne*N,:]
        Obj = cp.Maximize(a @ (K @ Ak_big @ ((L_xC_ext_big @ x_tilde_hor) + (L_x_big @ v_hor))))
        Const_w = [Aw_big @ w_hor <= bw_big]
        Const_v = [Av_big @ v_hor <= bv_big]
        Const = Const_w + Const_v
        prob = cp.Problem(Obj, Const)
        prob.solve(solver=cp.GUROBI, verbose=False)
        
        Phi_u_N_i = prob.value
        bu[i, :] = b - (1/(1 - alpha))*Phi_u_N_i

    return bx, bu