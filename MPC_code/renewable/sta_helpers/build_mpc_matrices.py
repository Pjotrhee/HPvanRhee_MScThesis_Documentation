import numpy as np
import scipy.io as sio
import cvxpy as cp
from scipy.linalg import solve_discrete_are
from scipy.linalg import block_diag
import control as ct
import os

from numpy.linalg import svd, lstsq, matrix_rank
from andes.models.renewable.sta_helpers.pred_matrices import pred_matrices
from andes.models.renewable.sta_helpers.xf_ellipsoidal import xf_ellipsoidal
from andes.models.renewable.sta_helpers.tighten_constraints import tighten_constraints
from andes.models.renewable.sta_helpers.cache_utils import make_cache_key, cache_path, load_cache, save_cache

def build_mpc_matrices(N, dt, Wf, Wr, Wp, Pmax, Pmin, Pref):

    # Open continuous model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "COI_models", "CEN_type2_2_model.mat")
    data = sio.loadmat(file_path)
    A_full  = data['A_ssm']
    B_full  = data['B_ssm']
    C_full  = data['C_ssm']
    D_full  = data['D_ssm']

    #Transform (to relative voltage angles)
    R_top = np.hstack([np.array([[1]]), np.zeros((1,7))])
    R_bottom = np.hstack([-np.ones((7,1)), np.eye(7)])
    R = np.vstack([R_top, R_bottom])   # 8x8 
    T = block_diag(R, np.eye(8), np.array([[1]]), np.eye(8), np.eye(8))
    T_inv = np.linalg.inv(T)

    A_t = T @ A_full @ T_inv
    B_t = T @ B_full
    C_t = C_full @ T_inv
    D_t = D_full.copy()

    #Reduce to remove first voltage angle
    E = np.hstack([np.zeros((32,1)), np.eye(32)])

    Ac = E @ A_t @ E.T
    Bc = E @ B_t
    Cc = C_t @ E.T
    Dc = D_t

    sysc = ct.ss(Ac,Bc[:,1:9],Cc,Dc[:,1:9])

    #Discretize
    sysd = ct.c2d(sysc, dt, 'zoh')
    A_pre = sysd.A
    B_pre_u = sysd.B
    C_pre = sysd.C
    D_pre = sysd.D

    #Dimensions
    dim_f = 8
    dim_theta = dim_f - 1
    dim_p = 8

    ### Add derivative states
    A_lb = A_pre[0:dim_theta+dim_f,0:dim_theta+dim_f]
    A_rb = A_pre[0:dim_theta+dim_f,dim_theta+dim_f:]
    A_lo = A_pre[dim_theta+dim_f:,0:dim_theta+dim_f]
    A_ro = A_pre[dim_theta+dim_f:,dim_theta+dim_f:] 
    A_1fdot = A_lb[dim_theta:,0:dim_theta]
    A_2fdot = A_lb[dim_theta:,dim_theta:] - np.eye(dim_f)
    A_3fdot = A_rb[dim_theta:,:] 

    eps = 1e-6
    A_boven = np.hstack([A_lb,np.zeros((dim_theta+dim_f,dim_f)),A_rb])
    A_midden = np.hstack([A_1fdot, A_2fdot,eps*np.eye(dim_f),A_3fdot])
    A_onder = np.hstack([A_lo,np.zeros((1+dim_p+dim_p,dim_f)),A_ro])

    B_boven_u = B_pre_u[0:dim_theta+dim_f,:]
    B_onder_u = B_pre_u[dim_theta+dim_f:,:]
    B_fdot_u = B_boven_u[dim_theta:,:]

    C_boven = np.hstack([np.zeros((dim_f+dim_f,dim_theta)),np.eye(dim_f+dim_f),np.zeros((dim_f+dim_f,1+dim_p+dim_p))])
    C_onder = np.hstack([np.zeros((dim_p,dim_theta+dim_f+dim_f+1)),np.eye(dim_p),np.zeros((dim_p,dim_p))])
    
    A = np.vstack([A_boven, A_midden, A_onder])
    Bu = np.vstack([B_boven_u,B_fdot_u,B_onder_u])
    C = np.vstack([C_boven, C_onder])
    D = np.zeros((dim_f+dim_f+dim_p,dim_p))

    #Dimensions
    dim_n = A.shape[0] # number of states
    dim_m = Bu.shape[1] # number of inputs
    dim_k = C.shape[0] # number of outputs
      
    #Discretize for disturbance channel Bd
    proto_Bd = Bc[:,0:1]
    sysc_d = ct.ss(Ac,proto_Bd,Cc,np.zeros((dim_f+dim_p,proto_Bd.shape[1])))
    sysd_d = ct.c2d(sysc_d, dt, 'zoh')
    B_pre_d = sysd_d.B
    D_pre_d = sysd_d.D

    dim_d = B_pre_d.shape[1] # number of disturbance channels
    
    B_boven_d = B_pre_d[0:dim_theta+dim_f,:]
    B_onder_d = B_pre_d[dim_theta+dim_f:,:]
    B_fdot_d = B_boven_d[dim_theta:,:]
    Bd = np.vstack([B_boven_d,B_fdot_d,B_onder_d])

    Dd = np.zeros((dim_f+dim_f+dim_p,dim_d))

    #Optimal setpoint
    H_ref = np.hstack([np.zeros((dim_p, dim_f+dim_f)),np.eye(dim_p)])

    vd = np.hstack([np.zeros((dim_d, dim_n)),np.eye(dim_d)]) # To get d_hat from xe_hat
    vx = np.hstack([np.eye(dim_n),np.zeros((dim_n,dim_d))]) # to get x_hat from xe_hat
    
    #Controllability check
    Ctrb = ct.ctrb(A, Bu)
    rank_C = np.linalg.matrix_rank(Ctrb)
    if rank_C != A.shape[0]:
        print("WARNING: Controllability condition not satisfied.")
        print(f"Rank = {rank_C}, expected = {A.shape[0]}")

        eigs = np.linalg.eigvals(A)
        stab = all(
            np.linalg.matrix_rank(np.hstack([lam*np.eye(A.shape[0]) - A, Bu])) == A.shape[0]
            for lam in eigs if abs(lam) >= 1 - 1e-8
        )
        if stab:
            print("BUT: Stabilisability condition is satisfied.")

    #Observability check
    Obsv = ct.obsv(A, C)
    rank_O = np.linalg.matrix_rank(Obsv)
    if rank_O != A.shape[0]:
        print("WARNING: Observability condition not satisfied.")
        print(f"Rank = {rank_O}, expected = {A.shape[0]}")

    #Observability for augmented/extended system check
    M_top = np.hstack((np.eye(dim_n) - A, -Bd))
    M_bottom = np.hstack((C, Dd))
    M = np.vstack((M_top, M_bottom))
    rank_M = np.linalg.matrix_rank(M)

    if rank_M != dim_n + dim_d:
        print("WARNING: Target selection rank condition not satisfied.")
        print(f"Rank = {rank_M}, expected = {dim_n + dim_d}")

    #Extend the system
    Ae = np.vstack([np.hstack([A, Bd]), np.hstack([np.zeros((dim_d,dim_n)),np.eye(dim_d)])])
    Be = np.vstack([Bu,np.zeros((dim_d, dim_m))])
    Ce = np.hstack([C, Dd])

    dim_ne = Ae.shape[0]

    #Build prediction matrices
    Tcx,Scu = pred_matrices(A,Bu,N)
    Tcx,Scd = pred_matrices(A,Bd,N)
    
    #Objective matrices
    Wf_bar = np.diag(Wf)
    Wr_bar = np.diag(Wr)
    Q = block_diag((1e-1)*np.eye(dim_theta), (1e3)*Wf_bar, (1/dt)*(1/dt)*(1e3)*Wr_bar, (1e-1)*np.eye(1), (1e3)*np.eye(dim_p), (1e-1)*np.eye(dim_p))

    Q_eig = np.linalg.eigvals(Q)
    if not all(qeig > 0 for qeig in Q_eig):
        print(f"WARNING: Q is NOT positive definit with eigenvalues: {Q_eig}")

    R = (1e3)*np.diag(Wp)
    R_eig = np.linalg.eigvals(R)
    if not all(reig > 0 for reig in R_eig):
        print(f"WARNING: R is NOT positive definit with eigenvalues: {R_eig}")

    P_dare, L_dare, K = ct.dare(A, Bu, Q, R)
    K = -K
    P_eig = np.linalg.eigvals(P_dare)
    #print(P_eig)
    if not all(peig > 0 for peig in P_eig):
        print(f"WARNING: P is NOT positive definit with eigenvalues: {P_eig}")
    

    Q_tilde = block_diag(np.kron(np.eye(N), Q),P_dare)
    R_tilde = np.kron(np.eye(N), R)

    H = (Scu.T @ Q_tilde @ Scu) + R_tilde
    H_eig = np.linalg.eigvals(H)
    if not all(heig >= 0 for heig in H_eig):
        print(f"WARNING: H is NOT positive semi definit with eigenvalues: {H_eig}")
    else:
        H = cp.psd_wrap(H) # Skips Postive semi-definite checks, only use when I now PSD is true    
    h = np.vstack([Tcx.T @ Q_tilde @ Scu, Scd.T @ Q_tilde @ Scu, Q_tilde @ Scu, R_tilde])

    # Some wide u Constraints
    bigg_limit = 10
    Au = np.vstack([np.eye(dim_m),-np.eye(dim_m)])
    bu = np.vstack([bigg_limit*np.ones((dim_m,1)),bigg_limit*np.ones((dim_m,1))])

    # Some x Constraints
    Ax = np.vstack([np.eye(dim_n),-np.eye(dim_n)]) #Not extended
    xmax = np.array(dim_theta*[bigg_limit] + dim_f*[bigg_limit] + dim_f*[bigg_limit] +  [bigg_limit] +  list((0.99*Pmax)-Pref) + dim_p*[bigg_limit])
    xmin = np.array(dim_theta*[-bigg_limit] + dim_f*[-bigg_limit] + dim_f*[-bigg_limit] +  [-bigg_limit] +  list(Pmin-Pref) + dim_p*[-bigg_limit])
    bx = np.vstack([xmax.reshape((dim_n,1)),-xmin.reshape((dim_n,1))])
    
    #Observer
    # Choose reasonable covariances and tune them:
    Q_obs = np.diag(dim_theta*[1e-4] + dim_f*[1e-4] + dim_f*[1] +  [1e-4] +  dim_p*[1e-2] + dim_p*[1e-2] + dim_d*[10])     # process noise covariance (tune)
    R_obs = np.diag(dim_f*[1]+dim_f*[1]+dim_p*[1e-1])     # measurement noise covariance (tune)

    # Solve discrete ARE for estimator (note we use Ae.T, Ce.T)
    P_obs = solve_discrete_are(Ae.T, Ce.T, Q_obs, R_obs)
    S_obs = Ce @ P_obs @ Ce.T + R_obs
    L_obs = P_obs @ Ce.T @ np.linalg.inv(S_obs)   # Kalman gain (discrete-time L)


    ### Tighten Constraints

    #Disturbance bound assumptions:
    Aw_x = np.vstack([np.eye(dim_n),-np.eye(dim_n)])
    w_xmax = np.array(dim_theta*[1e-6] + dim_f*[1e-4] + dim_f*[1e-4] +  [1e-5] +  dim_p*[1e-4] + dim_p*[1e-6])
    w_xmin = np.array(dim_theta*[-1e-6] + dim_f*[-1e-4] + dim_f*[-1e-4] +  [-1e-5] +  dim_p*[-1e-4] + dim_p*[-1e-6])
    bw_x = np.vstack([w_xmax.reshape((dim_n,1)),-w_xmin.reshape((dim_n,1))])

    Aw = np.vstack([np.eye(dim_ne),-np.eye(dim_ne)])
    wmax = np.hstack([w_xmax, dim_d*[1e-3]])
    wmin = np.hstack([w_xmin, dim_d*[-1e-3]])
    bw = np.vstack([wmax.reshape((dim_ne,1)),-wmin.reshape((dim_ne,1))])


    Av = np.vstack([np.eye(dim_k),-np.eye(dim_k)])
    vmax = np.array(dim_f*[1e-4] + dim_f*[1e-4] + dim_p*[1e-4])
    vmin = np.array(dim_f*[-1e-4] + dim_f*[-1e-4] + dim_p*[-1e-4])
    bv = np.vstack([vmax.reshape((dim_k,1)),-vmin.reshape((dim_k,1))])

    L_x = L_obs[0:dim_ne-dim_d,:]

    tight_data = dict(dim_n=dim_n, dim_m=dim_m, dim_k=dim_k, dim_ne=dim_ne, C=C, K=K, L=L_obs, L_x=L_x, A=A, Bu=Bu, A_ext=Ae, C_ext=Ce, Ax=Ax, bx=bx, Au=Au, bu=bu, Aw_x=Aw_x, bw_x=bw_x, Aw=Aw, bw=bw, Av=Av, bv=bv)

    tight_key = make_cache_key("tighten_constraints", tight_data)
    tight_file = cache_path("tighten_constraints", tight_key)

    cached = load_cache(tight_file)
    if cached is not None:
        bx = cached["bx"]
        bu = cached["bu"]
    else:
        bx, bu = tighten_constraints(tight_data)
        save_cache(tight_file, {"bx": bx, "bu": bu})

    Au_bar = np.kron(np.eye(N),Au)
    bu_bar = np.kron(np.ones((N,1)),bu)

    Ax_tilde = np.kron(np.eye(N),Ax)
    Ax_bar = Ax_tilde @ Scu[dim_n:dim_n*(N+1),:]
    bx_bar = np.kron(np.ones((N,1)),bx)
    Cx_bar = - Ax_tilde @ Tcx[dim_n:dim_n*(N+1),:]
    Dx_bar = - Ax_tilde @ Scd[dim_n:dim_n*(N+1),:]

    #Terminal Constraints
    ellipse_inputs = {
        "P_dare": P_dare,
        "K": K,
        "Au": Au,
        "bu": bu,
        "Ax": Ax,
        "bx": bx,
    }

    ellipse_key = make_cache_key("xf_ellipsoidal", ellipse_inputs)
    ellipse_file = cache_path("xf_ellipsoidal", ellipse_key)

    cached = load_cache(ellipse_file)
    if cached is not None:
        c_ellipse = cached["c_ellipse"]
    else:
        c_ellipse = xf_ellipsoidal(P_dare, K, Au, bu, Ax, bx)
        save_cache(ellipse_file, {"c_ellipse": c_ellipse})

    print(f'c_ellipse = {c_ellipse}')

    mpc_data = dict(dim_n=dim_n, dim_d=dim_d, dim_m=dim_m, Au=Au_bar, bu=bu_bar, Ax=Ax_bar, bx=bx_bar, Cx=Cx_bar, Dx=Dx_bar, H=H, h=h, A=Ae, B=Be, C=Ce, K=K, L=L_obs, Tcx=Tcx, Scu=Scu, Scd=Scd, c_ellipse=c_ellipse, P_dare=P_dare)
    ref_data = dict(A=A, B=Bu, Bd=Bd, C= C, Dd = Dd, H_ref = H_ref, vd=vd, vx=vx, Ax=Ax, bx=bx, Au=Au, bu=bu)
    
    return mpc_data, ref_data