import numpy as np
import scipy.io as sio
import cvxpy as cp
from scipy.linalg import solve_discrete_are
from scipy.linalg import block_diag
import control as ct
import os

from numpy.linalg import svd, lstsq, matrix_rank
from andes.models.renewable.cen_helpers.pred_matrices import pred_matrices

def build_mpc_matrices(N, dt, Wf, Wr, Wp, Pmax, Pmin, Pref):

    # Open continuous model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "COI_models", "CEN_model.mat")
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

    #sysc = ct.ss(Ac,Bc[:,8:16],Cc,Dc[:,8:16])
    sysc = ct.ss(Ac,Bc[:,1:9],Cc,Dc[:,1:9])

    #Discretize
    sysd = ct.c2d(sysc, dt, 'zoh')
    A = sysd.A
    Bu = sysd.B
    C = sysd.C
    D = sysd.D

    #Dimensions
    dim_n = A.shape[0] # number of states
    dim_m = Bu.shape[1] # number of inputs
    dim_k = C.shape[0] # number of outputs
    dim_f = 8
    dim_theta = dim_f -1
    dim_p = 8

      
    #Discretize for disturbance channels (Each frequency state gets its own disturbance estimate)
    proto_Bd = Bc[:,0:1]
    sysc_d = ct.ss(Ac,proto_Bd,Cc,np.zeros((dim_k,proto_Bd.shape[1])))
    sysd_d = ct.c2d(sysc_d, dt, 'zoh')
    Bd = sysd_d.B
    Dd = sysd_d.D

    dim_d = Bd.shape[1] # number of disturbance channels

    #Optimal setpoint
    H_ref = np.hstack([np.zeros((dim_k-dim_f, dim_f)),np.eye(dim_k-dim_f)])

    vd = np.hstack([np.zeros((dim_d, dim_n)),np.eye(dim_d)]) # To get d_hat from xe_hat

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

    #Observability of augmented/extended system check
    M_top = np.hstack((np.eye(dim_n) - A, -Bd))
    M_bottom = np.hstack((C, Dd))
    M = np.vstack((M_top, M_bottom))
    rank_M = np.linalg.matrix_rank(M)

    if rank_M != dim_n + dim_d:
        print("WARNING: Rank condition not satisfied.")
        print(f"Rank = {rank_M}, expected = {dim_n + dim_d}")

    #Extend the system
    Ae = np.vstack([np.hstack([A, Bd]), np.hstack([np.zeros((dim_d,dim_n)),np.eye(dim_d)])])
    Be = np.vstack([Bu,np.zeros((dim_d, dim_m))])
    Ce = np.hstack([C, Dd])
    De = D
    syse = ct.ss(Ae, Be, Ce, De, dt)
    dim_ne = Ae.shape[0]

    #Build prediction matrices
    Tc,Sc = pred_matrices(Ae,Be,N)

    u1 = np.hstack([np.zeros((dim_f,dim_theta)),-(1/dt)*np.eye(dim_f), np.zeros((dim_f, dim_ne - dim_f - dim_theta))])
    u2 = np.hstack([np.zeros((dim_f,dim_theta)),(1/dt)*np.eye(dim_f), np.zeros((dim_f, dim_ne - dim_f - dim_theta))])

    #For building f_hor, ROCOF_hor and P_hor
    Vf = np.hstack([np.zeros((N*dim_f,dim_ne)),np.kron(np.eye(N), np.hstack([np.zeros((dim_f,dim_theta)),np.eye(dim_f),np.zeros((dim_f, dim_ne - dim_f - dim_theta))]))])
    Vr = np.hstack([np.kron(np.eye(N), u1), np.zeros((N*dim_f,dim_ne))]) + np.hstack([np.zeros((N*dim_f,dim_ne)), np.kron(np.eye(N), u2)])
    Vp = np.hstack([np.zeros((N*dim_p,dim_ne)),np.kron(np.eye(N), np.hstack([np.zeros((dim_p, dim_ne - dim_p - dim_p - dim_d)),np.eye(dim_p), np.zeros((dim_p, dim_d + dim_p))]))])

    #Objective matrices
    Wf_bar = np.diag(Wf)
    Wr_bar = np.diag(Wr)
    Wp_bar = np.diag(Wp)

    Wf_tilde = Vf.T @ np.kron(np.eye(N), Wf_bar) @ Vf
    Wr_tilde = Vr.T @ np.kron(np.eye(N), Wr_bar) @ Vr

    Q_tilde = Wf_tilde + Wr_tilde
    R_tilde = np.kron(np.eye(N), Wp_bar)

    H = (Sc.T @ Q_tilde @ Sc) + R_tilde
    #eigenvalues = np.linalg.eigvals(H)
    #print(eigenvalues)
    H = cp.psd_wrap(H) # Skips Postive semi-definite checks, only use when I now PSD is true    
    h = np.vstack([Tc.T @ Q_tilde @ Sc, Q_tilde @ Sc, R_tilde])

    #Constraints matrices
    A_p = np.vstack([Vp @ Sc, - Vp @ Sc])
    b_p = np.vstack([-Vp @ Tc, Vp @ Tc])

    P_tilde = np.vstack([np.kron(np.ones((N,1)),((0.99*Pmax.reshape(-1,1))-Pref.reshape(-1,1))), np.kron(np.ones((N,1)), -(Pmin.reshape(-1,1)-Pref.reshape(-1,1)))]) # Pmax and Pref must be in p.u. Sbase

    # Choose reasonable covariances and tune them:
    Qw = np.diag(dim_theta*[1e-4] + dim_f*[1e-4] +  [1e-4] +  dim_p*[1e-1] + dim_p*[1e-4] + dim_d*[10])     # process noise covariance (tune)
    Rv = np.diag(dim_f*[1]+dim_p*[1e-1])     # measurement noise covariance (tune)

    # Solve discrete ARE for estimator (note we use Ae.T, Ce.T)
    P = solve_discrete_are(Ae.T, Ce.T, Qw, Rv)
    S = Ce @ P @ Ce.T + Rv
    L = P @ Ce.T @ np.linalg.inv(S)   # Kalman gain (discrete-time L)

    mpc_data = dict(dim_m=dim_m, A_p=A_p, b_p=b_p, P_tilde=P_tilde, H=H, h=h, A=Ae, B=Be, C=Ce, L=L)
    ref_data = dict(A=A, B=Bu, Bd=Bd, C= C, Dd = Dd, H_ref = H_ref, vd=vd, dPmax=Pmax.reshape(-1,1)-Pref.reshape(-1,1), dPmin=Pmin.reshape(-1,1)-Pref.reshape(-1,1))
    return mpc_data, ref_data