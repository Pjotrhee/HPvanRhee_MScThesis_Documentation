import numpy as np

def xf_ellipsoidal(P, K, Au, bu, Ax, bx):
    P_inv = np.linalg.inv(P)

    G = np.vstack([Ax,Au@K])
    H = np.vstack([bx,bu])

    c_list = []

    for i in range(G.shape[0]):
        g_i = G[i, :]
        h_i = H[i, :]
        c_i = (h_i*h_i)/(g_i.T @ P_inv @ g_i)
        c_list.append(c_i)

    c_star = min(c_list)
    return c_star[0]