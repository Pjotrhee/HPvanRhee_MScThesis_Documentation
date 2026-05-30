import numpy as np

def pred_matrices(A, B, N):
    """
    Compute prediction matrices T and S.

    Parameters
    ----------
    A : State matrix
    B : Input matrix
    N : Prediction horizon

    Returns
    -------
    T and S
    """

    n = A.shape[0]
    m = B.shape[1]

    # --- T matrix ---
    T = np.zeros((n * (N + 1), n))
    for k in range(N + 1):
        T[k*n:(k+1)*n, :] = np.linalg.matrix_power(A, k)

    # --- S matrix ---
    S = np.zeros((n * (N + 1), m * N))
    for k in range(1, N + 1):
        for j in range(k):
            S[k*n:(k+1)*n, j*m:(j+1)*m] = np.linalg.matrix_power(A, k-1-j) @ B

    return T, S
