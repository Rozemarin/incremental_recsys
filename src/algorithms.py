import scipy
import numpy as np


def sparse_svd(A, rank):
    np.random.seed(seed=42)
    U, S, Vt = scipy.sparse.linalg.svds(A, k=rank)
    U, S, Vt = U[:,::-1], S[::-1], Vt[::-1,:]
    return U, S, Vt


def psi_step_1order(dA, dAt, U0, S0, V0_T, r):
    '''
        Input
            dA: new data collected since the last time-step
            U0, S0, V0: last state of the model at time t − 1

        Output
            U1, S1, V1: new state of the model at time t
    '''
    if len(S0.shape) == 1: # for initial svd and output of incremental svd
        S0 = np.diag(S0)

    V0 = V0_T.T

    K1 = U0 @ S0 + dA.dot(V0) # dense
    U1, hat_S1 = np.linalg.qr(K1, mode='reduced') # dense
    wave_S0 = hat_S1 - U1.T @ dA.dot(V0) # dense
    L1 = V0 @ wave_S0.T + dAt.dot(U1)
    V1, S1_T = np.linalg.qr(L1, mode='reduced')  # scipy.linalg.qr(L1)

    S1 = S1_T.T

    return U1[:, :r], S1[:r, :r], V1.T[:r, :]


def psi_step_2order(dA, dAt, dA12, dA22, U0, S0, V0_T, r):
    '''
        Input
            dA: new data collected since the last time-step
            U0, S0, V0: last state of the model at time t − 1

        Output
            U1, S1, V1: new state of the model at time t
    '''
    # dA22 = (dA - dA12)
    if len(S0.shape) == 1: # for initial svd and output of incremental svd
        S0 = np.diag(S0)
        
    V0 = V0_T.T

    K12 = U0 @ S0 + dA12.dot(V0)
    U12, hat_S12 = np.linalg.qr(K12, mode='reduced') # scipy.linalg.qr(K12)
    wave_S0 = hat_S12 - U12.T @ dA12.dot(V0)
    L1 = V0 @ wave_S0.T + dAt.dot(U12)
    V1, hat_S1 = np.linalg.qr(L1, mode='reduced') # scipy.linalg.qr(L1)

    wave_S12 = hat_S1.T - U12.T @ dA22.dot(V1)
    K1 = U12 @ wave_S12 - dA22.dot(V1)
    U1, S1 = np.linalg.qr(K1, mode='reduced')
    
    return U1[:, :r], S1[:r, :r], V1.T[:r, :] 


def rand_svd(A, At, r, oversampling=10, power_iterations=0, seed=42):
    '''
        Input
            M: 2D numpy array
            r: rank value for truncation
            oversampling: number of extra random vectors to approximate range(M)

        Output
            Mr: 2D numpy array of rank r and of the same size as M
            rel_eps: relative error of rank-r approximation Mr
    '''
    np.random.seed(seed=seed)
    n, m = A.shape
    Omega = np.random.randn(n, r + oversampling, ) # n x (r + p)

    B = At.dot(Omega)

    for _ in range(power_iterations):
        B = At.dot(A.dot(B))     

    Q, R = np.linalg.qr(B, mode='reduced')

    U, S, Wt = np.linalg.svd(A @ Q, full_matrices=False)
    Vt = Wt @ Q.T

    return U[:, :(r + oversampling)], S[:(r + oversampling)], Vt[:(r + oversampling), :]


def add_random_embeddings(Q, diff):
    Q_additional = np.random.randn(diff, Q.shape[1])
    return np.vstack([Q, Q_additional])


def reused_projector_Vt(A, Vt, r):
    U, S, Wt = np.linalg.svd(A.dot(Vt.T), full_matrices=False)
    Vt = Wt @ Vt
    return U, S, Vt # [:, :r], S[:r], Vt[:r, :]
    

def reused_projector_U(At, U, r):
    W, S, Vt = np.linalg.svd((At.dot(U)).T, full_matrices=False)
    U = U @ W
    return U, S, Vt # U[:, :r], S[:r], Vt[:r, :]


def matthew_brand(C, U, S, Vt, r=None, update_rows=False, reorthogonalization=False, tol=1e-14):
    '''
    C - columns to edit
    '''

    if r is None:
        r = U.shape[1]

    if update_rows:
        C = C.T
        U, S, Vt = Vt.T, S.T, U.T
        L = scipy.sparse.csc_matrix.dot(U.T, C)
    else:
        L = scipy.sparse.csr_matrix.dot(U.T, C)

    if len(S.shape) == 1:
        S = np.diag(S)
    V = Vt.T

    H = C - U @ L
    J, K = np.linalg.qr(H)

    null = np.zeros(shape=(K.shape[0], S.shape[1]))
    Q = np.block([
        [S, L],
        [null, K]
    ])
    
    U_q, S_q, V_qt = np.linalg.svd(Q, full_matrices=False)
    V_q = V_qt.T

    U = np.block([[U, J]]) @ U_q

    null_ur = np.zeros(shape=(V.shape[0], V_q.shape[0] - V.shape[1]))
    null_bl = np.zeros(shape=(C.shape[1], V.shape[1]))
    eye = np.eye(C.shape[1], V_q.shape[0] - V.shape[1])
    V = np.block([
        [V, null_ur],
        [null_bl, eye]
    ])
    V = V @ V_q

    U = U[:, :r]
    S = S_q[:r]
    Vt = V[:, :r].T

    if reorthogonalization:
        # print('reort')
        U, _ = np.linalg.qr(U)
    
    if update_rows:
        U, S, Vt = Vt.T, S, U.T
        
    return np.asarray(U), np.asarray(S), np.asarray(Vt)


def small_svd(sdA, U, S, Vt, r):
    dn, dm = sdA.shape
    Q1 = np.random.randn(dn, r)
    Q2 = np.random.randn(dm, r)
    bZ = Q1 @ sdA.dot(Q2)
    
    u, s, vt = np.linalg.svd(bZ, full_matrices=False)
    
    #indices = 
