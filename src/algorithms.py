import scipy
import numpy as np


def sparse_svd(A, rank):
    U, S, Vt = scipy.sparse.linalg.svds(A, k=rank)
    U, S, Vt = U[:,::-1], S[::-1], Vt[::-1,:]
    return U, S, Vt


def psi_step_1order(dA, U0, S0, V0_T, r):
    '''
        Input
            dA: new data collected since the last time-step
            U0, S0, V0: last state of the model at time t − 1

        Output
            U1, S1, V1: new state of the model at time t
    '''
    if len(S0.shape) == 1:
        S0 = np.diag(S0)

    V0 = V0_T.T

    K1 = U0 @ S0 + dA @ V0
    U1, hat_S1 = np.linalg.qr(K1, mode='reduced') # scipy.linalg.qr(K1)
    wave_S0 = hat_S1 - U1.T @ dA @ V0
    L1 = V0 @ wave_S0.T + dA.T @ U1
    V1, S1_T = np.linalg.qr(L1, mode='reduced')  # scipy.linalg.qr(L1)

    S1 = S1_T.T
    return U1[:, :r], S1[:r, :r], V1.T[:r, :]


def psi_step_2order(dA, dA12, U0, S0, V0_T, r):
    '''
        Input
            dA: new data collected since the last time-step
            U0, S0, V0: last state of the model at time t − 1

        Output
            U1, S1, V1: new state of the model at time t
    '''
    if len(S0.shape) == 1:
        S0 = np.diag(S0)
        
    V0 = V0_T.T

    K12 = U0 @ S0 + dA12 @ V0
    U12, hat_S12 = np.linalg.qr(K12, mode='reduced') # scipy.linalg.qr(K12)
    wave_S0 = hat_S12 - U12.T @ dA12 @ V0
    L1 = V0 @ wave_S0.T + dA.T @ U12
    V1, hat_S1 = np.linalg.qr(L1, mode='reduced') # scipy.linalg.qr(L1)

    wave_S12 = hat_S1.T - U12.T @ (dA - dA12) @ V1
    K1 = U12 @ wave_S12 - (dA - dA12) @ V1
    U1, S1 = np.linalg.qr(K1, mode='reduced')
    
    return U1[:, :r], S1[:r, :r], V1.T[:r, :] 


def rand_svd(M, r, oversampling=10):
    '''
        Input
            M: 2D numpy array
            r: rank value for truncation
            oversampling: number of extra random vectors to approximate range(M)

        Output
            Mr: 2D numpy array of rank r and of the same size as M
            rel_eps: relative error of rank-r approximation Mr
    '''

    ny = M.shape[1]
    Omega = np.random.randn(ny, r + oversampling)
    Y = M @ Omega

    Q, R = np.linalg.qr(Y, mode='reduced') #, mode="economic")

    SVh = Q.T @ M

    W, S, Vh = np.linalg.svd(SVh, full_matrices=False)
    U = Q @ W

    return U[:, :r], np.diag(S[:r]), Vh[:r, :]


def add_random_embeddings(Q, diff):
    Q_additional = np.random.randn(diff, Q.shape[1])
    return np.vstack([Q, Q_additional])


def reused_projector_svd(M, U, r, v_form=False):
    '''
        Input
            M: 2D numpy array
            U: U from previous iteration
            r: rank value for truncation

        Output
            U_r, S_r, Vt_r: truncated svd decomposition
    '''
    Q = U 

    SVh = Q.T @ M

    if type(SVh) != type(np.array([1])):
        SVh = SVh.toarray()

    W, S, Vh = np.linalg.svd(SVh, full_matrices=False)
    U = Q @ W

    return U[:, :r], S[:r], Vh[:r, :]
