import numpy as np


def sigcov(H, M, N0):
    (Nr, Nt, K, B) = H.shape
    (Nt, Nsk, K, B) = M.shape

    C = np.zeros((Nr, Nr, K), dtype='complex')

    M0 = np.reshape(M, [Nt, Nsk*K, B])

    for k in range(K):
        C[:, :, k] = np.eye(Nr)*N0

        for b in range(B):
            T = np.dot(H[:, :, k, b], M0[:, :, b])
            C[:, :, k] += np.dot(T, T.conj().transpose())

    return C


def lmmse(H, M, N0, C=None):
    (Nr, Nt, K, B) = H.shape
    (Nt, Nsk, K, B) = M.shape

    U = np.zeros((Nr, Nsk, K, B), dtype='complex')

    # Signal covariance matrix
    if C is None:
        C = sigcov(H, M, N0)

    for k in range(K):
        for b in range(B):
            U[:, :, k, b] = np.dot(np.linalg.pinv(C[:, :, k]),
                                   np.dot(H[:, :, k, b], M[:, :, k, b]))

    return U


def mse(H, U, M, N0, C=None):
    (Nr, Nt, K, B) = H.shape
    (Nt, Nsk, K, B) = M.shape

    E = np.zeros((Nsk, Nsk, K, B), dtype='complex')

    M0 = np.reshape(M, [Nt, Nsk*K, B])

    # Signal covariance matrix
    if C is None:
        C = sigcov(H, M, N0)

    for k in range(K):
        for b in range(B):
            T = np.dot(U[:, :, k, b].conj().T,
                       np.dot(H[:, :, k, b], M[:, :, k, b]))

            E[:, :, k, b] = np.eye(Nsk) - 2*np.real(T)
            E[:, :, k, b] += N0 * np.dot(U[:, :, k, b].conj().T, U[:, :, k, b])

            for i in range(B):
                T = np.dot(U[:, :, k, b].conj().T,
                           np.dot(H[:, :, k, i], M0[:, :, i]))

                E[:, :, k, b] += np.dot(T, T.conj().T)
    return E


def rate(H, M, N0, C=None, E=None):
    (Nr, Nt, K, B) = H.shape
    (Nt, Nsk, K, B) = M.shape

    # Signal covariance matrix
    if C is None:
        C = sigcov(H, M, N0)

    U = lmmse(H, M, N0, C=C)

    if E is None:
        E = mse(H, U, M, N0, C=C)

    R = np.zeros((K, B))

    for k in range(K):
        for b in range(B):
            R[k, b] = -np.real(np.log2(np.linalg.det(E[:, :, k, b])))

    return R


def weighted_bisection(H, U, W, N0, threshold=1e-6):
    (Nr, Nt, K, B) = H.shape
    (Nr, Nsk, K, B) = U.shape

    # Weighted transmit covariance matrices
    C = np.zeros((Nt, Nt, B))

    for b in range(B):
        for i in range(B):
            for j in range(K):
                T = np.dot(np.dot(U[:, :, j, i], W[:, :, j, i]),
                           U[:, :, j, i].conj().T)

                C[:, :, b] += np.dot(np.dot(H[:, :, j, b].conj().T, T),
                                     H[:, :, j, b])

    for b in range(B):
        UB = 10
        lb = 0

        M0 = np.zeros((Nr, Nsk, K))

        while np.abs(UB - lb) > threshold:
            v = (UB + lb) / 2

            for k in range(K):
                T = np.dot(np.dot(H[:, :, k, b], U[:, :, k, b].conj().T),
                           W[:, :, k, b])

                M0[:, :, k] = np.dot(np.linalg.pinv(C[:, :, b] + np.eye(Nt)*v),
                                     T)
