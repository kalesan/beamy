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
            R[k,b] = -np.real(np.log2(np.linalg.det(E[:, :, k, b])))

    return R
