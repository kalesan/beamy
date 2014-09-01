import sys
sys.path.append('../marconi')

import numpy as np

import utils

RANDOM_REALIZATIONS = 5

Nr = 5
Nt = 7
K = 10
B = 3

N0 = 2


class TestSIGCOV:
    def slow_sigcov(self, H, M, N0):
        (Nr, Nt, K, B) = H.shape
        (Nt, Nsk, K, B) = M.shape

        C = np.zeros((Nr, Nr, K), dtype='complex')

        for k in range(K):
            C[:, :, k] = np.eye(Nr)*N0

            for b in range(B):
                for j in range(K):
                    T = np.dot(H[:, :, k, b], M[:, :, j, b])
                    C[:, :, k] += np.dot(T, T.conj().transpose())

        return C

    def test_random(self):
        for r in range(RANDOM_REALIZATIONS):
            H = np.random.randn(Nr, Nt, K, B) + 1j*np.random.randn(Nr, Nt, K, B)

            M = np.random.randn(Nt, Nt, K, B) + 1j*np.random.randn(Nt, Nt, K, B)

            C0 = self.slow_sigcov(H, M, N0)
            C = utils.sigcov(H, M, N0)

            np.testing.assert_almost_equal(C, C0)


class TestLMMSE:
    def slow_lmmse(self, H, M, N0):
        (Nr, Nt, K, B) = H.shape
        (Nt, Nsk, K, B) = M.shape

        U = np.zeros((Nr, Nsk, K, B), dtype='complex')

        # Signal covariance matrix
        C = utils.sigcov(H, M, N0)

        for k in range(K):
            for b in range(B):
                U[:, :, k, b] = np.dot(np.linalg.pinv(C[:, :, k]),
                                       np.dot(H[:, :, k, b], M[:, :, k, b]))

        return U

    def test_random(self):
        for r in range(RANDOM_REALIZATIONS):
            H = np.random.randn(Nr, Nt, K, B) + 1j*np.random.randn(Nr, Nt, K, B)

            M = np.random.randn(Nt, Nt, K, B) + 1j*np.random.randn(Nt, Nt, K, B)

            U0 = self.slow_lmmse(H, M, N0)
            U = utils.lmmse(H, M, N0)

            np.testing.assert_almost_equal(U, U0)


class TestMSE:
    def slow_mse(self, H, U, M, N0):
        (Nr, Nt, K, B) = H.shape
        (Nt, Nsk, K, B) = M.shape

        E = np.zeros((Nsk, Nsk, K, B), dtype='complex')

        for k in range(K):
            for b in range(B):
                T = np.dot(U[:, :, k, b].conj().T,
                           np.dot(H[:, :, k, b], M[:, :, k, b]))

                E[:, :, k, b] = np.eye(Nsk) - 2*np.real(T)
                E[:, :, k, b] += N0*np.dot(U[:, :, k, b].conj().T,
                                           U[:, :, k, b])

                for i in range(B):
                    for j in range(K):
                        T = np.dot(U[:, :, k, b].conj().T,
                                   np.dot(H[:, :, k, i], M[:, :, j, i]))

                        E[:, :, k, b] += np.dot(T, T.conj().T)
        return E

    def test_random(self):
        for r in range(RANDOM_REALIZATIONS):
            H = np.random.randn(Nr, Nt, K, B) + 1j*np.random.randn(Nr, Nt, K, B)

            M = np.random.randn(Nt, Nt, K, B) + 1j*np.random.randn(Nt, Nt, K, B)

            U = utils.lmmse(H, M, N0)

            E0 = self.slow_mse(H, U, M, N0)
            E = utils.mse(H, U, M, N0)

            np.testing.assert_almost_equal(E, E0)


class TestRATE:
    def slow_rate(self, H, M, N0):
        (Nr, Nt, K, B) = H.shape
        (Nt, Nsk, K, B) = M.shape

        # Signal covariance matrix
        C = utils.sigcov(H, M, N0)

        R = np.zeros((K, B))

        for k in range(K):
            for b in range(B):
                T = np.dot(H[:, :, k, b], M[:, :, k, b])
                S = np.dot(np.dot(T.conj().T, np.linalg.pinv(C[:, :, k])), T)

                R[k, b] = np.real(-np.log2(np.linalg.det(np.eye(Nsk) - S)))

        return R

    def test_random(self):
        for r in range(RANDOM_REALIZATIONS):
            H = np.random.randn(Nr, Nt, K, B) + 1j*np.random.randn(Nr, Nt, K, B)

            M = np.random.randn(Nt, Nt, K, B) + 1j*np.random.randn(Nt, Nt, K, B)

            U = utils.lmmse(H, M, N0)

            R0 = self.slow_rate(H, M, N0)
            R = utils.rate(H, M, N0)

            np.testing.assert_almost_equal(R, R0)
