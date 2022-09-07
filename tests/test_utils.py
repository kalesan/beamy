import pytest
import numpy as np

from beamy import utils

RANDOM_REALIZATIONS = 5

ACCURACY = 6

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

            np.testing.assert_almost_equal(np.linalg.norm(C - C0), 0, decimal=ACCURACY)

    def test_sigcov_benchmark(self, benchmark):
        H = np.random.randn(Nr, Nt, K, B) + 1j*np.random.randn(Nr, Nt, K, B)

        M = np.random.randn(Nt, Nt, K, B) + 1j*np.random.randn(Nt, Nt, K, B)

        benchmark(utils.sigcov, H, M, N0)


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

            np.testing.assert_almost_equal(np.linalg.norm(U - U0), 0, decimal=ACCURACY)

    def test_lmmse_benchmark(self, benchmark):
        H = np.random.randn(Nr, Nt, K, B) + 1j*np.random.randn(Nr, Nt, K, B)

        M = np.random.randn(Nt, Nt, K, B) + 1j*np.random.randn(Nt, Nt, K, B)

        benchmark(utils.lmmse, H, M, N0)


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

    @pytest.mark.skip(reason="WIP")
    def test_random(self):
        for r in range(RANDOM_REALIZATIONS):
            H = np.random.randn(Nr, Nt, K, B) + 1j*np.random.randn(Nr, Nt, K, B)
            M = np.random.randn(Nt, Nt, K, B) + 1j*np.random.randn(Nt, Nt, K, B)

            U = utils.lmmse(H, M, N0)

            E0 = self.slow_mse(H, U, M, N0)
            E = utils.mse(H, U, M, N0)

            np.testing.assert_almost_equal(np.linalg.norm(E - E0), 0, decimal=ACCURACY)

    def test_mse_benchmark(self, benchmark):
        H = np.random.randn(Nr, Nt, K, B) + 1j*np.random.randn(Nr, Nt, K, B)
        M = np.random.randn(Nt, Nt, K, B) + 1j*np.random.randn(Nt, Nt, K, B)

        U = utils.lmmse(H, M, N0)

        benchmark(utils.mse, H, U, M, N0)


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

            R0 = self.slow_rate(H, M, N0)
            R = utils.rate(H, M, N0)

            np.testing.assert_almost_equal(R, R0, decimal=ACCURACY)

    def test_rate_benchmark(self, benchmark):
        H = np.random.randn(Nr, Nt, K, B) + 1j*np.random.randn(Nr, Nt, K, B)
        M = np.random.randn(Nt, Nt, K, B) + 1j*np.random.randn(Nt, Nt, K, B)

        benchmark(utils.rate, H, M, N0)


class TestWB:
    def slow_wb(self, H, U, W, P, threshold=1e-6):
        (Nr, Nt, K, B) = H.shape
        (Nr, Nsk, K, B) = U.shape

        M = np.zeros((Nt, Nsk, K, B), dtype='complex')

        # Weighted transmit covariance matrices
        C = np.zeros((Nt, Nt, B), dtype='complex')

        for b in range(B):
            for i in range(B):
                for j in range(K):
                    T = np.dot(np.dot(U[:, :, j, i], W[:, :, j, i]),
                               U[:, :, j, i].conj().T)

                    C[:, :, b] += np.dot(np.dot(H[:, :, j, b].conj().T, T),
                                         H[:, :, j, b])

        for b in range(B):
            lb = 0.
            ub = 10.

            M0 = np.zeros((Nt, Nsk, K), dtype='complex')

            while np.abs(ub - lb) > threshold:
                v = (ub + lb) / 2

                for k in range(K):
                    T = np.dot(np.dot(H[:, :, k, b].conj().T, U[:, :, k, b]),
                               W[:, :, k, b])

                    M0[:, :, k] = np.dot(np.linalg.pinv(C[:, :, b] +
                                                        np.eye(Nt)*v), T)

                P_ = np.linalg.norm(M0[:])**2

                if P_ < P:
                    ub = v
                else:
                    lb = v

                if np.abs(lb-ub) < 1e-20:
                    ub *= 10

            M[:, :, :, b] = M0

        return M

    def test_random(self):
        for r in range(RANDOM_REALIZATIONS):
            Nsk = min(Nr, Nt)

            H = np.random.randn(Nr, Nt, K, B) + 1j*np.random.randn(Nr, Nt, K, B)

            M = np.random.randn(Nt, Nsk, K, B) + \
                1j*np.random.randn(Nt, Nsk, K, B)

            U = utils.lmmse(H, M, N0)

            W = np.zeros((Nsk, Nsk, K, B))
            for k in range(K):
                for b in range(B):
                    W[:, :, k, b] = np.random.rand(Nsk, Nsk)
                    W[:, :, k, b] = np.dot(W[:, :, k, b].T, W[:, :, k, b])

            M0 = self.slow_wb(H, U, W, 1)
            M = utils.weighted_bisection(H, U, W, 1)

            np.testing.assert_almost_equal(M, M0, decimal=ACCURACY)

    def test_wb_benchmark(self, benchmark):
        Nsk = min(Nr, Nt)

        H = np.random.randn(Nr, Nt, K, B) + 1j*np.random.randn(Nr, Nt, K, B)

        M = np.random.randn(Nt, Nsk, K, B) + \
            1j*np.random.randn(Nt, Nsk, K, B)

        U = utils.lmmse(H, M, N0)

        W = np.zeros((Nsk, Nsk, K, B))
        for k in range(K):
            for b in range(B):
                W[:, :, k, b] = np.random.rand(Nsk, Nsk)
                W[:, :, k, b] = np.dot(W[:, :, k, b].T, W[:, :, k, b])

        benchmark(utils.weighted_bisection, H, U, W, 1)
