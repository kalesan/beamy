import numpy as np


class Precoder(object):
    def __init__(self, Nr, Nt, K, B, P, N0):
        self.Nr = Nr
        self.Nt = Nt
        self.K = K
        self.B = B

        self.P = P
        self.N0 = N0

        self.Nsk = min(Nt, Nr)

    def normalize(self, M):
        for b in range(self.B):
            M0 = M[:, :, :, b]

            M[:, :, :, b] = self.P * (M0 / np.sqrt((M0[:]*M0[:].conj()).sum()))

        return M


class PrecoderGaussian(Precoder):
    def generate(self):
        M = np.random.randn(self.Nt, self.Nsk, self.K, self. B) + \
            np.random.randn(self.Nt, self.Nsk, self.K, self. B)*1j

        return self.normalize(M)
