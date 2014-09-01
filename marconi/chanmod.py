import numpy as np


class ChannelModel(object):
    def generate(self, Nr, Nt, K, B, realizations=1):
        pass


class GaussianModel(ChannelModel):
    def __init__(self):
        pass

    def generate(self, Nr, Nt, K, B, realizations=1):
        return (1/np.sqrt(2)) * (np.random.randn(Nr, Nt, K, B) +
                                 np.random.randn(Nr, Nt, K, B)*1j)
