import pytest
import numpy as np

from beamy import simulator
from beamy import precoder
from beamy import chanmod

RANDOM_REALIZATIONS = 5

ACCURACY = 6

Nr = 5
Nt = 7
K = 10
B = 3

N0 = 2


class TestIterationStats:
    def test_rate_types(self):
        Nsk = Nt

        iterations = 50

        prec = np.zeros((Nt, Nsk, K, B, iterations), dtype='complex')
        recv = np.zeros((Nr, Nsk, K, B, iterations), dtype='complex')

        channel = np.random.randn(Nr, Nt, K, B, iterations) + 1j*np.random.randn(Nr, Nt, K, B, iterations)

        sim1 = simulator.Simulator(precoder.PrecoderGaussian,  
            bs=B, users=K, nr=Nr, nt=Nt, rate_type='sum-rate')
        stats1 = sim1.iteration_stats(channel, recv, prec)

        sim2 = simulator.Simulator(precoder.PrecoderGaussian,  
            bs=B, users=K, nr=Nr, nt=Nt, rate_type='average-per-cell')
        stats2 = sim2.iteration_stats(channel, recv, prec)

        sim3 = simulator.Simulator(precoder.PrecoderGaussian,
            bs=B, users=K, nr=Nr, nt=Nt, rate_type='average-per-user')

        stats3 = sim3.iteration_stats(channel, recv, prec)

        np.testing.assert_almost_equal(stats1['rate'][iterations-1], stats2['rate'][iterations-1]*B, decimal=ACCURACY)
        np.testing.assert_almost_equal(stats1['rate'][iterations-1], stats3['rate'][iterations-1]*K, decimal=ACCURACY)
        np.testing.assert_almost_equal(stats2['rate'][iterations-1]*K, stats3['rate'][iterations-1]*B, decimal=ACCURACY)