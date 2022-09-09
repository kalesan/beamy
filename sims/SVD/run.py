import sys
import os
import numpy as np
import itertools

# from multiprocessing import Process
scriptdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(scriptdir, '..', '..'))
sys.path.append(os.path.join(scriptdir, '..', '..', 'beamy'))

from beamy.simulator import Simulator  # noqa: E402
import beamy.precoder  # noqa: E402


class PrecoderSVD(beamy.precoder.Precoder):
    def __init__(self, **kwargs):
        super(PrecoderSVD, self).__init__(**kwargs)

    def water_filling(self, gains):
        alpha_low = 0
        alpha_high = 1e5

        p = np.zeros(gains.shape)

        # Iterate while low/high bounds are further than stop_threshold
        while(np.abs(alpha_low - alpha_high) > self.precision):
            alpha = (alpha_low + alpha_high) / 2.0  # Test value in the middle of low/high

            # Solve the power allocation
            p = 1/alpha - self.noise_pwr / gains

            p[p < 0] = 0  # Consider only positive power allocation

            if (np.sum(p) > self.power):
                alpha_low = alpha
            else:
                alpha_high = alpha

        return np.sqrt(p)

    def generate(self, chan, recv, prec_prev, noise_pwr, **kwargs):
        prec = np.zeros(prec_prev.shape, dtype='complex')

        index = itertools.product(range(self.UE), range(self.BS))
        for (_ue, _bs) in index:
            [_, gains, V] = np.linalg.svd(chan[:, :, _ue, _bs])
            p = self.water_filling(gains)

            streams = len(gains)

            prec[:, :, _ue, _bs] = V[:, 0:streams] @ np.diag(p)

        return prec


####
realizations = 25
biterations = 50

if __name__ == '__main__':
    # The simulation cases
    B = 1
    K = 1

    tx = [2, 4, 8]
    rx = [2, 4, 8]

    SNR = [20, 25, 30, 35]

    sim = Simulator(PrecoderSVD(),
                    bs=B, users=K, nr=rx, nt=tx,
                    realizations=realizations, biterations=biterations,
                    SNR=SNR,
                    verbose_level=int(os.environ.get("VERBOSE", 0)))

    sim.run()
