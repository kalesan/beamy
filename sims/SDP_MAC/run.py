import os
import sys

# from multiprocessing import Process
scriptdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(scriptdir, '..', '..'))
sys.path.append(os.path.join(scriptdir, '..', '..', 'beamy'))

from beamy.simulator import Simulator  # noqa: E402
import precoder  # noqa: E402

####
realizations = 10
biterations = 150


def simulate(_rx, _tx, _K, _B, _SNR):
    sim = Simulator(precoder.PrecoderWMMSE(precision=1e-8),
                    bs=_B, users=K, nr=_rx, nt=_tx,
                    realizations=realizations, biterations=biterations,
                    SNR=_SNR, uplink=True,
                    verbose_level=int(os.environ.get("VERBOSE", 0)))

    sim.run()

    sim = Simulator(precoder.PrecoderWMMSE(precision=1e-8),
                    bs=_B, users=K, nr=_rx, nt=_tx,
                    realizations=realizations, biterations=biterations,
                    SNR=_SNR, uplink=True, txrxiter=5,
                    verbose_level=int(os.environ.get("VERBOSE", 0)))

    sim.run()

    sim = Simulator(precoder.PrecoderWMMSE(precision=1e-8),
                    bs=_B, users=K, nr=_rx, nt=_tx,
                    realizations=realizations, biterations=biterations,
                    SNR=_SNR, uplink=True, txrxiter=10,
                    verbose_level=int(os.environ.get("VERBOSE", 0)))

    sim.run()

    sim = Simulator(precoder.PrecoderSDP_MAC(),
                    bs=_B, users=K, nr=_rx, nt=_tx,
                    realizations=realizations, biterations=biterations,
                    SNR=_SNR, uplink=True,
                    verbose_level=int(os.environ.get("VERBOSE", 0)))
    sim.run()


if __name__ == '__main__':
    SNR = 25

    # The simulation cases
    (rx, tx, K, B) = (4, 4, 2, 1)
    simulate(rx, tx, K, B, SNR)

    # (rx, tx, K, B) = (2, 4, 4, 1)
    # simulate(rx, tx, K, B, SNR)

    # (rx, tx, K, B) = (4, 8, 4, 1)
    # simulate(rx, tx, K, B, SNR)
