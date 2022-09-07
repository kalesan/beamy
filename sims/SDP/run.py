import sys
import os

# from multiprocessing import Process
scriptdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(scriptdir, '..', '..'))
sys.path.append(os.path.join(scriptdir, '..', '..', 'beamy'))

from beamy.simulator import Simulator  # noqa: E402
import precoder  # noqa: E402

####
realizations = 25
biterations = 50


def simulate(_rx, _tx, _K, _B, _SNR):
    sim = Simulator(precoder.PrecoderWMMSE(), 
                    bs=_B, users=_K, nr=_rx, nt=_tx,
                    realizations=realizations, biterations=biterations, 
                    SNR=_SNR)

    sim.run()

    sim = Simulator(precoder.PrecoderSDP(), 
                    bs=_B, users=_K, nr=_rx, nt=_tx,
                    realizations=realizations, biterations=biterations, 
                    SNR=_SNR, 
                    verbose_level=int(os.environ.get("VERBOSE", 0)))
    sim.run()


if __name__ == '__main__':
    # The simulation cases
    B = 1
    K = 4
    tx = 4
    rx = 2

    SNR = 20

    simulate(rx, tx, K, B, SNR)
