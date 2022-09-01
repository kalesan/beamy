import sys
import os
import logging

# from multiprocessing import Process
scriptdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(scriptdir, '..', '..'))
sys.path.append(os.path.join(scriptdir, '..', '..', 'beamy'))

from beamy.simulator import Simulator  # noqa: E402
import precoder  # noqa: E402

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s - %(module)s - %(message)s")
handler.setFormatter(formatter)

while len(logger.handlers) > 0:
    logger.handlers.pop()

logger.addHandler(handler)

####
realizations = 25
biterations = 50


def simulate(_rx, _tx, _K, _B, _SNR):
    sim = Simulator(precoder.PrecoderWMMSE(), 
                    bs=_B, users=_K, nr=_rx, nt=_tx,
                    realizations=realizations, biterations=biterations, 
                    SNR=_SNR)

    sim.run()


if __name__ == '__main__':
    # The simulation cases
    B = 1
    K = 4
    tx = 4
    rx = 2

    SNR = [20, 25]

    simulate(rx, tx, K, B, SNR)
