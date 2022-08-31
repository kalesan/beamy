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
    sparams = (_rx, _tx, _K, _B)

    wmmse_res_file = "wmmse-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
    sim = Simulator(precoder.PrecoderWMMSE(), sysparams=sparams,
                    realizations=realizations, biterations=biterations,
                    resfile=wmmse_res_file, SNR=_SNR)

    sim.run()


if __name__ == '__main__':
    # The simulation cases
    B = 1
    K = 4
    tx = 4
    rx = 2

    SNR = 20

    simulate(rx, tx, K, B, SNR)
