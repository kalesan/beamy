import sys
sys.path.append("../../marconi")

# from multiprocessing import Process

import logging
from simulator import Simulator

import precoder

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

while len(logger.handlers) > 0:
    logger.handlers.pop()

formatter = logging.Formatter("%(levelname)s - %(module)s - %(message)s")

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)

logger.addHandler(handler)

handler = logging.FileHandler('WMMSE.log')
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)

logger.addHandler(handler)

####
realizations = 25
biterations = 50


def simulate(_rx, _tx, _K, _B, _SNR):
    sparams = (_rx, _tx, _K, _B)

    wmmse_res_file = "WMMSE-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
    sim = Simulator(precoder.PrecoderWMMSE(sparams), sysparams=sparams,
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
