import sys
sys.path.append("../../marconi")

# from multiprocessing import Process

import logging
from simulator import Simulator

import precoder

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
realizations = 2
biterations = 3


def simulate(_rx, _tx, _K, _B, _SNR):
    sparams = (_rx, _tx, _K, _B)

    sdp_res_file = "SDP2-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
    sim = Simulator(precoder.PrecoderSDP2(sparams), sysparams=sparams,
                    realizations=realizations, biterations=biterations,
                    resfile=sdp_res_file, SNR=_SNR)

    sim.run()

    sdp_res_file = "SDP-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
    sim = Simulator(precoder.PrecoderSDP(sparams), sysparams=sparams,
                    realizations=realizations, biterations=biterations,
                    resfile=sdp_res_file, SNR=_SNR)
    sim.run()


if __name__ == '__main__':
    # The simulation cases
    B = 1
    K = 4
    tx = 4
    rx = 2

    SNR = 20

    simulate(rx, tx, K, B, SNR)
