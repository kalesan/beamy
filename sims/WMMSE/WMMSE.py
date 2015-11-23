import sys
sys.path.append("../../marconi")

# from multiprocessing import Process

import logging
from simulator import Simulator

from precoder import *

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
realizations = 1
biterations = None

perantenna_power = True


def simulate(_rx, _tx, _K, _B, _SNR):
    sparams = (_rx, _tx, _K, _B)

    wmmse_res_file = "WMMSE-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
    sim = Simulator(PrecoderWMMSE.PrecoderWMMSE(sparams), sysparams=sparams,
                    realizations=realizations, biterations=biterations,
                    resfile=wmmse_res_file, SNR=_SNR)

    sim.run()

    sdp_res_file = "SDP-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
    sim = Simulator(PrecoderWMMSE_SDP.PrecoderWMMSE_SDP(sparams,
        perantenna_power=perantenna_power), sysparams=sparams,
                    realizations=realizations, biterations=biterations,
                    resfile=sdp_res_file, SNR=_SNR, 
                    perantenna_power=perantenna_power)
    sim.run()


if __name__ == '__main__':
    # The simulation cases
    B = 1
    K = 4
    tx = 4
    rx = 2

    SNR = 20

    simulate(rx, tx, K, B, SNR)
