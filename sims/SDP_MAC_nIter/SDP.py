import sys
sys.path.append("../../marconi")

# from multiprocessing import Process

import logging 
from simulator import Simulator

from precoder import *

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

while len(logger.handlers) > 0:
    logger.handlers.pop()

formatter = logging.Formatter("%(levelname)s - %(module)s - %(message)s")

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)

logger.addHandler(handler)

handler = logging.FileHandler('sdp.log')
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)

logger.addHandler(handler)

###
tol = 1e-10

realizations = 100
biterations = None

def simulate(_rx, _tx, _K, _B, _SNR):
    sparams = (_rx, _tx, _K, _B)

    sdp_res_file = "SDP-MAC-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
    sim = Simulator(PrecoderSDP_MAC.PrecoderSDP_MAC(sparams, solver_tolerance=tol),
                    sysparams=sparams, rate_conv_tol=rate_conv_tol, 
                    realizations=realizations, biterations=biterations,
                    resfile=sdp_res_file, SNR=_SNR)
    sim.run()

    wmmse_res_file = "WMMSE-MAC-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
    sim = Simulator(PrecoderWMMSE.PrecoderWMMSE(sparams, precision=tol),
                    sysparams=sparams, rate_conv_tol=rate_conv_tol,
                    realizations=realizations, biterations=biterations,
                    resfile=wmmse_res_file, SNR=_SNR)

    sim.run()


if __name__ == '__main__':
    SNR = 35

    # The simulation cases
    (rx, tx, K, B) = (4, 2, 1, 2)
    simulate(rx, tx, K, B, SNR)
