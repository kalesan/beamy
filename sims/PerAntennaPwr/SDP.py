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
biterations = 100


def simulate(_rx, _tx, _K, _B, _SNR):
    sparams = (_rx, _tx, _K, _B)

    sdp_res_file = "SDP-MAC-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
    sim = Simulator(PrecoderSDP_MAC.PrecoderSDP_MAC(sparams,
                        perantenna_power=True, solver_tolerance=1e-8),
                    sysparams=sparams,
                    realizations=realizations, biterations=biterations,
                    resfile=sdp_res_file, SNR=_SNR)
    sim.run()

    wmmse_res_file = "WMMSE-MAC-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
    sim = Simulator(PrecoderWMMSE_SDP.PrecoderWMMSE_SDP(sparams, 
                        perantenna_power=True), sysparams=sparams,
                    realizations=realizations, biterations=biterations,
                    resfile=wmmse_res_file, SNR=_SNR)

    sim.run()

    wmmse_res_file = "WMMSE-MAC-5-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
    sim = Simulator(PrecoderWMMSE_SDP.PrecoderWMMSE_SDP(sparams, 
                        perantenna_power=True), sysparams=sparams,
                    realizations=realizations, biterations=biterations,
                    resfile=wmmse_res_file, SNR=_SNR, txrxiter=5)

    sim.run()

    wmmse_res_file = "WMMSE-MAC-10-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
    sim = Simulator(PrecoderWMMSE_SDP.PrecoderWMMSE_SDP(sparams, 
                        perantenna_power=True), sysparams=sparams,
                    realizations=realizations, biterations=biterations,
                    resfile=wmmse_res_file, SNR=_SNR, txrxiter=10)

    sim.run()


if __name__ == '__main__':
    SNR = 30

    # The simulation cases
    (rx, tx, K, B) = (9, 3, 1, 3)
    simulate(rx, tx, K, B, SNR)

    # (rx, tx, K, B) = (2, 4, 4, 1)
    # simulate(rx, tx, K, B, SNR)

    # (rx, tx, K, B) = (4, 8, 4, 1)
    # simulate(rx, tx, K, B, SNR)
