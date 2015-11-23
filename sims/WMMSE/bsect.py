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
realizations = 10
biterations = 50
precision = 1e-6


def simulate(_rx, _tx, _K, _B, _SNR):
    sparams = (_rx, _tx, _K, _B)

    wmmse_res_file = "WMMSE-B-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
    sim = Simulator(PrecoderWMMSE.PrecoderWMMSE(sparams, method='fixed'), 
                    sysparams=sparams,
                    realizations=realizations, biterations=biterations,
                    resfile=wmmse_res_file, SNR=_SNR, rate_conv_tol=precision)

    sim.run()

    wmmse_res_file = "WMMSE-S-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
    sim = Simulator(PrecoderSDP.PrecoderSDP(sparams), 
                    sysparams=sparams,
                    realizations=realizations, biterations=biterations,
                    resfile=wmmse_res_file, SNR=_SNR, rate_conv_tol=precision)

    sim.run()

    wmmse_res_file = "WMMSE-F-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
    sim = Simulator(PrecoderSDP.PrecoderSDP(sparams, method='fixed'), 
                    sysparams=sparams,
                    realizations=realizations, biterations=biterations,
                    resfile=wmmse_res_file, SNR=_SNR, rate_conv_tol=precision)

    sim.run()

    wmmse_res_file = "WMMSE-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
    sim = Simulator(PrecoderWMMSE.PrecoderWMMSE(sparams,
                                method='bisection_only'), 
                    sysparams=sparams,
                    realizations=realizations, biterations=biterations,
                    resfile=wmmse_res_file, SNR=_SNR, rate_conv_tol=precision)

    sim.run()

    wmmse_res_file = "WMMSE-5-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
    sim = Simulator(PrecoderWMMSE.PrecoderWMMSE(sparams,
                                method='fixed'), 
                    sysparams=sparams, txrxiter=5,
                    realizations=realizations, biterations=biterations,
                    resfile=wmmse_res_file, SNR=_SNR, rate_conv_tol=precision)

    sim.run()

    wmmse_res_file = "WMMSE-10-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
    sim = Simulator(PrecoderWMMSE.PrecoderWMMSE(sparams,
                                method='fixed'), 
                    sysparams=sparams, txrxiter=10, 
                    realizations=realizations, biterations=biterations,
                    resfile=wmmse_res_file, SNR=_SNR,
                    rate_conv_tol=precision)

    sim.run()


if __name__ == '__main__':
    # The simulation cases
    B = 1
    K = 4
    tx = 4
    rx = 2

    SNR = 25

    simulate(rx, tx, K, B, SNR)
