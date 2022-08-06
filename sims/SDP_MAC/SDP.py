import logging
import sys
sys.path.append("../../beamy")

# from multiprocessing import Process

from simulator import Simulator  # noqa: E402

import precoder  # noqa: E402

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

####
realizations = 10
biterations = 150


def simulate(_rx, _tx, _K, _B, _SNR):
    sparams = (_rx, _tx, _K, _B)

    wmmse_res_file = "WMMSE-MAC-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
    sim = Simulator(precoder.PrecoderWMMSE(sparams, uplink=True, precision=1e-8),
                    sysparams=sparams,
                    realizations=realizations, biterations=biterations,
                    resfile=wmmse_res_file, SNR=_SNR, uplink=True)

    sim.run()

    wmmse_res_file = "WMMSE-MAC-5-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
    sim = Simulator(precoder.PrecoderWMMSE(sparams, uplink=True, precision=1e-8),
                    sysparams=sparams,
                    realizations=realizations, biterations=biterations,
                    resfile=wmmse_res_file, SNR=_SNR, uplink=True, txrxiter=5)

    sim.run()

    wmmse_res_file = "WMMSE-MAC-10-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
    sim = Simulator(precoder.PrecoderWMMSE(sparams, uplink=True, precision=1e-8),
                    sysparams=sparams,
                    realizations=realizations, biterations=biterations,
                    resfile=wmmse_res_file, SNR=_SNR, uplink=True, txrxiter=10)

    sim.run()

    sdp_res_file = "SDP-MAC-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
    sim = Simulator(precoder.PrecoderSDP_MAC(sparams, uplink=True),
                    sysparams=sparams,
                    realizations=realizations, biterations=biterations,
                    resfile=sdp_res_file, SNR=_SNR, uplink=True)
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
