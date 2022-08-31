import logging
import os
import sys

# from multiprocessing import Process
scriptdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(scriptdir, '..', '..'))
sys.path.append(os.path.join(scriptdir, '..', '..', 'beamy'))

from beamy.simulator import Simulator  # noqa: E402
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

    sim = Simulator(precoder.PrecoderWMMSE(sparams, uplink=True, precision=1e-8),
                    sysparams=sparams,
                    realizations=realizations, biterations=biterations, 
                    SNR=_SNR, uplink=True)

    sim.run()

    sim = Simulator(precoder.PrecoderWMMSE(sparams, uplink=True, precision=1e-8),
                    sysparams=sparams,
                    realizations=realizations, biterations=biterations, 
                    SNR=_SNR, uplink=True, txrxiter=5)

    sim.run()

    sim = Simulator(precoder.PrecoderWMMSE(sparams, uplink=True, precision=1e-8),
                    sysparams=sparams,
                    realizations=realizations, biterations=biterations, 
                    SNR=_SNR, uplink=True, txrxiter=10)

    sim.run()

    sim = Simulator(precoder.PrecoderSDP_MAC(sparams, uplink=True),
                    sysparams=sparams,
                    realizations=realizations, biterations=biterations,
                    SNR=_SNR, uplink=True)
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
