import sys
sys.path.append("../../marconi")

# from multiprocessing import Process

import logging
from simulator import Simulator
from chanmod import ClarkesModel

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
realizations = 50
biterations = 100


def simulate(_rx, _tx, _K, _B, _SNR, chmod):
    sparams = (_rx, _tx, _K, _B)

    act = {'BS': True, 'D2D': False}
    wmmse_res_file = "WMMSE-BS-%d-%d-%d-%d-%d-[%d].npz" % \
        (_rx, _tx, _K, _B, _SNR, chmod.termsep)

    sim = Simulator(precoder.PrecoderWMMSE(sparams), sysparams=sparams,
                    realizations=realizations, biterations=biterations,
                    resfile=wmmse_res_file, SNR=_SNR, channel_model=chmod,
                    active_links=act)

    sim.run()

    act = {'BS': False, 'D2D': True}
    wmmse_res_file = "WMMSE-D2D-%d-%d-%d-%d-%d-[%d].npz" % \
        (_rx, _tx, _K, _B, _SNR, chmod.termsep)

    sim = Simulator(precoder.PrecoderWMMSE(sparams), sysparams=sparams,
                    realizations=realizations, biterations=biterations,
                    resfile=wmmse_res_file, SNR=_SNR, channel_model=chmod,
                    active_links=act)

    sim.run()

    act = {'BS': True, 'D2D': True}
    wmmse_res_file = "WMMSE-D2D-%d-%d-%d-%d-%d-[%d].npz" % \
        (_rx, _tx, _K, _B, _SNR, chmod.termsep)

    sim = Simulator(precoder.PrecoderWMMSE(sparams), sysparams=sparams,
                    realizations=realizations, biterations=biterations,
                    resfile=wmmse_res_file, SNR=_SNR, channel_model=chmod,
                    active_links=act)

    sim.run()


if __name__ == '__main__':
    # The simulation cases
    (rx, tx, K, B, SNR, chmod) = (2, 4, 1, 1, 5, ClarkesModel(termsep=0))
    simulate(rx, tx, K, B, SNR, chmod)

    (rx, tx, K, B, SNR, chmod) = (2, 4, 1, 1, 5, ClarkesModel(termsep=-3))
    simulate(rx, tx, K, B, SNR, chmod)

    (rx, tx, K, B, SNR, chmod) = (2, 4, 2, 1, 5, ClarkesModel(termsep=0))
    simulate(rx, tx, K, B, SNR, chmod)
