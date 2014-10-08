import sys
sys.path.append("../../marconi")

# from multiprocessing import Process

import logging
from simulator import Simulator

from chanmod import ClarkesModel
from gainmod import Uniform1
from sysmodel import SystemModel

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


def simulate(sysmod):
    (_rx, _tx, _K, _B) = sysmod.sysparams
    _SNR = sysmod.SNR

    act = {'BS': True, 'D2D': True}
    wmmse_res_file = "WMMSE-Cell-%d-%d-%d-%d-%d.npz" % \
        (_rx, _tx, _K, _B, _SNR)

    sim = Simulator(sysmod, precoder.PrecoderCVX(sysmod.sysparams),
                    realizations=realizations, biterations=biterations,
                    resfile=wmmse_res_file, active_links=act)

    sim.run()

    act = {'BS': False, 'D2D': True}
    wmmse_res_file = "WMMSE-D2D-%d-%d-%d-%d-%d.npz" % \
        (_rx, _tx, _K, _B, _SNR)

    sim = Simulator(sysmod, precoder.PrecoderCVX(sysmod.sysparams),
                    realizations=realizations, biterations=biterations,
                    resfile=wmmse_res_file, active_links=act)

    sim.run()

    act = {'BS': True, 'D2D': False}
    wmmse_res_file = "WMMSE-BS-%d-%d-%d-%d-%d.npz" % \
        (_rx, _tx, _K, _B, _SNR)

    sim = Simulator(sysmod, precoder.PrecoderCVX(sysmod.sysparams),
                    realizations=realizations, biterations=biterations,
                    resfile=wmmse_res_file, active_links=act)

    sim.run()


if __name__ == '__main__':
    # The simulation cases
    (dx, bx, K, B, SNR) = (2, 4, 4, 1, 20)
    sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=Uniform1(K, B, SNR=SNR),
                         chanmod=ClarkesModel((dx, bx, K, B)))

    simulate(sysmod)
