import sys
sys.path.append("../../marconi")

import datetime

# from multiprocessing import Process

import logging
from simulator import Simulator

from chanmod import RicianModel, RicianModel
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

handler = logging.FileHandler('WMMSE-%s.log' % (datetime.datetime.now().time()))
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)

logger.addHandler(handler)

####

realizations = 100
biterations = 100

(dx, bx, K, B, SNR, radius, d2d_dist) = (2, 8, 4, 1, 15., 100., 20.)

if __name__ == '__main__':
    # The simulation cases

    radius = 15.0
    gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                        chanmod=RicianModel())

    sim = Simulator(sysmod, precoder.PrecoderWMMSE(sysmod.sysparams),
                    realizations=realizations, biterations=biterations)

    sim.run_iter(sysmod)
