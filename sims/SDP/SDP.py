import sys
sys.path.append("../../marconi")

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

B = 1
K = 4
tx = 2
rx = 2

realizations = 1
biterations = 15

sparams = (rx, tx, K, B)

sim = Simulator(precoder.PrecoderWMMSE(sparams), sysparams=sparams,
                realizations=realizations, biterations=biterations,
                resfile='wmmse.npz')
sim.run()

sim = Simulator(precoder.PrecoderSDP(sparams), sysparams=sparams,
                realizations=realizations, biterations=biterations,
                resfile='sdp.npz')
sim.run()
