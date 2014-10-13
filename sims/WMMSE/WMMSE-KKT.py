#!/usr/bin/env python2 

import argparse
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

parser = argparse.ArgumentParser()

parser.add_argument("radius", help="Cell radius in meters.", type=int)
#parser.add_argument('d2d_dist', "Distance of the D2D pairs in meters.", 
#                    type=int)

args = parser.parse_args()
radius = args.radius

realizations = 100
biterations = 100

(dx, bx, K, B, SNR, d2d_dist) = (2, 8, 4, 1, 15., 20.)

K_factor = 10

# Header
logger.info("Ndx: %d, Ntx: %d, K: %d, B: %d, SNR: %ddB, D2D dist: %dm " + 
            "and Radius: %dm", dx, bx, K, B, SNR, d2d_dist, radius)

gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                    chanmod=RicianModel(K_factor=K_factor))

sim = Simulator(sysmod, precoder.PrecoderWMMSE(sysmod.sysparams),
                realizations=realizations, biterations=biterations)

sim.run_iter(sysmod)
