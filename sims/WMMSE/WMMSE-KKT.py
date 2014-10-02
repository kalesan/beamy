import sys
sys.path.append("../../marconi")

import datetime

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

handler = logging.FileHandler('WMMSE-%s.log' % (datetime.datetime.now().time()))
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)

logger.addHandler(handler)

####
realizations = 100
biterations = 100


def simulate(sysmod):
    (_rx, _tx, _K, _B) = sysmod.sysparams
    _SNR = sysmod.SNR

    act = {'BS': True, 'D2D': True}
    wmmse_res_file = "WMMSE-Cell-%d-%d-%d-%d-%d-%d-%d.npz" % \
        (_rx, _tx, _K, _B, _SNR, sysmod.gainmod.radius, sysmod.gainmod.d2d_dist)

    sim = Simulator(sysmod, precoder.PrecoderWMMSE(sysmod.sysparams),
                    realizations=realizations, biterations=biterations,
                    resfile=wmmse_res_file, active_links=act)

    sim.run()

    act = {'BS': False, 'D2D': True}
    wmmse_res_file = "WMMSE-D2D-%d-%d-%d-%d-%d-%d-%d.npz" % \
        (_rx, _tx, _K, _B, _SNR, sysmod.gainmod.radius, sysmod.gainmod.d2d_dist)

    sim = Simulator(sysmod, precoder.PrecoderWMMSE(sysmod.sysparams),
                    realizations=realizations, biterations=biterations,
                    resfile=wmmse_res_file, active_links=act)

    sim.run()

    act = {'BS': True, 'D2D': False}
    wmmse_res_file = "WMMSE-BS-%d-%d-%d-%d-%d-%d-%d.npz" % \
        (_rx, _tx, _K, _B, _SNR, sysmod.gainmod.radius, sysmod.gainmod.d2d_dist)

    sim = Simulator(sysmod, precoder.PrecoderWMMSE(sysmod.sysparams),
                    realizations=realizations, biterations=biterations,
                    resfile=wmmse_res_file, active_links=act)

    sim.run()



if __name__ == '__main__':
    # The simulation cases
    (dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 4, 1, 20., 100., 5.)
    gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         chanmod=ClarkesModel((dx, bx, K, B)))
    simulate(sysmod)

    (dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 4, 1, 20., 90., 5.)
    gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         chanmod=ClarkesModel((dx, bx, K, B)))
    simulate(sysmod)

    (dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 4, 1, 20., 80., 5.)
    gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         chanmod=ClarkesModel((dx, bx, K, B)))
    simulate(sysmod)

    (dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 4, 1, 20., 70., 5.)
    gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         chanmod=ClarkesModel((dx, bx, K, B)))
    simulate(sysmod)

    (dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 4, 1, 20., 60., 5.)
    gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         chanmod=ClarkesModel((dx, bx, K, B)))
    simulate(sysmod)

    (dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 4, 1, 20., 50., 5.)
    gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         chanmod=ClarkesModel((dx, bx, K, B)))
    simulate(sysmod)

    (dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 4, 1, 20., 40., 5.)
    gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         chanmod=ClarkesModel((dx, bx, K, B)))
    simulate(sysmod)

    (dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 4, 1, 20., 30., 5.)
    gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         chanmod=ClarkesModel((dx, bx, K, B)))
    simulate(sysmod)

    (dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 4, 1, 20., 20., 5.)
    gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         chanmod=ClarkesModel((dx, bx, K, B)))
    simulate(sysmod)

    (dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 4, 1, 20., 10., 5.)
    gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         chanmod=ClarkesModel((dx, bx, K, B)))
    simulate(sysmod)


    (dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 4, 1, 20., 5., 5.)
    gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         chanmod=ClarkesModel((dx, bx, K, B)))
    simulate(sysmod)


    # K = 8
    (dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 8, 1, 20., 100., 5.)
    gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         chanmod=ClarkesModel((dx, bx, K, B)))
    simulate(sysmod)

    (dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 8, 1, 20., 90., 5.)
    gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         chanmod=ClarkesModel((dx, bx, K, B)))
    simulate(sysmod)

    (dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 8, 1, 20., 80., 5.)
    gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         chanmod=ClarkesModel((dx, bx, K, B)))
    simulate(sysmod)

    (dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 8, 1, 20., 70., 5.)
    gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         chanmod=ClarkesModel((dx, bx, K, B)))
    simulate(sysmod)

    (dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 8, 1, 20., 60., 5.)
    gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         chanmod=ClarkesModel((dx, bx, K, B)))
    simulate(sysmod)

    (dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 8, 1, 20., 50., 5.)
    gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         chanmod=ClarkesModel((dx, bx, K, B)))
    simulate(sysmod)

    (dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 8, 1, 20., 40., 5.)
    gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         chanmod=ClarkesModel((dx, bx, K, B)))
    simulate(sysmod)

    (dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 8, 1, 20., 30., 5.)
    gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         chanmod=ClarkesModel((dx, bx, K, B)))
    simulate(sysmod)

    (dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 8, 1, 20., 20., 5.)
    gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         chanmod=ClarkesModel((dx, bx, K, B)))
    simulate(sysmod)

    (dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 8, 1, 20., 10., 5.)
    gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         chanmod=ClarkesModel((dx, bx, K, B)))
    simulate(sysmod)


    (dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 8, 1, 20., 5., 5.)
    gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         chanmod=ClarkesModel((dx, bx, K, B)))
    simulate(sysmod)
