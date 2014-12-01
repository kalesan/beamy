import sys
sys.path.append("../../marconi")

import datetime

# from multiprocessing import Process

import logging
from simulator import Simulator

from chanmod import ClarkesModel, RicianModel, PreModel
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

realizations = 1
biterations = 25


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

    act = {'BS': True, 'D2D': False}
    wmmse_res_file = "WMMSE-BS-%d-%d-%d-%d-%d-%d-%d.npz" % \
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

if __name__ == '__main__':
    # The simulation cases
    #(dx, bx, K, B, SNR, radius, d2d_dist) = (2, 8, 8, 1, 10., 100., 20.)
    #gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    #sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         #chanmod=RicianModel())
    #simulate(sysmod)

    #(dx, bx, K, B, SNR, radius, d2d_dist) = (2, 8, 4, 1, 10., 90., 20.)
    #gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    #sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         #chanmod=RicianModel())
    #simulate(sysmod)

    #(dx, bx, K, B, SNR, radius, d2d_dist) = (2, 8, 4, 1, 10., 80., 20.)
    #gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    #sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         #chanmod=RicianModel())
    #simulate(sysmod)

    #(dx, bx, K, B, SNR, radius, d2d_dist) = (2, 8, 4, 1, 10., 70., 20.)
    #gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    #sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         #chanmod=RicianModel())
    #simulate(sysmod)

    #(dx, bx, K, B, SNR, radius, d2d_dist) = (2, 8, 4, 1, 10., 60., 20.)
    #gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    #sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         #chanmod=RicianModel())
    #simulate(sysmod)

    #(dx, bx, K, B, SNR, radius, d2d_dist) = (2, 8, 4, 1, 10., 50., 20.)
    #gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    #sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         #chanmod=RicianModel())
    #simulate(sysmod)

    #(dx, bx, K, B, SNR, radius, d2d_dist) = (2, 8, 4, 1, 10., 40., 20.)
    #gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    #sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         #chanmod=RicianModel())
    #simulate(sysmod)

    #(dx, bx, K, B, SNR, radius, d2d_dist) = (2, 8, 4, 1, 10., 30., 20.)
    #gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    #sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         #chanmod=RicianModel())
    #simulate(sysmod)

    (dx, bx, K, B, SNR, radius, d2d_dist) = (2, 8, 4, 1, 10., 20., 20.)
    gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         chanmod=RicianModel())
    simulate(sysmod)

    #(dx, bx, K, B, SNR, radius, d2d_dist) = (2, 8, 4, 1, 10., 10., 20.)
    #gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    #sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         #chanmod=RicianModel())
    #simulate(sysmod)

    #(dx, bx, K, B, SNR, radius, d2d_dist) = (2, 8, 4, 1, 10., 5., 20.)
    #gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    #sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         #chanmod=PreModel(chanfile='chan-2-8-4-1-1-150.mat'))
    #simulate(sysmod)




    # K = 8
    #(dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 8, 1, 10., 100., 5.)
    #gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    #sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         #chanmod=ClarkesModel((dx, bx, K, B)))
    #simulate(sysmod)

    #(dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 8, 1, 10., 90., 5.)
    #gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    #sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         #chanmod=ClarkesModel((dx, bx, K, B)))
    #simulate(sysmod)

    #(dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 8, 1, 10., 80., 5.)
    #gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    #sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         #chanmod=ClarkesModel((dx, bx, K, B)))
    #simulate(sysmod)

    #(dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 8, 1, 10., 70., 5.)
    #gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    #sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         #chanmod=ClarkesModel((dx, bx, K, B)))
    #simulate(sysmod)

    #(dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 8, 1, 10., 60., 5.)
    #gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    #sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         #chanmod=ClarkesModel((dx, bx, K, B)))
    #simulate(sysmod)

    #(dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 8, 1, 10., 50., 5.)
    #gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    #sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         #chanmod=ClarkesModel((dx, bx, K, B)))
    #simulate(sysmod)

    #(dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 8, 1, 10., 40., 5.)
    #gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    #sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         #chanmod=ClarkesModel((dx, bx, K, B)))
    #simulate(sysmod)

    #(dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 8, 1, 10., 30., 5.)
    #gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    #sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         #chanmod=ClarkesModel((dx, bx, K, B)))
    #simulate(sysmod)

    #(dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 8, 1, 10., 20., 5.)
    #gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    #sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         #chanmod=ClarkesModel((dx, bx, K, B)))
    #simulate(sysmod)

    #(dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 8, 1, 10., 10., 5.)
    #gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    #sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         #chanmod=ClarkesModel((dx, bx, K, B)))
    #simulate(sysmod)

    #(dx, bx, K, B, SNR, radius, d2d_dist) = (2, 4, 8, 1, 10., 5., 5.)
    #gainmod = Uniform1(K, B, SNR=SNR, radius=radius, d2d_dist=d2d_dist)
    #sysmod = SystemModel(dx, bx, K, B, SNR=SNR, gainmod=gainmod,
                         #chanmod=ClarkesModel((dx, bx, K, B)))
    #simulate(sysmod)
