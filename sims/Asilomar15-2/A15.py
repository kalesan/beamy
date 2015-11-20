import sys
sys.path.append("../../marconi")

import numpy as np

import logging
from simulator import Simulator

from precoder import *

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s - %(module)s - %(message)s")
handler.setFormatter(formatter)

fileHandler = logging.FileHandler('A15.log')
fileHandler.setLevel(logging.DEBUG)
fileHandler.setFormatter(formatter)

while len(logger.handlers) > 0:
    logger.handlers.pop()

logger.addHandler(handler)
logger.addHandler(fileHandler)

filename = 'done.npy'
done_cases = np.array([])

####
realizations = 50
biterations = 150
precision = 1e-5

def simulate(_rx, _tx, _K, _B, _SNR, biter):
    sparams = (_rx, _tx, _K, _B)

    global done_cases

    typ = 'iter'
    if biter is not None:
        typ = '%d' % (biter)

    if biter is not None:
        # SDP-F
        case = 'SDP-F-%d-%d-%d-%d-%d-%d-%s.npz' % \
                    (_rx, _tx, _K, _B, _SNR, realizations, typ)

        if not np.any(done_cases == np.array([case])):
            sdp_res_file = "SDP-F-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
            sim = Simulator(PrecoderSDP.PrecoderSDP(sparams, 
                            solver_tolerance=1e-8, precision=precision, 
                            method='fixed'),
                            sysparams=sparams,
                            realizations=realizations, biterations=biter,
                            resfile=sdp_res_file, SNR=_SNR)
            sim.run()

            done_cases = np.concatenate((np.array([case]), done_cases))
            np.save(filename, done_cases)
        else:
            print("Skipping: %s" % case)

        # WMMSE 
        case = 'WMMSE-%d-%d-%d-%d-%d-%d-%s.npz' % \
                    (_rx, _tx, _K, _B, _SNR, realizations, typ)

        if not np.any(done_cases == np.array([case])):
            wmmse_res_file = "WMMSE-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
            sim = Simulator(PrecoderWMMSE.PrecoderWMMSE(sparams, precision=precision), 
                            sysparams=sparams,
                            realizations=realizations, biterations=biter,
                            resfile=wmmse_res_file, SNR=_SNR)

            sim.run()

            done_cases = np.concatenate((np.array([case]), done_cases))
            np.save(filename, done_cases)
        else:
            print("Skipping: %s" % case)

        # WMMSE-5 
        case = 'WMMSE-5-%d-%d-%d-%d-%d-%d-%s.npz' % \
                    (_rx, _tx, _K, _B, _SNR, realizations, typ)

        if not np.any(done_cases == np.array([case])):
            wmmse_res_file = "WMMSE-5-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
            sim = Simulator(PrecoderWMMSE.PrecoderWMMSE(sparams, precision=precision), 
                            sysparams=sparams,
                            realizations=realizations, biterations=biter,
                            resfile=wmmse_res_file, SNR=_SNR, txrxiter=5)

            sim.run()

            done_cases = np.concatenate((np.array([case]), done_cases))
            np.save(filename, done_cases)
        else:
            print("Skipping: %s" % case)

        # WMMSE-10
        case = 'WMMSE-10-%d-%d-%d-%d-%d-%d-%s.npz' % \
                    (_rx, _tx, _K, _B, _SNR, realizations, typ)

        if not np.any(done_cases == np.array([case])):
            wmmse_res_file = "WMMSE-10-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
            sim = Simulator(PrecoderWMMSE.PrecoderWMMSE(sparams, precision=precision), 
                            sysparams=sparams,
                            realizations=realizations, biterations=biter,
                            resfile=wmmse_res_file, SNR=_SNR, txrxiter=10)

            sim.run()

            done_cases = np.concatenate((np.array([case]), done_cases))
            np.save(filename, done_cases)
        else:
            print("Skipping: %s" % case)

        # SDP-MAC
        case = 'SDP-%d-%d-%d-%d-%d-%d-%s.npz' % \
                    (_rx, _tx, _K, _B, _SNR, realizations, typ)

        if not np.any(done_cases == np.array([case])):
            sdp_res_file = "SDP-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
            sim = Simulator(PrecoderSDP.PrecoderSDP(sparams, 
                            solver_tolerance=1e-8, precision=precision, 
                            method='bisection'),
                            sysparams=sparams,
                            realizations=realizations, biterations=biter,
                            resfile=sdp_res_file, SNR=_SNR)
            sim.run()

            done_cases = np.concatenate((np.array([case]), done_cases))
            np.save(filename, done_cases)
        else:
            print("Skipping: %s" % case)

    else:
        # SDP-MAC
        case = 'SDP-%d-%d-%d-%d-%d-%d-%s.npz' % \
                    (_rx, _tx, _K, _B, _SNR, realizations, typ)

        if not np.any(done_cases == np.array([case])):
            sdp_res_file = "SDP-nIter-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
            sim = Simulator(PrecoderSDP.PrecoderSDP(sparams,
                            solver_tolerance=1e-8, precision=precision),
                            sysparams=sparams,
                            realizations=realizations, biterations=None,
                            resfile=sdp_res_file, SNR=_SNR)
            sim.run()

            done_cases = np.concatenate((np.array([case]), done_cases))
            np.save(filename, done_cases)
        else:
            print("Skipping: %s" % case)

        # WMMSE-MAC
        case = 'WMMSE-%d-%d-%d-%d-%d-%d-%s.npz' % \
                    (_rx, _tx, _K, _B, _SNR, realizations, typ)

        if not np.any(done_cases == np.array([case])):
            wmmse_res_file = "WMMSE-nIter-%d-%d-%d-%d-%d.npz" % (_rx, _tx, _K, _B, _SNR)
            sim = Simulator(PrecoderWMMSE.PrecoderWMMSE(sparams, precision=precision), 
                            sysparams=sparams,
                            realizations=realizations, biterations=None,
                            resfile=wmmse_res_file, SNR=_SNR)

            sim.run()

            done_cases = np.concatenate((np.array([case]), done_cases))
            np.save(filename, done_cases)
        else:
            print("Skipping: %s" % case)

def parf(parms):
    (rx, tx, K, B, SNR, biter) = parms
    simulate(rx, tx, K, B, SNR, biter)


if __name__ == '__main__':
    try:
        done_cases = np.load(filename)
    except:
        done_cases = np.array([])


    allargs = []

    SNR_Range = [5, 10, 15, 20, 25, 30]

    # The simulation cases
    # (rx, tx, K, B) = (3, 9, 3, 1)
    # allargs += [(rx, tx, K, B, SNR, None) for SNR in SNR_Range]
    # allargs += [(rx, tx, K, B, SNR, biterations) for SNR in SNR_Range]

    (rx, tx, K, B) = (2, 2, 2, 1)
    # allargs += [(rx, tx, K, B, SNR, None) for SNR in SNR_Range]
    allargs += [(rx, tx, K, B, SNR, biterations) for SNR in SNR_Range]

    (rx, tx, K, B) = (2, 4, 4, 1)
    # allargs += [(rx, tx, K, B, SNR, None) for SNR in SNR_Range]
    allargs += [(rx, tx, K, B, SNR, biterations) for SNR in SNR_Range]

    (rx, tx, K, B) = (2, 10, 5, 1)
    # allargs += [(rx, tx, K, B, SNR, None) for SNR in SNR_Range]
    allargs += [(rx, tx, K, B, SNR, biterations) for SNR in SNR_Range]

    # (rx, tx, K, B) = (9, 3, 3, 1)
    # allargs += [(rx, tx, K, B, SNR, None) for SNR in SNR_Range]
    # allargs += [(rx, tx, K, B, SNR, biterations) for SNR in SNR_Range]

    for a in allargs:
        parf(a)
