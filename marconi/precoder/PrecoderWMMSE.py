import itertools
import logging
from multiprocessing import Process, Queue

import picos as pic

import numpy as np
import scipy
import scipy.linalg

import utils

from precoder import Precoder


class PrecoderWMMSE(Precoder):
    """Weighted minimum MSE (WMMSE) precoder design."""

    def __init__(self, sysparams, precision=1e-6,
                          solver_tolerance=1e-6, method='bisection_only'):

        Precoder.__init__(self, sysparams, precision=precision,
                solver_tolerance=solver_tolerance)

        self.method = method

    def generate(self, *args, **kwargs):
        """ Generate the WMMSE precoders. """

        [chan, recv, prec_prev, noise_pwr] = [_a for _a in args]

        (n_rx, n_tx, n_ue, n_bs) = chan.shape

        pwr_lim = kwargs.get('pwr_lim', 1)

        errm = utils.mse(chan, recv, prec_prev, noise_pwr)

        weight = np.zeros((self.n_sk, self.n_sk, self.n_ue, self.n_bs),
                          dtype='complex')

        for (_ue, _bs) in itertools.product(range(self.n_ue), range(self.n_bs)):
            weight[:, :, _ue, _bs] = np.linalg.pinv(errm[:, :, _ue, _bs])

        # For these special conditions, we don't need bisection
        if (n_bs == 1) and (self.method is not "bisection_only"):
            nu = 0
            for (_ue, _bs) in itertools.product(range(self.n_ue), range(self.n_bs)):
                nu += np.trace(np.dot(np.dot(recv[:, :, _ue, _bs], 
                                             weight[:, :, _ue, _bs]),
                                             recv[:, :, _ue, _bs].conj().T))

            nu = np.real(noise_pwr * nu / pwr_lim)

            prec = utils.weighted_bisection(chan, recv, weight, pwr_lim, nu=nu)

            b = np.sqrt(pwr_lim) / np.linalg.norm(prec.flatten())

            prec *= b

            return prec

        return utils.weighted_bisection(chan, recv, weight, pwr_lim,
                                        threshold=self.precision)
