import itertools
import logging
from multiprocessing import Process, Queue

import picos as pic

import numpy as np
import scipy
import scipy.linalg

import utils


class Precoder(object):
    """ This is the base class for all precoder design. The generator function
    should be overridden to comply with the corresponding design."""

    def __init__(self, sysparams, precision=1e-6,
            solver_tolerance=1e-6):

        (self.n_rx, self.n_tx, self.n_ue, self.n_bs) = sysparams

        self.n_sk = min(self.n_tx, self.n_rx)

        self.logger = logging.getLogger(__name__)

        self.precision = precision

        self.solver_tolerance = solver_tolerance

        # Initialize (reset) precoder state
        self.reset()

    def normalize(self, prec, pwr_lim):
        """ Normalize the prec matrix to have per BS power constraint pwr_lim"""
        for _bs in range(self.n_bs):
            tmp = prec[:, :, :, _bs]

            prec[:, :, :, _bs] = tmp / np.sqrt((tmp[:]*tmp[:].conj()).sum())
            prec[:, :, :, _bs] *= pwr_lim

        return prec

    def generate(self, *_args, **kwargs):
        """ This method should be overriden by the actual precoder design."""
        pass

    def reset(self):
        """ Resets the precoder state and parameters. """
        pass




