import itertools
import logging
from multiprocessing import Process, Queue

import picos as pic

import numpy as np
import scipy
import scipy.linalg

import utils

from precoder import Precoder


class PrecoderGaussian(Precoder):
    """ This a simple Gaussian precoder design, which generates random Gaussian
        precoder matrices that are normalized according to the given power
        constraints."""

    def generate(self, *_args, **kwargs):
        """ Generate a Gaussian precoder"""

        prec = np.random.randn(self.n_tx, self.n_sk, self.n_ue, self.n_bs) + \
            np.random.randn(self.n_tx, self.n_sk, self.n_ue, self.n_bs)*1j

        return self.normalize(prec, kwargs.get('pwr_lim', 1))
