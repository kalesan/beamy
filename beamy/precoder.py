"""This module provides various precoder designs."""
import itertools
import logging

import numpy as np

import beamy


class Precoder(object):
    """ This is the base class for all precoder design. The generator function
    should be overridden to comply with the corresponding design."""

    def __init__(self, precision=1e-6):
        self.logger = logging.getLogger(__name__)

        self.precision = precision

    def init(self, Nr, Nt, UE, BS, uplink=False):
        (self.Nr, self.Nt, self.UE, self.BS) = (Nr, Nt, UE, BS)

        self.Nsk = min(self.Nt, self.Nr)

        self.uplink = uplink

        if uplink:
            Nt = self.Nt
            self.Nt = self.Nr
            self.Nr = Nt

            UE = self.UE
            self.UE = self.BS
            self.BS = UE

        # Initialize (reset) precoder state
        self.reset()

    def normalize(self, prec, pwr_lim):
        """ Normalize the prec matrix to have per BS power constraint pwr_lim"""
        for _bs in range(self.BS):
            tmp = prec[:, :, :, _bs]

            prec[:, :, :, _bs] = tmp / np.sqrt((tmp[:]*tmp[:].conj()).sum())
            prec[:, :, :, _bs] *= pwr_lim

        return prec

    def generate(self, *_, **kwargs):
        """ This method should be overriden by the actual precoder design."""
        pass

    def reset(self):
        """ Resets the precoder state and parameters. """
        pass


class PrecoderGaussian(Precoder):
    """ This a simple Gaussian precoder design, which generates random Gaussian
        precoder matrices that are normalized according to the given power
        constraints."""

    def __init__(self, **kwargs):
        super(PrecoderGaussian, self).__init__(**kwargs)

    def generate(self, *_, **kwargs):
        """ Generate a Gaussian precoder"""

        prec = np.random.randn(self.Nt, self.Nsk, self.UE, self.BS) + \
            np.random.randn(self.Nt, self.Nsk, self.UE, self.BS)*1j

        return self.normalize(prec, kwargs.get('pwr_lim', 1))


class PrecoderWMMSE(Precoder):
    """Weighted minimum MSE (WMMSE) precoder design."""

    def __init__(self, **kwargs):
        super(PrecoderWMMSE, self).__init__(**kwargs)

    def generate(self, chan, recv, prec_prev, noise_pwr, **kwargs):
        """ Generate the WMMSE precoders. """

        pwr_lim = kwargs.get('pwr_lim', 1)

        errm = beamy.utils.mse(chan, recv, prec_prev, noise_pwr)

        weight = np.zeros((self.Nsk, self.Nsk, self.UE, self.BS),
                          dtype='complex')

        for (_ue, _bs) in itertools.product(range(self.UE), range(self.BS)):
            weight[:, :, _ue, _bs] = np.linalg.pinv(errm[:, :, _ue, _bs])

        return beamy.utils.weighted_bisection(chan, recv, weight, pwr_lim,
                                              threshold=self.precision)
