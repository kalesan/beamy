"""This module provides various precoder designs."""
import itertools

import cvxopt as cvx
import picos as pic

import numpy as np
import scipy

import utils


class Precoder(object):
    """ This is the base class for all precoder design. The generator function
    should be overridden to comply with the corresponding design."""

    def __init__(self, sysparams):
        (self.n_rx, self.n_tx, self.n_ue, self.n_bs) = sysparams

        self.n_sk = min(self.n_tx, self.n_rx)

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


class PrecoderGaussian(Precoder):
    """ This a simple Gaussian precoder design, which generates random Gaussian
        precoder matrices that are normalized according to the given power
        constraints."""

    def generate(self, *_args, **kwargs):
        """ Generate a Gaussian precoder"""

        prec = np.random.randn(self.n_tx, self.n_sk, self.n_ue, self.n_bs) +  \
            np.random.randn(self.n_tx, self.n_sk, self.n_ue, self.n_bs)*1j

        return self.normalize(prec, kwargs.get('pwr_lim', 1))


class PrecoderWMMSE(Precoder):
    """Weighted minimum MSE (WMMSE) precoder design."""

    def generate(self, *args, **kwargs):
        """ Generate the WMMSE precoders. """

        [chan, recv, prec_prev, noise_pwr] = [_a for _a in args]

        pwr_lim = kwargs.get('pwr_lim', 1)

        errm = utils.mse(chan, recv, prec_prev, noise_pwr)

        weight = np.zeros((self.n_sk, self.n_sk, self.n_ue, self.n_bs),
                          dtype='complex')

        for _ue in range(self.n_ue):
            for _bs in range(self.n_bs):
                weight[:, :, _ue, _bs] = np.linalg.inv(errm[:, :, _ue, _bs])

        return utils.weighted_bisection(chan, recv, weight, noise_pwr, pwr_lim)


class PrecoderSDP(Precoder):
    """ Joint transceiver beamformer design based on SDP reformulation and
        successive linear approximation of the original problem. """

    def blkdiag(self, m_array):
        """ Block diagonalize [N1 N2 Y X] as X diagonal N1*Y-by-N2*Y blocks  """

        sizes = m_array.shape

        blksize = (sizes[0]*sizes[2], sizes[1]*sizes[2])

        ret_array = np.zeros((blksize[0], blksize[1], sizes[3]),
                             dtype='complex')

        for _bs in range(sizes[3]):
            blocks = [m_array[:, :, _ue, _bs] for _ue in range(sizes[2])]

            ret_array[:, :, _bs] = scipy.linalg.block_diag(*blocks)

        return ret_array

    def generate(self, *args, **kwargs):
        """ Generate the precoders. """

        [chan, recv, prec_prev, noise_pwr] = [_a for _a in args]

        pwr_lim = kwargs.get('pwr_lim', 1)

        # MSE and weights
        errm = utils.mse(chan, recv, prec_prev, noise_pwr)

        weights = np.zeros((self.n_sk, self.n_sk, self.n_ue, self.n_bs),
                           dtype='complex')

        for (_ue, _bs) in itertools.product(range(self.n_ue), range(self.n_bs)):
                weights[:, :, _ue, _bs] = np.linalg.inv(errm[:, :, _ue, _bs])

        # The final precoders
        prec = np.zeros((self.n_tx, self.n_sk, self.n_ue, self.n_bs),
                        dtype='complex')

        # Block diagonalize matrices
        recv = self.blkdiag(recv)
        weights = self.blkdiag(weights)

        # Composite channel
        chan = chan.transpose(1, 0, 2, 3) # [Nt Nr K B]
        chan = np.reshape(chan, [self.n_tx, self.n_rx*self.n_ue, self.n_bs])
        chan = chan.transpose(0, 1, 2) # [Nr*K Nt B]

        for _bs in range(self.n_bs):
            wsqrt = np.linalg.cholesky(weights[:, :, _bs])

            bounds = np.array([0.0, 10.0])

            while np.abs(bounds[0] - bounds[1]) > 1e-6:
                lvl = bounds.sum() / 2

                cov = np.dot(np.dot(chan[:, :, _bs], (1/lvl)*np.eye(self.n_tx)),
                            chan[:, :, _bs].conj().T)

                #######


                prob = pic.ProgressBar()

                recv = prob.add_variable('U', (self.n_rx*self.n_ue,
                                               self.n_sk*self.n_ue), 'complex')

                S = prob.add_variable('S', (self.n_sk*self.ue,
                                            self.n_sk*self.ue), 'hermitian')

                X = prob.add_variable('X', (self.n_sk*self.ue,
                                            self.n_sk*self.ue), 'hermitian')

                prob.add_constraint(X >> 0)
                prob.add_constraint(S >> 0)


        return prec
