""" This model defines different types of gain models to be used with the
    simulator and chanmod modules. """

import numpy as np

from itertools import product

import logging


class GainModel(object):
    """Docstring for ChannelModel. """

    def __init__(self, K, B, **kwargs):
        """ Constructor for the channel models."""

        self.K = K
        self.B = B

        self.logger = logging.getLogger(__name__)


class Uniform1(object):

    """Uniformly distributed users (pairs) in single-cell systesm. """

    def __init__(self, K, B, **kwargs):
        self.K = K
        self.B = B

        self.radius = kwargs.get('radius', 100.)
        r_ref = kwargs.get('r_ref', 1.0)

        # Path loss exponents
        # path_loss_exp_ue = kwargs.get('path_loss_exp_ue', 3.0)
        path_loss_exp_bs = kwargs.get('path_loss_exp_bs', 2.5)
        path_loss_exp_d2d = kwargs.get('path_loss_exp_d2d', 3.0)

        # Distance of the D2D pairs
        self.d2d_dist = kwargs.get('d2d_dist', 20)

        SNR_edge = kwargs.get('SNR', 20.)

        angles = np.r_[0:K] * 2*np.pi/K

        # Mid-points
        # locs = self.radius * (np.cos(angles) + 1j*np.sin(angles))
        angl = np.arcsin((self.d2d_dist / 2) / float(self.radius))

        coord_recv = self.radius * (np.cos(np.mod(angles+angl, 2*np.pi)) +
                                    1j*np.sin(np.mod(angles+angl, 2*np.pi)))
        coord_tran = self.radius * (np.cos(np.mod(angles-angl, 2*np.pi)) +
                                    1j*np.sin(np.mod(angles-angl, 2*np.pi)))

        # dist_BS = [np.abs(locs[k]) for k in range(K)]

        # Transmitter distance from receivers j from i
        dist_UE = np.zeros((K, K))

        for (i, j) in product(range(K), range(K)):
            dist_UE[i, j] = np.abs(coord_recv[i] - coord_tran[j])

        dist_UE[dist_UE == 0] = 1e-3

        # Gains
        self.gains = {}
        self.gains['D2D'] = np.zeros((K, K))

        for (i, j) in product(range(K), range(K)):
            self.gains['D2D'][i, j] = (r_ref/dist_UE[i, j])**path_loss_exp_d2d

        self.gains['B2D'] = np.ones((K, 1))
        self.gains['D2B'] = np.ones((K, 1))
        for i in range(K):
            self.gains['B2D'][i] = (r_ref/self.radius)**path_loss_exp_bs

        self.gains['D2B'] = self.gains['B2D']

        # Cell Edge SNR
        self.SNR_ue_dB = SNR_edge + \
            path_loss_exp_d2d * 10 * np.log10(self.d2d_dist)

        SNR_lin = 10**(self.SNR_ue_dB/10) * K

        self.SNR_dB = 10*np.log10(SNR_lin)
