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

        radius = kwargs.get('radius', 100)
        r_ref = kwargs.get('r_ref', 1.0)

        # Path loss exponents
        # path_loss_exp_ue = kwargs.get('path_loss_exp_ue', 3.0)
        path_loss_exp_bs = kwargs.get('path_loss_exp_bs', 2.5)
        path_loss_exp_d2d = kwargs.get('path_loss_exp_d2d', 3.0)

        SNR_edge = kwargs.get('SNR', 20)

        angles = np.r_[0:K] * 2*np.pi/K

        locs = radius * (np.cos(angles) + 1j*np.sin(angles))

        # dist_BS = [np.abs(locs[k]) for k in range(K)]

        dist_UE = [[np.abs(locs[i]-locs[j])
                    for i in range(K)] for j in range(K)]

        self.gains = {}
        self.gains['D2D'] = np.zeros((K, K))

        for (i, j) in product(range(K), range(K)):
            self.gains['D2D'][i, j] = (r_ref/dist_UE[i][j])**path_loss_exp_d2d

            if i == j:
                self.gains['D2D'][i, j] = 1

        self.gains['B2D'] = np.ones((K, 1))
        self.gains['D2B'] = np.ones((K, 1))
        for i in range(K):
            self.gains['B2D'][i] = (r_ref/radius)**path_loss_exp_bs

        self.gains['D2B'] = self.gains['B2D']

        SNR_dB = SNR_edge + path_loss_exp*10*log10(closeness*cellsep)

        # Cell Edge SNR

        self.SNR_dB = SNR_dB + path_loss_exp_bs*10*np.log10(radius)

        SNR_lin = 10**(self.SNR_dB/10) / K

        self.SNR_ue_dB = 10*np.log10(SNR_lin)

