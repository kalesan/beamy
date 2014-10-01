""" This model defines different types of channel models to be used with the
    simulator module. """

import numpy as np
import scipy.constants

import gainmod

import logging


class ChannelModel(object):
    """Docstring for ChannelModel. """

    def __init__(self, **kwargs):
        """ Constructor for the channel models.

        Kwargs:
            cellsep (double): Inter-cell separation [dB].
            intrasep (double): Intra-cell separation (BS-to-UE) [dB].
            termsep (double): Terminal separation (UE-to-UE) (pair-wise) [dB].

        """

        self.logger = logging.getLogger(__name__)


class GaussianModel(object):
    """ This class defines a Gaussian channel generator. """

    # pylint: disable=R0201
    def generate(self, sysparams, iterations=1):
        """ Generate a Gaussian channel with given system parameters """
        (Nrx, Ntx, K, B) = sysparams

        chan = np.random.randn(Nrx, Ntx, K, B, iterations) + \
            np.random.randn(Nrx, Ntx, K, B, iterations)*1j

        return (1/np.sqrt(2)) * chan


class ClarkesModel(ChannelModel):
    """Docstring for ClarkesModel. """

    def __init__(self, gainmod, **kwargs):
        """ Constructor for Clarke's channel model. All of the parameters are
        optional up to the defaults.

        Kwargs:
            speed_kmh (double): User terminal velocity [km/h] (default: 0).
            freq_sym_Hz (double): Symbol sampling frequency (default: 20e3).
            carrier_freq_Hz (double): Carrier frequency (default: 2e9).
            npath (int): Number of signal paths (default: 300).
        """
        super(ClarkesModel, self).__init__(**kwargs)

        speed_kmh = kwargs.get('speed_kmh', 0)
        freq_sym_Hz = kwargs.get('freq_sym_Hz', 20e3)
        carrier_freq_Hz = kwargs.get('carrier_freq_Hz', 2e9)
        npath = kwargs.get('npath', 300)

        self.gainmod = gainmod

        self.ts = 1 / freq_sym_Hz  # Sample rate
        self.vel = speed_kmh / 3.6  # Velocity [m/s]
        self.fd = carrier_freq_Hz * (self.vel / scipy.constants.c)

        self.npath = npath

        # Incident angles: avoid zero Doppler and exactly opposite Dopplers
        self.alpha = (np.r_[0:npath] / npath) * np.pi + np.pi / (4 * npath)
        self.alpha = self.alpha.reshape(npath, 1)

        # Angular Doppler frequencies
        self.phii = 2*np.pi * self.fd * np.cos(self.alpha)

    def genmat(self, sysparams, gains=None, iterations=1):
        """TODO: Docstring for genmat.

        Args:
            n_rx (int): Number of receive antennas.
            n_tx (int): Number of transmit antennas.
            gains (matrix): Channel gain matrix.
            iterations (int) Channel realizations.

        Returns: Channel array.

        """

        (n_rx, n_tx, K, B) = sysparams

        if gains is None:
            gains = np.zeros((K, B))

        gains = gains.reshape((K*B, 1), order='F')

        path_powers = np.kron(np.kron(1, gains), np.ones((n_rx*n_tx, 1)))

        # Timing vector
        tv = np.r_[0:iterations] * self.ts
        tv = tv.reshape(iterations, 1)

        # Subpath complex phase evolution (N x nsamples)
        theta_t = np.dot(self.phii, tv.T)

        paths = np.zeros((len(path_powers), iterations), dtype='complex')

        for pth in range(len(path_powers)):
            # add random complex init phases per subpath
            theta = theta_t + np.tile(2 * np.pi * np.random.rand(self.npath, 1),
                                      (1, iterations))

            paths[pth, :] = np.sqrt(path_powers[pth] / self.npath) * \
                np.sum(np.exp(1j*theta), 0)

        return paths.reshape((n_rx, n_tx, K, B, iterations), order='F')

    def generate(self, sysparams, **kwargs):
        """ Generate time-correlated Rayleigh fading channels.

        Args:
            sysparams (tuple): System parameters (RX, TX, K, B).

        Kwargs:
            iterations (int): Number of beamformer iterations.

        Returns: Array of channel matrices.

        """

        # TOOD: this should part of the initialization
        (n_dx, n_bx, K, B) = sysparams

        iterations = kwargs.get('iterations', 1)

        gains = np.zeros((K, B))

        gains = gainmod.unif_single_cell(K, 100)

        chan = {}

        # BS-UE channels
        self.logger.info("Generating channels")
        self.logger.info("* BS-UE")

        # gains = self.intrasep * np.ones((K, B))
        # gains = 10**(gains / 10)

        chan['B2D'] = self.genmat((n_dx, n_bx, K, B),
                                  gains=self.gainmod.gains['B2D'],
                                  iterations=iterations)

        self.logger.info("* UE-BS")
        chan['D2B'] = chan['B2D'].transpose(1, 0, 3, 2, 4)

        # BS-BS channels
        # gains = 0 * np.ones((B, B))
        # gains = 10**(gains / 10)

        self.logger.info("* BS-BS")
        chan['B2B'] = self.genmat((n_bx, n_bx, B, B), gains=np.array([1]),
                                  iterations=iterations)

        # UE-UE channels
        self.logger.info("* UE-UE")
        # gains = self.termsep * np.ones((K, K))
        # gains = 10**(gains / 10)
        # for k in range(K):
        #    gains[k, k] = 0

        chan['D2D'] = self.genmat((n_dx, n_dx, K, K), iterations=iterations,
                                  gains=self.gainmod.gains['D2D'])

        return chan
