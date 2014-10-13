""" This model defines different types of channel models to be used with the
    simulator module. """

import numpy as np
import scipy.constants
from scipy.io import loadmat

import itertools

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
    def generate(self, sysparams, gainmod, iterations=1):
        """ Generate a Gaussian channel with given system parameters """
        (Nrx, Ntx, K, B) = sysparams

        chan = np.random.randn(Nrx, Ntx, K, B, iterations) + \
            np.random.randn(Nrx, Ntx, K, B, iterations)*1j

        return (1/np.sqrt(2)) * chan


class PreModel(ChannelModel):
    """Docstring for ClarkesModel. """

    def __init__(self, chanfile):
        super(PreModel, self).__init__()

        self.chanfile = chanfile

        self.realization = 0

    def genmat(self, chan, gains=None, iterations=1):
        """TODO: Docstring for genmat.

        Args:
            chan (matrix): Channel base matrix.
            gains (matrix): Channel gain matrix.
            iterations (int) Channel realizations.

        Returns: Channel array.

        """

        (n_rx, n_tx, K, B) = chan.shape

        ret = np.zeros((n_rx, n_tx, K, B, iterations), dtype='complex')

        # TODO: Fix default gains
        if gains is None:
            gains = np.ones((K, B))

        for k in range(K):
            for b in range(B):
                for it in range(iterations):
                    ret[:, :, k, b, it] = np.sqrt(gains[k, b]) * \
                        chan[:, :, k, b]

        return ret

    def generate(self, sysparams, gainmod, **kwargs):
        """ Generate time-correlated Rayleigh fading channels.

        Args:
            sysparams (tuple): System parameters (RX, TX, K, B).

        Kwargs:
            iterations (int): Number of beamformer iterations.

        Returns: Array of channel matrices.

        """

        # TOOD: this should part of the initialization
        (n_dx, n_bx, K, B) = sysparams

        gains = gainmod.gains

        iterations = kwargs.get('iterations', 1)

        chan = {}

        # BS-UE channels
        self.logger.info("Generating channels")
        self.logger.info("* BS-UE")

        # gains = self.intrasep * np.ones((K, B))
        # gains = 10**(gains / 10)

        data = loadmat(self.chanfile)

        maxrel = data['H_b2d'].shape[-1]
        rel = self.realization

        chan['B2D'] = self.genmat(data['H_b2d'][:, :, :, :, 0, rel],
                                  gains=gains['B2D'],
                                  iterations=iterations)

        self.logger.info("* UE-BS")

        chan['D2B'] = self.genmat(data['H_d2b'][:, :, :, :, 0, rel],
                                  gains=gains['D2B'].transpose(),
                                  iterations=iterations)

        # BS-BS channels
        # gains = 0 * np.ones((B, B))
        # gains = 10**(gains / 10)

        self.logger.info("* BS-BS")
        chan['B2B'] = None

        # UE-UE channels
        self.logger.info("* UE-UE")
        # gains = self.termsep * np.ones((K, K))
        # gains = 10**(gains / 10)
        # for k in range(K):
        #    gains[k, k] = 0

        chan['D2D'] = self.genmat(data['H_d2d'][:, :, :, :, 0, rel],
                                  gains=gains['D2D'],
                                  iterations=iterations)

        self.realization = np.mod(rel + 1, maxrel)

        return chan


class RicianModel(ChannelModel):
    """Docstring for ClarkesModel. """

    def __init__(self, **kwargs):
        """ Constructor for Clarke's channel model. All of the parameters are
        optional up to the defaults.

        Kwargs:
            speed_kmh (double): User terminal velocity [km/h] (default: 0).
            freq_sym_Hz (double): Symbol sampling frequency (default: 20e3).
            carrier_freq_Hz (double): Carrier frequency (default: 2e9).
            npath (int): Number of signal paths (default: 300).
        """
        super(RicianModel, self).__init__(**kwargs)

        self.K_factor  = kwargs.get('K_factor', 10)
        speed_kmh = kwargs.get('speed_kmh', 0)

        freq_sym_Hz = kwargs.get('freq_sym_Hz', 20e3)
        carrier_freq_Hz = kwargs.get('carrier_freq_Hz', 2e9)
        npath = kwargs.get('npath', 300)

        self.ts = 1 / freq_sym_Hz  # Sample rate
        self.vel = speed_kmh / 3.6  # Velocity [m/s]
        self.fd = carrier_freq_Hz * (self.vel / scipy.constants.c)

        self.npath = npath

        # Incident angles: avoid zero Doppler and exactly opposite Dopplers
        self.alpha = (np.r_[0:npath] / npath) * np.pi + np.pi / (4 * npath)
        self.alpha = self.alpha.reshape(npath, 1)

        # Angular Doppler frequencies
        self.phii = 2*np.pi * self.fd * np.cos(self.alpha)

    def genmat(self, sysparams, gains, angles, iterations=1):
        """TODO: Docstring for genmat.

        Args:
            n_rx (int): Number of receive antennas.
            n_tx (int): Number of transmit antennas.
            gains (matrix): Channel gain matrix.
            iterations (int) Channel realizations.

        Returns: Channel array.

        """

        (n_rx, n_tx, K, B) = sysparams

        gains0 = gains.reshape((K*B, 1), order='F')

        path_powers = np.kron(np.kron(1, gains0), np.ones((n_rx*n_tx, 1)))

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

        # The rayleight fading component
        H = paths.reshape((n_rx, n_tx, K, B, iterations), order='F')

        ret = np.zeros((n_rx, n_tx, K, B, iterations), dtype='complex')

        if B == 1:
            angles = angles.reshape((K, 1), order='F')
        if K == 1:
            angles = angles.reshape((1, B), order='F')

        for (k, b) in itertools.product(range(K), range(B)):
            los_comp = np.zeros((1, n_tx), dtype='complex')

            for t in range(n_tx):
                los_comp[0, t] = np.exp(-1j*2*np.pi*t*1*angles[k, b])


            los_comp = los_comp/np.sqrt(n_tx)
            los_comp = np.tile(los_comp, (n_rx, 1))

            for itr in range(iterations):
                Kf = float(self.K_factor)

                ret[:, :, k, b, itr] = np.sqrt(gains[k, b]) * (Kf / (Kf + 1)) * los_comp
                ret[:, :, k, b, itr] += (1 / (Kf + 1)) * H[:, :, k, b, itr]

        return ret

    def generate(self, sysparams, gainmod, **kwargs):
        """ Generate time-correlated Rayleigh fading channels.

        Args:
            sysparams (tuple): System parameters (RX, TX, K, B).
            gainmod: Gain model.

        Kwargs:
            iterations (int): Number of beamformer iterations.

        Returns: Array of channel matrices.

        """

        # TOOD: this should part of the initialization
        (n_dx, n_bx, K, B) = sysparams

        gains = gainmod.gains

        iterations = kwargs.get('iterations', 1)

        chan = {}

        # BS-UE channels
        self.logger.info("Generating channels")
        self.logger.info("* BS-UE")

        # gains = self.intrasep * np.ones((K, B))
        # gains = 10**(gains / 10)

        chan['B2D'] = self.genmat((n_dx, n_bx, K, B),
                                  gains=gains['B2D'],
                                  iterations=iterations,
                                  angles=gainmod.angles['B2D'])

        self.logger.info("* UE-BS")
        #chan['D2B'] = self.genmat((n_bx, n_dx, B, K),
                                  #gains=gains['B2D'].transpose(),
                                  #iterations=iterations,
                                  #angles=gainmod.angles['B2D'].transpose())

        chan['D2B'] = chan['B2D'].transpose(1, 0, 3, 2, 4)

        # BS-BS channels
        # gains = 0 * np.ones((B, B))
        # gains = 10**(gains / 10)

        self.logger.info("* BS-BS")
        chan['B2B'] = self.genmat((n_bx, n_bx, B, B), gains=np.array([[1]]),
                                  iterations=iterations, angles=np.array([[0]]))

        # UE-UE channels
        self.logger.info("* UE-UE")
        # gains = self.termsep * np.ones((K, K))
        # gains = 10**(gains / 10)
        # for k in range(K):
        #    gains[k, k] = 0

        chan['D2D'] = self.genmat((n_dx, n_dx, K, K), iterations=iterations,
                                  gains=gains['D2D'],
                                  angles=gainmod.angles['D2D'])

        return chan

class ClarkesModel(ChannelModel):
    """Docstring for ClarkesModel. """

    def __init__(self, **kwargs):
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

        # TODO: Fix default gains
        if gains is None:
            gains = np.ones((K, B))

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

    def generate(self, sysparams, gainmod, **kwargs):
        """ Generate time-correlated Rayleigh fading channels.

        Args:
            sysparams (tuple): System parameters (RX, TX, K, B).
            gainmod (gainmod): Gain model.

        Kwargs:
            iterations (int): Number of beamformer iterations.

        Returns: Array of channel matrices.

        """

        # TOOD: this should part of the initialization
        (n_dx, n_bx, K, B) = sysparams

        gains = gainmod.gains

        iterations = kwargs.get('iterations', 1)

        chan = {}

        # BS-UE channels
        self.logger.info("Generating channels")
        self.logger.info("* BS-UE")

        # gains = self.intrasep * np.ones((K, B))
        # gains = 10**(gains / 10)

        chan['B2D'] = self.genmat((n_dx, n_bx, K, B),
                                  gains=gains['B2D'],
                                  iterations=iterations)

        self.logger.info("* UE-BS")
        chan['D2B'] = self.genmat((n_bx, n_dx, K, K),
                                  gains=gains['D2D'],
                                  iterations=iterations)

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
                                  gains=gains['D2D'])

        return chan
