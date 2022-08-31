""" This model defines different types of channel models to be used with the
    simulator module. """

import numpy as np
import scipy.constants


class ChannelModel(object):
    """Docstring for ChannelModel. """

    def __init__(self, **kwargs):
        """@todo: to be defined1.

        Kwargs:
            gain_model (str): Channel gain model. Supported models are "CellSep"
                and "Wrap7".
            cellsep (double): Cell separation in dB.

        """

        self.gain_model = kwargs.get('gain_model', 'CellSep')
        self.cellsep = kwargs.get('cellsep', 3)


class GaussianModel(ChannelModel):
    """ This class defines a Gaussian channel generator. """

    def __init__(self, **kwargs):
        super(GaussianModel, self).__init__(**kwargs)

    # pylint: disable=R0201
    def generate(self, Nr, Nt, K, B, iterations=1):
        """ Generate a Gaussian channel with given system parameters """
        chan = np.random.randn(Nr, Nt, K, B, iterations) + \
            np.random.randn(Nr, Nt, K, B, iterations)*1j

        return (1/np.sqrt(2)) * chan


class ClarkesModel(ChannelModel):
    """Docstring for ClarkesModel. """

    def __init__(self, **kwargs):
        """ Constructor for Clarke's channel model. All of the parameters are
        optional up to the defaults.

        Kwargs:
            speed_kmh (@todo): @todo
            freq_sym_Hz (@todo): @todo
            carrier_freq_Hz (@todo): @todo
            npath (@todo): @todo

        Returns: @todo

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

    def generate(self, Nr, Nt, K, B, iterations=1):
        """ Generate time-correlated Rayleigh fading channels.

        Args:
            Nr (int): Number of receive anntennas
            Nt (int): Number of receive anntennas
            K (int): Number of users
            B (int): Number of base stations (cells)

        Kwargs:
            iterations (int): Number of beamformer iterations.

        Returns: @todo

        """

        # Timing vector
        tv = np.r_[0:iterations] * self.ts
        tv = tv.reshape(iterations, 1)

        # Subpath complex phase evolution (N x nsamples)
        theta_t = np.dot(self.phii, tv.T)

        # Channel gains
        self.gains = np.ones((Nr * Nt * K * B, 1))

        paths = np.zeros((len(self.gains), iterations), dtype='complex')

        for pth in range(len(self.gains)):
            # add random complex init phases per subpath
            theta = theta_t + np.tile(2 * np.pi * np.random.rand(self.npath, 1),
                                      (1, iterations))

            paths[pth, :] = np.sqrt(self.gains[pth] / self.npath) * \
                np.sum(np.exp(1j*theta), 0)

        return paths.reshape(Nr, Nt, K, B, iterations, order='F')
