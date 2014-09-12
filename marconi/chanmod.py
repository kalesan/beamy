""" This model defines different types of channel models to be used with the
    simulator module. """

import numpy as np


class GaussianModel(object):
    """ This class defines a Gaussian channel generator. """

    # pylint: disable=R0903
    # pylint: disable=R0201
    def generate(self, sysparams):
        """ Generate a Gaussian channel with given system parameters """
        (n_rx, n_tx, n_ue, n_bs) = sysparams

        return (1/np.sqrt(2)) * (np.random.randn(n_rx, n_tx, n_ue, n_bs) +
                                 np.random.randn(n_rx, n_tx, n_ue, n_bs)*1j)
