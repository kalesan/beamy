""" This module gives plotting utilities to display the results. """

import numpy as np
import pylab as plt


def plotfile(filename):
    """ Plot a resource file contents. """

    data = np.load(filename)

    niters = data['R'].shape[0]

    plt.plot(range(niters), data['R'])

    plt.show()
