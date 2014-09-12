""" This is the top-level simulator module. The module consists of the the
Simulator class that is used to run the simulations for given environment and
precoder design. """

import logging

import numpy as np

import utils
import chanmod
import precoder


class Simulator(object):
    """ This is the top-level simulator class, which is used to define the
    simulation environments and run the simulations. """

    def __init__(self, prec, **kwargs):
        self.chanmod = kwargs.get('channel_model', chanmod.GaussianModel())
        self.sysparams = kwargs.get('sysparams', (2, 4, 10, 3))

        self.iterations = {'channel': kwargs.get('realizations', 20),
                           'beamformer': kwargs.get('biterations', 50)}

        self.seed = kwargs.get('seed', 1841)

        self.resfile = kwargs.get('resfile', 'res.npz')

        self.pwr_lim = 1
        self.noise_pwr = 10**-(kwargs.get('SNR', 20)/10)

        if prec is None:
            self.prec = precoder.PrecoderGaussian(self.sysparams)
        else:
            self.prec = prec

    def iterate_beamformers(self, chan):
        """ Iteratively generate the receive and transmit beaformers for the
        given channel matrix. """

        logger = logging.getLogger(__name__)

        n_bs = chan.shape[3]

        # Initialize beamformers
        rprec = precoder.PrecoderGaussian(self.sysparams)
        prec = rprec.generate(pwr_lim=self.pwr_lim)

        recv = utils.lmmse(chan, prec, self.noise_pwr)

        rate = np.zeros((self.iterations['beamformer'], 1))

        for ind in range(self.iterations['beamformer']):
            logger.info("Iteration %d/%d", ind+1, self.iterations['beamformer'])

            pwr = np.real(prec[:]*prec[:].conj()).sum()

            prec = self.prec.generate(chan, recv, prec, self.noise_pwr,
                                      pwr_lim=self.pwr_lim)

            cov = utils.sigcov(chan, prec, self.noise_pwr)

            recv = utils.lmmse(chan, prec, self.noise_pwr, cov=cov)

            errm = utils.mse(chan, recv, prec, self.noise_pwr, cov=cov)

            rate[ind] = (utils.rate(chan, prec, self.noise_pwr,
                                    errm=errm)[:]).sum() / n_bs

            logger.info("Rate: %.2f Power: %.2f%%", rate[ind][0],
                        100*(pwr/(n_bs*self.pwr_lim)))

        return rate

    def run(self):
        """ Run the simulator setup. """

        logger = logging.getLogger(__name__)

        # Initialize the random number generator
        np.random.seed(self.seed)

        rate = np.zeros((self.iterations['beamformer'], 1))

        for rel in range(self.iterations['channel']):
            logger.info("Realization %d/%d", rel+1,
                        self.iterations['channel'])

            chan = self.chanmod.generate(self.sysparams)

            rate += self.iterate_beamformers(chan)

            np.savez(self.resfile, R=rate/(rel+1))

if __name__ == "__main__":
    SPARAMS = (2, 8, 4, 1)

    SIM = Simulator(precoder.PrecoderSDP(SPARAMS), sysparams=SPARAMS)

    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.DEBUG)

    HANDLER = logging.StreamHandler()
    HANDLER.setLevel(logging.DEBUG)

    FORMATTER = logging.Formatter("%(levelname)s - %(module)s - %(message)s")
    HANDLER.setFormatter(FORMATTER)

    while len(LOGGER.handlers) > 0:
        LOGGER.handlers.pop()

    LOGGER.addHandler(HANDLER)

    SIM.run()
