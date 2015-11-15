""" This is the top-level simulator module. The module consists of the the
Simulator class that is used to run the simulations for given environment and
precoder design. """

import logging

import numpy as np
import pandas as pd

import utils
import chanmod
import precoder


class Simulator(object):
    """ This is the top-level simulator class, which is used to define the
    simulation environments and run the simulations. """

    def __init__(self, prec, **kwargs):
        self.sysparams = kwargs.get('sysparams', (2, 4, 10, 1))

        self.chanmod = kwargs.get('channel_model',
                                  chanmod.ClarkesModel(self.sysparams))

        self.iterations = {'channel': kwargs.get('realizations', 20),
                           'beamformer': kwargs.get('biterations', 50)}

        self.txrxiter = kwargs.get('txrxiter', 1)

        self.seed = kwargs.get('seed', 1841)

        self.resfile = kwargs.get('resfile', 'res.npz')

        self.pwr_lim = 1
        self.noise_pwr = 10**-(kwargs.get('SNR', 20)/10)

        self.static_channel = kwargs.get('static_channel', True)
        self.rate_conv_tol = 1e-6

        self.uplink = kwargs.get('uplink', False)

        if prec is None:
            self.prec = precoder.PrecoderGaussian(self.sysparams)
        else:
            self.prec = prec

    def iterate_beamformers_conv(self):
        """ Iteratively generate the receive and transmit beaformers for the
        given channel matrix until convergence. """

        self.prec.reset()

        logger = logging.getLogger(__name__)

        (n_rx, n_tx, n_ue, n_bs) = self.sysparams

        n_sk = min(n_rx, n_tx)

        # Initialize beamformers
        rprec = precoder.PrecoderGaussian((n_rx, n_tx, n_ue, n_bs))

        prec = rprec.generate(pwr_lim=self.pwr_lim)

        recv = np.random.randn(n_rx, n_sk, n_ue, n_bs) + \
                np.random.randn(n_rx, n_sk, n_ue, n_bs)*1j

        rate = []

        rate_prev = np.inf
        rate_cur = 0

        ind = 0

        while np.abs(rate_prev - rate_cur) > self.rate_conv_tol:
            chan = self.chanmod.generate()

            # For uplink simulations transpose the channel model
            if self.uplink:
                chan = chan.transpose(1, 0, 3, 2)

            logger.info("Iteration %d", ind)

            for txrxi in range(0, self.txrxiter):
                prec = self.prec.generate(chan, recv, prec, self.noise_pwr,
                                        pwr_lim=self.pwr_lim)

                cov = utils.sigcov(chan, prec, self.noise_pwr)

                recv = utils.lmmse(chan, prec, self.noise_pwr, cov=cov)

            errm = utils.mse(chan, recv, prec, self.noise_pwr, cov=cov)

            rate += [(utils.rate(chan, prec, self.noise_pwr,
                                    errm=errm)[:]).sum() / n_bs]

            pwr = np.real(prec[:]*prec[:].conj()).sum()

            logger.info("Rate: %.4f Power: %.2f%% (I: %f) ", rate[-1],
                        100*(pwr/(n_bs*self.pwr_lim)), rate[-1] - rate_prev)

            ind += 1

            if ind > 1:
                rate_prev = rate[-2]
                rate_cur = rate[-1]


        return rate

    def iterate_beamformers(self):
        """ Iteratively generate the receive and transmit beaformers for the
        given channel matrix. """

        self.prec.reset()

        logger = logging.getLogger(__name__)

        (n_rx, n_tx, n_ue, n_bs) = self.sysparams

        n_sk = min(n_rx, n_tx)

        # Initialize beamformers
        rprec = precoder.PrecoderGaussian((n_rx, n_tx, n_ue, n_bs))

        prec = rprec.generate(pwr_lim=self.pwr_lim)

        recv = np.random.randn(n_rx, n_sk, n_ue, n_bs) + \
                np.random.randn(n_rx, n_sk, n_ue, n_bs)*1j

        rate = []

        rate_prev = np.inf

        for ind in range(0, self.iterations['beamformer']):

            chan = self.chanmod.generate()

            # For uplink simulations transpose the channel model
            if self.uplink:
                chan = chan.transpose(1, 0, 3, 2)

            logger.info("Iteration %d/%d", ind, self.iterations['beamformer'])

            for txrxi in range(0, self.txrxiter):
                prec = self.prec.generate(chan, recv, prec, self.noise_pwr,
                                        pwr_lim=self.pwr_lim)

                cov = utils.sigcov(chan, prec, self.noise_pwr)

                recv = utils.lmmse(chan, prec, self.noise_pwr, cov=cov)

            errm = utils.mse(chan, recv, prec, self.noise_pwr, cov=cov)

            rate += [(utils.rate(chan, prec, self.noise_pwr,
                                    errm=errm)[:]).sum() / n_bs]

            pwr = np.real(prec[:]*prec[:].conj()).sum()

            logger.info("Rate: %.4f Power: %.2f%% (I: %f) ", rate[-1],
                        100*(pwr/(n_bs*self.pwr_lim)), rate[-1] - rate_prev)

            # Settle on convergence for static channels
            if self.static_channel and \
                np.abs(rate_prev - rate[-1]) < self.rate_conv_tol:

                itr = self.iterations['beamformer'] - ind
                while (itr > 0):
                    rate += [rate[-1]]
                    itr -= 1

                return rate

            rate_prev = rate[-1]

        return rate

    def run(self):
        """ Run the simulator setup. """

        logger = logging.getLogger(__name__)

        # Initialize the random number generator
        np.random.seed(self.seed)

        stats = None

        for rel in range(self.iterations['channel']):
            logger.info("Realization %d/%d", rel+1, self.iterations['channel'])

            self.chanmod.reset()

            if self.iterations['beamformer']:
                rate = self.iterate_beamformers()
            else:
                rate = self.iterate_beamformers_conv()

            print(len(rate))
            stat_t = pd.DataFrame({'rate': rate, 'iterations': float(len(rate))})

            if stats is None:
                stats = stat_t
            else:
                stats += stat_t

            np.savez(self.resfile, R=stats['rate']/(rel+1), I=stats['iterations'].iloc[-1]/(rel+1))
