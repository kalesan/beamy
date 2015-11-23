""" This is the top-level simulator module. The module consists of the the
Simulator class that is used to run the simulations for given environment and
precoder design. """

import logging

import numpy as np
import pandas as pd

import utils
import chanmod
import precoder

from precoder import PrecoderGaussian


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

        self.pwr_lim = 1.
        #self.pwr_lim = 10**(float(kwargs.get('SNR', 20))/10)
        self.noise_pwr = 10**(-float(kwargs.get('SNR', 20))/10)
        #self.noise_pwr = 1.0

        self.static_channel = kwargs.get('static_channel', True)
        self.rate_conv_tol = kwargs.get('rate_conv_tol', 1e-5)

        if prec is None:
            self.prec = PrecoderGaussian.PrecoderGaussian(self.sysparams)
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

        rprec = PrecoderGaussian.PrecoderGaussian(self.sysparams)

        prec = rprec.generate(pwr_lim=self.pwr_lim)

        rate = []

        rate_prev = np.Inf

        ind = 0
        err = np.Inf;

        while np.abs(err) > self.rate_conv_tol:
            chan = self.chanmod.generate()

            if ind == 0:
                recv = utils.lmmse(chan, prec, self.noise_pwr, cov=None)

            for txrxi in range(0, self.txrxiter):
                prec = self.prec.generate(chan, recv, prec, self.noise_pwr,
                                        pwr_lim=self.pwr_lim)

                cov = utils.sigcov(chan, prec, self.noise_pwr)

                recv = utils.lmmse(chan, prec, self.noise_pwr, cov=cov)

            errm = utils.mse(chan, recv, prec, self.noise_pwr, cov=cov)

            rate += [(utils.rate(chan, prec, self.noise_pwr,
                                    errm=errm)[:]).sum() / n_bs]

            pwr = np.real(prec[:]*prec[:].conj()).sum()

            logger.info("%d: R: %.4f P: %.2f%% (I: %.5g) ", ind+1, rate[-1],
                        100*(pwr/(n_bs*self.pwr_lim)), rate[-1] - rate_prev)

            ind += 1

            err = rate[-1] - rate_prev

            rate_prev = rate[-1]

        return rate

    def iterate_beamformers(self):
        """ Iteratively generate the receive and transmit beaformers for the
        given channel matrix. """

        self.prec.reset()

        logger = logging.getLogger(__name__)

        (n_rx, n_tx, n_ue, n_bs) = self.sysparams

        n_sk = min(n_rx, n_tx)

        # Initialize beamformers
        rprec = PrecoderGaussian.PrecoderGaussian(self.sysparams)

        prec = rprec.generate(pwr_lim=self.pwr_lim)

        recv = np.random.randn(n_rx, n_sk, n_ue, n_bs) + \
                np.random.randn(n_rx, n_sk, n_ue, n_bs)*1j

        rate = []

        rate_prev = np.inf

        for ind in range(0, self.iterations['beamformer']):

            chan = self.chanmod.generate()

            for txrxi in range(0, self.txrxiter):
                prec = self.prec.generate(chan, recv, prec, self.noise_pwr,
                                        pwr_lim=self.pwr_lim)

                cov = utils.sigcov(chan, prec, self.noise_pwr)

                recv = utils.lmmse(chan, prec, self.noise_pwr, cov=cov)

            errm = utils.mse(chan, recv, prec, self.noise_pwr, cov=cov)

            rate += [(utils.rate(chan, prec, self.noise_pwr,
                                    errm=errm)[:]).sum() / n_bs]

            pwr = np.real(prec[:]*prec[:].conj()).sum()

            logger.info("%d/%d: R: %.4f P: %.2f%% (I: %.5g) ", ind+1, 
                        self.iterations['beamformer'], rate[-1],
                        100*(pwr/(n_bs*self.pwr_lim)), rate[-1] - rate_prev)

            # Settle on convergence for static channels
            if np.abs(rate_prev - rate[-1]) < self.rate_conv_tol:

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

            stat_t = pd.DataFrame({'rate': rate, 
                        'iterations': float(len(rate))})

            if stats is None:
                stats = stat_t
            else:
                r1 = stats['rate'].iloc[-1]
                r2 = stat_t['rate'].iloc[-1]

                stats += stat_t

                # Concatenate different lenght rate vectors
                stats['rate'][stats['rate'].isnull()] = r1 + r2 

            np.savez(self.resfile, R=stats['rate']/(rel+1), 
                        I=stats['iterations'].iloc[0]/(rel+1))
