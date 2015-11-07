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
        self.rate_conv_tol = 1e-4

        self.uplink = kwargs.get('uplink', False)

        if prec is None:
            self.prec = precoder.PrecoderGaussian(self.sysparams)
        else:
            self.prec = prec

    def iteration_stats(self, chan_full, recv, prec):
        """ Collect iteration statistics from the given precoders and receivers.

        Args:
            chan (complex array): Channel matrix.
            recv (complex array): Receive beamformers for each iteration.
            prec (complex array): Transmit beamformers for each iteration.

        Returns: DataFrame containing the iteration statistics.

        """

        rate = np.zeros((self.iterations['beamformer']))
        mse = np.zeros((self.iterations['beamformer']))

        for ind in range(0, self.iterations['beamformer']):
            chan = chan_full[:, :, :, :, ind]

            cov = utils.sigcov(chan, prec[:, :, :, :, ind], self.noise_pwr)

            errm = utils.mse(chan, recv[:, :, :, :, ind], prec[:, :, :, :, ind],
                             self.noise_pwr, cov=cov)

            errm_tmp = errm.reshape((errm.shape[0], errm.shape[1],
                                     errm.shape[2]*errm.shape[3]), order='F')

            for tmp_ind in range(errm_tmp.shape[2]):
                mse[ind] += np.real(errm_tmp[:, :, tmp_ind].diagonal().sum())

            rate[ind] = (utils.rate(chan, prec[:, :, :, :, ind], self.noise_pwr,
                                    errm=errm)[:]).sum() / chan.shape[3]

        return pd.DataFrame({'rate': rate, 'mse': mse})

    def iterate_beamformers(self, chan_full):
        """ Iteratively generate the receive and transmit beaformers for the
        given channel matrix. """

        self.prec.reset()

        logger = logging.getLogger(__name__)

        (n_rx, n_tx, n_ue, n_bs) = chan_full.shape[0:4]

        n_sk = min(n_rx, n_tx)

        prec = np.zeros((n_tx, n_sk, n_ue, n_bs, self.iterations['beamformer']),
                        dtype='complex')

        recv = np.zeros((n_rx, n_sk, n_ue, n_bs, self.iterations['beamformer']),
                        dtype='complex')

        # Initialize beamformers
        rprec = precoder.PrecoderGaussian((n_rx, n_tx, n_ue, n_bs))

        prec[:, :, :, :, 0] = rprec.generate(pwr_lim=self.pwr_lim)

        recv[:, :, :, :, 0] = utils.lmmse(chan_full[:, :, :, :, 0],
                                          prec[:, :, :, :, 0], self.noise_pwr)

        rate = np.zeros((self.iterations['beamformer']))

        iprec = prec[:, :, :, :, 0]
        irecv = recv[:, :, :, :, 0]

        rate_prev = np.inf

        for ind in range(1, self.iterations['beamformer']):
            chan = chan_full[:, :, :, :, ind]

            logger.info("Iteration %d/%d", ind, self.iterations['beamformer'])

            for txrxi in range(0, self.txrxiter):
                iprec = self.prec.generate(chan, irecv, iprec, self.noise_pwr,
                                        pwr_lim=self.pwr_lim)

                cov = utils.sigcov(chan, iprec, self.noise_pwr)

                irecv = utils.lmmse(chan, iprec, self.noise_pwr, cov=cov)

            errm = utils.mse(chan, irecv, iprec, self.noise_pwr, cov=cov)

            rate[ind] = (utils.rate(chan, iprec, self.noise_pwr,
                                    errm=errm)[:]).sum() / n_bs

            pwr = np.real(iprec[:]*iprec[:].conj()).sum()

            logger.info("Rate: %.4f Power: %.2f%% (I: %f) ", rate[ind],
                        100*(pwr/(n_bs*self.pwr_lim)), rate[ind] - rate_prev)

            # Settle on convergence for static channels
            if self.static_channel and \
               np.abs(rate_prev - rate[ind]) < self.rate_conv_tol:

                itr = self.iterations['beamformer'] - ind

                # Add necessary singleton dimensions when B = 1
                if n_bs == 1:
                    prec[:, :, :, :, ind:] = np.tile(iprec[:, :, :, None],
                                                     (itr))
                    recv[:, :, :, :, ind:] = np.tile(irecv[:, :, :, None],
                                                     (itr))
                elif n_ue == 1:
                    iprec = np.array([iprec for _x in range(itr)])
                    irecv = np.array([irecv for _x in range(itr)])

                    prec[:, :, :, :, ind:] = iprec.transpose(1,2,3,4,0)
                    recv[:, :, :, :, ind:] = irecv.transpose(1,2,3,4,0)
                else:
                    prec[:, :, :, :, ind:] = np.tile(iprec, (itr))
                    recv[:, :, :, :, ind:] = np.tile(irecv, (itr))

                break

            prec[:, :, :, :, ind] = iprec
            recv[:, :, :, :, ind] = irecv

            rate_prev = rate[ind]

        return {'precoder': prec, 'receiver': recv}

    def run(self):
        """ Run the simulator setup. """

        logger = logging.getLogger(__name__)

        # Initialize the random number generator
        np.random.seed(self.seed)

        stats = None

        for rel in range(self.iterations['channel']):
            logger.info("Realization %d/%d", rel+1, self.iterations['channel'])

            chan = self.chanmod.generate(self.iterations['beamformer'])

            # For uplink simulations transpose the channel model
            if self.uplink:
                chan = chan.transpose(1, 0, 3, 2, 4)

            beamformers = self.iterate_beamformers(chan)

            stat_t = self.iteration_stats(chan, beamformers['receiver'],
                                          beamformers['precoder'])

            if stats is None:
                stats = stat_t
            else:
                stats += stat_t

            np.savez(self.resfile, R=stats['rate']/(rel+1),
                     E=stats['mse']/(rel+1))
