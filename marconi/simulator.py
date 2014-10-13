""" This is the top-level simulator module. The module consists of the the
Simulator class that is used to run the simulations for given environment and
precoder design. """

import logging

import itertools

import numpy as np
import pandas as pd

import utils
import chanmod
import precoder

class Simulator(object):
    """ This is the top-level simulator class, which is used to define the
    simulation environments and run the simulations. """

    def __init__(self, sysmodel, prec, **kwargs):
        logger = logging.getLogger(__name__)

        self.sysmodel = sysmodel
        self.sysparams = sysmodel.sysparams

        self.iterations = {'channel': kwargs.get('realizations', 20),
                           'beamformer': kwargs.get('biterations', 50)}

        self.active_links = kwargs.get('active_links',
                                       {'BS': True, 'D2D': True})

        self.seed = kwargs.get('seed', 1841)

        self.resfile = kwargs.get('resfile', 'res.npz')

        self.pwr_lim = {}
        self.pwr_lim['BS'] = 10**(sysmodel.gainmod.SNR_dB/10)
        self.pwr_lim['UE'] = 10**(sysmodel.gainmod.SNR_ue_dB/10)

        logger.info("BS TxPwr: %f, UE TxPwr: %f",
                    10*np.log10(self.pwr_lim['BS']),
                    10*np.log10(self.pwr_lim['UE']))

        self.noise_pwr = {}
        self.noise_pwr['BS'] = 1
        self.noise_pwr['UE'] = 1

        self.static_channel = kwargs.get('static_channel', True)
        self.rate_conv_tol = 1e-3

        self.uplink = kwargs.get('uplink', False)

        if prec is None:
            self.prec = precoder.PrecoderGaussian(self.sysparams)
        else:
            self.prec = prec

    def stats(self, chan, prec, recv):
        """Return performance statistics for the given beamformers.

        Args:
            chan (@todo): @todo
            prec (@todo): @todo
            recv (@todo): @todo

        Returns: dictionary of the statistics

        """
        (n_dx, n_bx, K, B) = chan['B2D'].shape

        n_sk = min(n_dx, n_bx)

        cov = utils.sigcov(chan, prec, self.noise_pwr)

        recv = utils.lmmse(chan, prec, self.noise_pwr, cov=cov)

        errm = utils.mse(chan, recv, prec, self.noise_pwr, cov=cov)

        # Rate
        rates = utils.rate(chan, prec, self.noise_pwr, errm=errm)

        rate = rates['D2D'][0].sum() + rates['D2D'][1].sum()
        for (_ue, _bs) in itertools.product(range(K), range(B)):
            rate += min(rates['B2D'][_ue, _bs], rates['D2B'][_bs, _ue])

        streams_B2D = 0.
        streams_D2B = 0.
        streams_D2D_1 = 0.
        streams_D2D_2 = 0.

        threshold = 1e-2

        for _bs in range(B):
            streams_B2D = np.sum(rates['B2D'][:, _bs] > threshold)
            streams_D2B = np.sum(rates['D2B'][_bs, :] > threshold)

        for _ue in range(K):
            streams_D2D_1 = np.sum(rates['D2D'][0][:] > threshold)
            streams_D2D_2 = np.sum(rates['D2D'][1][:] > threshold)

        return {'rate': rate, 'stream_alloc': {'B2D': streams_B2D,
                                               'D2B': streams_D2B,
                                               'D2D_1': streams_D2D_1,
                                               'D2D_2': streams_D2D_2}}

    def iteration_stats(self, chan_all, recv, prec):
        """ Collect iteration statistics from the given precoders and receivers.

        Args:
            chan_all (dictionary): Channel arrays.
            recv (complex array): Receive beamformers for each iteration.
            prec (complex array): Transmit beamformers for each iteration.

        Returns: DataFrame containing the iteration statistics.

        """

        # TODO: Generate rates for all scenarios and sum them up

        rate = np.zeros((self.iterations['beamformer']))
        streams_B2D = np.zeros((self.iterations['beamformer']))
        streams_D2B = np.zeros((self.iterations['beamformer']))
        streams_D2D_1 = np.zeros((self.iterations['beamformer']))
        streams_D2D_2 = np.zeros((self.iterations['beamformer']))

        for ind in range(0, self.iterations['beamformer']):
            iprec = {}
            iprec['D2B'] = prec['D2B'][:, :, :, :, ind]
            iprec['B2D'] = prec['B2D'][:, :, :, :, ind]
            iprec['D2D'] = np.array([None, None])
            iprec['D2D'][0] = prec['D2D'][0][:, :, :, ind]
            iprec['D2D'][1] = prec['D2D'][1][:, :, :, ind]

            irecv = {}
            irecv['D2B'] = recv['D2B'][:, :, :, :, ind]
            irecv['B2D'] = recv['B2D'][:, :, :, :, ind]
            irecv['D2D'] = np.array([None, None])
            irecv['D2D'][0] = recv['D2D'][0][:, :, :, ind]
            irecv['D2D'][1] = recv['D2D'][1][:, :, :, ind]

            ichan = {}
            ichan['B2D'] = chan_all['B2D'][:, :, :, :, 0]
            ichan['D2B'] = chan_all['D2B'][:, :, :, :, 0]
            ichan['D2D'] = chan_all['D2D'][:, :, :, :, 0]

            stats = self.stats(ichan, iprec, irecv)

            rate[ind] = stats['rate']
            streams_B2D[ind] = stats['stream_alloc']['B2D']
            streams_D2B[ind] = stats['stream_alloc']['D2B']
            streams_D2D_1[ind] = stats['stream_alloc']['D2D_1']
            streams_D2D_2[ind] = stats['stream_alloc']['D2D_2']

        return pd.DataFrame({'rate': rate, 'S_B2D': streams_B2D,
                             'S_D2B': streams_D2B, 'S_D2D_1': streams_D2D_1,
                             'S_D2D_2': streams_D2D_2})

    def iterate_beamformers(self, chan_all):
        """ Iteratively generate the receive and transmit beaformers for the
        given channel matrix. """

        self.prec.reset()

        logger = logging.getLogger(__name__)

        (n_dx, n_bx, n_ue, n_bs) = chan_all['B2D'].shape[0:4]

        prec = {}
        recv = {}

        n_sk = min(n_dx, n_bx)

        iters = self.iterations['beamformer']

        # Device to BS
        prec['D2B'] = np.zeros((n_dx, n_sk, n_bs, n_ue, iters), dtype='complex')
        recv['D2B'] = np.zeros((n_bx, n_sk, n_bs, n_ue, iters), dtype='complex')

        # Device to Device
        params = (n_dx, n_dx, n_ue, iters)

        prec['D2D'] = []
        prec['D2D'].append(np.zeros(params, dtype='complex'))
        prec['D2D'].append(np.zeros(params, dtype='complex'))

        recv['D2D'] = []
        recv['D2D'].append(np.zeros(params, dtype='complex'))
        recv['D2D'].append(np.zeros(params, dtype='complex'))

        # BS to Device
        prec['B2D'] = np.zeros((n_bx, n_sk, n_ue, n_bs, iters), dtype='complex')
        recv['B2D'] = np.zeros((n_dx, n_sk, n_ue, n_bs, iters), dtype='complex')

        # Initialize beamformers
        rnd_prec = precoder.PrecoderGaussian((n_dx, n_bx, n_ue, n_bs))
        iprec = rnd_prec.generate(pwr_lim_bs=self.pwr_lim)

        if self.active_links['BS']:
            prec['D2B'][:, :, :, :, 0] = iprec['D2B']
            prec['B2D'][:, :, :, :, 0] = iprec['B2D']
        else:
            iprec['D2B'] = prec['D2B'][:, :, :, :, 0]
            iprec['B2D'] = prec['B2D'][:, :, :, :, 0]

        if self.active_links['D2D']:
            prec['D2D'][0][:, :, :, 0] = iprec['D2D'][0]
            prec['D2D'][1][:, :, :, 0] = iprec['D2D'][1]
        else:
            iprec['D2D'][0] = prec['D2D'][0][:, :, :, 0]
            iprec['D2D'][1] = prec['D2D'][1][:, :, :, 0]

        ichan = {}
        ichan['B2D'] = chan_all['B2D'][:, :, :, :, 0]
        ichan['D2B'] = chan_all['D2B'][:, :, :, :, 0]
        ichan['D2D'] = chan_all['D2D'][:, :, :, :, 0]
        irecv = utils.lmmse(ichan, iprec, self.noise_pwr)

        rate_prev = np.inf

        for ind in range(1, self.iterations['beamformer']):
            ichan = {}
            ichan['B2D'] = chan_all['B2D'][:, :, :, :, ind]
            ichan['D2B'] = chan_all['D2B'][:, :, :, :, ind]
            ichan['D2D'] = chan_all['D2D'][:, :, :, :, ind]

            logger.info("Iteration %d/%d", ind, self.iterations['beamformer'])

            iprec = self.prec.generate(ichan, irecv, iprec, self.noise_pwr,
                                       pwr_lim=self.pwr_lim)

            irecv = utils.lmmse(ichan, iprec, self.noise_pwr)

            rates = utils.rate(ichan, iprec, self.noise_pwr)

            stats = self.stats(ichan, iprec, irecv)

            logger.info("Rate: %.4f (I: %f) ", stats['rate'],
                        stats['rate'] - rate_prev)
            logger.info("r0: %.4f r1: %.4f r2: %.4f r3: %.4f",
                        rates['D2B'][:].sum(), rates['B2D'][:].sum(),
                        rates['D2D'][0][:].sum(), rates['D2D'][1][:].sum())

            prec['B2D'][:, :, :, :, ind] = iprec['B2D']
            prec['D2B'][:, :, :, :, ind] = iprec['D2B']
            prec['D2D'][0][:, :, :, ind] = iprec['D2D'][0]
            prec['D2D'][1][:, :, :, ind] = iprec['D2D'][1]

            recv['B2D'][:, :, :, :, ind] = irecv['B2D']
            recv['D2B'][:, :, :, :, ind] = irecv['D2B']
            recv['D2D'][0][:, :, :, ind] = irecv['D2D'][0]
            recv['D2D'][1][:, :, :, ind] = irecv['D2D'][1]

            if np.abs(stats['rate'] - rate_prev) < self.rate_conv_tol:
                for ind2 in range(ind, self.iterations['beamformer']):
                    prec['B2D'][:, :, :, :, ind2] = iprec['B2D']
                    prec['D2B'][:, :, :, :, ind2] = iprec['D2B']
                    prec['D2D'][0][:, :, :, ind2] = iprec['D2D'][0]
                    prec['D2D'][1][:, :, :, ind2] = iprec['D2D'][1]

                    recv['B2D'][:, :, :, :, ind2] = irecv['B2D']
                    recv['D2B'][:, :, :, :, ind2] = irecv['D2B']
                    recv['D2D'][0][:, :, :, ind2] = irecv['D2D'][0]
                    recv['D2D'][1][:, :, :, ind2] = irecv['D2D'][1]

                return {'precoder': prec, 'receiver': recv}

            rate_prev = stats['rate']

        return {'precoder': prec, 'receiver': recv}

    def run(self):
        """ Run the simulator setup. """

        logger = logging.getLogger(__name__)

        # Initialize the random number generator
        np.random.seed(self.seed)

        stats = None

        for rel in range(self.iterations['channel']):
            logger.info("Realization %d/%d", rel+1, self.iterations['channel'])

            iterations = self.iterations['beamformer']

            chan = self.sysmodel.chanmod.generate(self.sysparams,
                                                  iterations=iterations,
                                                  gainmod=self.sysmodel.gainmod)

            beamformers = self.iterate_beamformers(chan)

            stat_t = self.iteration_stats(chan, beamformers['receiver'],
                                          beamformers['precoder'])

            if stats is None:
                stats = stat_t
            else:
                stats += stat_t

            np.savez(self.resfile, R=stats['rate']/(rel+1))

    def run_iter(self,sysmod):
        (_rx, _tx, _K, _B) = sysmod.sysparams
        _SNR = sysmod.SNR

        logger = logging.getLogger(__name__)

        # Initialize the random number generator
        np.random.seed(self.seed)

        stats_cell = None
        stats_BS = None
        stats_D2D = None

        proper_rel_cell = 1
        proper_rel_BS  = 1
        proper_rel_D2D = 1

        for rel in range(self.iterations['channel']):
            logger.info("Realization %d/%d", rel+1, self.iterations['channel'])

            iterations = self.iterations['beamformer']
            chan = self.sysmodel.chanmod.generate(self.sysparams,
                                                  iterations=iterations,
                                                  gainmod=self.sysmodel.gainmod)

            # Cell
            resfile = "WMMSE-Cell-%d-%d-%d-%d-%d-%d-%d.npz" % \
                (_rx, _tx, _K, _B, _SNR, sysmod.gainmod.radius, sysmod.gainmod.d2d_dist)

            self.active_links = {'BS': True, 'D2D': True}

            beamformers = self.iterate_beamformers(chan)

            stat_t = self.iteration_stats(chan, beamformers['receiver'],
                                          beamformers['precoder'])

            if not np.any(np.isnan(stat_t['rate'])):
                if stats_cell is None:
                    stats_cell = stat_t
                else:
                    stats_cell += stat_t

                np.savez(resfile, R=stats_cell['rate']/proper_rel_cell,
                        S_B2D=stats_cell['S_B2D']/proper_rel_cell,
                        S_D2B=stats_cell['S_D2B']/proper_rel_cell,
                        S_D2D_2=stats_cell['S_D2D_1']/proper_rel_cell,
                        S_D2D_1=stats_cell['S_D2D_2']/proper_rel_cell)

                proper_rell_cell += 1

            # BS
            resfile = "WMMSE-BS-%d-%d-%d-%d-%d-%d-%d.npz" % \
                (_rx, _tx, _K, _B, _SNR, sysmod.gainmod.radius, sysmod.gainmod.d2d_dist)

            self.active_links = {'BS': True, 'D2D': False}

            beamformers = self.iterate_beamformers(chan)

            stat_t = self.iteration_stats(chan, beamformers['receiver'],
                                          beamformers['precoder'])

            if not np.any(np.isnan(stat_t['rate'])):
                if stats_BS is None:
                    stats_BS = stat_t
                else:
                    stats_BS += stat_t

                np.savez(resfile, R=stats_BS['rate']/proper_rel_BS,
                        S_B2D=stats_BS['S_B2D']/proper_rel_BS,
                        S_D2B=stats_BS['S_D2B']/proper_rel_BS,
                        S_D2D_2=stats_BS['S_D2D_1']/proper_rel_BS,
                        S_D2D_1=stats_BS['S_D2D_2']/proper_rel_BS)

                proper_rel_BS += 1

            # D2D
            resfile = "WMMSE-D2D-%d-%d-%d-%d-%d-%d-%d.npz" % \
                (_rx, _tx, _K, _B, _SNR, sysmod.gainmod.radius, sysmod.gainmod.d2d_dist)

            self.active_links = {'BS': False, 'D2D': True}

            beamformers = self.iterate_beamformers(chan)

            stat_t = self.iteration_stats(chan, beamformers['receiver'],
                                          beamformers['precoder'])

            if not np.any(np.isnan(stat_t['rate'])):
                if stats_D2D is None:
                    stats_D2D = stat_t
                else:
                    stats_D2D += stat_t

                np.savez(resfile, R=stats_D2D['rate']/proper_rel_D2D,
                        S_B2D=stats_D2D['S_B2D']/proper_rel_D2D,
                        S_D2B=stats_D2D['S_D2B']/proper_rel_D2D,
                        S_D2D_2=stats_D2D['S_D2D_1']/proper_rel_D2D,
                        S_D2D_1=stats_D2D['S_D2D_2']/proper_rel_D2D)

                proper_rel_D2D += 1

