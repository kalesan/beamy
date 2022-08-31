""" This is the top-level simulator module. The module consists of the the
Simulator class that is used to run the simulations for given environment and
precoder design. """

import logging

import numpy as np
import pandas as pd
from datetime import datetime

from beamy import chanmod
from beamy import utils
from beamy import precoder


class Simulator(object):
    """ This is the top-level simulator class, which is used to define the
    simulation environments and run the simulations. """

    def __init__(self, prec, **kwargs):
        """
        Arguments:
            prec (Precoder): Precoding model

        Keywoard Args:
            sysparams (tupple): number of (Rx, Tx, UE, BS) (Default: (2, 4, 10, 1))
            channel_model (ChannelModel): Channel model (Default: ClarkesModel)
            realizations (int): Number of channel realizations (Default: 20)
            biterations (int): Maximum number of beamformer iterations (Default: 50)
            txrxiter (int): Number of TX/RX iterations per beamformer iteration (Default: 1)
            seed (int): Random number generator seed (Default: 1841)
            resfile (str): Result file path (Default: res.npz)
            SNR (float): Signal-to-noise ratio (in dB) (Default: 20)
            static_channel (bool): Does the channel remaing static duration beamformer iteration (default: True)
            uplink (bool): Simulate uplink (default: False)
            rate_type (str): How to present achievable rate (default: "average-per-cell"). Supported types
                             "average-per-cell", "average-per-user", "sum-rate"
        """
        self.sysparams = kwargs.get('sysparams', (2, 4, 10, 1))

        self.chanmod = kwargs.get('channel_model',
                                  chanmod.ClarkesModel(self.sysparams))

        self.iterations = {'channel': kwargs.get('realizations', 20),
                           'beamformer': kwargs.get('biterations', 50)}

        self.txrxiter = kwargs.get('txrxiter', 1)

        self.seed = kwargs.get('seed', 1841)

        self.resfile = kwargs.get('resfile', 'res.npz')

        self.SNR = kwargs.get('SNR', 20)

        self.pwr_lim = 1
        self.noise_pwr = 10**-(self.SNR/10)

        self.static_channel = kwargs.get('static_channel', True)
        self.rate_conv_tol = 1e-6

        self.uplink = kwargs.get('uplink', False)

        self.rate_type = kwargs.get('rate_type', "average-per-cell")

        if prec is None:
            self.prec = precoder.PrecoderGaussian()
        else:
            self.prec = prec

        self.write_info_csv()
        

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
                                    errm=errm)[:]).sum()

            if self.rate_type == "average-per-cell":
                rate[ind] /= chan.shape[3]
            elif self.rate_type == "average-per-user":
                rate[ind] /= chan.shape[2]
            elif self.rate_type == "sum-rate":
                pass
            else:
                print("Unsupported rate type. Defaultin to sum-rate")

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
        rprec = precoder.PrecoderGaussian()
        rprec.init((n_rx, n_tx, n_ue, n_bs))

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

                    prec[:, :, :, :, ind:] = iprec.transpose(1, 2, 3, 4, 0)
                    recv[:, :, :, :, ind:] = irecv.transpose(1, 2, 3, 4, 0)
                else:
                    prec[:, :, :, :, ind:] = np.tile(iprec, (itr))
                    recv[:, :, :, :, ind:] = np.tile(irecv, (itr))

                break

            prec[:, :, :, :, ind] = iprec
            recv[:, :, :, :, ind] = irecv

            rate_prev = rate[ind]

        return {'precoder': prec, 'receiver': recv}

    def write_info_csv(self):
        df = pd.DataFrame(data={
            'time': datetime.now().strftime('%c'),
            'B': self.sysparams[0],
            'K': self.sysparams[1],
            'Nr': self.sysparams[2],
            'Nt': self.sysparams[3],
            'realizations': self.iterations['channel'],
            'brealizations': self.iterations['beamformer'],
            'SNR': self.SNR,
            'uplink': self.uplink,
            'static_channel': self.static_channel
        }, index=[0])
        df.to_csv('info.csv')

    def write_csv(self, rate, mse):
        df = pd.DataFrame(data={'Rate': rate, 'MSE': mse})
        df.index.name = 'Iteration'
        df.to_csv('iteration.csv')

    def run(self):
        """ Run the simulator setup. """
        self.prec.init(self.sysparams, self.uplink)

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

        self.write_csv(stats['rate']/self.iterations['channel'], 
                       stats['mse']/self.iterations['channel'])
