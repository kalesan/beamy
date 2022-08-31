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
            bs (int): Number of base stations (cells) (Default: 1)
            users (int): Number of users (Default: 10)
            nr (int): Number of receive anntennas (Default: 4)
            nt (int): Number of receive anntennas (Default: 4)
            name: (str): Simulation name (Default: Simulation A) 
            channel_model (ChannelModel): Channel model (Default: ClarkesModel)
            realizations (int): Number of channel realizations (Default: 20)
            biterations (int): Maximum number of beamformer iterations (Default: 50)
            txrxiter (int): Number of TX/RX iterations per beamformer iteration (Default: 1)
            seed (int): Random number generator seed (Default: 1841)
            SNR (float): Signal-to-noise ratio (in dB) (Default: 20)
            static_channel (bool): Does the channel remaing static duration beamformer iteration (default: True)
            uplink (bool): Simulate uplink (default: False)
            rate_type (str): How to present achievable rate (default: "average-per-cell"). Supported types
                             "average-per-cell", "average-per-user", "sum-rate"
        """
        self.Nr = kwargs.get('nr', 2)
        self.Nt = kwargs.get('nt', 4)
        self.B = kwargs.get('bs', 1)
        self.K = kwargs.get('users', 10)

        self.name = kwargs.get('name', "Simulation A")

        self.chanmod = kwargs.get('channel_model',
                                  chanmod.ClarkesModel())

        self.iterations = {'channel': kwargs.get('realizations', 20),
                           'beamformer': kwargs.get('biterations', 50)}

        self.txrxiter = kwargs.get('txrxiter', 1)

        self.seed = kwargs.get('seed', 1841)

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
        rprec.init(n_rx, n_tx, n_ue, n_bs)

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
                    prec[:, :, :, :, ind:] = np.repeat(iprec[:, :, :, :, np.newaxis], itr, axis=4)
                    recv[:, :, :, :, ind:] = np.repeat(irecv[:, :, :, :, np.newaxis], itr, axis=4)

                break

            prec[:, :, :, :, ind] = iprec
            recv[:, :, :, :, ind] = irecv

            rate_prev = rate[ind]

        return {'precoder': prec, 'receiver': recv}


    def write_info_csv(self):
        df = pd.DataFrame(data={
            'time': datetime.now().strftime('%c'),
            'B': self.B, 'K': self.K, 'Nr': self.Nr, 'Nt': self.Nt,
            'realizations': self.iterations['channel'],
            'brealizations': self.iterations['beamformer'],
            'SNR': self.SNR,
            'uplink': self.uplink,
            'static_channel': self.static_channel
        }, index=[0])
        df.to_csv('info.csv')


    def write_iteration_csv(self, rate, mse):
        df = pd.DataFrame(data={'Rate': rate, 'MSE': mse, 'Name': self.name})
        df.index.name = 'Iteration'
        df.to_csv('iteration.csv')


    def write_result_csv(self, res):
        res.to_csv('result.csv')


    def run(self):
        """ Run the simulator setup. """
        logger = logging.getLogger(__name__)

        res = pd.DataFrame({})

        # Initialize the random number generator
        np.random.seed(self.seed)

        stats = None

        B = self.B
        K = self.K
        Nr = self.Nr
        Nt = self.Nt

        SNR = self.SNR

        self.prec.init(Nr, Nt, K, B, self.uplink)

        for rel in range(self.iterations['channel']):
            logger.info("Realization %d/%d", rel+1, self.iterations['channel'])

            chan = self.chanmod.generate(Nr, Nt, K, B, self.iterations['beamformer'])

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

            iter_res = pd.DataFrame(data={
                    'B': B, 'K': K, 'Nr': Nr, 'Nt': Nt, 'SNR': SNR,
                    'Rate': stats['rate'], 'MSE': stats['mse']
                }, index=[0])

            if res.empty:
                res = iter_res
            else:
                res += iter_res

        res /= self.iterations['channel']

        self.write_result_csv(res)

        self.write_iteration_csv(stats['rate'] / self.iterations['channel'], 
                                 stats['mse'] / self.iterations['channel'])
