"""This module provides various precoder designs."""
import itertools
import logging
from multiprocessing import Process, Queue

import cvxpy as cvx
import cvxopt

import picos as pic

import numpy as np
import scipy
import scipy.linalg

import utils


class Precoder(object):
    """ This is the base class for all precoder design. The generator function
    should be overridden to comply with the corresponding design."""

    def __init__(self, sysparams, uplink=False, precision=1e-5):
        (self.n_dx, self.n_bx, self.K, self.B) = sysparams

        self.n_sk = min(self.n_bx, self.n_dx)

        self.logger = logging.getLogger(__name__)

        self.precision = precision

        self.uplink = uplink

        if uplink:
            n_tx = self.n_tx
            self.n_tx = self.n_rx
            self.n_rx = n_tx

            K = self.K
            self.K = self.B
            self.B = K

        self.reset()

    def reset(self):
        """Reinitialize precoder parameters / state."""
        pass

    def normalize(self, prec, pwr_lim):
        """ Normalize the prec matrix along the last axis according to the given
            power constraint.
        """

        for ind in range(prec.shape[-1]):
            if len(prec.shape) == 3:
                tmp = prec[:, :, ind]

                prec[:, :, ind] = tmp / np.sqrt((tmp[:]*tmp[:].conj()).sum())
                prec[:, :, ind] *= pwr_lim
            else:
                tmp = prec[:, :, :, ind]

                prec[:, :, :, ind] = tmp / np.sqrt((tmp[:]*tmp[:].conj()).sum())
                prec[:, :, :, ind] *= pwr_lim

        return prec

    def generate(self, *_args, **kwargs):
        """ This method should be overriden by the actual precoder design."""
        pass


class PrecoderGaussian(Precoder):
    """ This a simple Gaussian precoder design, which generates random Gaussian
        precoder matrices that are normalized according to the given power
        constraints."""

    def generate(self, *_args, **kwargs):
        """ Generate a Gaussian precoder"""

        (n_dx, n_bx, K, B) = (self.n_dx, self.n_bx, self.K, self.B)

        n_sk = min(n_dx, n_bx)

        pwr_lim = kwargs.get('pwr_lim', {'BS': 1, 'UE': 1})

        prec = {}

        prec['B2D'] = np.random.randn(n_bx, n_sk, K, B) + \
            np.random.randn(n_bx, n_sk, K, B)*1j
        prec['B2D'] = self.normalize(prec['B2D'], pwr_lim['BS'])

        prec['D2B'] = np.random.randn(n_dx, n_sk, B, K) + \
            np.random.randn(n_dx, n_sk, B, K)*1j
        prec['D2B'] = self.normalize(prec['D2B'], pwr_lim['UE'])

        prec['D2D'] = [[], []]
        prec['D2D'][0] = np.random.randn(n_dx, n_dx, K) + \
            np.random.randn(n_dx, n_dx, K)*1j
        prec['D2D'][0] = self.normalize(prec['D2D'][0], pwr_lim['UE'])

        prec['D2D'][1] = np.random.randn(n_dx, n_dx, K) + \
            np.random.randn(n_dx, n_dx, K)*1j
        prec['D2D'][1] = self.normalize(prec['D2D'][0], pwr_lim['UE'])

        return prec


class PrecoderWMMSE(Precoder):
    """Weighted minimum MSE (WMMSE) precoder design."""

    def mse_weights(self, chan, recv, prec_prev, noise_pwr, errm=None):
        (n_dx, n_bx, K, B) = (self.n_dx, self.n_bx, self.K, self.B)
        n_sk = min(n_dx, n_bx)

        weights = {}

        weights['B2D'] = np.zeros((n_sk, n_sk, K, B), dtype='complex')
        weights['D2B'] = np.zeros((n_sk, n_sk, B, K), dtype='complex')
        weights['D2D'] = np.array([None, None])
        weights['D2D'][0] = np.zeros((n_dx, n_dx, K), dtype='complex')
        weights['D2D'][1] = np.zeros((n_dx, n_dx, K), dtype='complex')

        if errm is None:
            errm = utils.mse(chan, recv, prec_prev, noise_pwr)

        for (_ue, _bs) in itertools.product(range(self.K), range(self.B)):
            weights['B2D'][:, :, _ue, _bs] = np.linalg.pinv(
                errm['B2D'][:, :, _ue, _bs])

            weights['D2B'][:, :, _bs, _ue] = np.linalg.pinv(
                errm['D2B'][:, :, _bs, _ue])

            weights['D2D'][0][:, :, _ue] = np.linalg.pinv(
                errm['D2D'][0][:, :, _ue])

            weights['D2D'][1][:, :, _ue] = np.linalg.pinv(
                errm['D2D'][1][:, :, _ue])

        return weights

    def reset(self):
        """Reinitialize precoder parameters / state."""

        self.lvl1 = np.ones((self.K, self.B)) * 0.1
        self.lvl2 = np.ones((self.K, self.B)) * 0.1
        self.stepsize = 0.05
        self.iteration = 1
        self.itr = 1.

    def generate(self, *args, **kwargs):
        """ Generate the WMMSE precoders. """

        [chan, recv, prec_prev, noise_pwr] = [_a for _a in args]

        use_d2d = np.linalg.norm(recv['D2D'][:]) > 1e-10

        use_cell = (np.linalg.norm(recv['B2D'][:]) +
                    np.linalg.norm(recv['D2B'][:])) > 1e-10

        [n_dx, n_bx, K, B] = chan['B2D'].shape

        pwr_lim = kwargs.get('pwr_lim', {'BS': 1, 'UE': 1})

        err = [np.Inf]


        errm = utils.mse(chan, recv, prec_prev, noise_pwr)

        weight0 = self.mse_weights(chan, recv, prec_prev, noise_pwr,
                                   errm=errm)

        self.itr = 1

        err_prev = 0

        while (np.linalg.norm(err[:]) > 1e-3) and \
              (np.abs(err_prev - np.linalg.norm(err[:])) > self.precision):

            weight = {}
            weight['B2D'] = weight0['B2D'].copy()
            weight['D2B'] = weight0['D2B'].copy()
            weight['D2D'] = [[], []]
            weight['D2D'][0] = weight0['D2D'][0].copy()
            weight['D2D'][1] = weight0['D2D'][1].copy()

            for (_ue, _bs) in itertools.product(range(K), range(B)):
                weight['B2D'][:, :, _ue, _bs] *= self.lvl1[_ue, _bs]
                weight['D2B'][:, :, _bs, _ue] *= self.lvl2[_ue, _bs]

            prec = utils.weighted_bisection(chan, recv, weight, pwr_lim,
                                            self.precision)

            rates = utils.rate(chan, prec, noise_pwr)

            rates['D2B'] = rates['D2B'].transpose(1, 0)

            err_prev = np.linalg.norm(err[:])

            err = rates['D2B'] - rates['B2D']

            if self.iteration < 10:
                self.stepsize = 1.1**(-0.15*self.itr)
            else:
                self.stepsize = 1.1**(-self.itr)

            #if not use_d2d:
                #self.stepsize = 0.5 / np.sqrt(self.itr)

            t = np.minimum(rates['D2B'], rates['B2D'])

            self.lvl1 = self.lvl1 + self.stepsize*(rates['D2B'] - rates['B2D'])
            if not use_d2d:
                self.lvl1[self.lvl1 < 0] = 1e-2
            else:
                self.lvl1[self.lvl1 < 0] = 0

            self.lvl2 = 1 - self.lvl1
            if not use_d2d:
                self.lvl2[self.lvl2 < 0] = 1e-2
            else:
                self.lvl2[self.lvl2 < 0] = 0

            if np.mod(self.itr, 100) == 0:
                self.logger.debug("[%d] err: %f, lvl: %f - %f, (%f), r1: %f, " +
                                  "r2: %f r3: %f, r4: %f", self.itr,
                                  np.linalg.norm(err[:]),
                                  np.linalg.norm(self.lvl1.sum()),
                                  np.linalg.norm(self.lvl2.sum()),
                                  self.stepsize,
                                  rates['D2B'][:].sum(), rates['B2D'][:].sum(),
                                  rates['D2D'][0][:].sum(),
                                  rates['D2D'][1][:].sum())

            self.itr += 1

            if self.itr > 1000:
                break


            if rates['B2D'][:].sum() < 1e-2 and \
               rates['D2B'][:].sum() < 1e-2:
                break

        self.recv_prev = recv
        self.prec_prev = prec
        self.weight_prev = weight

        self.iteration += 1

        return prec


class PrecoderSDP_MAC(Precoder):
    """ Joint transceiver beamformer design for multiple access channel (MAC)
        based on SDP reformulation and successive linear approximation of the
        original problem. """

    def blkdiag(self, m_array):
        """ Block diagonalize [N1 N2 Y] as X diagonal N1*Y-by-N2*Y blocks  """

        sizes = m_array.shape

        blocks = [m_array[:, :, _bs] for _bs in range(sizes[2])]

        return scipy.linalg.block_diag(*blocks)

    def mse_weights(self, chan, recv, prec_prev, noise_pwr):
        weights = {}

        weights['B2D'] = np.zeros((n_sk, n_sk, K, B), dtype='complex')
        weights['D2B'] = np.zeros((n_sk, n_sk, B, K), dtype='complex')
        weights['D2D'] = [[], []]
        weights['D2D'][0] = np.zeros((n_dx, n_dx, K), dtype='complex')
        weights['D2D'][1] = np.zeros((n_dx, n_dx, K), dtype='complex')

        for (_ue, _bs) in itertools.product(range(self.K), range(self.B)):
            weights['B2D'][:, :, _ue, _bs] = np.linalg.pinv(
                errm['B2D'][:, :, _ue, _bs])

            weights['D2B'][:, :, _bs, _ue] = np.linalg.pinv(
                errm['D2B'][:, :, _bs, _ue])

            weights['D2D'][0][:, :, _ue] = np.linalg.pinv(
                errm['D2D'][0][:, :, _ue])

            weights['D2D'][1][:, :, _ue] = np.linalg.pinv(
                errm['D2D'][1][:, :, _ue])

        return weights

    def precoder(self, chan, recv, weights, lvl):
        # Weighted effective downlink channels and covariance
        wrecv = np.dot(np.dot(recv, weights), recv.conj().T)
        wrecv = np.squeeze(wrecv)

        wcov = np.dot(np.dot(chan.conj().T, wrecv), chan)

        # Concatenated transmitters
        prec = np.linalg.solve(wcov + lvl*np.eye(self.n_tx),
                               np.dot(np.dot(chan.conj().T, recv), np.squeeze(weights)))

        return prec.reshape(self.n_tx, self.n_sk, self.K, order='F')

    def solve(self, chan, prec, weights, pwr_lim, tol=1e-6, queue=None):
        """Solve the MAC transmit beamformers. Works only for B = 1.

        Args:
            chan (complex array): Channel matrix.
            prec (complex array): Previous iteration transmit beamformers.
            weights (complex array): MSE weights.
            pwr_lim (double): Sum transmit power limit

        Kwargs:
            tol (double): Solver solution toleranse.
            queue (bool): If not None, pass the return params through the queue.

        Returns: The transmit beamformer array.

        """
        # pylint: disable=R0914
        (n_rx, n_tx, K, n_sk) = (self.n_rx, self.n_tx, self.B, self.n_sk)

        cov = np.dot(np.dot(chan.conj().T, 1/self.noise_pwr * np.eye(n_rx)),
                     chan)

        prob = pic.Problem()

        popt = prob.add_variable('U', (n_tx*K, n_sk*K), 'complex')

        scov = prob.add_variable('S', (n_sk*K, n_sk*K), 'hermitian')

        scomp = prob.add_variable('X', (n_sk*K, n_sk*K), 'hermitian')

        eye_sk = pic.new_param('I', np.eye(n_sk*K))

        weights = pic.new_param('W', weights)

        # Objective
        objective = 'I' | (weights*scomp)

        prob.set_objective('min', objective)

        # Constraints
        prob.add_constraint(((scomp & eye_sk) //
                             (eye_sk & (scov + eye_sk))) >> 0)

        cpnt = pic.new_param('C0', np.dot(np.dot(prec.conj().T, cov), prec))

        peff = pic.new_param('U0', np.dot(prec.conj().T, cov))

        prob.add_constraint(cpnt + peff*(popt - prec) +
                            (popt.H - prec.conj().T)*peff.H >> scov)

        # Block diagonal structure constraint
        zind = np.kron(np.eye(K), np.ones((n_tx, n_sk)))
        zind = np.vstack(np.where(zind == 0))
        zind = [(zind[0, _ki], zind[1, _ki])
                for _ki in range(zind.shape[1])]

        prob.add_list_of_constraints(
            [popt.imag[_i, _j] == 0 for (_i, _j) in zind])
        prob.add_list_of_constraints(
            [popt.real[_i, _j] == 0 for (_i, _j) in zind])

        # Transmit power limit
        eye_k = pic.new_param('I', np.eye(n_tx))

        for _ue in range(K):
            pcomp = prob.add_variable('Y%i' % _ue, (n_sk, n_sk), 'hermitian')

            _r0 = (_ue)*n_tx
            _r1 = (_ue+1)*n_tx

            _c0 = _ue*n_sk
            _c1 = (_ue+1)*n_sk

            X = popt[_r0:_r1, _c0:_c1]

            prob.add_constraint(((pcomp & X.H) // (X & eye_k)) >> 0)
            prob.add_constraint(('I' | pcomp) <= pwr_lim)

        # Solve the problem
        prob.solve(verbose=0, noduals=True, tol=tol, solve_via_dual=False)

        popt = np.asarray(np.matrix(popt.value))

        prec = np.zeros((n_tx, n_sk, 1, K), dtype='complex')

        for _ue in range(K):
            _r0 = (_ue)*n_tx
            _r1 = (_ue+1)*n_tx

            _c0 = _ue*n_sk
            _c1 = (_ue+1)*n_sk

            prec[:, :, 0, _ue] = popt[_r0:_r1, _c0:_c1]

        if queue is None:
            return prec
        else:
            queue.put(prec)

    def generate(self, *args, **kwargs):
        """ Generate the precoders. """

        [chan_glob, recv_prev, prec_prev, noise_pwr] = [_a for _a in args]

        pwr_lim = kwargs.get('pwr_lim', 1)

        self.noise_pwr = noise_pwr
        self.pwr_lim = pwr_lim

        # MSE and weights
        weights = self.mse_weights(chan_glob, recv_prev, prec_prev, noise_pwr)

        # The new precoders
        prec = np.zeros((self.n_tx, self.n_sk, self.K, self.B),
                        dtype='complex')

        # Block diagonalize matrices
        recv_prev = self.blkdiag(np.squeeze(recv_prev, axis=(2,)))
        prec_prev = self.blkdiag(np.squeeze(prec_prev, axis=(2,)))
        weights = self.blkdiag(np.squeeze(weights, axis=(2,)))

        chan = np.squeeze(chan_glob)
        chan = chan.reshape((self.n_rx, self.n_tx*self.B),
                            order='F')

        # Picos is sandbox into separate process to ensure proper memory
        # management.
        queue = Queue()
        tol = 1e-6

        p_sandbox = Process(target=self.solve,
                            args=(chan, prec_prev, weights, pwr_lim, tol,
                                  queue))
        # prec = self.solve(chan, prec_prev, weights, pwr_lim)
        p_sandbox.start()
        p_sandbox.join()
        prec = queue.get()

        return prec


class PrecoderSDP(Precoder):
    """ Joint transceiver beamformer design based on SDP reformulation and
        successive linear approximation of the original problem. """

    def blkdiag(self, m_array):
        """ Block diagonalize [N1 N2 Y X] as X diagonal N1*Y-by-N2*Y blocks  """

        sizes = m_array.shape

        blksize = (sizes[0]*sizes[2], sizes[1]*sizes[2])

        ret_array = np.zeros((blksize[0], blksize[1], sizes[3]),
                             dtype='complex')

        for _bs in range(sizes[3]):
            blocks = [m_array[:, :, _ue, _bs] for _ue in range(sizes[2])]

            ret_array[:, :, _bs] = scipy.linalg.block_diag(*blocks)

        return ret_array

    def mse_weights(self, chan, recv, prec_prev, noise_pwr):
        weights = np.zeros((self.n_sk, self.n_sk, self.K, self.B),
                           dtype='complex')

        errm = utils.mse(chan, recv, prec_prev, noise_pwr)

        for (_ue, _bs) in itertools.product(range(self.K), range(self.B)):
            weights[:, :, _ue, _bs] = np.linalg.pinv(errm[:, :, _ue, _bs])

        return weights

    def precoder(self, chan, recv, weights, lvl):
        # Weighted effective downlink channels and covariance
        wrecv = np.dot(np.dot(recv, weights), recv.conj().T)
        wrecv = np.squeeze(wrecv)

        wcov = np.dot(np.dot(chan.conj().T, wrecv), chan)

        # Concatenated transmitters
        prec = np.dot(np.linalg.pinv(wcov + lvl*np.eye(self.n_tx)),
                      np.dot(np.dot(chan.conj().T, recv), np.squeeze(weights)))

        return prec.reshape(self.n_tx, self.n_sk, self.K, order='F')

    def solve(self, chan, recv, weights, lvl, tol=1e-4, queue=None):
        # pylint: disable=R0914

        (n_rx, n_tx, K, n_sk) = (self.n_rx, self.n_tx, self.K, self.n_sk)

        cov = np.dot(np.dot(chan, (1/lvl)*np.eye(n_tx)), chan.conj().T)

        prob = pic.Problem()

        ropt = prob.add_variable('U', (n_rx*K, n_sk*K), 'complex')

        scov = prob.add_variable('S', (n_sk*K, n_sk*K), 'hermitian')

        scomp = prob.add_variable('X', (n_sk*K, n_sk*K), 'hermitian')

        ncomp = prob.add_variable('Y', (n_rx*K, n_rx*K), 'hermitian')

        eye_sk = pic.new_param('I', np.eye(n_sk*K))
        eye_rx = pic.new_param('I', np.eye(n_rx*K))

        wsqrt = pic.new_param('W', np.linalg.cholesky(weights))
        winv = pic.new_param('W', np.linalg.pinv(weights))

        # Objective
        objective = 'I' | scomp
        objective += self.noise_pwr*('I' | ncomp)

        prob.set_objective('min', objective)

        # Constraints
        prob.add_constraint(((ncomp & ropt*wsqrt) //
                             (wsqrt.H*ropt.H & eye_rx)) >> 0)

        prob.add_constraint(((scomp & eye_sk) //
                             (eye_sk & (scov + winv))) >> 0)

        cpnt = pic.new_param('C0', np.dot(np.dot(recv.conj().T, cov), recv))

        reff = pic.new_param('U0', np.dot(recv.conj().T, cov))

        prob.add_constraint(cpnt + reff*(ropt - recv) +
                            (ropt.H - recv.conj().T)*reff.H >> scov)

        # Block diagonal structure constraint
        zind = np.kron(np.eye(K), np.ones((n_rx, n_sk)))
        zind = np.vstack(np.where(zind == 0))
        zind = [(zind[0, _ki], zind[1, _ki])
                for _ki in range(zind.shape[1])]

        prob.add_list_of_constraints(
            [ropt.imag[_i, _j] == 0 for (_i, _j) in zind])
        prob.add_list_of_constraints(
            [ropt.real[_i, _j] == 0 for (_i, _j) in zind])

        # Solve the problem
        prob.solve(verbose=0, noduals=True, tol=tol, solve_via_dual=False)

        ropt = np.asarray(np.matrix(ropt.value))

        if queue is None:
            return np.squeeze(ropt)
        else:
            queue.put(np.squeeze(ropt))

    def search(self, chan, recv, weights, method="bisection", step=0.5):
        """ Perform the primal-dual precoder optimization over the domain of
            dual variables.

        Args:
            chan (matrix): The concatenated channel matrix.
            recv (matrix): Block diagonal receivers from the previous iteration.
            weights (matrix): Block diagonal MSE weights.
            method (str): The dual variable update method. Supported update
                          methods are "bisection" and "subgradient".
            step (double): Step size for the subgradient method.

        Returns: The local precoder matrix.

        """

        upper_bound = 10.
        bounds = np.array([0, upper_bound])

        pnew = np.Inf

        tol = 1e-4

        itr = 1

        err = np.inf

        while err > self.precision:
            if method == "bisection":
                lvl = bounds.sum() / 2

            # Picos is sandbox into separate process to ensure proper memory
            # management.
            queue = Queue()
            p_sandbox = Process(target=self.solve,
                                args=(chan, recv, weights, lvl, tol, queue))
            # ropt = self.solve(chan, recv, weights, lvl, tol)
            p_sandbox.start()
            p_sandbox.join()
            ropt = queue.get()

            prec = self.precoder(chan, ropt, weights, lvl)

            # Compute power
            pnew = np.linalg.norm(prec[:])**2

            if method == "subgradient":
                lvl = max(1e-10, lvl + step/np.sqrt(itr)*(pnew - self.pwr_lim))
            else:
                if pnew > self.pwr_lim:
                    bounds[0] = lvl
                else:
                    bounds[1] = lvl

            err = np.abs(pnew - self.pwr_lim)

            self.logger.debug("%d: lvl: %f P: %f err: %f bnd: %f", itr, lvl,
                            pnew, err, np.abs(bounds[0] - bounds[1]))

            if np.abs(bounds[0] - bounds[1]) < 1e-10:
                if np.abs(upper_bound - bounds[1]) < 1e-9:
                    upper_bound *= 10
                    bounds = np.array([0, upper_bound])
                else:
                    tol *= 10
                    bounds = np.array([0, upper_bound])

            itr += 1

        return prec

    def generate(self, *args, **kwargs):
        """ Generate the precoders. """

        [chan_glob, recv, prec, noise_pwr] = [_a for _a in args]

        pwr_lim = kwargs.get('pwr_lim', 1)

        self.noise_pwr = noise_pwr
        self.pwr_lim = pwr_lim

        # MSE and weights
        weights = self.mse_weights(chan_glob, recv, prec, noise_pwr)

        # The new precoders
        prec = np.zeros((self.n_tx, self.n_sk, self.K, self.B),
                        dtype='complex')

        # Block diagonalize matrices
        recv = self.blkdiag(recv)
        weights = self.blkdiag(weights)

        for _bs in range(self.B):
            # Composite channel
            chan = np.dsplit(chan_glob[:, :, :, _bs], self.K)
            chan = np.squeeze(np.vstack(chan))

            prec[:, :, :, _bs] = self.search(chan, recv[:, :, _bs],
                                             weights[:, :, _bs])

        return prec


class PrecoderCVX(Precoder):
    """ Joint transceiver beamformer design based on SDP reformulation and
        successive linear approximation of the original problem. """


    def blkdiag(self, m_array):
        """ Block diagonalize [N1 N2 Y X] as X diagonal N1*Y-by-N2*Y blocks  """

        sizes = m_array.shape

        blksize = (sizes[0]*sizes[2], sizes[1]*sizes[2])

        ret_array = np.zeros((blksize[0], blksize[1], sizes[3]),
                             dtype='complex')

        for _bs in range(sizes[3]):
            blocks = [m_array[:, :, _ue, _bs] for _ue in range(sizes[2])]

            ret_array[:, :, _bs] = scipy.linalg.block_diag(*blocks)

        return ret_array

    def mse_weights(self, chan, recv, prec_prev, noise_pwr, errm=None):
        (n_dx, n_bx, K, B) = (self.n_dx, self.n_bx, self.K, self.B)
        n_sk = min(n_dx, n_bx)

        weights = {}

        weights['B2D'] = np.zeros((n_sk, n_sk, K, B), dtype='complex')
        weights['D2B'] = np.zeros((n_sk, n_sk, B, K), dtype='complex')
        weights['D2D'] = np.array([None, None])
        weights['D2D'][0] = np.zeros((n_dx, n_dx, K), dtype='complex')
        weights['D2D'][1] = np.zeros((n_dx, n_dx, K), dtype='complex')

        if errm is None:
            errm = utils.mse(chan, recv, prec_prev, noise_pwr)

        for (_ue, _bs) in itertools.product(range(self.K), range(self.B)):
            weights['B2D'][:, :, _ue, _bs] = np.linalg.pinv(
                errm['B2D'][:, :, _ue, _bs])

            weights['D2B'][:, :, _bs, _ue] = np.linalg.pinv(
                errm['D2B'][:, :, _bs, _ue])

            weights['D2D'][0][:, :, _ue] = np.linalg.pinv(
                errm['D2D'][0][:, :, _ue])

            weights['D2D'][1][:, :, _ue] = np.linalg.pinv(
                errm['D2D'][1][:, :, _ue])

        return weights

    def precoder(self, chan, recv, weights, lvl):
        # Weighted effective downlink channels and covariance
        wrecv = np.dot(np.dot(recv, weights), recv.conj().T)
        wrecv = np.squeeze(wrecv)

        wcov = np.dot(np.dot(chan.conj().T, wrecv), chan)

        # Concatenated transmitters
        prec = np.dot(np.linalg.pinv(wcov + lvl*np.eye(self.n_tx)),
                      np.dot(np.dot(chan.conj().T, recv), np.squeeze(weights)))

        return prec.reshape(self.n_tx, self.n_sk, self.n_ue, order='F')

    def solve_cvxpy(self, chan, recv, weights):
        # pylint: disable=R0914

        [n_dx, n_bx, K, B] = chan['B2D'].shape
        n_sk = min(n_dx, n_bx)

        use_d2d = np.linalg.norm(recv['D2D'][:]) > 1e-10

        use_cell = (np.linalg.norm(recv['B2D'][:]) +
                    np.linalg.norm(recv['D2B'][:])) > 1e-10


        recv['D2B'] = [cvxopt.matrix(recv['D2B'][:,:,0,k]) for k in range(K)]
        recv['B2D'] = [cvxopt.matrix(recv['B2D'][:,:,k,0]) for k in range(K)]
        recv['D2D'][0] = [cvxopt.matrix(recv['D2D'][0][:,:,k]) for k in range(K)]
        recv['D2D'][1] = [cvxopt.matrix(recv['D2D'][1][:,:,k]) for k in range(K)]

        wsqrt = {}
        wsqrt['D2B'] = [cvxopt.matrix(np.linalg.cholesky(weights['D2B'][:,:,0,k])) for k in range(K)]
        wsqrt['B2D'] = [cvxopt.matrix(np.linalg.cholesky(weights['B2D'][:,:,k,0])) for k in range(K)]
        wsqrt['D2D'] = [[], []]
        wsqrt['D2D'][0] = [cvxopt.matrix(np.linalg.cholesky(weights['D2D'][0][:,:,k])) for k in range(K)]
        wsqrt['D2D'][1] = [cvxopt.matrix(np.linalg.cholesky(weights['D2D'][1][:,:,k])) for k in range(K)]

        weights['D2B'] = [cvxopt.matrix(weights['D2B'][:,:,0,k])
                          for k in range(K)]
        weights['B2D'] = [cvxopt.matrix(weights['B2D'][:,:,k,0])
                          for k in range(K)]
        weights['D2D'][0] = [cvxopt.matrix(weights['D2D'][0][:,:,k])
                             for k in range(K)]
        weights['D2D'][1] = [cvxopt.matrix(weights['D2D'][1][:,:,k])
                             for k in range(K)]

        chan['D2B'] = [cvxopt.matrix(chan['D2B'][:,:,0,k])
                       for k in range(K)]
        chan['B2D'] = [cvxopt.matrix(chan['B2D'][:,:,k,0])
                       for k in range(K)]
        chan['D2D'] = [[cvxopt.matrix(chan['D2D'][:,:,i,j])
                        for j in range(K)] for i in range(K)]

        eye_sk = np.eye(n_sk)
        z_sk = np.zeros((n_sk, n_sk))

        eye_dk = np.eye(n_dx)
        z_dk = np.zeros((n_dx, n_dx))

        # Beamformer matrices
        prec_re = {}
        prec_re['D2D'] = []
        prec_re['D2D'].append([cvx.Variable(n_dx, n_dx) for k in range(K)])
        prec_re['D2D'].append([cvx.Variable(n_dx, n_dx) for k in range(K)])
        prec_re['B2D'] = [cvx.Variable(n_bx, n_sk) for k in range(K)]
        prec_re['D2B'] = [cvx.Variable(n_dx, n_sk) for k in range(K)]

        prec_im = {}
        prec_im['D2D'] = []
        prec_im['D2D'].append([cvx.Variable(n_dx, n_dx) for k in range(K)])
        prec_im['D2D'].append([cvx.Variable(n_dx, n_dx) for k in range(K)])
        prec_im['B2D'] = [cvx.Variable(n_bx, n_sk) for k in range(K)]
        prec_im['D2B'] = [cvx.Variable(n_dx, n_sk) for k in range(K)]

        # Quadratiac beamformer matrices
        quad_re = {}
        quad_re['D2D'] = []
        quad_re['D2D'].append([cvx.Variable(n_dx, n_dx) for k in range(K)])
        quad_re['D2D'].append([cvx.Variable(n_dx, n_dx) for k in range(K)])
        quad_re['B2D'] = [cvx.Variable(n_bx, n_bx) for k in range(K)]
        quad_re['D2B'] = [cvx.Variable(n_dx, n_dx) for k in range(K)]

        quad_im = {}
        quad_im['D2D'] = []
        quad_im['D2D'].append([cvx.Variable(n_dx, n_dx) for k in range(K)])
        quad_im['D2D'].append([cvx.Variable(n_dx, n_dx) for k in range(K)])
        quad_im['B2D'] = [cvx.Variable(n_bx, n_bx) for k in range(K)]
        quad_im['D2B'] = [cvx.Variable(n_dx, n_dx) for k in range(K)]

        # Rates
        R = {}
        R['D2D'] = []
        R['D2D'].append([cvx.Variable() for k in range(K)])
        R['D2D'].append([cvx.Variable() for k in range(K)])
        R['B2D'] = [cvx.Variable(name="Rd%d" % k) for k in range(K)]
        R['D2B'] = [cvx.Variable(name="Ru%d" % k) for k in range(K)]

        t = cvx.Variable(K, 1)

        objective = cvx.trace(0)
        for k in range(K):
            if use_cell:
                objective += cvx.sum_entries(t)

            if use_d2d:
                objective += R['D2D'][0][k] + R['D2D'][1][k]

        prob = cvx.Problem(cvx.Minimize(objective))

        for k in range(K):
            prob.constraints.append(R['B2D'][k] <= t[k])
            prob.constraints.append(R['D2B'][k] <= t[k])

        # Slot 1: D2B & D2D(0)

        # D2B
        for k in range(K):
            if not use_cell:
                continue

            C = cvx.trace(weights['D2B'][k].real())

            W_chan = weights['D2B'][k]*recv['D2B'][k].H*chan['D2B'][k]
            C -= cvx.trace(W_chan.real()*prec_re['D2B'][k])
            C += cvx.trace(W_chan.imag()*prec_im['D2B'][k])
            C -= cvx.trace(prec_re['D2B'][k].T*W_chan.H.real())
            C += cvx.trace(-prec_im['D2B'][k].T*W_chan.H.imag())

            C += cvx.trace((recv['D2B'][k]*weights['D2B'][k]*
                            recv['D2B'][k].H).real() * self.noise_pwr['BS'])

            for j in range(K):
                H_tmp = chan['D2B'][j].H*recv['D2B'][k]*wsqrt['D2B'][k]

                C += cvx.trace(H_tmp.T.real() * quad_re['D2B'][j] *
                               H_tmp.real())
                C += cvx.trace(H_tmp.T.imag() * quad_im['D2B'][j] *
                               H_tmp.real())
                C -= cvx.trace(H_tmp.T.real() * quad_im['D2B'][j] *
                               H_tmp.imag())
                C += cvx.trace(H_tmp.T.imag() * quad_re['D2B'][j] *
                               H_tmp.imag())

                if use_d2d:
                    C += cvx.trace(H_tmp.T.real() * quad_re['D2D'][0][j] *
                                H_tmp.real())
                    C += cvx.trace(H_tmp.T.imag() * quad_im['D2D'][0][j] *
                                H_tmp.real())
                    C -= cvx.trace(H_tmp.T.real() * quad_im['D2D'][0][j] *
                                H_tmp.imag())
                    C += cvx.trace(H_tmp.T.imag() * quad_re['D2D'][0][j] *
                                H_tmp.imag())

            prob.constraints.append(C <= R['D2B'][k])

            C = cvx.vstack(cvx.hstack(quad_re['D2B'][k], -quad_im['D2B'][k]),
                           cvx.hstack(quad_im['D2B'][k], quad_re['D2B'][k]))
            prob.constraints.append(C == cvx.Semidef(2*n_dx))

            C = cvx.vstack(cvx.hstack(quad_re['D2B'][k], prec_re['D2B'][k],
                                      -quad_im['D2B'][k], -prec_im['D2B'][k]),
                           cvx.hstack(prec_re['D2B'][k].T, eye_sk,
                                      prec_im['D2B'][k].T, z_sk),
                           cvx.hstack(quad_im['D2B'][k], prec_im['D2B'][k],
                                      quad_re['D2B'][k], prec_re['D2B'][k]),
                           cvx.hstack(-prec_im['D2B'][k].T, z_sk,
                                      prec_re['D2B'][k].T, eye_sk))

            prob.constraints.append(C == cvx.Semidef(2*(n_sk+n_dx)))

            # Skew-symmetry
            prob.constraints.append(quad_im['D2B'][k] + quad_im['D2B'][k].T
                                    == 0)

        # D2D - 1
        for k in range(K):
            if not use_d2d:
                continue

            C = cvx.trace(weights['D2D'][0][k].real())

            W_chan = weights['D2D'][0][k]*recv['D2D'][0][k].H*chan['D2D'][k][k]
            C -= cvx.trace(W_chan.real()*prec_re['D2D'][0][k])
            C += cvx.trace(W_chan.imag()*prec_im['D2D'][0][k])
            C -= cvx.trace(prec_re['D2D'][0][k].T*W_chan.H.real())
            C += cvx.trace(-prec_im['D2D'][0][k].T*W_chan.H.imag())

            C += cvx.trace((recv['D2D'][0][k] * weights['D2D'][0][k] *
                            recv['D2D'][0][k].H).real() * self.noise_pwr['UE'])

            for j in range(K):
                H_tmp = chan['D2D'][k][j].H*recv['D2D'][0][k]*wsqrt['D2D'][0][k]

                if use_cell:
                    C += cvx.trace(H_tmp.T.real() * quad_re['D2B'][j] *
                                H_tmp.real())
                    C += cvx.trace(H_tmp.T.imag() * quad_im['D2B'][j] *
                                H_tmp.real())
                    C -= cvx.trace(H_tmp.T.real() * quad_im['D2B'][j] *
                                H_tmp.imag())
                    C += cvx.trace(H_tmp.T.imag() * quad_re['D2B'][j] *
                                H_tmp.imag())

                C += cvx.trace(H_tmp.T.real() * quad_re['D2D'][0][j] *
                               H_tmp.real())
                C += cvx.trace(H_tmp.T.imag() * quad_im['D2D'][0][j] *
                               H_tmp.real())
                C -= cvx.trace(H_tmp.T.real() * quad_im['D2D'][0][j] *
                               H_tmp.imag())
                C += cvx.trace(H_tmp.T.imag() * quad_re['D2D'][0][j] *
                               H_tmp.imag())

            prob.constraints.append(C <= R['D2D'][0][k])

            C = cvx.vstack(cvx.hstack(quad_re['D2D'][0][k],
                                      -quad_im['D2D'][0][k]),
                           cvx.hstack(quad_im['D2D'][0][k],
                                      quad_re['D2D'][0][k]))
            prob.constraints.append(C == cvx.Semidef(2*n_dx))

            C = cvx.vstack(
                    cvx.hstack(quad_re['D2D'][0][k], prec_re['D2D'][0][k],
                               -quad_im['D2D'][0][k], -prec_im['D2D'][0][k]),
                    cvx.hstack(prec_re['D2D'][0][k].T, eye_sk,
                               prec_im['D2D'][0][k].T, z_sk),
                    cvx.hstack(quad_im['D2D'][0][k], prec_im['D2D'][0][k],
                               quad_re['D2D'][0][k], prec_re['D2D'][0][k]),
                    cvx.hstack(-prec_im['D2D'][0][k].T, z_sk,
                               prec_re['D2D'][0][k].T, eye_sk))

            prob.constraints.append(C == cvx.Semidef(2*(n_sk+n_dx)))

            # Skew-symmetry
            prob.constraints.append(quad_im['D2D'][0][k] +
                                    quad_im['D2D'][0][k].T  == 0)

        # B2D
        for k in range(K):
            if not use_cell:
                continue

            C = cvx.trace(weights['B2D'][k].real())

            W_chan = weights['B2D'][k]*recv['B2D'][k].H*chan['B2D'][k]

            C -= cvx.trace(W_chan.real()*prec_re['B2D'][k])
            C += cvx.trace(W_chan.imag()*prec_im['B2D'][k])
            C -= cvx.trace(prec_re['B2D'][k].T*W_chan.H.real())
            C += cvx.trace(-prec_im['B2D'][k].T*W_chan.H.imag())

            C += cvx.trace((recv['B2D'][k]*weights['B2D'][k]*
                            recv['B2D'][k].H).real() * self.noise_pwr['UE'])

            for j in range(K):
                H_tmp = chan['B2D'][k].H*recv['B2D'][k]*wsqrt['B2D'][k]

                C += cvx.trace(H_tmp.T.real() * quad_re['B2D'][j] *
                               H_tmp.real())
                C += cvx.trace(H_tmp.T.imag() * quad_im['B2D'][j] *
                               H_tmp.real())
                C -= cvx.trace(H_tmp.T.real() * quad_im['B2D'][j] *
                               H_tmp.imag())
                C += cvx.trace(H_tmp.T.imag() * quad_re['B2D'][j] *
                               H_tmp.imag())

            for j in range(K):
                if not use_d2d:
                    continue

                H_tmp = chan['D2D'][k][j].H*recv['B2D'][k]*wsqrt['B2D'][k]

                C += cvx.trace(H_tmp.T.real() * quad_re['D2D'][1][j] *
                               H_tmp.real())
                C += cvx.trace(H_tmp.T.imag() * quad_im['D2D'][1][j] *
                               H_tmp.real())
                C -= cvx.trace(H_tmp.T.real() * quad_im['D2D'][1][j] *
                               H_tmp.imag())
                C += cvx.trace(H_tmp.T.imag() * quad_re['D2D'][1][j] *
                               H_tmp.imag())

            prob.constraints.append(C <= R['B2D'][k])

            C = cvx.vstack(cvx.hstack(quad_re['B2D'][k], -quad_im['B2D'][k]),
                           cvx.hstack(quad_im['B2D'][k], quad_re['B2D'][k]))
            prob.constraints.append(C == cvx.Semidef(2*n_bx))

            C = cvx.vstack(
                    cvx.hstack(quad_re['B2D'][k], prec_re['B2D'][k],
                               -quad_im['B2D'][k], -prec_im['B2D'][k]),
                    cvx.hstack(prec_re['B2D'][k].T, eye_dk,
                               prec_im['B2D'][k].T, z_dk),
                    cvx.hstack(quad_im['B2D'][k], prec_im['B2D'][k],
                               quad_re['B2D'][k], prec_re['B2D'][k]),
                    cvx.hstack(-prec_im['B2D'][k].T, z_dk,
                               prec_re['B2D'][k].T, eye_dk))

            prob.constraints.append(C == cvx.Semidef(2*(n_sk+n_bx)))


            # Skew-symmetry
            prob.constraints.append(quad_im['B2D'][k] +
                                    quad_im['B2D'][k].T  == 0)

        # Power constraint
        for k in range(K):
            C = 0

            if use_d2d:
                C += cvx.sum_squares(prec_re['D2D'][0][k]) + \
                     cvx.sum_squares(prec_im['D2D'][0][k])

            if use_cell:
                C += cvx.sum_squares(prec_re['D2B'][k]) + \
                     cvx.sum_squares(prec_im['D2B'][k])

            prob.constraints.append(C <= self.pwr_lim['UE'])

        # D2D - 2
        for k in range(K):
            if not use_d2d:
                continue

            C = cvx.trace(weights['D2D'][1][k].real())

            W_chan = weights['D2D'][1][k]*recv['D2D'][1][k].H*chan['D2D'][k][k]
            C -= cvx.trace(W_chan.real()*prec_re['D2D'][1][k])
            C += cvx.trace(W_chan.imag()*prec_im['D2D'][1][k])
            C -= cvx.trace(prec_re['D2D'][1][k].T*W_chan.H.real())
            C += cvx.trace(-prec_im['D2D'][1][k].T*W_chan.H.imag())

            C += cvx.trace((recv['D2D'][1][k]*weights['D2D'][1][k]*
                            recv['D2D'][1][k].H).real() * self.noise_pwr['UE'])

            for j in range(K):
                if not use_cell:
                    continue

                H_tmp = chan['B2D'][k].H*recv['D2D'][1][k]*wsqrt['D2D'][1][k]

                C += cvx.trace(H_tmp.T.real() * quad_re['B2D'][j] *
                               H_tmp.real())
                C += cvx.trace(H_tmp.T.imag() * quad_im['B2D'][j] *
                               H_tmp.real())
                C -= cvx.trace(H_tmp.T.real() * quad_im['B2D'][j] *
                               H_tmp.imag())
                C += cvx.trace(H_tmp.T.imag() * quad_re['B2D'][j] *
                               H_tmp.imag())

            for j in range(K):
                H_tmp = chan['D2D'][k][j].H*recv['D2D'][1][k]*wsqrt['D2D'][1][k]

                C += cvx.trace(H_tmp.T.real() * quad_re['D2D'][1][j] *
                               H_tmp.real())
                C += cvx.trace(H_tmp.T.imag() * quad_im['D2D'][1][j] *
                               H_tmp.real())
                C -= cvx.trace(H_tmp.T.real() * quad_im['D2D'][1][j] *
                               H_tmp.imag())
                C += cvx.trace(H_tmp.T.imag() * quad_re['D2D'][1][j] *
                               H_tmp.imag())

            prob.constraints.append(C <= R['D2D'][1][k])

            C = cvx.vstack(cvx.hstack(quad_re['D2D'][1][k],
                                      -quad_im['D2D'][1][k]),
                           cvx.hstack(quad_im['D2D'][1][k],
                                      quad_re['D2D'][1][k]))

            prob.constraints.append(C == cvx.Semidef(2*n_dx))

            C = cvx.vstack(
                    cvx.hstack(quad_re['D2D'][1][k], prec_re['D2D'][1][k],
                               -quad_im['D2D'][1][k], -prec_im['D2D'][1][k]),
                    cvx.hstack(prec_re['D2D'][1][k].T, eye_sk,
                               prec_im['D2D'][1][k].T, z_sk),
                    cvx.hstack(quad_im['D2D'][1][k], prec_im['D2D'][1][k],
                               quad_re['D2D'][1][k], prec_re['D2D'][1][k]),
                    cvx.hstack(-prec_im['D2D'][1][k].T, z_sk,
                               prec_re['D2D'][1][k].T, eye_sk))

            prob.constraints.append(C == cvx.Semidef(2*(n_sk+n_dx)))

            # Skew-symmetry
            prob.constraints.append(quad_im['D2D'][1][k] +
                                    quad_im['D2D'][1][k].T  == 0)

        # Power constraints
        for k in range(K):
            if not use_d2d:
                continue

            prob.constraints.append(
                        cvx.sum_squares(prec_re['D2D'][1][k]) +
                        cvx.sum_squares(prec_im['D2D'][1][k])
                        <= self.pwr_lim['UE'])

        if use_cell:
            C = cvx.trace(0)
            for k in range(K):
                C += cvx.sum_squares(prec_re['B2D'][k])
                C += cvx.sum_squares(prec_im['B2D'][k])

            prob.constraints.append(C <= self.pwr_lim['BS'])

        # Solve the problem
        prob.solve(verbose=True, solver=cvx.SCS, max_iters=100000)

        prec = {}
        prec['D2B'] = np.zeros((n_dx, n_sk, B, K), dtype='complex')
        prec['B2D'] = np.zeros((n_bx, n_sk, K, B), dtype='complex')

        params = (n_dx, n_dx, K)
        prec['D2D'] = []
        prec['D2D'].append(np.zeros(params, dtype='complex'))
        prec['D2D'].append(np.zeros(params, dtype='complex'))

        for k in range(K):
            if use_cell:
                prec['D2B'][:,:,0,k] = prec_re['D2B'][k].value + \
                    1j*prec_im['D2B'][k].value

                prec['B2D'][:,:,k,0] = prec_re['B2D'][k].value + \
                    1j*prec_im['B2D'][k].value

            if use_d2d:
                prec['D2D'][0][:,:,k] = prec_re['D2D'][0][k].value + \
                    1j*prec_im['D2D'][0][k].value

                prec['D2D'][1][:,:,k] = prec_re['D2D'][1][k].value + \
                    1j*prec_im['D2D'][1][k].value

        print(np.linalg.norm(prec['B2D'][:]) / self.pwr_lim['BS'])
        print(np.linalg.norm(prec['D2B'][:]) / self.pwr_lim['UE'])
        print(np.linalg.norm(prec['D2D'][0][:]) / self.pwr_lim['UE'])
        print(np.linalg.norm(prec['D2D'][1][:]) / self.pwr_lim['UE'])

        return prec

    def generate(self, *args, **kwargs):
        """ Generate the precoders. """

        [chan, recv, prec_prev, noise_pwr] = [_a for _a in args]

        [n_dx, n_bx, K, B] = chan['B2D'].shape
        n_sk = min(n_dx, n_bx)

        pwr_lim = kwargs.get('pwr_lim', {'BS': 1, 'UE': 1})

        self.noise_pwr = noise_pwr
        self.pwr_lim = pwr_lim

        # MSE and weights
        errm = utils.mse(chan, recv, prec_prev, noise_pwr)

        weights = self.mse_weights(chan, recv, prec_prev, noise_pwr,
                                   errm=errm)

        # The new precoders
        prec = self.solve_cvxpy(chan.copy(), recv, weights)

        rates = utils.rate(chan, prec, noise_pwr)

        rates['D2B'] = rates['D2B'].transpose(1, 0)

        err = rates['D2B'][:].sum() - rates['B2D'][:].sum()

        self.logger.debug("err: %f, r1: %f, " "r2: %f r3: %f, r4: %f",
                          np.linalg.norm(err),
                          rates['D2B'][:].sum(), rates['B2D'][:].sum(),
                          rates['D2D'][0][:].sum(),
                          rates['D2D'][1][:].sum())

        return prec
