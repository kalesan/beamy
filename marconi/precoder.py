"""This module provides various precoder designs."""
import itertools
import logging
from multiprocessing import Process, Queue

import picos as pic

import numpy as np
import scipy
import scipy.linalg

import utils


class Precoder(object):
    """ This is the base class for all precoder design. The generator function
    should be overridden to comply with the corresponding design."""

    def __init__(self, sysparams, uplink=False, precision=1e-6):
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

    def mse_weights(self, chan, recv, prec_prev, noise_pwr):
        (n_dx, n_bx, K, B) = (self.n_dx, self.n_bx, self.K, self.B)
        n_sk = min(n_dx, n_bx)

        weights = {}

        weights['B2D'] = np.zeros((n_sk, n_sk, K, B), dtype='complex')
        weights['D2B'] = np.zeros((n_sk, n_sk, B, K), dtype='complex')
        weights['D2D'] = np.array([None, None])
        weights['D2D'][0] = np.zeros((n_dx, n_dx, K), dtype='complex')
        weights['D2D'][1] = np.zeros((n_dx, n_dx, K), dtype='complex')

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

    def generate(self, *args, **kwargs):
        """ Generate the WMMSE precoders. """

        [chan, recv, prec_prev, noise_pwr] = [_a for _a in args]

        [n_dx, n_bx, K, B] = chan['B2D'].shape
        n_sk = min(n_dx, n_bx)

        pwr_lim = kwargs.get('pwr_lim', {'BS': 1, 'UE': 1})

        err = np.Inf
        err_prev = np.Inf

        lvl = np.ones((K, B)) * 0.5
        STEP_BASE = 1e-5
        stepsize = 1

        itr = 1

        while np.linalg.norm(err) > self.precision:
            errm = utils.mse(chan, recv, prec_prev, noise_pwr)

            weight = self.mse_weights(chan, recv, prec_prev, noise_pwr)

            for (_ue, _bs) in itertools.product(range(K), range(B)):
                weight['B2D'][:, :, _ue, _bs] *= lvl[_ue, _bs]
                weight['D2B'][:, :, _bs, _ue] *= (1 - lvl[_ue, _bs])

            prec = utils.weighted_bisection(chan, recv, weight, pwr_lim,
                                            threshold=self.precision)

            rates = utils.rate(chan, prec, noise_pwr)

            rates['D2B'] = rates['D2B'].transpose(1, 0)

            err = rates['D2B'][:].sum() - rates['B2D'][:].sum()

            if np.abs(err - err_prev) < 1e-4:
                stepsize = 1.01*stepsize
            else:
                stepsize = 1/np.sqrt(itr)

            lvl = lvl + stepsize*(err)

            # Mostly like D2D rates are zero and we can stop
            if np.linalg.norm(lvl[:]) > 1e10:
                break

            if np.mod(itr, 100) == 0:
                self.logger.debug("err: %f, lvl: %f (%f), r1: %f, r2: %f " +
                                  "r3: %f, r4: %f", np.linalg.norm(err),
                                   np.linalg.norm(lvl[:]), stepsize,
                                   rates['D2B'][:].sum(), rates['B2D'][:].sum(),
                                   rates['D2D'][0][:].sum(),
                                   rates['D2D'][1][:].sum())

            itr += 1
            err_prev = err

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
        prec = np.dot(np.linalg.pinv(wcov + lvl*np.eye(self.n_tx)),
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
