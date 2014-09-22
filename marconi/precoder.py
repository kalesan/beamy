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
        (self.n_rx, self.n_tx, self.n_ue, self.n_bs) = sysparams

        self.n_sk = min(self.n_tx, self.n_rx)

        self.logger = logging.getLogger(__name__)

        self.precision = precision

        self.uplink = uplink

        if uplink:
            n_tx = self.n_tx
            self.n_tx = self.n_rx
            self.n_rx = n_tx

            n_ue = self.n_ue
            self.n_ue = self.n_bs
            self.n_bs = n_ue

        # Initialize (reset) precoder state
        self.reset()

    def normalize(self, prec, pwr_lim):
        """ Normalize the prec matrix to have per BS power constraint pwr_lim"""
        for _bs in range(self.n_bs):
            tmp = prec[:, :, :, _bs]

            prec[:, :, :, _bs] = tmp / np.sqrt((tmp[:]*tmp[:].conj()).sum())
            prec[:, :, :, _bs] *= pwr_lim

        return prec

    def generate(self, *_args, **kwargs):
        """ This method should be overriden by the actual precoder design."""
        pass

    def reset(self):
        """ Resets the precoder state and parameters. """
        pass


class PrecoderGaussian(Precoder):
    """ This a simple Gaussian precoder design, which generates random Gaussian
        precoder matrices that are normalized according to the given power
        constraints."""

    def generate(self, *_args, **kwargs):
        """ Generate a Gaussian precoder"""

        prec = np.random.randn(self.n_tx, self.n_sk, self.n_ue, self.n_bs) + \
            np.random.randn(self.n_tx, self.n_sk, self.n_ue, self.n_bs)*1j

        return self.normalize(prec, kwargs.get('pwr_lim', 1))


class PrecoderWMMSE(Precoder):
    """Weighted minimum MSE (WMMSE) precoder design."""

    def generate(self, *args, **kwargs):
        """ Generate the WMMSE precoders. """

        [chan, recv, prec_prev, noise_pwr] = [_a for _a in args]

        pwr_lim = kwargs.get('pwr_lim', 1)

        errm = utils.mse(chan, recv, prec_prev, noise_pwr)

        weight = np.zeros((self.n_sk, self.n_sk, self.n_ue, self.n_bs),
                          dtype='complex')

        for (_ue, _bs) in itertools.product(range(self.n_ue), range(self.n_bs)):
            weight[:, :, _ue, _bs] = np.linalg.pinv(errm[:, :, _ue, _bs])

        return utils.weighted_bisection(chan, recv, weight, pwr_lim,
                                        threshold=self.precision)


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
        weights = np.zeros((self.n_sk, self.n_sk, self.n_ue, self.n_bs),
                           dtype='complex')

        errm = utils.mse(chan, recv, prec_prev, noise_pwr)

        for (_ue, _bs) in itertools.product(range(self.n_ue), range(self.n_bs)):
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

        return prec.reshape(self.n_tx, self.n_sk, self.n_ue, order='F')

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
        (n_rx, n_tx, n_ue, n_sk) = (self.n_rx, self.n_tx, self.n_bs, self.n_sk)

        cov = np.dot(np.dot(chan.conj().T, 1/self.noise_pwr * np.eye(n_rx)),
                     chan)

        prob = pic.Problem()

        popt = prob.add_variable('U', (n_tx*n_ue, n_sk*n_ue), 'complex')

        scov = prob.add_variable('S', (n_sk*n_ue, n_sk*n_ue), 'hermitian')

        scomp = prob.add_variable('X', (n_sk*n_ue, n_sk*n_ue), 'hermitian')

        eye_sk = pic.new_param('I', np.eye(n_sk*n_ue))

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
        zind = np.kron(np.eye(n_ue), np.ones((n_tx, n_sk)))
        zind = np.vstack(np.where(zind == 0))
        zind = [(zind[0, _ki], zind[1, _ki])
                for _ki in range(zind.shape[1])]

        prob.add_list_of_constraints(
            [popt.imag[_i, _j] == 0 for (_i, _j) in zind])
        prob.add_list_of_constraints(
            [popt.real[_i, _j] == 0 for (_i, _j) in zind])

        # Transmit power limit
        eye_k = pic.new_param('I', np.eye(n_tx))

        for _ue in range(n_ue):
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

        prec = np.zeros((n_tx, n_sk, 1, n_ue), dtype='complex')

        for _ue in range(n_ue):
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
        prec = np.zeros((self.n_tx, self.n_sk, self.n_ue, self.n_bs),
                        dtype='complex')

        # Block diagonalize matrices
        recv_prev = self.blkdiag(np.squeeze(recv_prev, axis=(2,)))
        prec_prev = self.blkdiag(np.squeeze(prec_prev, axis=(2,)))
        weights = self.blkdiag(np.squeeze(weights, axis=(2,)))

        chan = np.squeeze(chan_glob)
        chan = chan.reshape((self.n_rx, self.n_tx*self.n_bs),
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

    def reset(self):
        """ Resets the precoder state and parameters. """
        self.lvl = None

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
        weights = np.zeros((self.n_sk, self.n_sk, self.n_ue, self.n_bs),
                           dtype='complex')

        errm = utils.mse(chan, recv, prec_prev, noise_pwr)

        for (_ue, _bs) in itertools.product(range(self.n_ue), range(self.n_bs)):
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

        return prec.reshape(self.n_tx, self.n_sk, self.n_ue, order='F')

    def solve(self, chan, recv, weights, lvl, tol=1e-4, queue=None):
        # pylint: disable=R0914

        (n_rx, n_tx, n_ue, n_sk) = (self.n_rx, self.n_tx, self.n_ue, self.n_sk)

        cov = np.dot(np.dot(chan, (1/lvl)*np.eye(n_tx)), chan.conj().T)

        prob = pic.Problem()

        ropt = prob.add_variable('U', (n_rx*n_ue, n_sk*n_ue), 'complex')

        scov = prob.add_variable('S', (n_sk*n_ue, n_sk*n_ue), 'hermitian')

        scomp = prob.add_variable('X', (n_sk*n_ue, n_sk*n_ue), 'hermitian')

        ncomp = prob.add_variable('Y', (n_rx*n_ue, n_rx*n_ue), 'hermitian')

        eye_sk = pic.new_param('I', np.eye(n_sk*n_ue))
        eye_rx = pic.new_param('I', np.eye(n_rx*n_ue))

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
        zind = np.kron(np.eye(n_ue), np.ones((n_rx, n_sk)))
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

        if self.lvl is not None:
            # 10%-bounds around the previous point
            upper_bound = self.lvl*1.1
            lower_bound = self.lvl*0.9
            bounds = np.array([lower_bound, upper_bound])
        else:
            upper_bound = 10.
            lower_bound = 0.
            bounds = np.array([lower_bound, upper_bound])

        pnew = np.Inf

        tol = 1e-4

        itr = 1

        err = np.inf

        while err > self.precision:
            if method == "bisection":
                self.lvl = bounds.sum() / 2

            # Picos is sandbox into separate process to ensure proper memory
            # management.
            queue = Queue()
            p_sandbox = Process(target=self.solve,
                                args=(chan, recv, weights, self.lvl, tol, queue))
            # ropt = self.solve(chan, recv, weights, self.lvl, tol)
            p_sandbox.start()
            p_sandbox.join()
            ropt = queue.get()

            prec = self.precoder(chan, ropt, weights, self.lvl)

            # Compute power
            pnew = np.linalg.norm(prec[:])**2

            if method == "subgradient":
                self.lvl = max(1e-10, self.lvl + step/np.sqrt(itr)*(pnew - self.pwr_lim))
            else:
                if pnew > self.pwr_lim:
                    bounds[0] = self.lvl
                else:
                    bounds[1] = self.lvl

            err = np.abs(pnew - self.pwr_lim)

            self.logger.debug("%d: self.lvl: %f P: %f err: %f bnd: %f", itr, self.lvl,
                              pnew, err, np.abs(bounds[0] - bounds[1]))

            if np.abs(bounds[0] - bounds[1]) < 1e-10:
                if np.abs(upper_bound - bounds[1]) < 1e-9:
                    upper_bound *= 10
                    lower_bound /= 10
                    bounds = np.array([lower_bound, upper_bound])
                elif np.abs(lower_bound - bounds[0]) < 1e-9:
                    upper_bound *= 10
                    lower_bound /= 10
                    bounds = np.array([lower_bound, upper_bound])
                else:
                    tol *= 10
                    bounds = np.array([lower_bound, upper_bound])

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
        prec = np.zeros((self.n_tx, self.n_sk, self.n_ue, self.n_bs),
                        dtype='complex')

        # Block diagonalize matrices
        recv = self.blkdiag(recv)
        weights = self.blkdiag(weights)

        for _bs in range(self.n_bs):
            # Composite channel
            chan = np.dsplit(chan_glob[:, :, :, _bs], self.n_ue)
            chan = np.squeeze(np.vstack(chan))

            prec[:, :, :, _bs] = self.search(chan, recv[:, :, _bs],
                                             weights[:, :, _bs])

        return prec
