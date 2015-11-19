import itertools
import logging
from multiprocessing import Process, Queue

import picos as pic

import numpy as np
import scipy
import scipy.linalg

import utils

from precoder import Precoder


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

        prob.add_constraint(peff*popt + popt.H*peff.H - cpnt == scov)

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
            _r0 = (_ue)*n_tx
            _r1 = (_ue+1)*n_tx

            _c0 = _ue*n_sk
            _c1 = (_ue+1)*n_sk

            X = popt[_r0:_r1, _c0:_c1]

            prob.add_constraint(abs(X) <= np.sqrt(pwr_lim))

        # Solve the problem
        prob.solve(verbose=False, noduals=True, tol=tol, solve_via_dual=False)

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

        # Picos is sandboxed into separate process to ensure proper memory
        # management.
        queue = Queue()

        p_sandbox = Process(target=self.solve,
                            args=(chan, prec_prev, weights, pwr_lim,
                                self.solver_tolerance, queue))

        # prec = self.solve(chan, prec_prev, weights, pwr_lim)
        p_sandbox.start()
        p_sandbox.join()
        prec = queue.get()

        return prec
