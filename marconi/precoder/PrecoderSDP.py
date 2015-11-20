import itertools
import logging
from multiprocessing import Process, Queue

import picos as pic

import numpy as np
import scipy
import scipy.linalg

import utils

from precoder import Precoder


class PrecoderSDP(Precoder):
    """ Joint transceiver beamformer design based on SDP reformulation and
        successive linear approximation of the original problem. """

    def __init__(self, sysparams, precision=1e-6,
                          solver_tolerance=1e-6, method='bisection'):

        Precoder.__init__(self, sysparams, precision=precision,
                solver_tolerance=solver_tolerance)

        self.method = method

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
                             (wsqrt.H*ropt.H & eye_sk)) >> 0)

        prob.add_constraint(((scomp & eye_sk) //
                             (eye_sk & (scov + winv))) >> 0)

        cpnt = pic.new_param('C0', np.dot(np.dot(recv.conj().T, cov), recv))

        reff = pic.new_param('U0', np.dot(recv.conj().T, cov))

        prob.add_constraint(cpnt + reff*(ropt - recv) +
                            (ropt.H - recv.conj().T)*reff.H == scov)

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
            upper_bound = self.lvl*2
            lower_bound = self.lvl/2
            bounds = np.array([lower_bound, upper_bound])
        else:
            upper_bound = 10.
            lower_bound = 0.
            bounds = np.array([lower_bound, upper_bound])

        pnew = np.Inf

        itr = 1

        err = np.inf

        if method == "fixed":
            v = np.trace(np.dot(np.dot(recv, weights), recv.conj().T))

            # Picos is sandbox into separate process to ensure proper memory
            # management.
            queue = Queue()
            p_sandbox = Process(target=self.solve,
                                args=(chan, recv, weights, v,
                                      self.solver_tolerance, queue))
            #ropt = self.solve(chan, recv, weights, self.lvl,
            #        self.solver_tolerance)
            p_sandbox.start()
            p_sandbox.join()
            ropt = queue.get()

            v = np.trace(np.dot(np.dot(ropt, weights), recv.conj().T))

            prec = self.precoder(chan, ropt, weights, v)

            ropt /= np.sqrt(np.linalg.norm(prec.flatten()))

            return (self.pwr_lim / np.linalg.norm(prec.flatten())) * prec;

        while (pnew >= self.pwr_lim) or (err > self.precision):
            if method == "bisection":
                self.lvl = bounds.sum() / 2

            # Picos is sandbox into separate process to ensure proper memory
            # management.
            queue = Queue()
            p_sandbox = Process(target=self.solve,
                                args=(chan, recv, weights, self.lvl,
                                    self.solver_tolerance, queue))
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
                                             weights[:, :, _bs],
                                             method=self.method)

        return prec
