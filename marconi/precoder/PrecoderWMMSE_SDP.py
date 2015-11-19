import itertools
import logging
from multiprocessing import Process, Queue

import picos as pic

import numpy as np
import scipy
import scipy.linalg

import utils

from precoder import Precoder


class PrecoderWMMSE_SDP(Precoder):
    """ WMMSE Algorith using SDP formulation that allows the use of any convex
        power constraints."""

    def __init__(self, sysparams, precision=1e-6,
                          solver_tolerance=1e-6, perantenna_power=False):

        Precoder.__init__(self, sysparams, precision=precision,
                solver_tolerance=solver_tolerance)

        self.perantenna_power = perantenna_power


    def solve(self, recv, chan, weights, pwr_lim, tol=1e-6, queue=None):
        """Solve the WMMSE transmit beamformers. 

        Args:
            chan (complex array): Channel matrix.
            weights (complex array): MSE weights.
            pwr_lim (double): Sum transmit power limit
            noise_pwr (double): The noise power

        Kwargs:
            tol (double): Solver solution toleranse.
            queue (bool): If not None, pass the return params through the queue.

        Returns: The transmit beamformer array.

        """
        # pylint: disable=R0914
        (n_rx, n_tx, n_bs, n_sk, n_ue) = (self.n_rx, self.n_tx, self.n_bs,
                self.n_sk, self.n_ue)

        prob = pic.Problem()

        mse = np.ndarray((n_ue, n_bs), object)
        for (k, b) in itertools.product(range(n_ue), range(n_bs)):
            mse[k, b] = prob.add_variable('E(%d,%d)' % (k, b), (n_sk, n_sk),
                    'hermitian')

        popt = np.ndarray((n_ue, n_bs), object)
        for (k, b) in itertools.product(range(n_ue), range(n_bs)):
            popt[k, b] = prob.add_variable('M(%d,%d)' % (k, b), (n_tx, n_sk),
                    'complex')

        qvar = np.ndarray((n_ue, n_bs), object)
        for (k, b) in itertools.product(range(n_ue), range(n_bs)):
            qvar[k, b] = prob.add_variable('Q(%d,%d)' % (k, b), (n_tx, n_tx),
                    'hermitian')

        eye_sk = pic.new_param('I', np.eye(n_sk))

        W = np.ndarray((n_ue, n_bs), object);
        for (k, b) in itertools.product(range(n_ue), range(n_bs)):
            W[k, b] = pic.new_param('W(%d,%d)' % (k, b), weights[:, :, k, b])

        U = np.ndarray((n_ue, n_bs), object);
        for (k, b) in itertools.product(range(n_ue), range(n_bs)):
            U[k, b] = pic.new_param('U(%d,%d)' % (k, b), recv[:, :, k, b])

        H = np.ndarray((n_ue, n_bs), object);
        for (k, b) in itertools.product(range(n_ue), range(n_bs)):
            H[k, b] = pic.new_param('H(%d,%d)' % (k, b), chan[:, :, k, b])

        # Objective
        objective = 0
        for (k, b) in itertools.product(range(n_ue), range(n_bs)):
            objective += W[k, b] * mse[k, b]

        prob.set_objective('min', 'I' | (objective))

        # MSE Constraints
        for (k, b) in itertools.product(range(n_ue), range(n_bs)):
            E = eye_sk - U[k, b].H * H[k, b] * popt[k, b] - \
                         popt[k, b].H * H[k, b].H * U[k, b]

            for (i, j) in itertools.product(range(n_ue), range(n_bs)):
                E += U[k, b].H * H[k, j] * qvar[i, j] * H[k, j].H * U[k, b]

            prob.add_constraint(mse[k, b] - E >> 0)

            prob.add_constraint(((eye_sk & popt[k, b].H) // 
                                 (popt[k, b] & qvar[k, b])) >> 0)

        # Power Constraint
        if self.perantenna_power:
            Pwr = np.ndarray((n_tx, n_ue, n_bs), pic.AffinExp)

            for (t, k, b) in itertools.product(range(n_tx), range(n_ue), 
                                            range(n_bs)):
                p = prob.add_variable('p(%d,%d,%d)' % (t, k, b))
                prob.add_constraint(abs(popt[k,b][t,:])**2 <= p) 

                Pwr[t, k, b] = p

            for (t, b) in itertools.product(range(n_tx), range(n_bs)):
                    prob.add_constraint(
                            pic.sum([Pwr[t, k, b] for k in range(n_ue)]) 
                            <= pwr_lim/n_tx)

        else:
            P = prob.add_variable('P', (n_ue, n_bs))
            for (k, b) in itertools.product(range(n_ue), range(n_bs)):
                prob.add_constraint(abs(popt[k, b])**2 <= P[k, b])

            for b in range(n_bs):
                prob.add_constraint(pic.sum([P[k, b] for k in range(n_ue)]) 
                        <= pwr_lim)

        # Solve the problem
        prob.solve(solver='cvxopt', verbose=False, noduals=True, tol=tol, 
                solve_via_dual=False)

        prec = np.zeros((n_tx, n_sk, n_ue, n_bs), dtype='complex')

        for (k, b) in itertools.product(range(n_ue), range(n_bs)):
            prec[:, :, k, b] = np.asarray(np.matrix(popt[k, b].value))

        if queue is None:
            return prec
        else:
            queue.put(prec)

    def generate(self, *args, **kwargs):
        """ Generate the WMMSE precoders. """

        [chan, recv, prec_prev, noise_pwr] = [_a for _a in args]

        pwr_lim = kwargs.get('pwr_lim', 1)

        errm = utils.mse(chan, recv, prec_prev, noise_pwr)

        weights = np.zeros((self.n_sk, self.n_sk, self.n_ue, self.n_bs),
                          dtype='complex')

        for (_ue, _bs) in itertools.product(range(self.n_ue), range(self.n_bs)):
            weights[:, :, _ue, _bs] = np.linalg.pinv(errm[:, :, _ue, _bs])

        # Picos is sandboxed into separate process to ensure proper memory
        # management.
        # queue = Queue()

        # p_sandbox = Process(target=self.solve,
                            # args=(recv, chan, weights, pwr_lim,
                                # self.solver_tolerance, queue))

        # p_sandbox.start()
        # p_sandbox.join()
        # prec = queue.get()
        prec = self.solve(recv, chan, weights, pwr_lim)

        return prec #queue.get()
