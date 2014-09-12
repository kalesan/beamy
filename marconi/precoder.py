"""This module provides various precoder designs."""
import itertools

import picos as pic

import numpy as np
import scipy
import scipy.linalg

import utils


class Precoder(object):
    """ This is the base class for all precoder design. The generator function
    should be overridden to comply with the corresponding design."""

    def __init__(self, sysparams):
        (self.n_rx, self.n_tx, self.n_ue, self.n_bs) = sysparams

        self.n_sk = min(self.n_tx, self.n_rx)

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


class PrecoderGaussian(Precoder):
    """ This a simple Gaussian precoder design, which generates random Gaussian
        precoder matrices that are normalized according to the given power
        constraints."""

    def generate(self, *_args, **kwargs):
        """ Generate a Gaussian precoder"""

        prec = np.random.randn(self.n_tx, self.n_sk, self.n_ue, self.n_bs) +  \
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

        for _ue in range(self.n_ue):
            for _bs in range(self.n_bs):
                weight[:, :, _ue, _bs] = np.linalg.inv(errm[:, :, _ue, _bs])

        return utils.weighted_bisection(chan, recv, weight, pwr_lim,
                                        threshold=1e-12)


class PrecoderSDP(Precoder):
    """ Joint transceiver beamformer design based on SDP reformulation and
        successive linear approximation of the original problem. """

    def __init__(self, sysparams):
        (self.n_rx, self.n_tx, self.n_ue, self.n_bs) = sysparams

        self.n_sk = min(self.n_tx, self.n_rx)

        #self.recv_prev = None

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

    def generate(self, *args, **kwargs):
        """ Generate the precoders. """

        [chan_glob, recv, prec_prev, noise_pwr] = [_a for _a in args]

        pwr_lim = kwargs.get('pwr_lim', 1)

        # MSE and weights
        errm = utils.mse(chan_glob, recv, prec_prev, noise_pwr)

        weights = np.zeros((self.n_sk, self.n_sk, self.n_ue, self.n_bs),
                           dtype='complex')

        for (_ue, _bs) in itertools.product(range(self.n_ue), range(self.n_bs)):
                weights[:, :, _ue, _bs] = np.linalg.inv(errm[:, :, _ue, _bs])

        # The final precoders
        prec = np.zeros((self.n_tx, self.n_sk, self.n_ue, self.n_bs),
                        dtype='complex')

        # Block diagonalize matrices
        recv = self.blkdiag(recv)
        #if self.recv_prev is None:
            #recv = self.blkdiag(recv)
            #self.recv_prev = recv.copy()
        #else:
            #recv = self.recv_prev.copy()

        weights = self.blkdiag(weights)

        for _bs in range(self.n_bs):
            # Composite channel
            chan = np.dsplit(chan_glob[:, :, :, _bs], self.n_ue)
            chan = np.squeeze(np.vstack(chan))
            #import ipdb; ipdb.set_trace()

            # Local receivers
            recvl = recv[:, :, _bs]

            for ind in range(self.n_ue*self.n_sk):
                weights[ind, ind, _bs] = np.real(weights[ind, ind, _bs])

            wsqrt = np.linalg.cholesky(weights[:, :, _bs])

            bounds = np.array([0.0, 10.0])

            P_ = np.Inf

            lvl = 2

            step = 1

            iter = 1

            while np.abs(P_ - pwr_lim) > 1e-6:
                #lvl = bounds.sum() / 2

                cov = np.dot(np.dot(chan, (1/lvl)*np.eye(self.n_tx)),
                             chan.conj().T)

                #######
                prob = pic.Problem()

                U = prob.add_variable('U', (self.n_rx*self.n_ue,
                                            self.n_sk*self.n_ue), 'complex')

                S = prob.add_variable('S', (self.n_sk*self.n_ue,
                                            self.n_sk*self.n_ue), 'hermitian')

                X = prob.add_variable('X', (self.n_sk*self.n_ue,
                                            self.n_sk*self.n_ue), 'hermitian')

                Xn = prob.add_variable('Xn', (self.n_rx*self.n_ue,
                                              self.n_rx*self.n_ue), 'hermitian')

                Z = pic.new_param('I', np.eye(self.n_sk*self.n_ue))

                I = pic.new_param('I', np.eye(self.n_rx*self.n_ue))

                W = pic.new_param('W', wsqrt)
                WI = pic.new_param('W', np.linalg.inv(weights[:, :, _bs]))

                objective = 'I' | X
                objective += noise_pwr*('I' | Xn)

                prob.set_objective('min', objective)

                prob.add_constraint(((Xn & U*W) // (W.H*U.H & I)) >> 0)
                prob.add_constraint(((X & Z) // (Z & (S + WI))) >> 0)

                C0 = pic.new_param('C0', np.dot(np.dot(recvl.conj().T, cov),
                                                recvl))
                U0 = pic.new_param('U0', np.dot(recvl.conj().T, cov))

                prob.add_constraint(C0 + U0*(U - recvl) +
                                    (U.H - recvl.conj().T)*U0.H >> S)

                Zi = np.kron(np.eye(self.n_ue), np.ones((self.n_rx, self.n_sk)))
                Zi = np.vstack(np.where(Zi == 0))
                Zi = [(Zi[0, k], Zi[1, k]) for k in range(Zi.shape[1])]

                #prob.add_list_of_constraints(
                    #[U.imag[i, j] == 0 for (i, j) in Zi])
                #prob.add_list_of_constraints(
                    #[U.real[i, j] == 0 for (i, j) in Zi])

                prob.solve(verbose=0, noduals=True, gaplim=1e-6, tol=1e-12,
                           harmonic_steps=3, step_sqp=3)

                U = np.asarray(np.matrix(U.value))

                #self.recv_prev[:, :, _bs] = U.copy()

                U = np.squeeze(U)

                wrecv = np.dot(np.dot(U, weights[:, :, _bs]), U.conj().T)
                wrecv = np.squeeze(wrecv)
                wcov = np.dot(np.dot(chan.conj().T, wrecv), chan)

                A = np.linalg.pinv(wcov + lvl*np.eye(self.n_tx))
                B = np.dot(np.dot(chan.conj().T, U),
                                  np.squeeze(weights[:, :, _bs]))

                tmp = np.dot(np.linalg.pinv(wcov + lvl*np.eye(self.n_tx)),
                             np.dot(np.dot(chan.conj().T, U),
                                    np.squeeze(weights[:, :, _bs])))

                tmp = tmp.reshape(self.n_tx, self.n_sk, self.n_ue, order='F')

                prec[:, :, :, _bs] = tmp.copy()

                P_ = np.linalg.norm(tmp[:])**2

                step = 1./(iter**0.4)

                iter += 1

                lvl = max(1e-10, lvl + step*(P_ - pwr_lim))

                print("lvl: %f P: %f : %f - %f" % (lvl, P_,
                                                   np.abs(P_ - pwr_lim), step))

                if P_ <= pwr_lim:
                    bounds[1] = lvl
                else:
                    bounds[0] = lvl

        return prec.copy()
