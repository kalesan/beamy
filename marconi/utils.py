""" This is a utility function module consisting of help functions that can be
used to compute varying signal properties. """

import itertools

import numpy as np


def sigcov(chan, prec, noise_pwr):
    """ Signal received covariance for given set of precoders. """

    (n_rx, n_tx, n_ue, n_bs) = chan.shape
    (n_tx, n_sk, n_ue, n_bs) = prec.shape

    cov = np.zeros((n_rx, n_rx, n_ue), dtype='complex')

    # Concatenated precoders
    prec_cat = np.reshape(prec, [n_tx, n_sk*n_ue, n_bs], order='F')

    # Generate the covariance matrices
    for _ue in range(n_ue):
        cov[:, :, _ue] = np.eye(n_rx)*noise_pwr

        for _bs in range(n_bs):
            _pc = np.dot(chan[:, :, _ue, _bs], prec_cat[:, :, _bs])
            cov[:, :, _ue] += np.dot(_pc, _pc.conj().transpose())

    return cov


def lmmse(chan, prec, noise_pwr, cov=None):
    """ Generate linear MMSE receive beamformers. """

    n_rx = chan.shape[0]
    (n_sk, n_ue, n_bs) = prec.shape[1:]

    recv = np.zeros((n_rx, n_sk, n_ue, n_bs), dtype='complex')

    # Signal covariance matrix
    if cov is None:
        cov = sigcov(chan, prec, noise_pwr)

    for _ue in range(n_ue):
        for _bs in range(n_bs):
            recv[:, :, _ue, _bs] = \
                np.linalg.solve(cov[:, :, _ue], np.dot(chan[:, :, _ue, _bs],
                                                       prec[:, :, _ue, _bs]))

    return recv


def mse(chan, recv, prec, noise_pwr, cov=None):
    """ Compute the user specific mean-squared error matrices. """

    (n_tx, n_ue, n_bs) = chan.shape[1:]
    (n_tx, n_sk, n_ue, n_bs) = prec.shape

    errm = np.zeros((n_sk, n_sk, n_ue, n_bs), dtype='complex')

    prec_cat = np.reshape(prec, [n_tx, n_sk*n_ue, n_bs], order='F')

    # Signal covariance matrix
    if cov is None:
        cov = sigcov(chan, prec, noise_pwr)

    for _ue in range(n_ue):
        for _bs in range(n_bs):
            _tmp = np.dot(recv[:, :, _ue, _bs].conj().T,
                          np.dot(chan[:, :, _ue, _bs], prec[:, :, _ue, _bs]))

            errm[:, :, _ue, _bs] = np.eye(n_sk) - _tmp - _tmp.conj().T
            errm[:, :, _ue, _bs] += noise_pwr * \
                np.dot(recv[:, :, _ue, _bs].conj().T, recv[:, :, _ue, _bs])

            for _int_bs in range(n_bs):
                _tmp = np.dot(recv[:, :, _ue, _bs].conj().T,
                              np.dot(chan[:, :, _ue, _int_bs],
                                     prec_cat[:, :, _int_bs]))

                errm[:, :, _ue, _bs] += np.dot(_tmp, _tmp.conj().T)

    return errm


def rate(chan, prec, noise_pwr, cov=None, errm=None):
    """ Compute the user/BS pair specific rates.

        The rate is computed via the corresponding MSE assuming LMMSE receive
        beamformers.
    """

    (n_ue, n_bs) = chan.shape[2:]

    # Signal covariance matrix
    if cov is None:
        cov = sigcov(chan, prec, noise_pwr)

    recv = lmmse(chan, prec, noise_pwr, cov=cov)

    if errm is None:
        errm = mse(chan, recv, prec, noise_pwr, cov=cov)

    rates = np.zeros((n_ue, n_bs))

    for _ue in range(n_ue):
        for _bs in range(n_bs):
            _tmp = -np.log2(np.linalg.det(errm[:, :, _ue, _bs]))
            rates[_ue, _bs] = np.real(_tmp)

    return rates


def weighted_bisection(chan, recv, weights, pwr_lim, threshold=1e-6):
    """ Utilize the weighted bisection algorithm to solve the weighted MSE
        minimizing transmit beamformers subject to given per-BS sum power
        constraint. """

    # System configuration
    cfg = {'TX': chan.shape[1], 'UE': chan.shape[2], 'BS': chan.shape[3],
           'SK': recv.shape[1]}

    # The final precoders
    prec = np.zeros((cfg['TX'], cfg['SK']*cfg['UE'], cfg['BS']),
                    dtype='complex')

    # Weighted transmit covariance matrices
    wcov = np.zeros((cfg['TX'], cfg['TX'], cfg['BS']), dtype='complex')

    for (_bs, _int_bs, _ue) in itertools.product(range(cfg['BS']),
                                                 range(cfg['BS']),
                                                 range(cfg['UE'])):

        _tmp = np.dot(np.dot(recv[:, :, _ue, _int_bs],
                             weights[:, :, _ue, _int_bs]),
                      recv[:, :, _ue, _int_bs].conj().T)

        wcov[:, :, _bs] += np.dot(np.dot(chan[:, :, _ue, _bs].conj().T,
                                         _tmp), chan[:, :, _ue, _bs])

    # Effective downlink channels
    wchan = np.zeros((cfg['TX'], cfg['SK'], cfg['UE'], cfg['BS']),
                     dtype='complex')

    for (_ue, _bs) in itertools.product(range(cfg['UE']), range(cfg['BS'])):
        wchan[:, :, _ue, _bs] = np.dot(np.dot(chan[:, :, _ue, _bs].conj().T,
                                              recv[:, :, _ue, _bs]),
                                       weights[:, :, _ue, _bs])

    wchan = np.reshape(wchan, [cfg['TX'], cfg['SK']*cfg['UE'], cfg['BS']],
                       order='F')

    # Perform the power bisection for each BS separately
    for _bs in range(cfg['BS']):
        prec[:, :, _bs] = np.dot(np.linalg.pinv(wcov[:, :, _bs]), wchan[:, :, _bs])

        if np.linalg.norm(prec[:, :, _bs][:]) <= np.sqrt(pwr_lim):
            continue

        bounds = np.array([0, 10.])

        err = np.inf

        while err > threshold:
            lvl = (bounds.sum()) / 2

            prec[:, :, _bs] = np.linalg.solve((wcov[:, :, _bs] +
                                               np.eye(cfg['TX'])*lvl),
                                              wchan[:, :, _bs])

            err = np.abs(np.linalg.norm(prec[:, :, _bs][:]) - np.sqrt(pwr_lim))

            if np.linalg.norm(prec[:, :, _bs][:]) < np.sqrt(pwr_lim):
                bounds[1] = lvl
            else:
                bounds[0] = lvl

            if lvl < 1e-12:
                break

            # Re-adjust the boundaries if the upper limit seems to low
            if np.abs(bounds[0] - bounds[1]) < 1e-14:
                bounds[1] *= 10

    return np.reshape(prec, [cfg['TX'], cfg['SK'], cfg['UE'], cfg['BS']],
                      order='F')
