""" This is a utility function module consisting of help functions that can be
used to compute varying signal properties. """

import itertools

import numpy as np


def sigcov(chan, prec, noise_pwr):
    """ Signal received covariance for given set of precoders. """

    (n_dx, n_bx, K, B) = chan['B2D'].shape
    (n_bx, n_sk, K, B) = prec['B2D'].shape

    cov = {}

    cov['BS'] = np.zeros((n_bx, n_bx, B), dtype='complex')

    cov['UE'] = np.array([None, None])
    cov['UE'][0] = np.zeros((n_dx, n_dx, K), dtype='complex')
    cov['UE'][1] = np.zeros((n_dx, n_dx, K), dtype='complex')

    # Concatenated precoders
    prec_cat = {}
    prec_cat['B2D'] = np.reshape(prec['B2D'], [n_bx, n_sk*K, B], order='F')
    prec_cat['D2B'] = np.reshape(prec['D2B'], [n_dx, n_sk*B, K], order='F')
    prec_cat['D2D'] = np.array([None, None])
    prec_cat['D2D'][0] = np.reshape(prec['D2D'][0], [n_dx, n_dx, K], order='F')
    prec_cat['D2D'][1] = np.reshape(prec['D2D'][1], [n_dx, n_dx, K], order='F')

    # Generate the covariance matrices

    # Slot 1
    for _bs in range(B):
        cov['BS'][:, :, _bs] = np.eye(n_bx)*noise_pwr['BS']

        for _ue_tran in range(K):
            _pc = np.dot(chan['D2B'][:, :, _bs, _ue_tran],
                         prec_cat['D2D'][0][:, :, _ue_tran])

            cov['BS'][:, :, _bs] += np.dot(_pc, _pc.conj().transpose())

            _pc = np.dot(chan['D2B'][:, :, _bs, _ue_tran],
                         prec_cat['D2B'][:, :, _ue_tran])

            cov['BS'][:, :, _bs] += np.dot(_pc, _pc.conj().transpose())

    for _ue in range(K):
        cov['UE'][0][:, :, _ue] = np.eye(n_dx)*noise_pwr['UE']

        for _ue_tran in range(K):
            _pc = np.dot(chan['D2D'][:, :, _ue, _ue_tran],
                         prec_cat['D2D'][0][:, :, _ue_tran])

            cov['UE'][0][:, :, _ue] += np.dot(_pc, _pc.conj().transpose())

            _pc = np.dot(chan['D2D'][:, :, _ue, _ue_tran],
                         prec_cat['D2B'][:, :, _ue_tran])

            cov['UE'][0][:, :, _ue] += np.dot(_pc, _pc.conj().transpose())

    # Slot 2
    for _ue in range(K):
        cov['UE'][1][:, :, _ue] = np.eye(n_dx)*noise_pwr['UE']

        for _bs in range(B):
            _pc = np.dot(chan['B2D'][:, :, _ue, _bs],
                         prec_cat['B2D'][:, :, _bs])

            cov['UE'][1][:, :, _ue] += np.dot(_pc, _pc.conj().transpose())

        for _ue_tran in range(K):
            _pc = np.dot(chan['D2D'][:, :, _ue, _ue_tran],
                         prec_cat['D2D'][1][:, :, _ue_tran])

            cov['UE'][1][:, :, _ue] += np.dot(_pc, _pc.conj().transpose())

    return cov


def lmmse(chan, prec, noise_pwr, cov=None):
    """ Generate linear MMSE receive beamformers. """

    (n_dx, n_bx, K, B) = chan['B2D'].shape
    (n_bx, n_sk, K, B) = prec['B2D'].shape

    recv = {}

    recv['B2D'] = np.zeros((n_dx, n_sk, K, B), dtype='complex')
    recv['D2B'] = np.zeros((n_bx, n_sk, B, K), dtype='complex')
    recv['D2D'] = [[], []]
    recv['D2D'][0] = np.zeros((n_dx, n_dx, K), dtype='complex')
    recv['D2D'][1] = np.zeros((n_dx, n_dx, K), dtype='complex')

    # Signal covariance matrix
    if cov is None:
        cov = sigcov(chan, prec, noise_pwr)

    # Slot 1
    for (_ue, _bs) in itertools.product(range(K), range(B)):
        recv['D2B'][:, :, _bs, _ue] = \
            np.dot(np.linalg.pinv(cov['BS'][:, :, _bs]),
                   np.dot(chan['D2B'][:, :, _bs, _ue],
                          prec['D2B'][:, :, _bs, _ue]))

    for _ue in range(K):
        recv['D2D'][0][:, :, _ue] = \
            np.dot(np.linalg.pinv(cov['UE'][0][:, :, _ue]),
                   np.dot(chan['D2D'][:, :, _ue, _ue],
                          prec['D2D'][0][:, :, _ue]))

    # Slot 2
    for (_ue, _bs) in itertools.product(range(K), range(B)):
        recv['B2D'][:, :, _ue, _bs] = \
            np.dot(np.linalg.pinv(cov['UE'][1][:, :, _ue]),
                   np.dot(chan['B2D'][:, :, _ue, _bs],
                          prec['B2D'][:, :, _ue, _bs]))

    for _ue in range(K):
        recv['D2D'][1][:, :, _ue] = \
            np.dot(np.linalg.pinv(cov['UE'][1][:, :, _ue]),
                   np.dot(chan['D2D'][:, :, _ue, _ue],
                          prec['D2D'][1][:, :, _ue]))

    return recv


def mse(chan, recv, prec, noise_pwr, cov=None):
    """ Compute the user specific mean-squared error matrices. """

    (n_dx, n_bx, K, B) = chan['B2D'].shape
    (n_bx, n_sk, K, B) = prec['B2D'].shape

    errm = {}

    errm['B2D'] = np.zeros((n_sk, n_sk, K, B), dtype='complex')
    errm['D2B'] = np.zeros((n_sk, n_sk, B, K), dtype='complex')
    errm['D2D'] = [[], []]
    errm['D2D'][0] = np.zeros((n_dx, n_dx, K), dtype='complex')
    errm['D2D'][1] = np.zeros((n_dx, n_dx, K), dtype='complex')

    # Concatenated precoders
    prec_cat = {}
    prec_cat['B2D'] = np.reshape(prec['B2D'], [n_bx, n_sk*K, B], order='F')
    prec_cat['D2B'] = np.reshape(prec['D2B'], [n_dx, n_sk*B, K], order='F')
    prec_cat['D2D'] = [[], []]
    prec_cat['D2D'][0] = np.reshape(prec['D2D'][0], [n_dx, n_dx, K], order='F')
    prec_cat['D2D'][1] = np.reshape(prec['D2D'][1], [n_dx, n_dx, K], order='F')

    # Signal covariance matrix
    if cov is None:
        cov = sigcov(chan, prec, noise_pwr)

    ## Slot 1 ##

    # Device to base station
    for (_ue, _bs) in itertools.product(range(K), range(B)):
        # Intended signal
        _tmp = np.dot(recv['D2B'][:, :, _bs, _ue].conj().T,
                        np.dot(chan['D2B'][:, :, _bs, _ue],
                               prec['D2B'][:, :, _bs, _ue]))

        errm['D2B'][:, :, _bs, _ue] = np.eye(n_sk) - _tmp - _tmp.conj().T
        errm['D2B'][:, :, _bs, _ue] += noise_pwr['BS'] * \
            np.dot(recv['D2B'][:, :, _bs, _ue].conj().T,
                   recv['D2B'][:, :, _bs, _ue])

        for _int_ue in range(K):
            _tmp = np.dot(recv['D2B'][:, :, _bs, _ue].conj().T,
                            np.dot(chan['D2B'][:, :, _bs, _int_ue],
                                   prec_cat['D2B'][:, :, _int_ue]))

            errm['D2B'][:, :, _bs, _ue] += np.dot(_tmp, _tmp.conj().T)

            _tmp = np.dot(recv['D2B'][:, :, _bs, _ue].conj().T,
                            np.dot(chan['D2B'][:, :, _bs, _int_ue],
                                   prec_cat['D2D'][0][:, :, _int_ue]))

            errm['D2B'][:, :, _bs, _ue] += np.dot(_tmp, _tmp.conj().T)

    # Device to device
    for _ue in range(K):
        # Intended signal
        _tmp = np.dot(recv['D2D'][0][:, :, _ue].conj().T,
                        np.dot(chan['D2D'][:, :, _ue, _ue],
                               prec['D2D'][0][:, :, _ue]))

        errm['D2D'][0][:, :, _ue] = np.eye(n_dx) - _tmp - _tmp.conj().T
        errm['D2D'][0][:, :, _ue] += noise_pwr['UE'] * \
            np.dot(recv['D2D'][0][:, :, _ue].conj().T,
                   recv['D2D'][0][:, :, _ue])

        for _int_ue in range(K):
            _tmp = np.dot(recv['D2D'][0][:, :, _ue].conj().T,
                            np.dot(chan['D2D'][:, :, _ue, _int_ue],
                                   prec_cat['D2D'][0][:, :, _int_ue]))

            errm['D2D'][0][:, :, _ue] += np.dot(_tmp, _tmp.conj().T)

    ## Slot 2 ##

    # Base station to device
    for (_ue, _bs) in itertools.product(range(K), range(B)):
        # Intended signal
        _tmp = np.dot(recv['B2D'][:, :, _ue, _bs].conj().T,
                        np.dot(chan['B2D'][:, :, _ue, _bs],
                               prec['B2D'][:, :, _ue, _bs]))

        errm['B2D'][:, :, _ue, _bs] = np.eye(n_sk) - _tmp - _tmp.conj().T
        errm['B2D'][:, :, _ue, _bs] += noise_pwr['UE'] * \
            np.dot(recv['B2D'][:, :, _ue, _bs].conj().T,
                   recv['B2D'][:, :, _ue, _bs])

        for _int_bs in range(B):
            _tmp = np.dot(recv['B2D'][:, :, _ue, _bs].conj().T,
                            np.dot(chan['B2D'][:, :, _ue, _int_bs],
                                   prec_cat['B2D'][:, :, _int_bs]))

            errm['B2D'][:, :, _ue, _bs] += np.dot(_tmp, _tmp.conj().T)

        for _int_ue in range(K):
            _tmp = np.dot(recv['B2D'][:, :, _ue, _bs].conj().T,
                            np.dot(chan['D2D'][:, :, _ue, _int_ue],
                                   prec_cat['D2D'][1][:, :, _int_ue]))

            errm['B2D'][:, :, _ue, _bs] += np.dot(_tmp, _tmp.conj().T)

    # Device to device
    for _ue in range(K):
        # Intended signal
        _tmp = np.dot(recv['D2D'][1][:, :, _ue].conj().T,
                        np.dot(chan['D2D'][:, :, _ue, _ue],
                               prec['D2D'][1][:, :, _ue]))

        errm['D2D'][1][:, :, _ue] = np.eye(n_dx) - _tmp - _tmp.conj().T
        errm['D2D'][1][:, :, _ue] += noise_pwr['UE'] * \
            np.dot(recv['D2D'][1][:, :, _ue].conj().T,
                   recv['D2D'][1][:, :, _ue])

        for _int_bs in range(B):
            _tmp = np.dot(recv['D2D'][1][:, :, _ue].conj().T,
                            np.dot(chan['B2D'][:, :, _ue, _int_bs],
                                   prec_cat['B2D'][:, :, _int_bs]))

            errm['D2D'][1][:, :, _ue] += np.dot(_tmp, _tmp.conj().T)

        for _int_ue in range(K):
            _tmp = np.dot(recv['D2D'][1][:, :, _ue].conj().T,
                            np.dot(chan['D2D'][:, :, _ue, _int_ue],
                                   prec_cat['D2D'][1][:, :, _int_ue]))

            errm['D2D'][1][:, :, _ue] += np.dot(_tmp, _tmp.conj().T)

    return errm


def rate(chan, prec, noise_pwr, cov=None, errm=None):
    """ Compute the user/BS pair specific rates.

        The rate is computed via the corresponding MSE assuming LMMSE receive
        beamformers.
    """

    (n_dx, n_bx, K, B) = chan['B2D'].shape
    (n_bx, n_sk, K, B) = prec['B2D'].shape

    rates = {}

    rates['B2D'] = np.zeros((K, B))
    rates['D2B'] = np.zeros((B, K))
    rates['D2D'] = [[], []]
    rates['D2D'][0] = np.zeros(K)
    rates['D2D'][1] = np.zeros(K)

    # Signal covariance matrix
    if cov is None:
        cov = sigcov(chan, prec, noise_pwr)

    recv = lmmse(chan, prec, noise_pwr, cov=cov)

    if errm is None:
        errm = mse(chan, recv, prec, noise_pwr, cov=cov)

    for (_ue, _bs) in itertools.product(range(K), range(B)):
        _tmp = -np.log2(np.linalg.det(errm['B2D'][:, :, _ue, _bs]))
        rates['B2D'][_ue, _bs] = np.real(_tmp)
        if np.real(_tmp) < 0:
            import ipdb; ipdb.set_trace()

        _tmp = -np.log2(np.linalg.det(errm['D2B'][:, :, _bs, _ue]))
        rates['D2B'][_bs, _ue] = np.real(_tmp)
        if np.real(_tmp) < 0:
            import ipdb; ipdb.set_trace()

    for _ue in range(K):
        _tmp = -np.log2(np.linalg.det(errm['D2D'][0][:, :, _ue]))
        rates['D2D'][0][_ue] = np.real(_tmp)
        if np.real(_tmp) < 0:
            import ipdb; ipdb.set_trace()

        _tmp = -np.log2(np.linalg.det(errm['D2D'][1][:, :, _ue]))
        rates['D2D'][1][_ue] = np.real(_tmp)
        if np.real(_tmp) < 0:
            import ipdb; ipdb.set_trace()

    return rates


def weighted_bisection1(chan, recv, weights, pwr_lim, threshold=1e-10):
    """ Utilize the weighted bisection algorithm to solve the weighted MSE
        minimizing transmit beamformers for slot 1 subject to given per-BS sum
        power constraint. """

    # System configuration
    (n_dx, n_bx, K, B) = chan['B2D'].shape
    n_sk = min(n_dx, n_bx)

    # The final precoders
    prec = {}
    prec['D2B'] = np.zeros((n_dx, n_sk*B, K), dtype='complex')
    prec['D2D'] = np.zeros((n_dx, n_dx, K), dtype='complex')

    # Weighted transmit covariance matrices
    wcov = np.zeros((n_dx, n_dx, K), dtype='complex')

    for (_ue, _int_ue, _bs) in itertools.product(range(K), range(K),
                                                 range(B)):

        _tmp = np.dot(np.dot(recv['D2B'][:, :, _bs, _int_ue],
                             weights['D2B'][:, :, _bs, _int_ue]),
                      recv['D2B'][:, :, _bs, _int_ue].conj().T)

        tmp_chan = chan['D2B'][:, :, _bs, _ue]
        wcov[:, :, _ue] += np.dot(np.dot(tmp_chan.conj().T, _tmp), tmp_chan)

    for (_ue, _ue_pri) in itertools.product(range(K), range(K)):

        _tmp = np.dot(np.dot(recv['D2D'][0][:, :, _ue_pri],
                             weights['D2D'][0][:, :, _ue_pri]),
                      recv['D2D'][0][:, :, _ue_pri].conj().T)

        tmp_chan = chan['D2D'][:, :, _ue_pri, _ue]
        wcov[:, :, _ue] += np.dot(np.dot(tmp_chan.conj().T, _tmp), tmp_chan)

    # Effective downlink channels
    wchan = {}
    wchan['D2B'] = np.zeros((n_dx, n_sk, B, K), dtype='complex')
    wchan['D2D'] = np.zeros((n_dx, n_dx, K), dtype='complex')

    for (_ue, _bs) in itertools.product(range(K), range(B)):
        tmp_chan = chan['D2B'][:, :, _bs, _ue]
        _wc = np.dot(np.dot(tmp_chan.conj().T, recv['D2B'][:, :, _bs, _ue]),
                     weights['D2B'][:, :, _bs, _ue])

        wchan['D2B'][:, :, _bs, _ue] = _wc

    for _ue in range(K):
        tmp_chan = chan['D2D'][:, :, _ue, _ue]
        _wc = np.dot(np.dot(tmp_chan.conj().T, recv['D2D'][0][:, :, _ue]),
                     weights['D2D'][0][:, :, _ue])

        wchan['D2D'][:, :, _ue] = _wc

    wchan['D2B'] = wchan['D2B'].reshape((n_dx, n_sk*B, K), order='F')

    # Perform the power bisection for each BS separately
    for _ue in range(K):
        prec['D2B'][:, :, _ue] = np.dot(np.linalg.pinv(wcov[:, :, _ue]),
                                        wchan['D2B'][:, :, _ue])
        prec['D2D'][:, :, _ue] = np.dot(np.linalg.pinv(wcov[:, :, _ue]),
                                        wchan['D2D'][:, :, _ue])

        pwr = np.linalg.norm(prec['D2B'][:, :, _ue][:])**2 + \
              np.linalg.norm(prec['D2D'][:, :, _ue][:])**2

        if pwr <= pwr_lim['UE']:
            continue

        upper_bound = 10.
        bounds = np.array([0, upper_bound])

        err = np.inf

        while err > threshold:
            lvl = (bounds.sum()) / 2

            _wcov = wcov[:, :, _ue] + np.eye(n_dx)*lvl

            prec['D2B'][:, :, _ue] = np.dot(np.linalg.pinv(_wcov),
                                            wchan['D2B'][:, :, _ue])
            prec['D2D'][:, :, _ue] = np.dot(np.linalg.pinv(_wcov),
                                            wchan['D2D'][:, :, _ue])

            pwr = np.linalg.norm(prec['D2B'][:, :, _ue][:])**2 + \
                  np.linalg.norm(prec['D2D'][:, :, _ue][:])**2

            err = np.linalg.norm(pwr - pwr_lim['UE'])**2

            if pwr < pwr_lim['UE']:
                bounds[1] = lvl
            else:
                bounds[0] = lvl

            # Re-adjust the boundaries if the upper limit seems to low
            if np.abs(upper_bound - bounds[1]) < 1e-8:
                upper_bound *= 10
                bounds = np.array([0, upper_bound])

    prec['D2B'] = prec['D2B'].reshape((n_dx, n_sk, B, K), order='F')
    # prec['D2D'] = prec['D2D'].reshape((n_dx, n_dx, K), order='F')

    return prec


def weighted_bisection2(chan, recv, weights, pwr_lim, threshold=1e-10):
    """ Utilize the weighted bisection algorithm to solve the weighted MSE
        minimizing transmit beamformers for slot 2 subject to given per-BS sum
        power constraint. """

    # System configuration
    (n_dx, n_bx, K, B) = chan['B2D'].shape
    n_sk = min(n_dx, n_bx)

    # The final precoders
    prec = {}
    prec['B2D'] = np.zeros((n_bx, n_sk*K, B), dtype='complex')
    prec['D2D'] = np.zeros((n_dx, n_dx, K), dtype='complex')

    # Weighted transmit covariance matrices
    wcov = {}
    wcov['BS'] = np.zeros((n_bx, n_bx, B), dtype='complex')
    wcov['UE'] = np.zeros((n_dx, n_dx, K), dtype='complex')


    # Base station to device
    for (_ue, _int_bs, _bs) in itertools.product(range(K), range(B), range(B)):

        _tmp = np.dot(np.dot(recv['B2D'][:, :, _ue, _int_bs],
                             weights['B2D'][:, :, _ue, _int_bs]),
                      recv['B2D'][:, :, _ue, _int_bs].conj().T)

        tmp_chan = chan['B2D'][:, :, _ue, _bs]
        wcov['BS'][:, :, _bs] += np.dot(np.dot(tmp_chan.conj().T, _tmp),
                                        tmp_chan)

    for (_ue, _bs) in itertools.product(range(K), range(B)):

        _tmp = np.dot(np.dot(recv['D2D'][1][:, :, _ue],
                             weights['D2D'][1][:, :, _ue]),
                      recv['D2D'][1][:, :, _ue].conj().T)

        tmp_chan = chan['B2D'][:, :, _ue, _bs]
        wcov['BS'][:, :, _bs] += np.dot(np.dot(tmp_chan.conj().T, _tmp),
                                        tmp_chan)

    # Device to device
    for (_ue, _ue_pri) in itertools.product(range(K), range(K)):

        _tmp = np.dot(np.dot(recv['D2D'][1][:, :, _ue_pri],
                             weights['D2D'][1][:, :, _ue_pri]),
                      recv['D2D'][1][:, :, _ue_pri].conj().T)

        tmp_chan = chan['D2D'][:, :, _ue_pri, _ue]
        wcov['UE'][:, :, _ue] += np.dot(np.dot(tmp_chan.conj().T, _tmp),
                                        tmp_chan)

    for (_ue_pri, _bs, _ue) in itertools.product(range(K), range(B),
                                                 range(K)):

        _tmp = np.dot(np.dot(recv['B2D'][:, :, _ue_pri, _bs],
                             weights['B2D'][:, :, _ue_pri, _bs]),
                      recv['B2D'][:, :, _ue_pri, _bs].conj().T)

        tmp_chan = chan['D2D'][:, :, _ue_pri, _ue]
        wcov['UE'][:, :, _ue] += np.dot(np.dot(tmp_chan.conj().T, _tmp),
                                        tmp_chan)

    # Effective downlink channels
    wchan = {}
    wchan['B2D'] = np.zeros((n_bx, n_sk, K, B), dtype='complex')
    wchan['D2D'] = np.zeros((n_dx, n_dx, K), dtype='complex')

    for (_ue, _bs) in itertools.product(range(K), range(B)):
        tmp_chan = chan['B2D'][:, :, _ue, _bs]
        _wc = np.dot(np.dot(tmp_chan.conj().T, recv['B2D'][:, :, _ue, _bs]),
                     weights['B2D'][:, :, _ue, _bs])

        wchan['B2D'][:, :, _ue, _bs] = _wc

    for _ue in range(K):
        tmp_chan = chan['D2D'][:, :, _ue, _ue]
        _wc = np.dot(np.dot(tmp_chan.conj().T, recv['D2D'][1][:, :, _ue]),
                     weights['D2D'][1][:, :, _ue])

        wchan['D2D'][:, :, _ue] = _wc

    wchan['B2D'] = wchan['B2D'].reshape((n_bx, n_sk*K, B), order='F')

    # Perform the power bisection for each BS separately
    for _bs in range(B):
        prec['B2D'][:, :, _bs] = np.dot(np.linalg.pinv(wcov['BS'][:, :, _bs]),
                                        wchan['B2D'][:, :, _bs])

        pwr = np.linalg.norm(prec['B2D'][:, :, _bs][:])**2

        if pwr <= pwr_lim['BS']:
            continue

        upper_bound = 10.
        bounds = np.array([0, upper_bound])

        err = np.inf

        while err > threshold:
            lvl = (bounds.sum()) / 2

            _wcov = wcov['BS'][:, :, _bs] + np.eye(n_bx)*lvl

            prec['B2D'][:, :, _bs] = np.linalg.solve(_wcov,
                                                     wchan['B2D'][:, :, _bs])

            pwr = np.linalg.norm(prec['B2D'][:, :, _bs][:])**2

            err = np.abs(pwr - pwr_lim['BS'])

            if pwr < pwr_lim['BS']:
                bounds[1] = lvl
            else:
                bounds[0] = lvl

            # Re-adjust the boundaries if the upper limit seems to low
            if np.abs(upper_bound - bounds[1]) < 1e-8:
                upper_bound *= 10
                bounds = np.array([0, upper_bound])

    # Perform the power bisection for each BS separately
    for _ue in range(K):
        prec['D2D'][:, :, _ue] = np.linalg.solve(wcov['UE'][:, :, _ue],
                                                 wchan['D2D'][:, :, _ue])

        pwr = np.linalg.norm(prec['D2D'][:, :, _ue][:])**2

        if pwr <= pwr_lim['UE']:
            continue

        upper_bound = 10.
        bounds = np.array([0, upper_bound])

        err = np.inf

        while err > threshold:
            lvl = (bounds.sum()) / 2

            _wcov = wcov['UE'][:, :, _ue] + np.eye(n_dx)*lvl

            prec['D2D'][:, :, _ue] = np.dot(np.linalg.pinv(_wcov),
                                            wchan['D2D'][:, :, _ue])

            pwr = np.linalg.norm(prec['D2D'][:, :, _ue][:])**2

            err = np.abs(pwr - pwr_lim['UE'])

            if pwr < pwr_lim['UE']:
                bounds[1] = lvl
            else:
                bounds[0] = lvl

            # Re-adjust the boundaries if the upper limit seems to low
            if np.abs(upper_bound - bounds[1]) < 1e-8:
                upper_bound *= 10
                bounds = np.array([0, upper_bound])

    prec['B2D'] = prec['B2D'].reshape((n_bx, n_sk, K, B), order='F')
    # prec['D2D'] = prec['D2D'].reshape((n_dx, n_dx, K), order='F')

    return prec


def weighted_bisection(chan, recv, weights, pwr_lim, threshold=1e-6):
    """ Utilize the weighted bisection algorithm to solve the weighted MSE
    minimizing transmit beamformers subject to given sum power constraints. """

    prec = {}
    prec['D2D'] = [[], []]

    prec1 = weighted_bisection1(chan, recv, weights, pwr_lim,
                                threshold=threshold)
    prec2 = weighted_bisection2(chan, recv, weights, pwr_lim,
                                threshold=threshold)

    prec['D2B'] = prec1['D2B']
    prec['B2D'] = prec2['B2D']
    prec['D2D'][0] = prec1['D2D']
    prec['D2D'][1] = prec2['D2D']

    return prec
