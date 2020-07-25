import numpy as np
import timeit
from tqdm import tqdm
import numpy.linalg as lin
from tvpVAR.utils.solve_struct import solve_struct


def ir_vecm_sv(beta, sigma, cidx, t_start, s, a1, p, d, cum_ir=False, tax_first=True):
    """
    This function computes impulses responses for the VECM models with
    stochastic volatility. Structural parameters are recovered from the
    estimated parameters using the procedure described in the appendix.
    :param beta:
    :param sigma:
    :param cidx:
    :param t_start:
    :param s:
    :param al:
    :param p:
    :param d:
    :param cum_ir:
    :param tax_first:
    :return:
    """
    # Allocate space
    nsims = sigma.shape[2]
    n = sigma.shape[0]
    k = int((beta.shape[0] - 0.5 * n * (n-1)) / n)
    bidx = np.array(1-cidx, dtype=bool)

    # For constructing the contemporaneus coefficients
    b0t = np.eye(n)  # this is actually B0_t.T
    ltidx = np.array(np.triu(np.ones((n, n))) - np.eye(n), dtype = bool)

    ab2 = np.zeros((1, nsims))
    c = np.zeros((2, nsims))
    svars = np.zeros((3, nsims))
    err = np.zeros((1, nsims))
    ir = np.zeros((n, n, 1 + s, nsims))

    print('Simulating impulse responses...')

    start = timeit.default_timer()
    for i in tqdm(range(nsims)):
        # Decompose the reduced form cov matrix using identification scheme in BP2002 - to be changed
        # to follow Primiceri (2005)
        psi = -np.vstack((a1[0], 0))
        b0t = b0t.T
        b0t[ltidx.T] = beta[cidx.ravel(), t_start, i]
        b0t = b0t.T
        qt = b0t @ np.diag(1 / (sigma[:, t_start, i]).ravel()) @ b0t.T
        [kap, neg_c, delta, xi, s2_3, err[0, i]] = solve_struct(qt, psi, tax_first)

        # save draws of structural params and variances
        ab2[0, i] = delta[0, 1] + delta[1, 0]  # note: either delta 01 or delta 10 is 0
        c[:, i] = -neg_c
        svars[:, i] = np.vstack((np.diag(xi).reshape((-1,1)), s2_3.reshape((-1, 1)))).ravel()

        # compute impulse response for all combinations
        m = d.shape[0]  # the cointegration dim
        ikap = lin.inv(kap)  # note: this is just 3x3
        normkap = ikap @ np.diag((1 / np.diag(ikap)))  # normalize shocks
        f = np.eye(n * (p + 1))  # note: VECM induces an extra lag in the VAR, for VAR p+1 to be replaced by p
        ir[:, :, 0, i] = normkap  # simultaneous response
        for dt in range(s):
            b0t = b0t.T
            b0t[ltidx.T] = beta[cidx.ravel(), t_start + dt, i]
            b0t = b0t.T
            bt = (np.reshape(beta[bidx.ravel(), t_start + dt, i], (k, n), order='F') @ lin.inv(b0t)).T
            ct = np.hstack((bt[:, k - p * n:], np.zeros((n, n))))
            ct = ct - np.hstack((-(np.eye(n) +bt[:, k - p * n - m:k - p * n ] @ d), bt[:, k - p * n:]))
            f = np.vstack((ct, np.hstack((np.eye(n * p), np.zeros((n * p, n)))))) @ f
            ir[:, :, 1 + dt, i] = cum_ir * ir[:, :, dt, i] + f[:n, :n] @ normkap

    stop = timeit.default_timer()
    print('Impulse Response simulation completed adter', stop - start)

    return ab2, c, svars, ir, err


