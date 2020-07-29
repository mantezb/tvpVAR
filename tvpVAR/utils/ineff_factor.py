import numpy as np
from typing import Tuple
from statsmodels.tsa.stattools import acf
import warnings


def ineff_factor1(x, lag):

    if lag == 0:
        lag = np.max([get_lag(x), 20])
    r, _ = _get_acf(x, nlags=lag)
    ef1 = 1 + 2 * np.sum(r[1:], axis=0)

    return ef1, lag


def acorr_simple(x, lag):

    r = np.zeros((lag, 1))

    for i in range(lag):
        y = x[:(-i - 1)]
        z = x[(i + 1):]
        r[i] = np.mean((y-np.mean(x, axis=0)) * (z - np.mean(z - np.mean(x, axis=0))), axis=0)

    r = np.vstack((1, r / np.var(x, axis=0)))

    return r


def _get_acf(x, alpha=0.05, nlags=None, unbiased=False, mode='matlab'):
    n = len(x)
    if nlags:
        r, b = acf(x, alpha=alpha, nlags=nlags, unbiased=unbiased)
    elif n < 41:
        r, b = acf(x, alpha=alpha, nlags=n - 1, unbiased=unbiased)
    else:
        r, b = acf(x, alpha=alpha, unbiased=unbiased)

    if mode == 'matlab':
        b = [2 / np.sqrt(n), -2 / np.sqrt(n)]

    return r, b


def get_lag(x):
    lag = 0
    n = len(x)
    r, b = _get_acf(x, alpha=0.05, nlags=n-1, mode='matlab')
    lag_new = np.min(np.argwhere(np.abs(r) < b[0]))

    i = 0
    while (i <= 100) and (lag != lag_new):
        lag = lag_new
        r, b = _get_acf(x, alpha=0.05, nlags=n - 1, mode='matlab')
        lag_new = np.min(np.argwhere(np.abs(r) < b[0]))
        i = i + 1

    if lag != lag_new:
        warnings.warn('Could not find appropriate truncation lag. Lag will be set to 0.', UserWarning)
        lag = 0

    return lag


def ineff_factor(x: np.ndarray, lag=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function computes the inefficiency factors for a set of MCMC chains.
    If x is a two-dimensional matrix, it is assumed that each row contains
    a separate chain.  If the lag cut-off L is not provided, an attempt will
    be made to caculate it based on an auto-correlation "tappering-off"
    rule-of-thumb. In this case, it will also be returned as an additional
    output variable.
    :param x: a matrix where each row contains a seperate chain
    :param l: lag cut-off
    :return:
    """
    n = x.shape[0]
    ef = np.zeros((n, 1))

    if lag is None:
        lag = np.zeros((n, 1))

    for i in range(n):
        ef[i], lag[i] = ineff_factor1(x[i, :], lag[i])

    return ef, lag

