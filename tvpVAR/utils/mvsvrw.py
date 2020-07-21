import numpy as np
from scipy.stats import norm
from scipy.linalg import block_diag
import numpy.linalg as lin
from typing import Tuple


def mvsvrw(y_star: np.ndarray, h: np.ndarray, iSig: np.ndarray, iVh: np.ndarray) -> np.ndarray:
    """
    This function simulates log-volatilities for a multivariate stochastic
    volatility model with independent random-walk transitions.
    :param y_star:
    :param h:
    :param iSig:
    :param iVh:
    :return h, S:
    """

    n = iSig.shape[0]
    tn = h.shape[0]

    # Normal mixture
    pi = np.array([0.0073, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.2575])
    mi = np.array([-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819]) - 1.2704  # means already adjusted!!
    sigi = np.array([5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261])
    sqrtsigi = np.sqrt(sigi)

    # Sample S from a 7-point discrete distribution
    temprand = np.random.rand(tn, 1)
    q = np.tile(pi, (tn, 1)) * norm.pdf(np.tile(y_star, (1, 7)), np.tile(h, (1, 7)) + np.tile(mi, (tn, 1)), np.tile(sqrtsigi, (tn, 1)))
    q = q / np.tile(np.sum(q, axis=1), (1, 7))
    S = 7 - np.sum(np.tile(temprand, (1, 7)) < np.cumsum(q, axis=1), axis=1) + 1

    # Sample h
    Hh = np.diag(-np.ones(tn-n), -n) + np.eye(tn)
    invSh = block_diag(iVh, np.kron(np.eye(tn / n - 1), iSig))
    dconst = mi[S].T
    invOmega = np.diag(1/sigi[S].T, 0)
    Kh = Hh.T @ invSh @ Hh
    Ph = Kh + invOmega
    Ch = np.linalg.cholesky(Ph)
    hhat = lin.lstsq(Ph, invOmega @ (y_star - dconst))[0]
    h = hhat + lin.lstsq(Ch, np.random.randn(tn, 1))

    return np.hstack((h, S))




